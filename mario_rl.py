import warnings
from _queue import Empty
warnings.filterwarnings("ignore")

import random
import time
from time import sleep
import os
from multiprocessing import Queue, Event, Lock
from multiprocessing import Process
import torch.multiprocessing as mp

import torch
from collections import deque

from environment import create_env
from dqn_agent import DQN, SharedReplayBuffer, MarioAgent, DEVICE

mp.set_start_method('spawn', force=True)

# maximum size of the queue where collector processes store replays,
# the limit is for when the collector threads outpace the main thread
REP_Q_SIZE = 20000

# the size of the replay buffer, where the agent stores its memories,
# bigger memory -> old replays stay longer in memory -> more stable gradient updates
BUFFER_SIZE = 200000

# on how many epochs we want to train, this is basically forever
NUM_EPOCHS = 10000

# if an agent does not improve (x-position) for this amount of steps, the run gets canceled
DEADLOCK_STEPS = 25

# the amount of steps a run can last at max
MAX_STEPS_PER_RUN = 1200

# the batch size for the agents policy training
BATCH_SIZE = 256

# the amount of batches we train per epoch
EPISODES_PER_EPOCH = 40

# interval at which the model will be saved
SAVE_INTERVAL = 10

# how much epsilon decays each training epoch, high epsilon means high chance to randomly explore the environment
EPSILON_DECAY = 0.001

EPSILON_MIN = 0.01

# gamma describes how much the agent should look for future rewards vs immediate ones.
# gamma = 1 future rewards are as valuable as immediate ones
# gamma = 0 only immediate rewards matter
GAMMA = 0.9

def save_checkpoint(agent, epoch, checkpoint_dir='checkpoints'):
    """Save the agent's state and training progress."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Clear CUDA cache before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Only save essential data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': agent.q_network.state_dict(),
        'epsilon': agent.epsilon,
        # Don't save the entire memory buffer to reduce memory usage
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'mario_agent_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


# returns the current checkpoint
def load_checkpoint(agent, checkpoint_path):
    """Load the agent's state and training progress."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0  # Return initial epoch

    checkpoint = torch.load(checkpoint_path)

    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = checkpoint['epsilon']

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('mario_agent_epoch_')]
    if not checkpoints:
        return None

    # Extract epoch numbers and find the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


# this function will be run by each collector process
def collector_process(experience_queue, model_queue, stop_event, epsilon, id, queue_lock):
    env = create_env()
    print(f"[Collector {id}] Environment created successfully")
    
    # Initialize model with minimal memory footprint
    with torch.no_grad():
        local_model = DQN((128, 128), env.action_space.n).to(DEVICE)
        local_model.eval()  # Set to evaluation mode to reduce memory usage
    print(f"[Collector {id}] Local model initialized")

    # Initialize reward tracking
    import time
    recent_rewards = deque(maxlen=100)  # Store last 100 episode rewards
    last_print_time = time.time()
    print_interval = 20  # Print every 10 seconds

    # respond to stop event from parent process
    episode_count = 0
    best_reward = 0
    while not stop_event.is_set():
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and not stop_event.is_set():
            # check for random update of model weights and epsilon
            try:
                if not model_queue.empty():
                    weights, new_epsilon = model_queue.get_nowait()
                    local_model.load_state_dict(weights)
                    epsilon = new_epsilon
                    print(f"[Collector {id}] Model weights and epsilon updated successfully. New epsilon: {epsilon:.2f}")
            except Empty:
                pass

            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():  # Disable gradient calculation
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    q_values = local_model(state_tensor)
                    action = q_values.argmax().item()

            # Take action and send experience
            next_state, reward, done, info = env.step(action)
            
            # Acquire lock before putting experience in queue
            with queue_lock:
                experience_queue.put((state, action, reward, next_state, done))

            if experience_queue.qsize() > REP_Q_SIZE: print(f"[Collector {id}] experience queue full...")
            while experience_queue.qsize() > REP_Q_SIZE:
                sleep(2)

            state = next_state
            total_reward += reward
            steps += 1

            if steps >= MAX_STEPS_PER_RUN:  # Max steps per episode
                done = True

        # update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        # Store episode reward and update statistics
        recent_rewards.append(total_reward)
        episode_count += 1
        
        # Print statistics periodically
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            print(f"[Collector {id}] Episode {episode_count} - Average reward (last {len(recent_rewards)} episodes): {avg_reward:.2f} - Total best reward: {best_reward:.2f}")
            last_print_time = current_time

    print(f"[Collector {id}] Process stopping due to stop event")


def wait_for_buffer(agent, experience_queue, min_buffer_size=REP_Q_SIZE/4):
    """Wait until the replay buffer has enough samples, processing experiences while waiting."""
    last_print_time = time.time()
    print_interval = 5  # Print status every 5 seconds
    
    while len(agent.memory.buffer) < min_buffer_size:
        # Process experiences continuously
        try:
            experience = experience_queue.get(0.01)
            agent.memory.push(*experience)
        except Empty:
            pass

        # Print status periodically
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"[MAIN] Filling buffer: {len(agent.memory.buffer)}/{min_buffer_size:.0f} samples (Queue size: {experience_queue.qsize()})")
            last_print_time = current_time


def main():
    # Create and wrap the environment
    env = create_env(deadlock_steps=DEADLOCK_STEPS)

    # Create shared resources
    experience_queue = Queue()
    model_queue = Queue()
    stop_event = Event()
    queue_lock = Lock()  # Create a lock for the queue

    # Initialize agent
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue, memory_size=BUFFER_SIZE, gamma=GAMMA, epsilon_decay=EPSILON_DECAY, epsilon_min=0.3)

    # Try to load the latest checkpoint
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")

    # Spawning collector processes to continuously collect memories
    num_collectors = 2
    collectors = []

    # Start collector processes
    for i in range(num_collectors):
        p = Process(target=collector_process,
                    args=(experience_queue, model_queue, stop_event, agent.epsilon, i+1, queue_lock))
        collectors.append(p)
        p.start()

    # Send initial model weights to all collectors
    for _ in range(num_collectors):
        model_queue.put((agent.q_network.state_dict(), agent.epsilon))

    # initially fill up the buffer partially (25%)
    wait_for_buffer(agent, experience_queue)

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\nEpoch {epoch}")
            sleep(1)

            # Process any new experiences from the queue
            processed = 0
            with queue_lock:  # Acquire lock to process all items
                while True:
                    try:
                        # Use a small timeout to allow for proper synchronization
                        experience = experience_queue.get(timeout=0.01)
                        agent.memory.push(*experience)
                        processed += 1
                        # limit processed experience to the replay queue size
                        if processed >= REP_Q_SIZE:
                            break
                    except Empty:
                        break
            if processed > 0:
                print(f"[MAIN] Processed {processed} experiences from queue")

            print("[MAIN] Training agent...")
            agent.replay(batch_size=BATCH_SIZE, episodes=EPISODES_PER_EPOCH)

            # Send updated model to collectors less frequently
            if epoch % 1 == 0:  # Send updates every 1 epochs
                for _ in range(num_collectors):
                    model_queue.put((agent.q_network.state_dict(), agent.epsilon))
            with agent.memory.lock:
                print(f"Replay buffer size: {len(agent.memory.buffer)}")

            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0 and epoch != 0:  # Save every 10 epochs
                save_checkpoint(agent, epoch)

    except KeyboardInterrupt:
        print("\nStopping training...")
    finally:
        # Cleanup
        stop_event.set()
        for p in collectors:
            p.join()
        env.close()


if __name__ == "__main__":
    main()
