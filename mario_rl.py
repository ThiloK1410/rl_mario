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
import pandas as pd

import torch
from collections import deque

from environment import create_env
from dqn_agent import DQN, MarioAgent, DEVICE
from config import (
    DATA_FILE, REP_Q_SIZE, BUFFER_SIZE, NUM_EPOCHS, DEADLOCK_STEPS,
    MAX_STEPS_PER_RUN, BATCH_SIZE, EPISODES_PER_EPOCH, LEARNING_RATE,
    SAVE_INTERVAL, EPSILON_START, EPSILON_DECAY, EPSILON_MIN, GAMMA, AGENT_FOLDER
)

mp.set_start_method('spawn', force=True)

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


def write_log(epoch, data_dict):
    """Appends a dictionary of metrics to a CSV file."""
    # Add the epoch to the dictionary
    data_dict['epoch'] = epoch

    # Check if the file already exists
    file_exists = os.path.exists(DATA_FILE)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame([data_dict])

    # If the file doesn't exist, write with the header.
    # Otherwise, append without the header.
    df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)


# this function will be run by each collector process
def collector_process(experience_queue, model_queue, stop_event, epsilon, id, queue_lock):
    try:
        env = create_env(deadlock_steps=DEADLOCK_STEPS)
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
            try:
                state = env.reset()
                done = False
                total_reward = 0
                steps = 0

                while not done and not stop_event.is_set():
                    try:
                        # check for random update of model weights and epsilon
                        if not model_queue.empty():
                            weights, new_epsilon = model_queue.get_nowait()
                            local_model.load_state_dict(weights)
                            epsilon = new_epsilon
                            print(f"[Collector {id}] Model weights and epsilon updated successfully. New epsilon: {epsilon:.2f}")
                    except Empty:
                        pass
                    except Exception as e:
                        print(f"[Collector {id}] Error updating model: {str(e)}")

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

                    if MAX_STEPS_PER_RUN != 0 and steps >= MAX_STEPS_PER_RUN:  # Max steps per episode
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

            except Exception as e:
                print(f"[Collector {id}] Error in episode loop: {str(e)}")
                continue

        print(f"[Collector {id}] Process stopping due to stop event")
    except Exception as e:
        print(f"[Collector {id}] Fatal error: {str(e)}")
        print(f"[Collector {id}] Process stopping due to error")
    finally:
        if 'env' in locals():
            env.close()


def wait_for_buffer(agent, experience_queue, min_buffer_size=BUFFER_SIZE/32):
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
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue, memory_size=BUFFER_SIZE, gamma=GAMMA, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, lr=LEARNING_RATE, epsilon=EPSILON_START)

    # Try to load the latest checkpoint
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")

    # Spawning collector processes to continuously collect memories
    num_collectors = 6
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

            # Check if we have enough samples for training
            if len(agent.memory.buffer) < BATCH_SIZE:
                print(f"[MAIN] Not enough samples for training (have {len(agent.memory.buffer)}, need {BATCH_SIZE})")
                continue

            print("[MAIN] Training agent...")
            lr, avg_reward, loss, td_error = agent.replay(batch_size=BATCH_SIZE, episodes=EPISODES_PER_EPOCH)

            for _ in range(num_collectors // 2):
                model_queue.put((agent.q_network.state_dict(), agent.epsilon))
            print(f"Replay buffer size: {len(agent.memory.buffer)}")

            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0 and epoch != 0:  # Save every 10 epochs
                save_checkpoint(agent, epoch, checkpoint_dir=AGENT_FOLDER)

            log_data = {
                'td_error': td_error,
                'loss': loss,
                'average_reward': avg_reward,
                'learning_rate': lr,
                'buffer_size': len(agent.memory.buffer)
            }
            write_log(epoch, log_data)

    except KeyboardInterrupt:
        print("\nStopping training...")
    except Exception as e:
        print(f"[MAIN] Error during training: {str(e)}")
    finally:
        # Cleanup
        stop_event.set()
        for p in collectors:
            p.join()
        env.close()


if __name__ == "__main__":
    main()
