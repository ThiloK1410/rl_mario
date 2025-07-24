from multiprocessing import Process, Queue, Event, Lock, Value
from queue import Empty, Full
import time
from environment import create_env_new
from dqn_agent import DQN, DuelingDQN, MarioAgent, DEVICE
from time import sleep
import signal
import torch
import numpy as np
from tensorboard_logger import create_logger
from collections import deque
from mario_rl_common import save_checkpoint, load_checkpoint, find_latest_checkpoint

EXPERIMENT_NAME = "test_5"

AGENT_FOLDER = "checkpoints"

# Basic Training Parameters
USED_MOVESET = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up']
]

LEARNING_RATE = 0.001
GAMMA = 0.95
BUFFER_SIZE = 100000
BATCH_SIZE = 256
NUM_BATCHES = 16

# Multiprocessing Parameters
CHUNK_SIZE = 100
NUM_COLLECTORS = 2
CONSUMED_EXP = BATCH_SIZE * NUM_BATCHES
REUSE_FACTOR = 10

# Network Architecture Parameters
USE_DUELING_NETWORK = True
STACKED_FRAMES = 4
DOWNSCALE_RESOLUTION = 128

# Training Parameters  
LR_DECAY_RATE = 100
LR_DECAY_FACTOR = 0.9
AGENT_TAU = 0.05

# Prioritized Experience Replay Parameters
PER_ALPHA = 0.8
PER_BETA = 0.4
PER_BETA_INCREMENT = 0.001

# Epsilon Scheduler Parameters
EPSILON_START = 1.0
EPSILON_MIN = 0.2
EPSILON_DECAY = 0.0005
EPSILON_FINE_TUNE_THRESHOLD = 5
EPSILON_FINE_TUNE_DECAY = 0.0001
EPSILON_FINE_TUNE_MIN = 0.01

# Environment Parameters
DEADLOCK_PENALTY = 0.5
DEADLOCK_STEPS = 20
SKIPPED_FRAMES = 8
SPARSE_FRAME_INTERVAL = 4


def create_env():
    """Create Mario environment with mario_rl.py configuration parameters."""
    return create_env_new(
        used_moveset=USED_MOVESET,
        downscale_resolution=DOWNSCALE_RESOLUTION,
        deadlock_penalty=DEADLOCK_PENALTY,
        deadlock_steps=DEADLOCK_STEPS,
        skipped_frames=SKIPPED_FRAMES,
        stacked_frames=STACKED_FRAMES,
        sparse_frame_interval=SPARSE_FRAME_INTERVAL
    )


class CollectorProcess(Process):
    def __init__(self, id, replay_chunk_queue, model_queue, logging_queue, close_event, epoch, total_flag_completions):
        Process.__init__(self)
        # Create model with correct parameters from mario_rl.py
        if USE_DUELING_NETWORK:
            self.model = DuelingDQN(n_actions=len(USED_MOVESET), stacked_frames=STACKED_FRAMES, input_resolution=DOWNSCALE_RESOLUTION).to(DEVICE)
        else:
            self.model = DQN(n_actions=len(USED_MOVESET), stacked_frames=STACKED_FRAMES, input_resolution=DOWNSCALE_RESOLUTION).to(DEVICE)
        
        self.replay_chunk_queue = replay_chunk_queue
        self.model_queue = model_queue
        self.logging_queue = logging_queue
        self.epsilon = 1
        self.close_event = close_event
        self.id = id
        self.epoch = epoch
        self.env = None
        self.done = True
        self.state = None
        self.total_flag_completions = total_flag_completions  # Shared across all collectors

        self.last_epoch = -1
        
        # distance tracking
        self.distance = 0
        self.average_distance = deque(maxlen=100)


    def run(self):
        # Ignore keyboard interrupts - only respond to close_event
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        print(f"[Collector {self.id}] process started")
        
        self.env = create_env()
        print(f"[Collector {self.id}] Environment created successfully")

        try:
            chunk = []
            while True:
                if self.close_event.is_set():
                    print(f"[Collector {self.id}] closing")
                    break
                # try to load model parameters from the model queue
                try:
                    self.epsilon, model_state = self.model_queue.get_nowait()
                    self.model.load_state_dict(model_state)
                    print(f"[Collector {self.id}] loaded model and epsilon {self.epsilon}")
                except Empty:
                    pass
                
                for _ in range(10):
                    # get experience from the environment
                    chunk.append(self.get_experience())
                    # if we have a chunk of the desired size, send it to the replay chunk queue
                    if len(chunk) == CHUNK_SIZE:
                        # try to put the chunk in the queue, if it's full, wait
                        self.replay_chunk_queue.put(chunk)
                        chunk = []

                if self.last_epoch != self.epoch.value:
                    self.logging_queue.put(("Performance/average_distance", np.mean(self.average_distance), self.epoch.value))
                    self.last_epoch = self.epoch.value
                

        except Exception as e:
            print(f"[Collector {self.id}] error: {e}")
            raise e

    # sample a single experience from the environment, handle the done flag
    def get_experience(self):
        if self.done:
            self.state = self.env.reset()
            self.average_distance.append(self.distance)
            self.distance = 0
            self.done = False
        
        # Convert state to tensor and get action using proper preprocessing
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, len(USED_MOVESET))
        else:
            state_tensor = torch.FloatTensor(np.array(self.state)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = self.model(state_tensor).argmax().item()
        
        next_state, reward, self.done, info = self.env.step(action)

        self.distance = max(self.distance, info['x_pos'])
        
        # Track flag completions (only from level start, not recorded positions)
        if self.done and info.get('flag_get', False):
            # Note: This is a simplified approach - ideally we'd track if this episode 
            # started from level beginning vs recorded position
            with self.total_flag_completions.get_lock():
                self.total_flag_completions.value += 1
                print(f"[Collector {self.id}] 🎯 FLAG COMPLETED! Total: {self.total_flag_completions.value}")

        state = self.state
        self.state = next_state
        return state, action, reward, next_state, self.done
    
class TrainingProcess(Process):
    def __init__(self, close_event, epoch, replay_chunk_queue, model_queue, logging_queue, total_flag_completions):
        Process.__init__(self)
        self.close_event = close_event
        self.epoch = epoch
        self.replay_chunk_queue = replay_chunk_queue
        self.model_queue = model_queue
        self.logging_queue = logging_queue
        self.total_flag_completions = total_flag_completions  # Shared value
        self.agent = None
    def run(self):
        # Ignore keyboard interrupts - only respond to close_event
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        try:
            self.agent = create_agent()
        except Exception as e:
            print(f"[TRAINING] ERROR: Could not create agent: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Share initial model parameters with collectors at startup
        try:
            initial_model_state = self.agent.q_network.state_dict()
            
            # Try to put initial model parameters in queue for each collector
            for i in range(NUM_COLLECTORS):
                try:
                    self.model_queue.put((self.agent.epsilon, initial_model_state), timeout=1)
                except Full:
                    print(f"[TRAINING] model queue full at collector {i}, collectors will use default parameters initially")
                    break
        except Exception as e:
            print(f"[TRAINING] Warning: Could not share initial model parameters: {e}")
            import traceback
            traceback.print_exc()
        
        print("[TRAINING] Starting main training loop...")
        memories_per_epoch = 0
        while True:
            if self.close_event.is_set():
                print("[TRAINING] closing process")
                # Save final checkpoint before closing
                try:
                    save_checkpoint(self.agent, self.epoch.value, EXPERIMENT_NAME, checkpoint_dir=AGENT_FOLDER)
                    print(f"[TRAINING] Final checkpoint saved at epoch {self.epoch.value}")
                except Exception as e:
                    print(f"[TRAINING] Warning: Could not save final checkpoint: {e}")
                break

            # collect experience chunks from the replay chunk queue
            try:
                # block until a chunk is available, but with timeout to check close_event
                chunk = self.replay_chunk_queue.get(timeout=1)
                self.agent.remember_batch(chunk)
                memories_per_epoch += len(chunk)
            except Empty:
                print(f"[TRAINING] waiting for enough experiences: {memories_per_epoch} / {CONSUMED_EXP / REUSE_FACTOR}")
                # Continue to check close_event

            # check if we can start a training run
            if memories_per_epoch > CONSUMED_EXP / REUSE_FACTOR:
                print(f"[TRAINING] Replaying with {memories_per_epoch} new experiences")
                start_time = time.time()
                # replay the agent with a memory lock to prevent race conditions
                current_lr, avg_reward, returned_loss, returned_td_error = self.agent.replay(batch_size=BATCH_SIZE, episodes=NUM_BATCHES)
                print(f"[TRAINING] finished Epoch {self.epoch.value} in {time.time() - start_time:.2f} seconds")
                # log the training metrics to tensorboard
                epoch = self.epoch.value
                self.logging_queue.put(("Profiling/training_time", time.time() - start_time, epoch))
                self.logging_queue.put(("Hyperparameters/current_lr", current_lr, epoch))
                self.logging_queue.put(("Hyperparameters/epsilon", self.agent.epsilon, epoch))
                self.logging_queue.put(("Performance/average_reward", avg_reward, epoch))
                self.logging_queue.put(("Performance/loss", returned_loss, epoch))
                self.logging_queue.put(("Performance/td_error", returned_td_error, epoch))
                self.logging_queue.put(("Random/buffer_size", len(self.agent.memory), epoch))

                # Update epsilon scheduler with flag completions (aggregate from all collectors)
                with self.total_flag_completions.get_lock():
                    current_flags = self.total_flag_completions.value
                self.agent.update_flag_completions(current_flags)
                
                # Log flag completions for monitoring
                self.logging_queue.put(("Performance/total_flag_completions", current_flags, epoch))

                self.epoch.value += 1
                memories_per_epoch = 0

                # Save checkpoint every 100 epochs
                if self.epoch.value % 100 == 0:
                    try:
                        save_checkpoint(self.agent, self.epoch.value, EXPERIMENT_NAME, checkpoint_dir=AGENT_FOLDER)
                        print(f"[TRAINING] Checkpoint saved at epoch {self.epoch.value}")
                    except Exception as e:
                        print(f"[TRAINING] Warning: Could not save checkpoint at epoch {self.epoch.value}: {e}")

                # Share updated model parameters with collectors after training
                try:
                    # Get current model state dict and share with collectors
                    model_state = self.agent.q_network.state_dict()
                    # Try to put model parameters in queue (non-blocking)
                    self.model_queue.put((self.agent.epsilon, model_state))
                    print(f"[TRAINING] shared model parameters for epoch {self.epoch.value}")
                except Full:
                    pass  # Queue is full, collectors haven't consumed recent updates yet

            

# this thread will manage the interaction between the collector processes and the training process
# it will collect the experience chunks from the collector proceeses and store them in the buffer
class HeadProcess(Process):
    def __init__(self, close_event):
        Process.__init__(self)
        self.replay_chunk_queue = Queue(maxsize=20)
        self.model_queue = Queue(maxsize=NUM_COLLECTORS)
        self.logging_queue = Queue()
        self.close_event = close_event
        
        # Shared value for flag completions across all processes
        self.total_flag_completions = Value('i', 0)
        
        # Check for existing checkpoint to determine starting epoch
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=EXPERIMENT_NAME)
        start_epoch = 0
        if latest_checkpoint:
            try:
                # Load checkpoint just to get the epoch number
                import torch
                checkpoint_data = torch.load(latest_checkpoint, map_location=torch.device("cpu"), weights_only=False)
                start_epoch = checkpoint_data.get('epoch', 0)
                print(f"[HEAD] Found checkpoint at epoch {start_epoch}, will resume from there")
            except Exception as e:
                print(f"[HEAD] Warning: Could not read epoch from checkpoint: {e}")
                start_epoch = 0
        
        self.epoch = Value('i', start_epoch)

    def run(self):
        # Ignore keyboard interrupts - only respond to close_event
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        # create collector processes
        collector_processes = []
        for i in range(NUM_COLLECTORS):
            collector = CollectorProcess(i, self.replay_chunk_queue, self.model_queue, self.logging_queue, self.close_event, self.epoch, self.total_flag_completions)
            collector.start()
            collector_processes.append(collector)
        # create training process with shared agent
        training_process = TrainingProcess(self.close_event, self.epoch, self.replay_chunk_queue, self.model_queue, self.logging_queue, self.total_flag_completions)
        training_process.start()

        logger = create_logger(process_name=None, experiment_name=EXPERIMENT_NAME)

        # Main coordination loop - just manage process lifecycle
        while True:
            # handle closure of program
            if self.close_event.is_set():
                print("[HEAD] Shutdown signal received, closing child processes...")
                
                # Give processes time to finish current work and shut down gracefully
                shutdown_timeout = 30  # seconds
                
                # Wait for all collector processes to finish
                for i, collector in enumerate(collector_processes):
                    print(f"[HEAD] Waiting for collector {i} to finish...")
                    collector.join(timeout=shutdown_timeout)
                    if collector.is_alive():
                        print(f"[HEAD] Collector {i} didn't finish in time, terminating...")
                        collector.terminate()
                        collector.join(timeout=5)
                
                # Wait for training process to finish
                print("[HEAD] Waiting for training process to finish...")
                training_process.join(timeout=shutdown_timeout)
                if training_process.is_alive():
                    print("[HEAD] Training process didn't finish in time, terminating...")
                    training_process.terminate()
                    training_process.join(timeout=5)
                
                print("All processes closed, HeadProcess closing")
                break

            # handle logging
            while True:
                try:
                    metric, value, epoch = self.logging_queue.get(timeout=1)
                    logger.log(metric, value, epoch)
                except Empty:
                    break


def create_agent():
    agent = MarioAgent(  
            n_actions=len(USED_MOVESET), 
            lr=LEARNING_RATE,
            gamma=GAMMA,
            epsilon=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            memory_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            # Network architecture parameters
            use_dueling_network=USE_DUELING_NETWORK,
            stacked_frames=STACKED_FRAMES,
            input_resolution=DOWNSCALE_RESOLUTION,
            # Training parameters
            tau=AGENT_TAU,
            lr_decay_rate=LR_DECAY_RATE,
            lr_decay_factor=LR_DECAY_FACTOR,
            # PER parameters
            per_alpha=PER_ALPHA,
            per_beta=PER_BETA,
            per_beta_increment=PER_BETA_INCREMENT,
            # Epsilon scheduler parameters
            fine_tune_threshold=EPSILON_FINE_TUNE_THRESHOLD,
            fine_tune_decay=EPSILON_FINE_TUNE_DECAY,
            fine_tune_min=EPSILON_FINE_TUNE_MIN
    )
    
    # Check for existing checkpoint and load it
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=EXPERIMENT_NAME)
    if latest_checkpoint:
        try:
            start_epoch, loaded_experiment_name = load_checkpoint(agent, latest_checkpoint)
            print(f"[AGENT] Loaded existing checkpoint from epoch {start_epoch}")
            print(f"[AGENT] Continuing training for experiment: {EXPERIMENT_NAME}")
        except Exception as e:
            print(f"[AGENT] Warning: Could not load checkpoint {latest_checkpoint}: {e}")
            print(f"[AGENT] Starting fresh training")
    else:
        print(f"[AGENT] No existing checkpoint found for experiment: {EXPERIMENT_NAME}")
        print(f"[AGENT] Starting fresh training")
    
    return agent

if __name__ == "__main__":
    try:
        close_event = Event()
        head_process = HeadProcess(close_event)
        head_process.start()
        while True:
            sleep(100)
    # graceful shutdown on keyboard interrupt
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt received, initiating graceful shutdown...")
        close_event.set()
        
        # Give head process time to shut down gracefully
        print("[MAIN] Waiting for head process to finish...")
        head_process.join(timeout=60)  # 60 seconds should be enough
        
        if head_process.is_alive():
            print("[MAIN] Head process didn't finish in time, terminating...")
            head_process.terminate()
            head_process.join(timeout=10)
        
        print("[MAIN] Graceful shutdown completed")
        
    # try graceful shutdown on exception
    except Exception as e:
        print(f"[MAIN] Error: {e}")
        close_event.set()
        head_process.join(timeout=30)
        
        if head_process.is_alive():
            print("[MAIN] Head process didn't finish in time during error shutdown, terminating...")
            head_process.terminate()
            head_process.join(timeout=10)
            
        raise e
    
