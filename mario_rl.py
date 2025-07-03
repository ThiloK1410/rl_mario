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
import threading

import torch
from collections import deque

from environment import create_env
from dqn_agent import DQN, MarioAgent, DEVICE
from config import (
    DATA_FILE, REP_Q_SIZE, BUFFER_SIZE, NUM_EPOCHS, DEADLOCK_STEPS,
    MAX_STEPS_PER_RUN, BATCH_SIZE, EPISODES_PER_EPOCH, LEARNING_RATE,
    SAVE_INTERVAL, EPSILON_START, EPSILON_DECAY, EPSILON_MIN, GAMMA, AGENT_FOLDER, NUM_PROCESSES, RANDOM_STAGES,
    RANDOM_SAVES, MODEL_UPDATE_FREQUENCY, USE_MIXED_PRECISION, ASYNC_MODEL_UPDATES
)

mp.set_start_method('spawn', force=True)


class ExperienceProcessor:
    """
    Dedicated thread for continuously processing experiences from queue to buffer.
    Uses priority-based locking to prevent collector interruptions.
    """
    
    def __init__(self, agent, experience_queue, queue_lock, 
                 batch_size=200, process_interval=0.01, stats_interval=30):
        self.agent = agent
        self.experience_queue = experience_queue
        self.queue_lock = queue_lock
        self.batch_size = batch_size
        self.process_interval = process_interval
        self.stats_interval = stats_interval
        
        # Thread control
        self.running = False
        self.thread = None
        self._shutdown_event = threading.Event()
        
        # Priority locking system to reduce collector interruptions
        self._processing_flag = threading.Event()  # Signals when processor is working
        self._priority_lock = threading.Lock()     # Additional lock for priority control
        
        # CPU-friendly adaptive parameters
        self.min_sleep_time = 0.005  # Minimum 5ms sleep to prevent CPU spinning
        self.max_sleep_time = 0.1    # Maximum 100ms sleep when idle
        self.adaptive_factor = 1.5   # Multiplier for adaptive sleep
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'total_time': 0,
            'avg_batch_size': 0,
            'avg_rate': 0,
            'queue_full_events': 0,
            'empty_polls': 0,
            'lock_contentions': 0,
            'successful_batch_drains': 0,
            'emergency_drains': 0,
            'timeout_errors': 0,
            'max_queue_size_seen': 0,
            'critical_pressure_events': 0,
            'cpu_friendly_sleeps': 0,
            'adaptive_sleep_adjustments': 0
        }
        self._stats_lock = threading.Lock()
        self._last_stats_time = time.time()
        
    def start(self):
        """Start the processing thread."""
        if self.running:
            print("[PROCESSOR] Already running")
            return
        
        self.running = True
        self._shutdown_event.clear()
        self.thread = threading.Thread(target=self._process_loop, daemon=True, name="ExperienceProcessor")
        self.thread.start()
        print("[PROCESSOR] Experience processing thread started with priority locking")
    
    def stop(self, timeout=10.0):
        """Stop the processing thread gracefully."""
        if not self.running:
            return
        
        print("[PROCESSOR] Stopping experience processing thread...")
        self.running = False
        self._shutdown_event.set()
        self._processing_flag.clear()  # Release any waiting collectors
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print("[PROCESSOR] Warning: Thread did not stop cleanly")
            else:
                print("[PROCESSOR] Thread stopped cleanly")
        
        self._print_final_stats()
    
    def _process_loop(self):
        """Main processing loop with priority-based lock acquisition."""
        print("[PROCESSOR] Processing loop started with priority locking")
        
        while self.running and not self._shutdown_event.is_set():
            try:
                processed = self._priority_process_batch()
                
                if processed > 0:
                    with self._stats_lock:
                        self.stats['total_processed'] += processed
                        self.stats['total_batches'] += 1
                        self.stats['successful_batch_drains'] += 1
                else:
                    with self._stats_lock:
                        self.stats['empty_polls'] += 1
                
                # Print periodic stats
                current_time = time.time()
                if current_time - self._last_stats_time >= self.stats_interval:
                    self._print_stats()
                    self._last_stats_time = current_time
                
                # CPU-FRIENDLY adaptive sleep based on queue pressure and processing results
                queue_size = self.experience_queue.qsize()
                queue_pressure = queue_size / REP_Q_SIZE if REP_Q_SIZE > 0 else 0
                
                # Calculate CPU-friendly sleep time
                if processed == 0:
                    # No experiences processed - increase sleep time to reduce CPU usage
                    if queue_pressure > 0.5:
                        # Queue has items but couldn't process - moderate sleep to retry
                        sleep_time = max(self.min_sleep_time, self.process_interval * 2)  # Min 5ms
                    else:
                        # Queue likely empty - much longer sleep to save CPU
                        sleep_time = max(self.min_sleep_time, self.process_interval * 5)  # Min 5ms, up to 50ms
                elif queue_pressure > 0.8:
                    # HIGH: Queue building up - still use minimum sleep to prevent CPU spinning
                    sleep_time = self.min_sleep_time  # 5ms minimum - no CPU spinning!
                    print(f"[PROCESSOR] High queue pressure {queue_pressure:.1%} - using min sleep {sleep_time*1000:.1f}ms")
                elif queue_pressure > 0.5:
                    # MEDIUM: Some queue pressure - reduced but still CPU-friendly sleep
                    sleep_time = max(self.min_sleep_time, self.process_interval * 0.8)  # Min 5ms
                else:
                    # LOW: Normal processing - standard sleep
                    sleep_time = max(self.min_sleep_time, self.process_interval)  # Min 5ms, standard 10ms
                
                # Apply the CPU-friendly sleep
                time.sleep(sleep_time)
                
                # Track CPU-friendly behavior
                with self._stats_lock:
                    self.stats['cpu_friendly_sleeps'] += 1
                    if sleep_time >= self.min_sleep_time:
                        self.stats['adaptive_sleep_adjustments'] += 1
                    
            except Exception as e:
                print(f"[PROCESSOR] Error in processing loop: {e}")
                time.sleep(0.1)  # Prevent tight error loops
                
        print("[PROCESSOR] Processing loop ended")
    
    def _priority_process_batch(self):
        """
        Process a batch with priority-based lock acquisition and timeout handling.
        Uses a two-phase approach to minimize lock contention.
        """
        experiences = []
        
        # Phase 1: Try to acquire priority lock first
        with self._priority_lock:
            # Signal that processor is working (collectors should yield)
            self._processing_flag.set()
            
            # Phase 2: Acquire queue lock and drain aggressively with timeout handling
            lock_acquired = False
            try:
                # Use timeout to avoid indefinite blocking
                lock_acquired = self.queue_lock.acquire(timeout=0.01)  # 10ms timeout
                
                if lock_acquired:
                    # Drain as many experiences as possible while we have the lock
                    start_drain_time = time.time()
                    max_drain_time = 0.05  # Maximum 50ms draining time to prevent blocking
                    
                    for _ in range(self.batch_size * 3):  # Increased to drain 3x batch size when queue is full
                        # Check if we've been draining too long
                        if time.time() - start_drain_time > max_drain_time:
                            print(f"[PROCESSOR] Max drain time reached, processed {len(experiences)} experiences")
                            break
                            
                        try:
                            # Use CPU-friendly timeout for queue access
                            experience = self.experience_queue.get(timeout=0.01)  # Increased to 10ms for CPU efficiency
                            experiences.append(experience)
                        except Empty:
                            break
                        except Exception as e:
                            print(f"[PROCESSOR] Error getting experience from queue: {e}")
                            break
                    
                    drain_time = time.time() - start_drain_time
                    
                    # Check queue status and warn if it's getting too full
                    queue_size = self.experience_queue.qsize()
                    if queue_size > REP_Q_SIZE * 0.8:  # Warn at 80% capacity
                        print(f"[PROCESSOR] WARNING: Queue at {queue_size}/{REP_Q_SIZE} ({queue_size/REP_Q_SIZE*100:.1f}%) - "
                              f"drained {len(experiences)} experiences in {drain_time*1000:.1f}ms")
                        
                        # If queue is very full, try emergency drain with longer timeout
                        if queue_size > REP_Q_SIZE * 0.95:  # Emergency at 95% capacity
                            print(f"[PROCESSOR] EMERGENCY DRAIN: Queue critical at {queue_size}")
                            emergency_start = time.time()
                            emergency_count = 0
                            
                            # Emergency drain with longer timeout
                            while time.time() - emergency_start < 0.1:  # 100ms emergency drain
                                try:
                                    experience = self.experience_queue.get(timeout=0.002)  # 2ms timeout
                                    experiences.append(experience)
                                    emergency_count += 1
                                    
                                    # Stop if queue drops below critical level
                                    if self.experience_queue.qsize() < REP_Q_SIZE * 0.8:
                                        break
                                except Empty:
                                    break
                                except Exception as e:
                                    print(f"[PROCESSOR] Emergency drain error: {e}")
                                    break
                            
                            if emergency_count > 0:
                                print(f"[PROCESSOR] Emergency drain completed: {emergency_count} additional experiences")
                    
                    # Update contention stats
                    with self._stats_lock:
                        if len(experiences) > 0:
                            self.stats['successful_batch_drains'] += 1
                            
                            # Track queue pressure
                            if queue_size > REP_Q_SIZE * 0.8:
                                self.stats['queue_full_events'] += 1
                        
                else:
                    # Failed to acquire lock - collector interference
                    with self._stats_lock:
                        self.stats['lock_contentions'] += 1
                    
            finally:
                if lock_acquired:
                    self.queue_lock.release()
                
                # Clear processing flag to allow collectors to proceed
                self._processing_flag.clear()
        
        # Phase 3: Process experiences outside all locks
        if not experiences:
            return 0
        
        start_process_time = time.time()
        
        # Use batch operations if available for maximum efficiency
        if hasattr(self.agent.memory, 'push_batch'):
            try:
                self.agent.memory.push_batch(experiences)
            except Exception as e:
                print(f"[PROCESSOR] Batch push failed, falling back to individual: {e}")
                # Fallback to individual pushes - use explicit unpacking
                self._push_experiences_individually(experiences)
        else:
            # Individual pushes
            self._push_experiences_individually(experiences)
        
        process_time = time.time() - start_process_time
        total_time = process_time  # drain_time already accounted for in stats
        
        # Update timing stats
        with self._stats_lock:
            self.stats['total_time'] += total_time  # type: ignore
            if self.stats['total_batches'] > 0:
                self.stats['avg_batch_size'] = float(self.stats['total_processed']) / float(self.stats['total_batches'])  # type: ignore
            if self.stats['total_time'] > 0:
                self.stats['avg_rate'] = float(self.stats['total_processed']) / float(self.stats['total_time'])  # type: ignore
        
        return len(experiences)
    
    def _push_experiences_individually(self, experiences):
        """Helper method to push experiences individually with proper validation."""
        for i, experience in enumerate(experiences):
            try:
                # Validate experience format
                if not experience:
                    print(f"[PROCESSOR] Empty experience at index {i}")
                    continue
                    
                # Check if it's a valid tuple/list with 5 elements
                if not hasattr(experience, '__getitem__') or not hasattr(experience, '__len__'):
                    print(f"[PROCESSOR] Invalid experience type at index {i}: {type(experience)}")
                    continue
                    
                if len(experience) != 5:
                    print(f"[PROCESSOR] Invalid experience length at index {i}: {len(experience)}")
                    continue
                
                # Explicit unpacking to avoid linter issues
                state, action, reward, next_state, done = experience  # type: ignore
                self.agent.memory.push(state, action, reward, next_state, done)
                
            except (TypeError, ValueError, IndexError) as e:
                print(f"[PROCESSOR] Failed to push experience at index {i}: {e}")
            except Exception as e:
                print(f"[PROCESSOR] Unexpected error pushing experience at index {i}: {e}")
    
    def _print_stats(self):
        """Print processing statistics with contention info."""
        with self._stats_lock:
            queue_size = self.experience_queue.qsize()
            buffer_size = len(self.agent.memory)
            contention_ratio = self.stats['lock_contentions'] / max(self.stats['total_batches'] + self.stats['lock_contentions'], 1)
            
            print(f"[PROCESSOR] Stats - Processed: {self.stats['total_processed']:,}, "
                  f"Batches: {self.stats['total_batches']:,}, "
                  f"Avg batch: {self.stats['avg_batch_size']:.1f}, "
                  f"Rate: {self.stats['avg_rate']:.0f} exp/s, "
                  f"Queue: {queue_size}, Buffer: {buffer_size:,}")
            
            print(f"[PROCESSOR] Contention - Lock conflicts: {self.stats['lock_contentions']:,} "
                  f"({contention_ratio:.1%}), Successful drains: {self.stats['successful_batch_drains']:,}")
    
    def _print_final_stats(self):
        """Print final statistics when stopping."""
        with self._stats_lock:
            contention_ratio = self.stats['lock_contentions'] / max(self.stats['total_batches'] + self.stats['lock_contentions'], 1)
            
            print(f"[PROCESSOR] Final Stats:")
            print(f"  Total experiences processed: {self.stats['total_processed']:,}")
            print(f"  Total batches: {self.stats['total_batches']:,}")
            print(f"  Average batch size: {self.stats['avg_batch_size']:.1f}")
            print(f"  Average processing rate: {self.stats['avg_rate']:.0f} exp/s")
            print(f"  Lock contentions: {self.stats['lock_contentions']:,} ({contention_ratio:.1%})")
            print(f"  Successful batch drains: {self.stats['successful_batch_drains']:,}")
            print(f"  Empty polls: {self.stats['empty_polls']:,}")
            if self.stats['total_time'] > 0:
                print(f"  Total processing time: {self.stats['total_time']:.1f}s")
    
    def get_stats(self):
        """Get current statistics (thread-safe)."""
        with self._stats_lock:
            return {
                **self.stats,
                'queue_size': self.experience_queue.qsize(),
                'buffer_size': len(self.agent.memory),
                'is_running': self.running
            }
    
    def is_processing(self):
        """Check if processor is currently working (for collector coordination)."""
        return self._processing_flag.is_set()


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

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))

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
def collector_process(experience_queue, model_queue, distance_queue, stop_event, epsilon, id, queue_lock):
    try:
        env = create_env(sample_random_stages=RANDOM_STAGES, use_random_saves=RANDOM_SAVES)
        print(f"[Collector {id}] Environment created successfully")

        # Initialize model with minimal memory footprint
        with torch.no_grad():
            local_model = DQN((128, 128), env.action_space.n).to(DEVICE)  # type: ignore
            local_model.eval()  # Set to evaluation mode to reduce memory usage
        print(f"[Collector {id}] Local model initialized")

        # Initialize reward tracking
        import time
        recent_rewards = deque(maxlen=100)  # Store last 100 episode rewards
        last_print_time = time.time()
        print_interval = 20  # Print every 20 seconds

        # Performance optimization: batch action inference to reduce GPU switching overhead
        action_batch_size = 8  # Process multiple states together
        state_buffer = []
        
        # Performance tracking
        gpu_inference_count = 0
        random_action_count = 0
        model_update_count = 0
        last_performance_print = time.time()

        # respond to stop event from parent process
        episode_count = 0
        best_reward = 0
        # keeping track of distance reached in episode
        best_x = 0
        while not stop_event.is_set():
            try:
                # At the start of each episode
                best_x = 0
                state = env.reset()  # env.reset() only returns observation
                
                # Take a no-op step to get initial position info
                _, _, _, info = env.step(0)  # No-op action to get info
                start_x = info['x_pos']
                
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
                            model_update_count += 1
                            print(f"[Collector {id}] Model weights and epsilon updated successfully. New epsilon: {epsilon:.2f}")
                    except Empty:
                        pass
                    except Exception as e:
                        print(f"[Collector {id}] Error updating model: {str(e)}")

                    # Optimized epsilon-greedy action selection
                    if random.random() <= epsilon:
                        action = env.action_space.sample()
                        random_action_count += 1
                    else:
                        # GPU inference - try to batch when possible for efficiency
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                            
                            # Use mixed precision for faster inference
                            with torch.cuda.amp.autocast():
                                q_values = local_model(state_tensor)
                            
                            action = q_values.argmax().item()
                            gpu_inference_count += 1

                    # Take action and send experience
                    next_state, reward, done, info = env.step(action)
                    # update best x
                    best_x = max(best_x, info['x_pos'])

                    # Acquire lock before putting experience in queue
                    # NOTE: This is now optimized - the dedicated processor thread
                    # will handle draining this queue continuously
                    with queue_lock:
                        experience_queue.put((state, action, reward, next_state, done))

                    # Monitor queue size but don't block as aggressively
                    queue_size = experience_queue.qsize()
                    if queue_size > REP_Q_SIZE:
                        if queue_size > REP_Q_SIZE * 2:  # Only warn at 2x capacity
                            print(f"[Collector {id}] Queue very full: {queue_size}")
                        # Shorter sleep since dedicated processor is handling it
                        sleep(0.1)  # Reduced from 2 seconds

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
                
                # Print performance stats every 30 seconds  
                if current_time - last_performance_print >= 30:
                    total_actions = gpu_inference_count + random_action_count
                    gpu_ratio = gpu_inference_count / max(total_actions, 1) * 100
                    print(f"[Collector {id}] Performance - GPU inference: {gpu_inference_count:,} ({gpu_ratio:.1f}%), "
                          f"Random: {random_action_count:,}, Model updates: {model_update_count}")
                    last_performance_print = current_time

                if best_x != start_x:
                    distance_queue.put(best_x - start_x)

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


def wait_for_buffer_with_processor(processor, min_buffer_size, timeout=300):
    """
    Wait until the replay buffer has enough samples.
    The dedicated processor handles all the work.
    """
    print(f"[MAIN] Waiting for buffer to reach {min_buffer_size} samples...")
    start_time = time.time()
    last_print_time = time.time()
    print_interval = 5
    
    while len(processor.agent.memory) < min_buffer_size:
        current_time = time.time()
        
        # Timeout check
        if current_time - start_time > timeout:
            print(f"[MAIN] Timeout waiting for buffer after {timeout}s")
            break
        
        # Print status periodically
        if current_time - last_print_time >= print_interval:
            stats = processor.get_stats()
            buffer_size = stats['buffer_size']
            queue_size = stats['queue_size']
            rate = stats['avg_rate']
            
            print(f"[MAIN] Buffer filling: {buffer_size:,}/{min_buffer_size:,} "
                  f"(Queue: {queue_size}, Rate: {rate:.0f} exp/s)")
            last_print_time = current_time
        
        time.sleep(1)  # Check every second
    
    final_size = len(processor.agent.memory)
    print(f"[MAIN] Buffer ready with {final_size:,} samples")


def main():
    # Set CUDA memory management for large models
    import os
    # Optimize CUDA memory allocation as suggested by PyTorch docs
    if not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("[MAIN] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # Create and wrap the environment
    env = create_env(sample_random_stages=RANDOM_STAGES, use_random_saves=RANDOM_SAVES)

    # Create shared resources
    experience_queue = Queue()
    model_queue = Queue()
    ep_distance_queue = Queue()
    stop_event = Event()
    queue_lock = Lock()  # Create a lock for the queue

    ep_dist_deque = deque(maxlen=50)

    # Initialize agent
    agent = MarioAgent((128, 128), env.action_space.n, experience_queue, memory_size=BUFFER_SIZE, gamma=GAMMA, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN, lr=LEARNING_RATE, epsilon=EPSILON_START)  # type: ignore

    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[MAIN] CUDA device: {DEVICE}")
        print(f"[MAIN] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")

    # Try to load the latest checkpoint
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")

    # Initialize dedicated experience processor with CPU-friendly settings
    processor = ExperienceProcessor(
        agent=agent,
        experience_queue=experience_queue,
        queue_lock=queue_lock,
        batch_size=200,  # Reduced batch size to process more frequently with less CPU load
        process_interval=0.02,  # Increased to 20ms for CPU-friendly polling
        stats_interval=30  # Print stats every 30 seconds
    )
    
    # Start the dedicated processing thread
    processor.start()

    # Spawning collector processes to continuously collect memories
    collectors = []
    # Start collector processes
    for i in range(NUM_PROCESSES):  # type: ignore
        p = Process(target=collector_process,
                    args=(experience_queue, model_queue, ep_distance_queue, stop_event, agent.epsilon, i+1, queue_lock))
        collectors.append(p)
        p.start()

    # Send initial model weights to all collectors
    for _ in range(NUM_PROCESSES):  # type: ignore
        model_queue.put((agent.q_network.state_dict(), agent.epsilon))

    # Wait for buffer to fill - the processor handles everything automatically
    wait_for_buffer_with_processor(processor, BUFFER_SIZE//32)

    def print_cuda_memory_stats():
        """Print current CUDA memory usage for monitoring."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[MEMORY] GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    try:
        # Experience tracking for logging
        epoch_start_time = time.time()
        epoch_start_buffer_size = len(agent.memory)
        
        # Experience reuse ratio tracking
        reuse_ratio_threshold = 20.0  # Maximum allowed reuse ratio
        
        for epoch in range(start_epoch, NUM_EPOCHS):  # type: ignore
            print(f"\nEpoch {epoch}")
            
            # Experience rate tracking
            current_time = time.time()
            current_buffer_size = len(agent.memory)
            
            # Calculate experience metrics for this epoch
            if epoch > start_epoch:
                epoch_duration = current_time - epoch_start_time
                experiences_gained = current_buffer_size - epoch_start_buffer_size
                
                # Experience rates
                experiences_per_second = experiences_gained / max(epoch_duration, 0.001)
                experiences_per_epoch = experiences_gained
                
                # Calculate experience reuse ratio
                experiences_consumed_per_epoch = BATCH_SIZE * EPISODES_PER_EPOCH
                reuse_ratio = experiences_consumed_per_epoch / max(experiences_gained, 1)
                
                print(f"[EXPERIENCE] Gained {experiences_gained:,} experiences in {epoch_duration:.1f}s "
                      f"({experiences_per_second:.1f} exp/s)")
                print(f"[REUSE RATIO] Consumption: {experiences_consumed_per_epoch:,}, "
                      f"Collection: {experiences_gained:,}, "
                      f"Ratio: {reuse_ratio:.2f} (threshold: {reuse_ratio_threshold})")
                
                # Check if reuse ratio is too high - block and wait for more experiences
                if reuse_ratio > reuse_ratio_threshold:
                    print(f"[THROTTLE] Reuse ratio {reuse_ratio:.2f} exceeds threshold {reuse_ratio_threshold}!")
                    print(f"[THROTTLE] Blocking main thread to yield CPU for experience collection...")
                    
                    # Wait for collectors to gather enough new experiences
                    wait_start_time = time.time()
                    target_new_experiences = experiences_consumed_per_epoch // 2  # Wait for at least half of what we consume
                    
                    # Use processor's total processed count instead of buffer size (works even when buffer is full)
                    initial_processed_count = processor.get_stats()['total_processed']
                    
                    print(f"[THROTTLE] Waiting for {target_new_experiences:,} new experiences...")
                    print(f"[THROTTLE] Buffer full detection: Using processor count instead of buffer size")
                    
                    while True:
                        # Sleep to yield CPU to collector processes
                        time.sleep(1.0)  # 1 second sleep to give collectors CPU time
                        
                        current_processed_count = processor.get_stats()['total_processed']
                        new_experiences_collected = current_processed_count - initial_processed_count
                        wait_duration = time.time() - wait_start_time
                        
                        print(f"[THROTTLE] Waiting... {new_experiences_collected:,}/{target_new_experiences:,} experiences processed "
                              f"({wait_duration:.1f}s elapsed)")
                        
                        # Check if we've collected enough new experiences
                        if new_experiences_collected >= target_new_experiences:
                            print(f"[THROTTLE] Target reached! Processed {new_experiences_collected:,} new experiences in {wait_duration:.1f}s")
                            break
                        
                        # Safety timeout to prevent infinite waiting
                        if wait_duration > 120:  # 2 minutes maximum wait
                            print(f"[THROTTLE] Timeout reached ({wait_duration:.1f}s), proceeding with training")
                            break
                        
                        # Print processor stats during wait to monitor progress
                        if int(wait_duration) % 10 == 0:  # Every 10 seconds
                            processor_stats = processor.get_stats()
                            print(f"[THROTTLE] Processor stats - Rate: {processor_stats['avg_rate']:.1f} exp/s, "
                                  f"Queue: {processor_stats['queue_size']}")
                    
                    print(f"[THROTTLE] Resuming training after {wait_duration:.1f}s wait")
                    
                    # Reset tracking since we've waited and buffer has grown
                    epoch_start_time = time.time()
                    epoch_start_buffer_size = len(agent.memory)
                
            else:
                # First epoch - no previous data
                experiences_per_second = 0.0
                experiences_per_epoch = 0
                epoch_duration = 0.0
                reuse_ratio = 0.0
            
            # Print memory stats every 10 epochs for monitoring
            if epoch % 10 == 0:
                print_cuda_memory_stats()
                
                # Print processor stats too
                processor_stats = processor.get_stats()
                print(f"[PROCESSOR] Total processed: {processor_stats['total_processed']:,}, "
                      f"Rate: {processor_stats['avg_rate']:.1f} exp/s, "
                      f"Queue: {processor_stats['queue_size']}")
            
            # NO MORE EXPERIENCE PROCESSING HERE!
            # The dedicated thread handles it continuously in the background
            
            # Process episode distance stats (still needed)
            processed = 0
            while True:
                try:
                    ep_dist_deque.append(ep_distance_queue.get(timeout=0.01))
                    processed += 1
                except Empty:
                    break
            if processed > 0:
                print(f"[MAIN] Processed {processed} episode stats from queue")

            # Check if we have enough samples for training
            buffer_size = len(agent.memory)
            if buffer_size < BATCH_SIZE:
                print(f"[MAIN] Not enough samples for training (have {buffer_size:,}, need {BATCH_SIZE})")
                # Update tracking for next epoch even if we don't train
                epoch_start_time = current_time
                epoch_start_buffer_size = current_buffer_size
                continue

            print(f"[MAIN] Training agent... (Buffer: {buffer_size:,})")
            
            try:
                # Enable mixed precision training if configured
                if USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        lr, avg_reward, loss, td_error = agent.replay(batch_size=BATCH_SIZE, episodes=EPISODES_PER_EPOCH)
                else:
                    lr, avg_reward, loss, td_error = agent.replay(batch_size=BATCH_SIZE, episodes=EPISODES_PER_EPOCH)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[MAIN] CUDA OOM during training: {e}")
                    print("[MAIN] Clearing CUDA cache and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Try with smaller batch size
                    smaller_batch = BATCH_SIZE // 2
                    print(f"[MAIN] Retrying with smaller batch size: {smaller_batch}")
                    try:
                        if USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                lr, avg_reward, loss, td_error = agent.replay(batch_size=smaller_batch, episodes=EPISODES_PER_EPOCH)
                        else:
                            lr, avg_reward, loss, td_error = agent.replay(batch_size=smaller_batch, episodes=EPISODES_PER_EPOCH)
                    except RuntimeError as e2:
                        print(f"[MAIN] Still failing with smaller batch: {e2}")
                        print("[MAIN] Skipping this training step")
                        # Update tracking for next epoch
                        epoch_start_time = current_time
                        epoch_start_buffer_size = current_buffer_size
                        continue
                else:
                    raise e

            # OPTIMIZATION: Only update collector models every N epochs to reduce overhead
            if ASYNC_MODEL_UPDATES and epoch % MODEL_UPDATE_FREQUENCY == 0:
                print(f"[MAIN] Updating collector models (epoch {epoch}, frequency: every {MODEL_UPDATE_FREQUENCY})")
                # give each collector process a model update
                for _ in range(NUM_PROCESSES):  # type: ignore
                    model_queue.put((agent.q_network.state_dict(), agent.epsilon))
            elif not ASYNC_MODEL_UPDATES:
                # Original behavior - update every epoch
                for _ in range(NUM_PROCESSES):  # type: ignore
                    model_queue.put((agent.q_network.state_dict(), agent.epsilon))

            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0 and epoch != 0:
                # Clear cache before saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                save_checkpoint(agent, epoch, checkpoint_dir=AGENT_FOLDER)

            avg_dist = sum(ep_dist_deque) / len(ep_dist_deque) if ep_dist_deque else 0
            
            # Enhanced log data with experience collection and reuse ratio metrics
            log_data = {
                'td_error': td_error,
                'loss': loss,
                'average_reward': avg_reward,
                'average distance': avg_dist,
                'learning_rate': lr,
                'epsilon': agent.epsilon,
                'buffer_size': len(agent.memory),
                'experiences_per_second': experiences_per_second,
                'experiences_per_epoch': experiences_per_epoch,
                'epoch_duration_seconds': epoch_duration,
                'reuse_ratio': reuse_ratio,
                'throttled': False
            }
            write_log(epoch, log_data)
            
            # Update tracking for next epoch
            epoch_start_time = current_time
            epoch_start_buffer_size = current_buffer_size
            
            # Periodic cleanup to prevent memory leaks
            if epoch % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nStopping training...")
    except Exception as e:
        print(f"[MAIN] Error during training: {str(e)}")
        # Print memory stats on error for debugging
        print_cuda_memory_stats()
    finally:
        # Cleanup - stop processor first, then collectors
        print("[MAIN] Stopping experience processor...")
        processor.stop()
        
        print("[MAIN] Stopping collector processes...")
        stop_event.set()
        for p in collectors:  # type: ignore
            p.join(timeout=10)
        
        env.close()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[MAIN] Final CUDA cache cleanup completed")


if __name__ == "__main__":
    main()
