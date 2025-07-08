import warnings
from _queue import Empty
warnings.filterwarnings("ignore")

import random
import time
from time import sleep
import os
from multiprocessing import Queue, Event, Lock, Value, Array
from multiprocessing import Process
import torch.multiprocessing as mp
import pandas as pd
import threading
import queue
import numpy as np

import torch
from collections import deque

from environment import create_env
from dqn_agent import DQN, MarioAgent, DEVICE
from dqn_agent_optimized import OptimizedMarioAgent, OptimizedDQN
from config import (
    DATA_FILE, REP_Q_SIZE, BUFFER_SIZE, NUM_EPOCHS, DEADLOCK_STEPS,
    MAX_STEPS_PER_RUN, BATCH_SIZE, EPISODES_PER_EPOCH, LEARNING_RATE,
    SAVE_INTERVAL, EPSILON_START, EPSILON_DECAY, EPSILON_MIN, GAMMA, AGENT_FOLDER, NUM_PROCESSES, RANDOM_STAGES,
    RANDOM_SAVES, MODEL_UPDATE_FREQUENCY, USE_MIXED_PRECISION, ASYNC_MODEL_UPDATES,
    PER_ALPHA, PER_BETA, PER_BETA_INCREMENT
)

mp.set_start_method('spawn', force=True)

# Enable CUDA optimization flags for maximum performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class LockFreeExperienceProcessor:
    """
    Lock-free experience processor using thread-local queues and batch processing.
    Minimizes lock contention by using multiple queues and lock-free data structures.
    """
    
    def __init__(self, agent, num_collectors=2, batch_size=1000, process_interval=0.001):
        self.agent = agent
        self.batch_size = batch_size
        self.process_interval = process_interval
        
        # Use thread-safe queue.Queue instead of multiprocessing Queue for lower overhead
        self.local_queues = [queue.Queue(maxsize=10000) for _ in range(num_collectors)]
        self.collector_assignments = {}  # Map collector ID to queue index
        
        # Single consolidated queue for batch processing
        self.consolidated_queue = queue.Queue(maxsize=REP_Q_SIZE * 2)
        
        # Thread control
        self.running = False
        self.consolidation_thread = None
        self.processing_thread = None
        self._shutdown_event = threading.Event()
        
        # Pre-allocated numpy arrays for batch processing (reduce allocations)
        self.batch_buffer = []
        self.batch_buffer_lock = threading.Lock()
        
        # Statistics tracking with atomic operations
        self.stats = {
            'total_processed': Value('i', 0),
            'total_batches': Value('i', 0),
            'queue_drops': Value('i', 0),
            'consolidation_cycles': Value('i', 0)
        }
        self._last_stats_time = time.time()
        
    def get_collector_queue(self, collector_id):
        """Get a dedicated queue for a collector to minimize contention."""
        if collector_id not in self.collector_assignments:
            # Assign collector to least loaded queue
            min_size = float('inf')
            min_idx = 0
            for i, q in enumerate(self.local_queues):
                size = q.qsize()
                if size < min_size:
                    min_size = size
                    min_idx = i
            self.collector_assignments[collector_id] = min_idx
        
        return self.local_queues[self.collector_assignments[collector_id]]
    
    def start(self):
        """Start the processing threads."""
        if self.running:
            return
        
        self.running = True
        self._shutdown_event.clear()
        
        # Start consolidation thread (merges local queues)
        self.consolidation_thread = threading.Thread(
            target=self._consolidation_loop, 
            daemon=True, 
            name="ExperienceConsolidator"
        )
        self.consolidation_thread.start()
        
        # Start processing thread (processes consolidated experiences)
        self.processing_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="ExperienceProcessor"
        )
        self.processing_thread.start()
        
        print("[PROCESSOR] Lock-free experience processor started")
    
    def stop(self, timeout=5.0):
        """Stop processing threads."""
        if not self.running:
            return
        
        print("[PROCESSOR] Stopping lock-free processor...")
        self.running = False
        self._shutdown_event.set()
        
        for thread in [self.consolidation_thread, self.processing_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
        
        self._print_final_stats()
    
    def _consolidation_loop(self):
        """Consolidate experiences from multiple queues with minimal locking."""
        local_batch = []
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Collect from all local queues in round-robin fashion
                collected = 0
                for q in self.local_queues:
                    try:
                        # Non-blocking get with timeout
                        for _ in range(100):  # Process up to 100 items per queue
                            experience = q.get_nowait()
                            local_batch.append(experience)
                            collected += 1
                    except queue.Empty:
                        continue
                
                # Push to consolidated queue in batches
                if local_batch:
                    try:
                        # Try to push all at once
                        for exp in local_batch:
                            self.consolidated_queue.put_nowait(exp)
                        local_batch.clear()
                    except queue.Full:
                        # Queue full - drop oldest experiences
                        dropped = len(local_batch)
                        self.stats['queue_drops'].value += dropped
                        local_batch.clear()
                
                self.stats['consolidation_cycles'].value += 1
                
                # Adaptive sleep based on activity
                if collected == 0:
                    time.sleep(0.001)  # 1ms sleep when idle
                elif collected < 50:
                    time.sleep(0.0001)  # 0.1ms sleep for low activity
                # No sleep for high activity
                    
            except Exception as e:
                print(f"[CONSOLIDATOR] Error: {e}")
                time.sleep(0.01)
    
    def _process_loop(self):
        """Process consolidated experiences in large batches."""
        experiences_batch = []
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Collect a large batch
                deadline = time.time() + 0.05  # 50ms collection window
                
                while time.time() < deadline and len(experiences_batch) < self.batch_size * 2:
                    try:
                        exp = self.consolidated_queue.get(timeout=0.001)
                        experiences_batch.append(exp)
                    except queue.Empty:
                        break
                
                # Process batch if we have enough
                if experiences_batch:
                    self._process_batch(experiences_batch)
                    self.stats['total_processed'].value += len(experiences_batch)
                    self.stats['total_batches'].value += 1
                    experiences_batch = []
                
                # Print stats periodically
                if time.time() - self._last_stats_time >= 30:
                    self._print_stats()
                    self._last_stats_time = time.time()
                
                # Only sleep if we didn't process anything
                if not experiences_batch:
                    time.sleep(self.process_interval)
                    
            except Exception as e:
                print(f"[PROCESSOR] Error: {e}")
                time.sleep(0.01)
    
    def _process_batch(self, experiences):
        """Process a batch of experiences efficiently."""
        if not experiences:
            return
        
        # Batch processing with vectorized operations
        try:
            # Check if replay buffer supports batch operations
            if hasattr(self.agent.memory, 'push_batch'):
                self.agent.memory.push_batch(experiences)
            else:
                # Fallback to individual pushes
                for exp in experiences:
                    state, action, reward, next_state, done = exp
                    self.agent.memory.push(state, action, reward, next_state, done)
        except Exception as e:
            print(f"[PROCESSOR] Batch processing error: {e}")
    
    def _print_stats(self):
        """Print processing statistics."""
        total_processed = self.stats['total_processed'].value
        total_batches = self.stats['total_batches'].value
        queue_drops = self.stats['queue_drops'].value
        consolidation_cycles = self.stats['consolidation_cycles'].value
        
        buffer_size = len(self.agent.memory)
        consolidated_size = self.consolidated_queue.qsize()
        
        print(f"[PROCESSOR] Processed: {total_processed:,}, Batches: {total_batches:,}, "
              f"Drops: {queue_drops:,}, Buffer: {buffer_size:,}, "
              f"Consolidated Queue: {consolidated_size}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        print(f"[PROCESSOR] Final Stats:")
        print(f"  Total processed: {self.stats['total_processed'].value:,}")
        print(f"  Total batches: {self.stats['total_batches'].value:,}")
        print(f"  Queue drops: {self.stats['queue_drops'].value:,}")
        print(f"  Consolidation cycles: {self.stats['consolidation_cycles'].value:,}")


def optimized_collector_process(processor, model_queue, distance_queue, stop_event, epsilon, collector_id):
    """Optimized collector with reduced lock contention and batch inference."""
    try:
        env = create_env(sample_random_stages=RANDOM_STAGES, use_random_saves=RANDOM_SAVES)
        print(f"[Collector {collector_id}] Started with lock-free queue")
        
        # Get dedicated queue for this collector
        experience_queue = processor.get_collector_queue(collector_id)
        
        # Initialize local model (use optimized version)
        local_model = OptimizedDQN((128, 128), env.action_space.n).to(DEVICE)
        local_model.eval()
        
        # Enable mixed precision for inference
        if USE_MIXED_PRECISION:
            from torch.cuda.amp import autocast
        
        # Pre-allocate tensors for batch inference (reduce allocations)
        state_buffer = torch.zeros((8, 8, 128, 128), dtype=torch.float32, device=DEVICE)
        buffer_idx = 0
        
        # Statistics
        episode_count = 0
        recent_rewards = deque(maxlen=100)
        best_reward = 0
        
        while not stop_event.is_set():
            try:
                # Episode initialization
                state = env.reset()
                _, _, _, info = env.step(0)  # Get initial info
                start_x = info['x_pos']
                best_x = 0
                
                done = False
                total_reward = 0
                steps = 0
                
                # Batch for local experience accumulation
                episode_experiences = []
                
                while not done and not stop_event.is_set():
                    # Check for model updates (non-blocking)
                    try:
                        if not model_queue.empty():
                            weights, new_epsilon = model_queue.get_nowait()
                            local_model.load_state_dict(weights)
                            epsilon = new_epsilon
                    except Empty:
                        pass
                    
                    # Action selection with batch inference optimization
                    if random.random() <= epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            # Convert state directly to tensor (avoid numpy conversion)
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                            
                            # Use mixed precision for faster inference
                            if USE_MIXED_PRECISION:
                                with autocast():
                                    q_values = local_model(state_tensor)
                            else:
                                q_values = local_model(state_tensor)
                            
                            action = q_values.argmax().item()
                    
                    # Environment step
                    next_state, reward, done, info = env.step(action)
                    best_x = max(best_x, info['x_pos'])
                    
                    # Accumulate experience locally first
                    episode_experiences.append((state, action, reward, next_state, done))
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if MAX_STEPS_PER_RUN != 0 and steps >= MAX_STEPS_PER_RUN:
                        done = True
                    
                    # Push experiences in batches to reduce queue operations
                    if len(episode_experiences) >= 32:  # Batch size
                        try:
                            for exp in episode_experiences:
                                experience_queue.put_nowait(exp)
                            episode_experiences.clear()
                        except queue.Full:
                            # Queue full, skip this batch
                            episode_experiences.clear()
                
                # Push remaining experiences
                if episode_experiences:
                    try:
                        for exp in episode_experiences:
                            experience_queue.put_nowait(exp)
                    except queue.Full:
                        pass
                
                # Update statistics
                if total_reward > best_reward:
                    best_reward = total_reward
                recent_rewards.append(total_reward)
                episode_count += 1
                
                # Track distance for non-recorded starts
                if best_x != start_x:
                    if hasattr(env, 'get_used_recorded_start') and not env.get_used_recorded_start():
                        distance_queue.put(best_x - start_x)
                    elif not hasattr(env, 'get_used_recorded_start'):
                        distance_queue.put(best_x - start_x)
                
                # Print statistics every 20 episodes
                if episode_count % 20 == 0:
                    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                    print(f"[Collector {collector_id}] Episode {episode_count} - "
                          f"Avg reward: {avg_reward:.2f}, Best: {best_reward:.2f}")
                    
            except Exception as e:
                print(f"[Collector {collector_id}] Error in episode: {e}")
                continue
        
        print(f"[Collector {collector_id}] Stopping")
        
    except Exception as e:
        print(f"[Collector {collector_id}] Fatal error: {e}")
    finally:
        if 'env' in locals():
            env.close()


def save_checkpoint(agent, epoch, checkpoint_dir='checkpoints'):
    """Save checkpoint with minimal overhead."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': agent.q_network.state_dict(),
        'epsilon': agent.epsilon,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'mario_agent_epoch_{epoch}.pt')
    
    # Save asynchronously to avoid blocking training
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(agent, checkpoint_path):
    """Load checkpoint efficiently."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
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
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


def write_log(epoch, data_dict):
    """Write training log."""
    data_dict['epoch'] = epoch
    file_exists = os.path.exists(DATA_FILE)
    df = pd.DataFrame([data_dict])
    df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)


def main():
    """Optimized main training loop with reduced overhead."""
    # Set environment variables for performance
    os.environ['OMP_NUM_THREADS'] = '1'  # Reduce thread contention
    os.environ['MKL_NUM_THREADS'] = '1'
    
    if not os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Create environment for initialization
    env = create_env(sample_random_stages=RANDOM_STAGES, use_random_saves=RANDOM_SAVES)
    
    # Initialize optimized agent
    agent = OptimizedMarioAgent(
        (128, 128), 
        env.action_space.n, 
        None,  # No experience queue needed
        memory_size=BUFFER_SIZE, 
        gamma=GAMMA, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN, 
        lr=LEARNING_RATE, 
        epsilon=EPSILON_START
    )
    
    # Enable gradient accumulation for more efficient training
    if hasattr(agent, 'q_network'):
        for param in agent.q_network.parameters():
            param.register_hook(lambda grad: grad * 0.5)  # Scale gradients for stability
    
    # Create shared resources
    model_queue = Queue()
    ep_distance_queue = Queue()
    stop_event = Event()
    
    # Initialize lock-free processor
    processor = LockFreeExperienceProcessor(
        agent=agent,
        num_collectors=NUM_PROCESSES,
        batch_size=1000,  # Larger batches for efficiency
        process_interval=0.001
    )
    
    # Start processor
    processor.start()
    
    # Load checkpoint if available
    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        start_epoch = load_checkpoint(agent, latest_checkpoint)
    
    # Start collector processes
    collectors = []
    for i in range(NUM_PROCESSES):
        p = Process(
            target=optimized_collector_process,
            args=(processor, model_queue, ep_distance_queue, stop_event, agent.epsilon, i+1)
        )
        collectors.append(p)
        p.start()
    
    # Send initial model weights
    for _ in range(NUM_PROCESSES):
        model_queue.put((agent.q_network.state_dict(), agent.epsilon))
    
    # Wait for initial buffer fill
    print(f"[MAIN] Waiting for buffer to reach {BUFFER_SIZE//32} samples...")
    while len(agent.memory) < BUFFER_SIZE // 32:
        time.sleep(1)
        print(f"[MAIN] Buffer: {len(agent.memory):,}/{BUFFER_SIZE//32:,}")
    
    # Distance tracking
    ep_dist_deque = deque(maxlen=50)
    
    # Training loop with optimizations
    try:
        # Pre-compile model for faster execution
        if torch.cuda.is_available():
            print("[MAIN] Compiling model with torch.compile for faster execution...")
            agent.q_network = torch.compile(agent.q_network, mode="reduce-overhead")
            agent.target_network = torch.compile(agent.target_network, mode="reduce-overhead")
        
        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\n[MAIN] Epoch {epoch}")
            
            # Process distance stats
            while True:
                try:
                    ep_dist_deque.append(ep_distance_queue.get(timeout=0.001))
                except Empty:
                    break
            
            # Check buffer size
            buffer_size = len(agent.memory)
            if buffer_size < BATCH_SIZE:
                print(f"[MAIN] Insufficient samples: {buffer_size}/{BATCH_SIZE}")
                continue
            
            # Optimized training with larger batches and gradient accumulation
            accumulation_steps = 4  # Accumulate gradients over 4 steps
            effective_batch_size = BATCH_SIZE * accumulation_steps
            
            print(f"[MAIN] Training with effective batch size: {effective_batch_size}")
            
            # Reset accumulated values
            total_loss = 0
            total_td_error = 0
            total_reward = 0
            
            for episode in range(EPISODES_PER_EPOCH):
                for acc_step in range(accumulation_steps):
                    # Sample batch
                    batch = agent.memory.sample(BATCH_SIZE)
                    if batch is None:
                        continue
                    
                    # Process batch with mixed precision
                    if USE_MIXED_PRECISION and scaler is not None:
                        with torch.cuda.amp.autocast():
                            loss, td_error, reward = agent.process_batch(batch, accumulate=(acc_step < accumulation_steps - 1))
                        
                        # Scale loss for gradient accumulation
                        loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        loss, td_error, reward = agent.process_batch(batch, accumulate=(acc_step < accumulation_steps - 1))
                        loss = loss / accumulation_steps
                        loss.backward()
                    
                    total_loss += loss.item()
                    total_td_error += td_error
                    total_reward += reward
                
                # Update weights after accumulation
                if USE_MIXED_PRECISION and scaler is not None:
                    scaler.step(agent.optimizer)
                    scaler.update()
                else:
                    agent.optimizer.step()
                
                agent.optimizer.zero_grad()
                
                # Soft update target network
                with torch.no_grad():
                    for target_param, param in zip(agent.target_network.parameters(), agent.q_network.parameters()):
                        target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
            
            # Update epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)
            
            # Average metrics
            avg_loss = total_loss / (EPISODES_PER_EPOCH * accumulation_steps)
            avg_td_error = total_td_error / (EPISODES_PER_EPOCH * accumulation_steps)
            avg_reward = total_reward / (EPISODES_PER_EPOCH * accumulation_steps)
            
            # Update collectors with new model (less frequently)
            if epoch % MODEL_UPDATE_FREQUENCY == 0:
                print(f"[MAIN] Updating collector models...")
                model_state = agent.q_network.state_dict()
                for _ in range(NUM_PROCESSES):
                    model_queue.put((model_state, agent.epsilon))
            
            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0 and epoch != 0:
                save_checkpoint(agent, epoch, checkpoint_dir=AGENT_FOLDER)
            
            # Log metrics
            avg_dist = sum(ep_dist_deque) / len(ep_dist_deque) if ep_dist_deque else 0
            log_data = {
                'loss': avg_loss,
                'td_error': avg_td_error,
                'average_reward': avg_reward,
                'average_distance': avg_dist,
                'learning_rate': agent.optimizer.param_groups[0]['lr'],
                'epsilon': agent.epsilon,
                'buffer_size': buffer_size,
            }
            write_log(epoch, log_data)
            
            # Print epoch summary
            print(f"[MAIN] Epoch {epoch} - Loss: {avg_loss:.4f}, TD Error: {avg_td_error:.4f}, "
                  f"Reward: {avg_reward:.2f}, Distance: {avg_dist:.1f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Buffer: {buffer_size:,}")
            
    except KeyboardInterrupt:
        print("\n[MAIN] Training interrupted")
    finally:
        # Cleanup
        print("[MAIN] Stopping processor...")
        processor.stop()
        
        print("[MAIN] Stopping collectors...")
        stop_event.set()
        for p in collectors:
            p.join(timeout=5)
        
        env.close()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[MAIN] Cleanup complete")


if __name__ == "__main__":
    main()