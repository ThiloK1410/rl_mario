import warnings
warnings.filterwarnings("ignore")

import time
import torch
from collections import deque
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from environment import create_env
from dqn_agent import MarioAgent, DEVICE
from config import (
    BUFFER_SIZE, NUM_EPOCHS, MAX_STEPS_PER_RUN, BATCH_SIZE, 
    EPISODES_PER_EPOCH, LEARNING_RATE, SAVE_INTERVAL, EPSILON_START, 
    EPSILON_DECAY, EPSILON_MIN, GAMMA, AGENT_FOLDER, RANDOM_STAGES
)

# Configuration: Set to False, to use Standard Replay Buffer (for comparison)
USE_PRIORITIZED_REPLAY = True
from tensorboard_logger import TensorBoardLogger, create_experiment_config
import config

# Import common functions
from mario_rl_common import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint, list_available_experiments,
    setup_experiment_and_checkpoint, initialize_agent, log_training_metrics, 
    cleanup_and_save, get_global_stats_with_lock, update_global_stats, 
    increment_level_start_episodes
)


class ExperienceCollector:
    """Thread-safe experience collector that runs environments in parallel."""
    
    def __init__(self, agent, experience_queue, num_collectors=2):
        self.agent = agent
        self.experience_queue = experience_queue
        self.num_collectors = num_collectors
        self.running = False
        self.collectors = []
        self.executor = None
        
    def start(self):
        """Start the experience collection threads."""
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.num_collectors)
        
        # Start collector threads
        for i in range(self.num_collectors):
            future = self.executor.submit(self._collect_experiences, i)
            self.collectors.append(future)
        
        # Only print once at startup
        if self.num_collectors == 1:
            print(f"[COLLECTOR] Started experience collection")
        else:
            print(f"[COLLECTOR] Started {self.num_collectors} parallel collectors")
    
    def stop(self):
        """Stop the experience collection threads."""
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=True)
        # No print on stop to reduce noise
    
    def _collect_experiences(self, collector_id):
        """Collect experiences in a single thread."""
        # Create environment for this collector
        env = create_env()
        
        try:
            while self.running:
                # Initialize environment
                state = env.reset()
                episode_reward = 0
                episode_distance = 0
                step_count = 0
                
                # Get initial x_pos to establish starting position
                try:
                    _, _, done, initial_info = env.step(0)  # No-op to get initial info
                    if done:
                        state = env.reset()
                        _, _, _, initial_info = env.step(0)
                except ValueError as e:
                    if "cannot step in a done environment" in str(e):
                        state = env.reset()
                        _, _, _, initial_info = env.step(0)
                    else:
                        raise e
                
                start_x_pos = initial_info.get('x_pos', 0)
                max_x_pos = start_x_pos
                
                # Track if current episode started from beginning of level
                started_from_beginning = not env.get_used_recorded_start()
                
                # Thread-safe increment of level start episodes
                if started_from_beginning:
                    increment_level_start_episodes()
                
                # Collect experiences for this episode
                while self.running:
                    # Get action from agent (thread-safe)
                    action = self.agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    
                    # Add experience to queue for training thread
                    experience = (state, action, reward, next_state, done)
                    try:
                        self.experience_queue.put(experience, timeout=0.1)
                    except queue.Full:
                        # Queue is full, skip this experience
                        pass
                    
                    # Update tracking
                    episode_reward += reward
                    current_x_pos = info.get('x_pos', 0)
                    max_x_pos = max(max_x_pos, current_x_pos)
                    step_count += 1
                    
                    # Check for episode end
                    if done or (0 < MAX_STEPS_PER_RUN <= step_count):
                        # Calculate distance traveled
                        episode_distance = max_x_pos - start_x_pos
                        
                        # Update global statistics
                        flag_completed = info.get('flag_get', False)
                        update_global_stats(episode_distance, started_from_beginning, flag_completed)
                        
                        break
                    else:
                        state = next_state
                
        except Exception as e:
            print(f"[ERROR] Collector {collector_id} failed: {e}")
        finally:
            env.close()


def parallel_collect_and_train(agent, epoch, experiences_consumed_per_epoch, reuse_ratio_threshold):
    """Collect experiences and train in parallel using threading."""
    # Create experience queue for communication between threads
    experience_queue = queue.Queue(maxsize=1000)  # Buffer experiences
    
    # Start experience collection in background
    collector = ExperienceCollector(agent, experience_queue, num_collectors=2)
    collector.start()
    
    # Training variables
    experiences_added_since_training = 0
    training_occurred = False
    
    try:
        # Collect experiences until we have enough for training
        while True:
            try:
                # Get experience from queue (blocking with timeout)
                experience = experience_queue.get(timeout=1.0)
                
                # Add to agent's memory
                state, action, reward, next_state, done = experience
                agent.remember(state, action, reward, next_state, done)
                experiences_added_since_training += 1
                
                # Check if we have enough experiences for training
                if len(agent.memory) >= BATCH_SIZE and experiences_added_since_training > 0:
                    # Calculate reuse ratio
                    reuse_ratio = experiences_consumed_per_epoch / experiences_added_since_training
                    
                    # Train if reuse ratio is acceptable
                    if reuse_ratio <= reuse_ratio_threshold:
                        # Perform training (no print here to reduce noise)
                        lr, avg_reward, loss, td_error = agent.replay(
                            batch_size=BATCH_SIZE, 
                            episodes=EPISODES_PER_EPOCH
                        )
                        
                        training_occurred = True
                        experiences_added_since_training = 0
                        break
                        
            except queue.Empty:
                # No experiences available, continue
                continue
                
    finally:
        # Stop experience collection
        collector.stop()
    
    if not training_occurred:
        return None  # No training occurred
    
    # Get global statistics
    global_stats = get_global_stats_with_lock()
    
    return {
        'lr': lr,
        'avg_reward': avg_reward,
        'loss': loss,
        'td_error': td_error,
        'episode_count': 0,  # Not tracked in parallel version
        'experiences_added': experiences_added_since_training,
        'reuse_ratio': experiences_consumed_per_epoch / max(experiences_added_since_training, 1),
        **global_stats  # Add all global statistics
    }


def main():
    print("[PARALLEL MARIO RL] Starting parallel training pipeline")
    
    tb_logger, latest_checkpoint, loaded_experiment_name = setup_experiment_and_checkpoint()
    
    # Training parameters
    reuse_ratio_threshold = config.REUSE_FACTOR
    
    # Initialize agent (no environment needed here - created in collector threads)
    agent, start_epoch, hparams = initialize_agent(
        tb_logger, loaded_experiment_name, latest_checkpoint, USE_PRIORITIZED_REPLAY
    )
    
    # Calculate experiences consumed per epoch
    experiences_consumed_per_epoch = BATCH_SIZE * EPISODES_PER_EPOCH

    print(f"[TRAINING] Starting from epoch {start_epoch}, target: {NUM_EPOCHS}")
    print(f"[TRAINING] Using {2} parallel collectors, reuse ratio: {reuse_ratio_threshold}")
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Update global epoch counter for curriculum learning
            config.update_training_epoch(epoch)
            
            # Collect experiences and train in parallel
            metrics = parallel_collect_and_train(agent, epoch, experiences_consumed_per_epoch, reuse_ratio_threshold)
            
            # Skip logging if no training occurred
            if not metrics:
                continue
            
            # Update agent's epsilon scheduler with flag completions
            agent.update_flag_completions(metrics['flag_completions'])
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            
            # Save checkpoint (only print every SAVE_INTERVAL epochs)
            if epoch % SAVE_INTERVAL == 0:
                save_checkpoint(agent, epoch, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
            
            # Simplified console logging - only essential info
            epsilon_info = agent.get_epsilon_info()
            epsilon_phase_indicator = ""
            if epsilon_info['phase'] == 'Fine-tuning':
                epsilon_phase_indicator = "🎯"
            elif epsilon_info['phase'] == 'Regular (Ready for Fine-tuning)':
                epsilon_phase_indicator = "⏳"
            
            # Single line per epoch with key metrics
            level_vs_recorded = f"L:{metrics['avg_level_start_distance']:.1f} R:{metrics['avg_recorded_start_distance']:.1f}" if metrics['level_start_episodes'] > 0 and metrics['recorded_start_episodes'] > 0 else f"Best:{metrics['best_level_start_distance']:.1f}"
            flag_info = f"F:{metrics['flag_completions']}/{metrics['level_start_episodes']}" if metrics['level_start_episodes'] > 0 else "F:0/0"
            
            print(f"[{epoch:4d}] Loss:{metrics['loss']:.3f} Dist:{metrics['avg_distance']:.1f} {level_vs_recorded} {flag_info} ε:{agent.epsilon:.3f}{epsilon_phase_indicator} Buf:{len(agent.memory)} ({epoch_duration:.1f}s)")
            
            # Log to TensorBoard (but suppress the verbose console output)
            # We'll modify the common function to be less verbose
            tb_logger.log_metrics({
                'Training/Loss': metrics['loss'],
                'Training/TD_Error': metrics['td_error'],
                'Performance/Average_Reward': metrics['avg_reward'],
                'Performance/Average_Distance': metrics['avg_distance'],
                'Performance/Episode_Count': metrics['episode_count'],
                'Performance/Best_Level_Start_Distance': metrics['best_level_start_distance'],
                'Performance/Avg_Level_Start_Distance': metrics['avg_level_start_distance'],
                'Performance/Avg_Recorded_Start_Distance': metrics['avg_recorded_start_distance'],
                'Performance/Level_Start_Episodes': metrics['level_start_episodes'],
                'Performance/Recorded_Start_Episodes': metrics['recorded_start_episodes'],
                'Performance/Flag_Completions': metrics['flag_completions'],
                'Performance/Flag_Completion_Rate': metrics['flag_completion_rate'],
                'Hyperparameters/Learning_Rate': metrics['lr'],
                'Hyperparameters/Epsilon': agent.epsilon,
                'Hyperparameters/Epsilon_Phase': 1 if epsilon_info['phase'] == 'Regular' else 2,
                'Hyperparameters/Recorded_Start_Probability': config.get_recorded_start_probability(epoch),
                'System/Buffer_Size': len(agent.memory),
                'System/Epoch_Duration': epoch_duration,
                'System/Experiences_Added': metrics['experiences_added'],
                'System/Reuse_Ratio': metrics['reuse_ratio']
            }, epoch)
            
            # Log model parameters periodically (but don't print about it)
            if epoch % 50 == 0:
                tb_logger.log_model_parameters(agent.q_network, epoch)
    
    except KeyboardInterrupt:
        print("\n[TRAINING] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup and save final checkpoint
        final_epoch = epoch if 'epoch' in locals() else 0
        cleanup_and_save(tb_logger, agent, final_epoch, loaded_experiment_name, hparams)
        
        print("[TRAINING] Completed")


if __name__ == "__main__":
    main() 