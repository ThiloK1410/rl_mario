import warnings
warnings.filterwarnings("ignore")

import time
import torch
from collections import deque
import os

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
    setup_experiment_and_checkpoint, initialize_environment_and_agent, 
    log_training_metrics, cleanup_and_save, get_global_stats_with_lock,
    update_global_stats, increment_level_start_episodes
)


def collect_experiences_and_train(agent, env, epoch, experiences_consumed_per_epoch, reuse_ratio_threshold):
    """Collect experiences and perform training when conditions are met."""
    # Initialize environment
    state = env.reset()
    episode_reward = 0
    episode_distance = 0
    episode_count = 0
    step_count = 0
    
    # Get initial x_pos to establish starting position
    # Handle case where environment might be in done state after reset
    try:
        _, _, done, initial_info = env.step(0)  # No-op to get initial info
        if done:
            # Environment is done after reset, reset again
            state = env.reset()
            _, _, _, initial_info = env.step(0)  # Try no-op again
    except ValueError as e:
        if "cannot step in a done environment" in str(e):
            # Environment is in done state, reset and try again
            state = env.reset()
            _, _, _, initial_info = env.step(0)
        else:
            raise e
    
    start_x_pos = initial_info.get('x_pos', 0)
    max_x_pos = start_x_pos
    
    # Track if current episode started from beginning of level (not recorded position)
    started_from_beginning = not env.get_used_recorded_start()
    if started_from_beginning:
        increment_level_start_episodes()
    
    # Tracking variables
    experiences_added_since_training = 0  # Simple counter for experiences added
    
    training_occurred = False
    
    # Collect experiences until we have enough for training
    while True:
        # Agent takes action
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Store experience and count it
        agent.remember(state, action, reward, next_state, done)
        experiences_added_since_training += 1
        
        # Update tracking
        episode_reward += reward
        current_x_pos = info.get('x_pos', 0)
        max_x_pos = max(max_x_pos, current_x_pos)
        step_count += 1
        
        # Check for episode end
        if done or (MAX_STEPS_PER_RUN > 0 and step_count >= MAX_STEPS_PER_RUN):
            # Calculate distance traveled from start to maximum reached position
            episode_distance = max_x_pos - start_x_pos
            episode_count += 1
            
            # Update global statistics
            flag_completed = info.get('flag_get', False)
            update_global_stats(episode_distance, started_from_beginning, flag_completed)
            
            # Reset for next episode
            state = env.reset()
            # Get starting position for new episode
            # Handle case where environment might be in done state after reset
            try:
                _, _, done, initial_info = env.step(0)  # No-op to get initial info
                if done:
                    # Environment is done after reset, reset again
                    state = env.reset()
                    _, _, _, initial_info = env.step(0)  # Try no-op again
            except ValueError as e:
                if "cannot step in a done environment" in str(e):
                    # Environment is in done state, reset and try again
                    state = env.reset()
                    _, _, _, initial_info = env.step(0)
                else:
                    raise e
            
            start_x_pos = initial_info.get('x_pos', 0)
            max_x_pos = start_x_pos
            episode_reward = 0
            episode_distance = 0
            step_count = 0
            
            # Track if new episode started from beginning of level
            started_from_beginning = not env.get_used_recorded_start()
            if started_from_beginning:
                increment_level_start_episodes()
        else:
            state = next_state
        
        # Check if we have enough experiences for training
        if len(agent.memory) >= BATCH_SIZE and experiences_added_since_training > 0:
            # Calculate reuse ratio
            reuse_ratio = experiences_consumed_per_epoch / experiences_added_since_training
            
            # Train if reuse ratio is acceptable
            if reuse_ratio <= reuse_ratio_threshold:
                print(f"\n[EPOCH {epoch}] Training triggered")
                print(f"[RATIO] Collected: {experiences_added_since_training}, "
                      f"Will consume: {experiences_consumed_per_epoch}, "
                      f"Ratio: {reuse_ratio:.2f}")
                
                # Perform training
                lr, avg_reward, loss, td_error = agent.replay(
                    batch_size=BATCH_SIZE, 
                    episodes=EPISODES_PER_EPOCH
                )
                
                training_occurred = True
                experiences_added_since_training = 0  # Reset counter
                break
    
    if not training_occurred:
        return None  # No training occurred
    
    # Get global statistics
    global_stats = get_global_stats_with_lock()
    
    return {
        'lr': lr,
        'avg_reward': avg_reward,
        'loss': loss,
        'td_error': td_error,
        'episode_count': episode_count,
        'experiences_added': experiences_added_since_training,
        'reuse_ratio': experiences_consumed_per_epoch / max(experiences_added_since_training, 1),
        **global_stats  # Add all global statistics
    }


def main():
    print("[SIMPLE MARIO RL] Starting non-parallel training pipeline")
    
    tb_logger, latest_checkpoint, loaded_experiment_name = setup_experiment_and_checkpoint()
    
    # Training parameters - ADJUSTED for more frequent training
    reuse_ratio_threshold = config.REUSE_FACTOR
    
    # Initialize environment and agent
    env, agent, start_epoch, hparams = initialize_environment_and_agent(
        tb_logger, loaded_experiment_name, latest_checkpoint, USE_PRIORITIZED_REPLAY
    )
    
    # Calculate experiences consumed per epoch
    experiences_consumed_per_epoch = BATCH_SIZE * EPISODES_PER_EPOCH

    print(f"[TRAINING] Reuse ratio threshold: {reuse_ratio_threshold}")
    print(f"[TRAINING] Collecting minimum experiences...")
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Update global epoch counter for curriculum learning
            config.update_training_epoch(epoch)
            
            # Collect experiences and train
            metrics = collect_experiences_and_train(agent, env, epoch, experiences_consumed_per_epoch, reuse_ratio_threshold)
            
            # Skip logging if no training occurred
            if not metrics:
                continue
            
            # Update agent's epsilon scheduler with flag completions
            agent.update_flag_completions(metrics['flag_completions'])
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            
            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0:
                save_checkpoint(agent, epoch, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
            
            # Log training metrics
            log_training_metrics(tb_logger, agent, epoch, metrics, epoch_duration, prefix="SIMPLE ")
    
    except KeyboardInterrupt:
        print("\n[TRAINING] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
    finally:
        # Cleanup and save final checkpoint
        final_epoch = epoch if 'epoch' in locals() else 0
        cleanup_and_save(tb_logger, agent, final_epoch, loaded_experiment_name, hparams)
        
        env.close()
        print("[TRAINING] Environment closed")


if __name__ == "__main__":
    main() 