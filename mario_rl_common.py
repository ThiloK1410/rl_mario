"""
Common utilities for Mario RL training pipelines.
Shared functions used by both mario_rl.py and mario_rl_simple.py.
"""

import os
import glob
import torch
from collections import deque
import threading

from dqn_agent import MarioAgent, DEVICE
from environment import create_env, create_env_new
from tensorboard_logger import TensorBoardLogger, create_experiment_config
from config import (
    BUFFER_SIZE, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, 
    EPSILON_START, AGENT_FOLDER, RANDOM_STAGES
)
import config


# Global distance tracker for smooth averaging across epochs
_global_distance_tracker = deque(maxlen=100)  # Track last 100 episode distances across all epochs

# Global progress tracking for episodes that started from beginning vs recorded positions
_global_level_start_distances = deque(maxlen=100)  # Track distances from level start episodes
_global_recorded_start_distances = deque(maxlen=100)  # Track distances from recorded start episodes
_global_best_level_start_distance = 0  # Track best distance ever achieved from level start

# Global flag completion tracker for episodes that started from beginning of level (as specifically requested)
_global_flag_completions = 0  # Count total flag completions from level start
_global_level_start_episodes = 0  # Count total episodes that started from level start

# Thread-safe locks for global variables
_global_stats_lock = threading.Lock()


def save_checkpoint(agent, epoch, experiment_name, checkpoint_dir='checkpoints'):
    """Save agent checkpoint without replay buffer."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Use experiment name in checkpoint filename to avoid conflicts between parallel experiments
    safe_name = experiment_name.replace('/', '_').replace('\\', '_')  # Make filename safe
    checkpoint_path = f"{checkpoint_dir}/mario_agent_{safe_name}.pt"
    
    # Get epsilon scheduler state
    epsilon_info = agent.get_epsilon_info()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': agent.q_network.state_dict(),
        'target_model_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'epsilon_scheduler_state': {
            'current_epsilon': agent.epsilon_scheduler.current_epsilon,
            'phase': agent.epsilon_scheduler.phase,
            'flag_completions': agent.epsilon_scheduler.flag_completions
        },
        'memory_size': len(agent.memory),
        'experiment_name': experiment_name  # Save experiment name for resuming
    }, checkpoint_path)
    
    print(f"[CHECKPOINT] Saved at epoch {epoch} (buffer size: {len(agent.memory)})")


def load_checkpoint(agent, checkpoint_path):
    """Load agent checkpoint without replay buffer and return starting epoch and experiment name."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load epsilon scheduler state if available (for backward compatibility)
    if 'epsilon_scheduler_state' in checkpoint:
        scheduler_state = checkpoint['epsilon_scheduler_state']
        agent.epsilon_scheduler.current_epsilon = scheduler_state['current_epsilon']
        agent.epsilon_scheduler.phase = scheduler_state['phase']
        agent.epsilon_scheduler.flag_completions = scheduler_state['flag_completions']
        print(f"[CHECKPOINT] Loaded epsilon scheduler: Phase {scheduler_state['phase']}, "
              f"Epsilon: {scheduler_state['current_epsilon']:.3f}, "
              f"Flags: {scheduler_state['flag_completions']}")
    else:
        # Fallback for old checkpoints - set epsilon manually
        agent.epsilon_scheduler.current_epsilon = checkpoint['epsilon']
        print(f"[CHECKPOINT] Legacy checkpoint - set epsilon to {checkpoint['epsilon']:.3f}")
    
    # Return both epoch and experiment name (if available)
    experiment_name = checkpoint.get('experiment_name', None)
    print(f"[CHECKPOINT] Loaded from epoch {checkpoint['epoch']} (buffer will be rebuilt from scratch)")
    if experiment_name:
        print(f"[CHECKPOINT] Resuming experiment: {experiment_name}")
    
    return checkpoint['epoch'], experiment_name


def find_latest_checkpoint(checkpoint_dir='checkpoints', experiment_name=None):
    """Find the checkpoint file for a specific experiment."""
    if experiment_name:
        # Look for experiment-specific checkpoint
        safe_name = experiment_name.replace('/', '_').replace('\\', '_')
        checkpoint_path = f"{checkpoint_dir}/mario_agent_{safe_name}.pt"
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    # Fallback: look for the old format (for backward compatibility)
    old_checkpoint_path = f"{checkpoint_dir}/mario_agent.pt"
    if os.path.exists(old_checkpoint_path):
        return old_checkpoint_path
    
    return None


def list_available_experiments(checkpoint_dir='checkpoints'):
    """List all available experiments from checkpoint files."""
    experiments = []
    
    if not os.path.exists(checkpoint_dir):
        return experiments
    
    # Find all experiment-specific checkpoints
    pattern = os.path.join(checkpoint_dir, "mario_agent_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract experiment name from filename
            filename = os.path.basename(checkpoint_file)
            if filename.startswith("mario_agent_") and filename.endswith(".pt"):
                exp_name = filename[12:-3]  # Remove "mario_agent_" and ".pt"
                exp_name = exp_name.replace('_', '/')  # Restore original name
                experiments.append(exp_name)
        except Exception:
            continue
    
    # Check for old-style checkpoint
    old_checkpoint = os.path.join(checkpoint_dir, "mario_agent.pt")
    if os.path.exists(old_checkpoint):
        try:
            checkpoint_data = torch.load(old_checkpoint, map_location=torch.device('cpu'), weights_only=False)
            exp_name = checkpoint_data.get('experiment_name', 'legacy_experiment')
            experiments.append(exp_name)
        except Exception:
            experiments.append('legacy_experiment')
    
    return list(set(experiments))  # Remove duplicates


def setup_experiment_and_checkpoint():
    """Setup experiment logging and load checkpoints if available."""
    # Try to load latest checkpoint first to get experiment name
    start_epoch = 0
    loaded_experiment_name = None
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        print(f"[CHECKPOINT] Found existing checkpoint: {latest_checkpoint}")
        # Load just the checkpoint metadata to get experiment name
        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
            loaded_experiment_name = checkpoint_data.get('experiment_name', None)
            if loaded_experiment_name:
                print(f"[CHECKPOINT] Found saved experiment name: {loaded_experiment_name}")
                # Now look for the experiment-specific checkpoint
                experiment_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=loaded_experiment_name)
                if experiment_checkpoint and experiment_checkpoint != latest_checkpoint:
                    latest_checkpoint = experiment_checkpoint
                    print(f"[CHECKPOINT] Using experiment-specific checkpoint: {latest_checkpoint}")
        except Exception as e:
            print(f"[CHECKPOINT] Warning: Could not read experiment name from checkpoint: {e}")
    
    # Create TensorBoard logger with experiment name
    print("\n[TENSORBOARD] Setting up experiment logging...")
    if loaded_experiment_name:
        print(f"[TENSORBOARD] Resuming experiment: {loaded_experiment_name}")
        experiment_name = loaded_experiment_name
    else:
        # Show available experiments for reference
        available_experiments = list_available_experiments(checkpoint_dir=AGENT_FOLDER)
        if available_experiments:
            print(f"[INFO] Available experiments with checkpoints: {', '.join(available_experiments)}")
        
        try:
            experiment_name = input("Enter experiment name (or press Enter for auto-generated): ").strip()
            if not experiment_name:
                experiment_name = None
        except (EOFError, KeyboardInterrupt):
            # Handle non-interactive environments
            experiment_name = None
            print("Using auto-generated experiment name")
    
    tb_logger = TensorBoardLogger(experiment_name=experiment_name)
    
    # Now that we have the final experiment name, look for its specific checkpoint
    final_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=tb_logger.experiment_name)
    if final_checkpoint:
        latest_checkpoint = final_checkpoint
        print(f"[CHECKPOINT] Using checkpoint for experiment '{tb_logger.experiment_name}': {latest_checkpoint}")
    
    return tb_logger, latest_checkpoint, loaded_experiment_name


def initialize_agent(tb_logger, loaded_experiment_name, latest_checkpoint, use_prioritized_replay=True, n_actions=None):
    """Initialize the agent and load checkpoint if available."""
    # Log hyperparameters at the start (only for new experiments)
    if not loaded_experiment_name:
        hparams = create_experiment_config(config)
        hparams['USE_PRIORITIZED_REPLAY'] = use_prioritized_replay
        hparams['REUSE_RATIO_THRESHOLD'] = config.REUSE_FACTOR
        print(f"[TENSORBOARD] Logging hyperparameters: {hparams}")
        tb_logger.log_hyperparameters(hparams)
    else:
        print(f"[TENSORBOARD] Resuming experiment - skipping hyperparameter logging")
        hparams = None
    
    # Determine number of actions
    if n_actions is None:
        n_actions = len(config.USED_MOVESET)
    
    # Initialize agent
    agent = MarioAgent(
        n_actions=n_actions,
        memory_size=BUFFER_SIZE, 
        gamma=GAMMA, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN, 
        lr=LEARNING_RATE, 
        epsilon=EPSILON_START
    )
    
    # Override buffer type if requested
    if not use_prioritized_replay:
        from dqn_agent import StandardReplayBuffer
        agent.memory = StandardReplayBuffer(BUFFER_SIZE)
        print(f"[AGENT] Switched to Standard Replay Buffer for comparison")
    
    buffer_type = "Standard Replay" if not use_prioritized_replay else "TorchRL Prioritized Replay"
    print(f"[AGENT] Initialized agent with buffer size {BUFFER_SIZE}")
    print(f"[AGENT] Using {buffer_type} buffer")
    if torch.cuda.is_available():
        print(f"[DEVICE] Using CUDA: {DEVICE}")
    
    # Load the checkpoint if we found one earlier
    start_epoch = 0
    if latest_checkpoint:
        start_epoch, _ = load_checkpoint(agent, latest_checkpoint)  # We already have the experiment name
        print(f"[CHECKPOINT] Resuming from epoch {start_epoch}")
    
    return agent, start_epoch, hparams


def initialize_environment_and_agent(tb_logger, loaded_experiment_name, latest_checkpoint, use_prioritized_replay=True):
    """Initialize the environment and agent, and load checkpoint if available."""
    # Create single environment
    env = create_env_new()
    print(f"[ENV] Created environment (random_stages={RANDOM_STAGES})")
    
    # Initialize agent
    agent, start_epoch, hparams = initialize_agent(
        tb_logger, loaded_experiment_name, latest_checkpoint, 
        use_prioritized_replay, env.action_space.n
    )
    
    return env, agent, start_epoch, hparams


def log_training_metrics(tb_logger, agent, epoch, metrics, epoch_duration, prefix=""):
    """Log training metrics to TensorBoard and console."""
    buffer_size = len(agent.memory)
    
    # Get current recorded start probability for logging
    current_recorded_probability = config.get_recorded_start_probability(epoch)
    
    # Get epsilon scheduler information
    epsilon_info = agent.get_epsilon_info()
    
    # Calculate curriculum learning metric: avg_level_start_distance / best_level_start_distance
    curriculum_ratio = 0.0
    if metrics['best_level_start_distance'] > 0 and metrics['avg_level_start_distance'] > 0:
        curriculum_ratio = metrics['avg_level_start_distance'] / metrics['best_level_start_distance']
    
    # Log comprehensive metrics to TensorBoard
    tb_metrics = {
        'Training/Loss': metrics['loss'],
        'Training/TD_Error': metrics['td_error'],
        'Performance/Average_Reward': metrics['avg_reward'],
        'Performance/Average_Distance': metrics['avg_distance'],  # Smooth average (last 100 episodes)
        'Performance/Episode_Count': metrics['episode_count'],
        'Performance/Best_Level_Start_Distance': metrics['best_level_start_distance'],
        'Performance/Avg_Level_Start_Distance': metrics['avg_level_start_distance'],
        'Performance/Avg_Recorded_Start_Distance': metrics['avg_recorded_start_distance'],
        'Performance/Level_Start_Episodes': metrics['level_start_episodes'],
        'Performance/Recorded_Start_Episodes': metrics['recorded_start_episodes'],
        'Performance/Flag_Completions': metrics['flag_completions'],
        'Performance/Flag_Completion_Rate': metrics['flag_completion_rate'],
        'Curriculum/Average_Distance_Ratio': curriculum_ratio,  # avg_distance / max_distance for curriculum tracking
        'Hyperparameters/Learning_Rate': metrics['lr'],
        'Hyperparameters/Epsilon': agent.epsilon,
        'Hyperparameters/Epsilon_Phase': 1 if epsilon_info['phase'] == 'Regular' else 2,
        'Hyperparameters/Recorded_Start_Probability': current_recorded_probability,
        'System/Buffer_Size': buffer_size,
        'System/Epoch_Duration': epoch_duration,
        'System/Experiences_Added': metrics['experiences_added'],
        'System/Reuse_Ratio': metrics['reuse_ratio']
    }
    tb_logger.log_metrics(tb_metrics, epoch)
    
    # Log model parameters periodically
    if epoch % 50 == 0:
        tb_logger.log_model_parameters(agent.q_network, epoch)
        print(f"[TENSORBOARD] Logged model parameters at epoch {epoch}")
    
    # Log to console with epoch info
    print(f"[TENSORBOARD] Epoch {epoch} metrics logged")
    
    # Simple progress logging
    level_vs_recorded = f"Level:{metrics['avg_level_start_distance']:.1f} vs Recorded:{metrics['avg_recorded_start_distance']:.1f}" if metrics['level_start_episodes'] > 0 and metrics['recorded_start_episodes'] > 0 else f"Best Level Start:{metrics['best_level_start_distance']:.1f}"
    flag_info = f"Flags:{metrics['flag_completions']}/{metrics['level_start_episodes']} ({metrics['flag_completion_rate']:.1%})" if metrics['level_start_episodes'] > 0 else "Flags:0/0 (0.0%)"
    epsilon_phase_indicator = ""
    if epsilon_info['phase'] == 'Fine-tuning':
        epsilon_phase_indicator = "🎯"
    elif epsilon_info['phase'] == 'Regular (Ready for Fine-tuning)':
        epsilon_phase_indicator = "⏳"
    print(f"[{prefix}EPOCH {epoch}] Loss: {metrics['loss']:.4f}, TD-Error: {metrics['td_error']:.4f}, "
          f"Avg Reward: {metrics['avg_reward']:.2f}, Avg Distance: {metrics['avg_distance']:.1f}, "
          f"{level_vs_recorded}, {flag_info}, "
          f"Epsilon: {agent.epsilon:.3f}{epsilon_phase_indicator}, RecordedStart: {current_recorded_probability:.3f}, Buffer: {buffer_size}")


def cleanup_and_save(tb_logger, agent, epoch, loaded_experiment_name, hparams):
    """Handle final cleanup and save final checkpoint."""
    # Save final checkpoint to prevent loss of progress
    try:
        save_checkpoint(agent, epoch, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
        print("[TRAINING] Final checkpoint saved")
    except Exception as e:
        print(f"[TRAINING] Warning: Could not save final checkpoint: {e}")
    
    # Log final hyperparameters with results (only for new experiments)
    if not loaded_experiment_name and hparams:
        try:
            if epoch > 0:
                final_metrics = {
                    'hparam/final_loss': 0,  # Will be updated if metrics are available
                    'hparam/final_avg_reward': 0,
                    'hparam/final_avg_distance': 0,
                    'hparam/total_epochs': epoch,
                    'hparam/final_buffer_size': len(agent.memory) if agent else 0
                }
                tb_logger.log_hyperparameters(hparams, final_metrics)
                print(f"[TENSORBOARD] Final hyperparameters logged with results")
        except Exception as e:
            print(f"[TENSORBOARD] Warning: Could not log final hyperparameters: {e}")
    
    # Close TensorBoard logger
    tb_logger.close()


def get_global_stats_with_lock():
    """Get global statistics in a thread-safe manner."""
    global _global_distance_tracker, _global_level_start_distances, _global_recorded_start_distances
    global _global_best_level_start_distance, _global_flag_completions, _global_level_start_episodes
    
    with _global_stats_lock:
        # Use global distance tracker for smooth averaging across epochs
        avg_distance = sum(_global_distance_tracker) / len(_global_distance_tracker) if _global_distance_tracker else 0
        
        # Calculate progress metrics comparing level start vs recorded start performance
        avg_level_start_distance = sum(_global_level_start_distances) / len(_global_level_start_distances) if _global_level_start_distances else 0
        avg_recorded_start_distance = sum(_global_recorded_start_distances) / len(_global_recorded_start_distances) if _global_recorded_start_distances else 0
        
        # Calculate flag completion rate for episodes that started from beginning
        flag_completion_rate = (_global_flag_completions / _global_level_start_episodes) if _global_level_start_episodes > 0 else 0
        
        # Copy values for return
        return {
            'avg_distance': avg_distance,
            'best_level_start_distance': _global_best_level_start_distance,
            'avg_level_start_distance': avg_level_start_distance,
            'avg_recorded_start_distance': avg_recorded_start_distance,
            'level_start_episodes': len(_global_level_start_distances),
            'recorded_start_episodes': len(_global_recorded_start_distances),
            'flag_completions': _global_flag_completions,
            'flag_completion_rate': flag_completion_rate
        }


def update_global_stats(episode_distance, started_from_beginning, flag_completed):
    """Update global statistics in a thread-safe manner."""
    global _global_distance_tracker, _global_level_start_distances, _global_recorded_start_distances
    global _global_best_level_start_distance, _global_flag_completions, _global_level_start_episodes
    
    with _global_stats_lock:
        _global_distance_tracker.append(episode_distance)
        
        if started_from_beginning:
            _global_level_start_distances.append(episode_distance)
            if episode_distance > _global_best_level_start_distance:
                _global_best_level_start_distance = episode_distance
            if flag_completed:
                _global_flag_completions += 1
        else:
            _global_recorded_start_distances.append(episode_distance)


def increment_level_start_episodes():
    """Increment level start episode counter in a thread-safe manner."""
    global _global_level_start_episodes
    
    with _global_stats_lock:
        _global_level_start_episodes += 1 