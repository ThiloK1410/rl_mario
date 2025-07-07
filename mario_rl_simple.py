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


def save_checkpoint(agent, epoch, experiment_name, checkpoint_dir='checkpoints'):
    """Save agent checkpoint without replay buffer."""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Use experiment name in checkpoint filename to avoid conflicts between parallel experiments
    safe_name = experiment_name.replace('/', '_').replace('\\', '_')  # Make filename safe
    checkpoint_path = f"{checkpoint_dir}/mario_agent_{safe_name}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': agent.q_network.state_dict(),
        'target_model_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
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
    agent.epsilon = checkpoint['epsilon']
    
    # Return both epoch and experiment name (if available)
    experiment_name = checkpoint.get('experiment_name', None)
    print(f"[CHECKPOINT] Loaded from epoch {checkpoint['epoch']} (buffer will be rebuilt from scratch)")
    if experiment_name:
        print(f"[CHECKPOINT] Resuming experiment: {experiment_name}")
    
    return checkpoint['epoch'], experiment_name


def find_latest_checkpoint(checkpoint_dir='checkpoints', experiment_name=None):
    """Find the checkpoint file for a specific experiment."""
    import os
    
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
    import os
    import glob
    
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





def main():
    print("[SIMPLE MARIO RL] Starting non-parallel training pipeline")
    
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
    
    # Training parameters - ADJUSTED for more frequent training
    reuse_ratio_threshold = 40.0
    
    # Log hyperparameters at the start (only for new experiments)
    if not loaded_experiment_name:
        hparams = create_experiment_config(config)
        hparams['USE_PRIORITIZED_REPLAY'] = USE_PRIORITIZED_REPLAY
        hparams['REUSE_RATIO_THRESHOLD'] = reuse_ratio_threshold
        print(f"[TENSORBOARD] Logging hyperparameters: {hparams}")
        tb_logger.log_hyperparameters(hparams)
    else:
        print(f"[TENSORBOARD] Resuming experiment - skipping hyperparameter logging")
    
    # Create single environment
    env = create_env()
    print(f"[ENV] Created environment (random_stages={RANDOM_STAGES})")
    
    # Initialize agent (no experience_queue needed for simple version)
    agent = MarioAgent(
        state_shape=(128, 128), 
        n_actions=env.action_space.n,  # type: ignore
        experience_queue=None,  # Not needed for single-threaded version
        memory_size=BUFFER_SIZE, 
        gamma=GAMMA, 
        epsilon_decay=EPSILON_DECAY, 
        epsilon_min=EPSILON_MIN, 
        lr=LEARNING_RATE, 
        epsilon=EPSILON_START
    )
    
    # Override buffer type if requested
    if not USE_PRIORITIZED_REPLAY:
        from dqn_agent import StandardReplayBuffer
        agent.memory = StandardReplayBuffer(BUFFER_SIZE)
        print(f"[AGENT] Switched to Standard Replay Buffer for comparison")
    
    buffer_type = "Standard Replay" if not USE_PRIORITIZED_REPLAY else "TorchRL Prioritized Replay"
    print(f"[AGENT] Initialized agent with buffer size {BUFFER_SIZE}")
    print(f"[AGENT] Using {buffer_type} buffer")
    if torch.cuda.is_available():
        print(f"[DEVICE] Using CUDA: {DEVICE}")
    
    # Load the checkpoint if we found one earlier
    if latest_checkpoint:
        start_epoch, _ = load_checkpoint(agent, latest_checkpoint)  # We already have the experiment name
        print(f"[CHECKPOINT] Resuming from epoch {start_epoch}")
    
    # Calculate experiences consumed per epoch
    experiences_consumed_per_epoch = BATCH_SIZE * EPISODES_PER_EPOCH
    
    # Initialize environment
    state = env.reset()
    episode_reward = 0
    episode_distance = 0
    episode_count = 0
    step_count = 0
    
    # Tracking variables
    experiences_added_since_training = 0  # Simple counter for experiences added
    distances = deque(maxlen=100)  # Track last 100 episode distances

    print(f"[TRAINING] Reuse ratio threshold: {reuse_ratio_threshold}")
    print(f"[TRAINING] Collecting minimum experiences...")
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()
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
                episode_distance = info.get('x_pos', 0)
                step_count += 1
                
                # Check for episode end
                if done or (MAX_STEPS_PER_RUN > 0 and step_count >= MAX_STEPS_PER_RUN):
                    distances.append(episode_distance)
                    episode_count += 1
                    
                    # Reset for next episode
                    state = env.reset()
                    episode_reward = 0
                    episode_distance = 0
                    step_count = 0
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
            
            # Skip logging if no training occurred
            if not training_occurred:
                continue
            
            # Calculate epoch metrics
            epoch_duration = time.time() - epoch_start_time
            avg_distance = sum(distances) / len(distances) if distances else 0
            buffer_size = len(agent.memory)
            
            # Save checkpoint
            if epoch % SAVE_INTERVAL == 0:
                save_checkpoint(agent, epoch, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
            

            
            # Log comprehensive metrics to TensorBoard
            tb_metrics = {
                'Training/Loss': loss,
                'Training/TD_Error': td_error,
                'Performance/Average_Reward': avg_reward,
                'Performance/Average_Distance': avg_distance,
                'Performance/Episode_Count': episode_count,
                'Hyperparameters/Learning_Rate': lr,
                'Hyperparameters/Epsilon': agent.epsilon,
                'System/Buffer_Size': buffer_size,
                'System/Epoch_Duration': epoch_duration,
                'System/Experiences_Added': experiences_added_since_training,
                'System/Reuse_Ratio': experiences_consumed_per_epoch / max(experiences_added_since_training, 1)
            }
            tb_logger.log_metrics(tb_metrics, epoch)
            
            # Log model parameters periodically
            if epoch % 50 == 0:
                tb_logger.log_model_parameters(agent.q_network, epoch)
                print(f"[TENSORBOARD] Logged model parameters at epoch {epoch}")
            
            # Log to console with epoch info
            print(f"[TENSORBOARD] Epoch {epoch} metrics logged")
            
            # Simple progress logging
            print(f"[EPOCH {epoch}] Loss: {loss:.4f}, TD-Error: {td_error:.4f}, "
                  f"Avg Reward: {avg_reward:.2f}, Avg Distance: {avg_distance:.1f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Buffer: {buffer_size}")
    
    except KeyboardInterrupt:
        print("\n[TRAINING] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
    finally:
        # Save final checkpoint to prevent loss of progress
        try:
            save_checkpoint(agent, epoch if 'epoch' in locals() else 0, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
            print("[TRAINING] Final checkpoint saved")
        except Exception as e:
            print(f"[TRAINING] Warning: Could not save final checkpoint: {e}")
        
        # Log final hyperparameters with results (only for new experiments)
        if not loaded_experiment_name:
            try:
                final_epoch = epoch if 'epoch' in locals() else 0
                if final_epoch > 0 and 'hparams' in locals():
                    final_metrics = {
                        'hparam/final_loss': loss if 'loss' in locals() else 0,
                        'hparam/final_avg_reward': avg_reward if 'avg_reward' in locals() else 0,
                        'hparam/final_avg_distance': avg_distance if 'avg_distance' in locals() else 0,
                        'hparam/total_epochs': final_epoch,
                        'hparam/final_buffer_size': len(agent.memory) if 'agent' in locals() else 0
                    }
                    tb_logger.log_hyperparameters(hparams, final_metrics)
                    print(f"[TENSORBOARD] Final hyperparameters logged with results")
            except Exception as e:
                print(f"[TENSORBOARD] Warning: Could not log final hyperparameters: {e}")
        
        # Close TensorBoard logger
        tb_logger.close()
        
        env.close()
        print("[TRAINING] Environment closed")


if __name__ == "__main__":
    main() 