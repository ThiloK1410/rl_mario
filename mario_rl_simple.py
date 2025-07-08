import warnings
warnings.filterwarnings("ignore")

import time
import torch
from collections import deque
import os
import threading
from queue import Queue

from environment import create_env
from dqn_agent import MarioAgent, DEVICE
from config import (
    BUFFER_SIZE, NUM_EPOCHS, MAX_STEPS_PER_RUN, BATCH_SIZE,
    EPISODES_PER_EPOCH, LEARNING_RATE, SAVE_INTERVAL, EPSILON_START,
    EPSILON_DECAY, EPSILON_MIN, GAMMA, AGENT_FOLDER, RANDOM_STAGES, REUSE_FACTOR,
    MIN_BUFFER_SIZE, NUM_COLLECTION_THREADS
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


def collect_experiences_threaded(agent, total_target_count, max_episodes=None, num_threads=None, use_random_actions=False):
    """
    Collect experiences using multiple threads.
    Each thread gets its own environment and collects experiences in parallel.
    """
    # Use config default if not specified
    if num_threads is None:
        num_threads = NUM_COLLECTION_THREADS
    
    # Calculate experiences per thread
    experiences_per_thread = total_target_count // num_threads
    remainder = total_target_count % num_threads
    
    action_type = "random" if use_random_actions else "agent"
    print(f"[THREADING] Starting {num_threads} threads to collect {total_target_count} experiences ({action_type} actions)")
    print(f"[THREADING] Each thread collects ~{experiences_per_thread} experiences")
    
    # Create environments for each thread
    environments = []
    for i in range(num_threads):
        env = create_env()
        environments.append(env)
    
    # Create queue for results
    result_queue = Queue()
    threads = []
    
    # Start threads
    for i in range(num_threads):
        # Give remainder experiences to first threads
        target_count = experiences_per_thread + (1 if i < remainder else 0)
        
        thread = threading.Thread(
            target=collect_experiences_worker,
            args=(agent, environments[i], target_count, max_episodes, result_queue, i, use_random_actions)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results from all threads
    all_experiences = []
    all_episode_distances = []
    
    for _ in range(num_threads):
        experiences, episode_distances, worker_id = result_queue.get()
        all_experiences.extend(experiences)
        all_episode_distances.extend(episode_distances)
    
    # Clean up environments
    for env in environments:
        env.close()
    
    print(f"[THREADING] Collected {len(all_experiences)} total experiences from {num_threads} threads")
    return all_experiences, all_episode_distances


def collect_experiences_worker(agent, env, target_count, max_episodes, result_queue, worker_id, use_random_actions=False):
    """
    Worker function for threaded experience collection.
    Collects experiences and puts results in the queue.
    """
    try:
        experiences, episode_distances = collect_experiences_batch(agent, env, target_count, max_episodes, use_random_actions)
        result_queue.put((experiences, episode_distances, worker_id))
        print(f"[THREAD-{worker_id}] Collected {len(experiences)} experiences")
    except Exception as e:
        print(f"[THREAD-{worker_id}] Error: {e}")
        result_queue.put(([], [], worker_id))


def collect_experiences_batch(agent, env, target_count, max_episodes=None, use_random_actions=False):
    """
    Collect experiences in batches for better performance.
    Returns (experiences, episode_stats)
    """
    experiences = []
    episode_stats = []
    
    state = env.reset()
    episode_reward = 0
    episode_distance = 0
    step_count = 0
    episode_count = 0
    
    # Pre-allocate lists for better performance
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    
    while len(experiences) < target_count:
        # Collect experience
        if use_random_actions:
            action = env.action_space.sample()
        else:
            action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        # Store in batch lists
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        # Update episode tracking
        episode_reward += reward
        episode_distance = info.get('x_pos', 0)
        step_count += 1
        
        # Check for episode end
        if done or (MAX_STEPS_PER_RUN > 0 and step_count >= MAX_STEPS_PER_RUN):
            # Track episode stats
            if hasattr(env, 'get_used_recorded_start') and not env.get_used_recorded_start():
                episode_stats.append(episode_distance)
            elif not hasattr(env, 'get_used_recorded_start'):
                episode_stats.append(episode_distance)
            
            episode_count += 1
            
            # Reset for next episode
            state = env.reset()
            episode_reward = 0
            episode_distance = 0
            step_count = 0
            
            # Stop if we've reached max episodes
            if max_episodes and episode_count >= max_episodes:
                break
        else:
            state = next_state
        
        # Process batch when it gets large enough
        if len(states) >= 100:  # Process in chunks of 100
            # Convert to experiences
            for i in range(len(states)):
                experiences.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
            
            # Clear batch lists
            states.clear()
            actions.clear()
            rewards.clear()
            next_states.clear()
            dones.clear()
    
    # Process remaining experiences
    for i in range(len(states)):
        experiences.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
    
    return experiences[:target_count], episode_stats


def setup_experiment():
    """Setup experiment, checkpoints, and TensorBoard logging."""
    print("[SIMPLE MARIO RL] Starting non-parallel training pipeline")
    
    # Try to load latest checkpoint first to get experiment name
    start_epoch = 0
    loaded_experiment_name = None
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER)
    if latest_checkpoint:
        print(f"[CHECKPOINT] Found existing checkpoint: {latest_checkpoint}")
        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
            loaded_experiment_name = checkpoint_data.get('experiment_name', None)
            if loaded_experiment_name:
                print(f"[CHECKPOINT] Found saved experiment name: {loaded_experiment_name}")
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
        available_experiments = list_available_experiments(checkpoint_dir=AGENT_FOLDER)
        if available_experiments:
            print(f"[INFO] Available experiments with checkpoints: {', '.join(available_experiments)}")
        
        try:
            experiment_name = input("Enter experiment name (or press Enter for auto-generated): ").strip()
            if not experiment_name:
                experiment_name = None
        except (EOFError, KeyboardInterrupt):
            experiment_name = None
            print("Using auto-generated experiment name")
    
    tb_logger = TensorBoardLogger(experiment_name=experiment_name)
    
    # Now that we have the final experiment name, look for its specific checkpoint
    final_checkpoint = find_latest_checkpoint(checkpoint_dir=AGENT_FOLDER, experiment_name=tb_logger.experiment_name)
    if final_checkpoint:
        latest_checkpoint = final_checkpoint
        print(f"[CHECKPOINT] Using checkpoint for experiment '{tb_logger.experiment_name}': {latest_checkpoint}")
    
    # Log hyperparameters at the start (only for new experiments)
    if not loaded_experiment_name:
        hparams = create_experiment_config(config)
        hparams['USE_PRIORITIZED_REPLAY'] = USE_PRIORITIZED_REPLAY
        hparams['REUSE_RATIO_THRESHOLD'] = REUSE_FACTOR
        print(f"[TENSORBOARD] Logging hyperparameters: {hparams}")
        tb_logger.log_hyperparameters(hparams)
    else:
        print(f"[TENSORBOARD] Resuming experiment - skipping hyperparameter logging")
    
    return tb_logger, latest_checkpoint, start_epoch, loaded_experiment_name

def fill_initial_buffer(agent, env, target_size):
    """Fill the replay buffer with initial experiences before training starts using multiple threads."""
    print(f"[BUFFER] Filling buffer with {target_size} initial experiences using {NUM_COLLECTION_THREADS} threads...")
    
    # Use threaded collection for initial buffer filling with random actions
    experiences, episode_distances = collect_experiences_threaded(agent, target_size, max_episodes=None, use_random_actions=True)
    
    # Add experiences to memory
    if hasattr(agent.memory, 'push_batch'):
        agent.memory.push_batch(experiences)
    else:
        for exp in experiences:
            agent.remember(*exp)
    
    print(f"[BUFFER] Buffer filled with {len(experiences)} experiences")
    print(f"[BUFFER] Current buffer size: {len(agent.memory)}")
    
    # Close the training environment since we created new ones for threading
    env.close()


def setup_agent_and_environment():
    """Initialize agent and environment."""
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
    
    return agent, env

def training_loop(agent, env, tb_logger, start_epoch):
    """Main training loop."""
    # Tracking variables
    distances = deque(maxlen=100)  # Track last 100 episode distances
    episode_count = 0
    
    # Calculate fixed number of experiences to collect per epoch
    experiences_per_epoch = BATCH_SIZE * EPISODES_PER_EPOCH // int(REUSE_FACTOR)
    
    print(f"[TRAINING] Collecting {experiences_per_epoch} experiences per epoch")
    print(f"[TRAINING] Training with {EPISODES_PER_EPOCH} episodes per epoch")
    
    epoch = start_epoch
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Collect experiences for this epoch using threads
            new_experiences, episode_distances = collect_experiences_threaded(
                agent, experiences_per_epoch, max_episodes=None, num_threads=NUM_COLLECTION_THREADS
            )
            
            # Add experiences to memory
            if hasattr(agent.memory, 'push_batch'):
                agent.memory.push_batch(new_experiences)
            else:
                for exp in new_experiences:
                    agent.remember(*exp)
            
            distances.extend(episode_distances)
            episode_count += len(episode_distances)
            
            # Train the agent
            lr, avg_reward, loss, td_error = agent.replay(
                batch_size=BATCH_SIZE, 
                episodes=EPISODES_PER_EPOCH
            )
            
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
                'System/Experiences_Added': len(new_experiences)
            }
            tb_logger.log_metrics(tb_metrics, epoch)
            
            # Log model parameters periodically
            if epoch % 50 == 0:
                tb_logger.log_model_parameters(agent.q_network, epoch)
            
            # Concise progress logging
            print(f"Epoch {epoch}: Loss={loss:.4f}, Reward={avg_reward:.2f}, Distance={avg_distance:.1f}, "
                  f"Epsilon={agent.epsilon:.3f}, Buffer={buffer_size}")
    
    except KeyboardInterrupt:
        print("\n[TRAINING] Training loop interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training loop failed: {str(e)}")
    
    return epoch

def cleanup_and_finalize(agent, env, tb_logger, loaded_experiment_name, final_epoch):
    """Handle cleanup and finalization."""
    # Save final checkpoint to prevent loss of progress
    try:
        save_checkpoint(agent, final_epoch, tb_logger.experiment_name, checkpoint_dir=AGENT_FOLDER)
        print("[TRAINING] Final checkpoint saved")
    except Exception as e:
        print(f"[TRAINING] Warning: Could not save final checkpoint: {e}")
    
    # Log final hyperparameters with results (only for new experiments)
    if not loaded_experiment_name:
        try:
            if final_epoch > 0:
                hparams = create_experiment_config(config)
                hparams['USE_PRIORITIZED_REPLAY'] = USE_PRIORITIZED_REPLAY
                
                final_metrics = {
                    'hparam/final_loss': 0,  # Would need to pass these from training loop
                    'hparam/final_avg_reward': 0,
                    'hparam/final_avg_distance': 0,
                    'hparam/total_epochs': final_epoch,
                    'hparam/final_buffer_size': len(agent.memory)
                }
                tb_logger.log_hyperparameters(hparams, final_metrics)
                print(f"[TENSORBOARD] Final hyperparameters logged with results")
        except Exception as e:
            print(f"[TENSORBOARD] Warning: Could not log final hyperparameters: {e}")
    
    # Close TensorBoard logger
    tb_logger.close()
    
    env.close()
    print("[TRAINING] Environment closed")


def main():
    """Main function that orchestrates the training pipeline."""
    agent = None
    env = None
    tb_logger = None
    loaded_experiment_name = None
    final_epoch = 0
    
    try:
        # Setup experiment and logging
        tb_logger, latest_checkpoint, start_epoch, loaded_experiment_name = setup_experiment()
        
        # Initialize agent and environment
        agent, env = setup_agent_and_environment()
        
        # Load the checkpoint if we found one earlier
        if latest_checkpoint:
            start_epoch, _ = load_checkpoint(agent, latest_checkpoint)
            print(f"[CHECKPOINT] Resuming from epoch {start_epoch}")
        
        # Fill initial buffer
        fill_initial_buffer(agent, env, MIN_BUFFER_SIZE)
        
        # Create new environment for training since initial buffer fill closed the old one
        env = create_env()
        
        # Run training loop
        final_epoch = training_loop(agent, env, tb_logger, start_epoch)
        
    except KeyboardInterrupt:
        print("\n[TRAINING] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
    finally:
        # Cleanup and finalize
        if agent is not None and env is not None and tb_logger is not None:
            cleanup_and_finalize(agent, env, tb_logger, loaded_experiment_name, final_epoch)


if __name__ == "__main__":
    main() 