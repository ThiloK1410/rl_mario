import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from multiprocessing import Lock, Event
import bisect

# TorchRL imports
try:
    from tensordict import TensorDict
    from torchrl.data.replay_buffers import TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import PrioritizedSampler
    from torchrl.data.replay_buffers.storages import LazyTensorStorage, ListStorage
    TORCHRL_AVAILABLE = True
except ImportError:
    TORCHRL_AVAILABLE = False
    print("Warning: TorchRL not available, falling back to standard replay buffer")

from config import LR_DECAY_RATE, LR_DECAY_FACTOR, AGENT_TAU, PER_ALPHA, PER_BETA, PER_BETA_INCREMENT

# Global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class OptimizedDQN(nn.Module):
    """Optimized DQN with performance improvements."""
    def __init__(self, input_shape, n_actions, input_channels=8):
        super(OptimizedDQN, self).__init__()
        
        # Use more efficient convolution parameters
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),  # Add batch norm for faster convergence
            nn.ReLU(inplace=True),  # Inplace operations save memory
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        conv_out_size = self._get_conv_out(input_shape, input_channels)
        
        # Optimized fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512, bias=False),
            nn.LayerNorm(512),  # Layer norm for stability
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _get_conv_out(self, shape, channels):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, channels, *shape))
        return int(np.prod(o.size()))
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input normalization moved to preprocessing for efficiency
        fx = x.float() / 255.0
        conv_out = self.conv(fx).flatten(start_dim=1)
        return self.fc(conv_out)


class OptimizedMarioAgent(MarioAgent):
    """Optimized version of MarioAgent with performance improvements."""
    
    def __init__(self, state_shape, n_actions, experience_queue, lr=0.00025, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.02, epsilon_decay=0.001, memory_size=200000, batch_size=32):
        # Initialize parent class
        super().__init__(state_shape, n_actions, experience_queue, lr, gamma, epsilon,
                        epsilon_min, epsilon_decay, memory_size, batch_size)
        
        # Replace networks with optimized versions
        self.q_network = OptimizedDQN(state_shape, n_actions).to(DEVICE)
        self.target_network = OptimizedDQN(state_shape, n_actions).to(DEVICE)
        
        # Use AdamW optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Add learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2, eta_min=lr * 0.01
        )
        
        # Pre-allocate tensors for efficiency
        self._preallocated_batch_size = batch_size * 4  # For gradient accumulation
        self._state_buffer = torch.zeros((self._preallocated_batch_size, 8, 128, 128), 
                                       dtype=torch.float32, device=DEVICE)
        self._next_state_buffer = torch.zeros((self._preallocated_batch_size, 8, 128, 128), 
                                            dtype=torch.float32, device=DEVICE)
        
    def process_batch(self, batch, accumulate=False):
        """
        Process a single batch with optional gradient accumulation.
        Returns loss, td_error, and average reward.
        """
        if batch is None:
            return torch.tensor(0.0), 0.0, 0.0
        
        # Handle different buffer types
        try:
            # TorchRL TensorDict format
            states = batch["state"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_states = batch["next_state"]
            dones = batch["done"]
            indices = batch.get("index", None)
            weights = batch.get("_weight", None)
            if weights is None:
                weights = torch.ones(len(states), device=DEVICE)
            
        except (TypeError, AttributeError, KeyError):
            # Legacy tuple format
            states, actions, rewards, next_states, dones, indices, weights = batch
            
            if states is None:
                return torch.tensor(0.0), 0.0, 0.0
            
            # Convert to tensors if needed
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(DEVICE)
            if not isinstance(actions, torch.Tensor):
                actions = torch.LongTensor(actions).to(DEVICE)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.FloatTensor(rewards).to(DEVICE)
            if not isinstance(next_states, torch.Tensor):
                next_states = torch.FloatTensor(next_states).to(DEVICE)
            if not isinstance(dones, torch.Tensor):
                dones = torch.BoolTensor(dones).to(DEVICE)
            
            if weights is None:
                weights = torch.ones(len(states), device=DEVICE)
            elif not isinstance(weights, torch.Tensor):
                weights = torch.FloatTensor(weights).to(DEVICE)
        
        # Double DQN update
        with torch.no_grad():
            # Get best actions from online network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            # Compute targets
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values).detach()
        
        # Weighted Huber loss (more stable than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (weights * loss).mean()
        
        # Update priorities if using prioritized replay
        if indices is not None and hasattr(self.memory, 'update_priorities'):
            self.memory.update_priorities(indices, td_errors.cpu().numpy() + 1e-6)
        
        return loss, td_errors.mean().item(), rewards.mean().item()
    
    def optimized_replay(self, batch_size=32, episodes=1, accumulation_steps=4):
        """
        Optimized replay with gradient accumulation and mixed precision support.
        """
        if len(self.memory) < batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        total_loss = 0.0
        total_td_error = 0.0
        total_reward = 0.0
        steps_processed = 0
        
        # Process multiple episodes
        for episode in range(episodes):
            # Gradient accumulation over multiple batches
            for acc_step in range(accumulation_steps):
                batch = self.memory.sample(batch_size)
                if batch is None:
                    continue
                
                # Process batch
                loss, td_error, reward = self.process_batch(batch, accumulate=True)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                total_loss += loss.item()
                total_td_error += td_error
                total_reward += reward
                steps_processed += 1
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            
            # Optimizer step after accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Soft update target network
            self._soft_update_target_network()
        
        # Update learning rate
        self.scheduler.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        
        # Average metrics
        if steps_processed > 0:
            avg_loss = total_loss / steps_processed
            avg_td_error = total_td_error / steps_processed
            avg_reward = total_reward / steps_processed
        else:
            avg_loss = avg_td_error = avg_reward = 0.0
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return current_lr, avg_reward, avg_loss, avg_td_error
    
    def _soft_update_target_network(self):
        """Efficient soft update of target network."""
        with torch.no_grad():
            # Vectorized soft update
            for target_param, param in zip(self.target_network.parameters(), 
                                         self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
    
    def act_batch(self, states, epsilon_override=None):
        """
        Efficient batch action selection for multiple states.
        """
        epsilon = epsilon_override if epsilon_override is not None else self.epsilon
        batch_size = len(states)
        
        # Generate random mask for epsilon-greedy
        random_mask = torch.rand(batch_size) <= epsilon
        
        # Random actions
        random_actions = torch.randint(0, self.n_actions, (batch_size,))
        
        # Greedy actions
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(DEVICE)
            q_values = self.q_network(states_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu()
        
        # Combine using mask
        actions = torch.where(random_mask, random_actions, greedy_actions)
        
        return actions.numpy()


# Import base classes before defining optimized versions
import sys
import os
# Ensure we can import from the same directory
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# Now import after path is set
from dqn_agent import MarioAgent

# Export the optimized classes
__all__ = ['OptimizedDQN', 'OptimizedMarioAgent']