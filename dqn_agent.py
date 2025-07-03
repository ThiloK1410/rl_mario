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

from config import LR_DECAY_RATE, LR_DECAY_FACTOR, AGENT_TAU

# Global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, input_channels=8):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            # The input is a stack of 4 grayscale 84x84 frames
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape, input_channels)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape, channels):
        # Create a dummy tensor with a batch size of 1 to pass through the conv layers
        o = self.conv(torch.zeros(1, channels, *shape))
        # Flatten the output and get the total number of elements
        return int(np.prod(o.size()))

    def forward(self, x):
        # The input x is expected to be a tensor of integers (0-255)
        # with shape (batch_size, channels, height, width)

        # Normalize pixel values to the [0, 1] range
        fx = x.float() / 255.0

        # Pass through convolutional layers and flatten the output
        conv_out = self.conv(fx).flatten(start_dim=1)

        # Pass through the fully-connected layers to get Q-values
        return self.fc(conv_out)

class StandardReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None, None, None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, None, None

    def update_priorities(self, indices, new_priorities):
        """No-op for compatibility with prioritized replay buffers."""
        pass

    def __len__(self):
        return len(self.buffer)


class TorchRLPrioritizedReplayBuffer:
    """
    TorchRL-based PrioritizedReplayBuffer that stores data on CPU
    and only moves sampled batches to GPU for maximum memory efficiency.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        if not TORCHRL_AVAILABLE:
            raise ImportError("TorchRL is not available. Please install it with: pip install torchrl")
        
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        self._current_size = 0
        
        # Store buffer data on CPU to save GPU memory
        self.cpu_device = torch.device('cpu')
        
        # Use ListStorage instead of LazyTensorStorage to avoid memory allocation issues
        self._storage = ListStorage(capacity)
        self._sampler = PrioritizedSampler(
            max_capacity=capacity,
            alpha=alpha,
            beta=beta
        )
        
        self._replay_buffer = TensorDictReplayBuffer(
            storage=self._storage,
            sampler=self._sampler,
            batch_size=32,  # Default batch size
            priority_key="td_error"
        )
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer - stored on CPU."""
        # Convert LazyFrames to numpy arrays if needed
        def to_numpy(obj):
            if hasattr(obj, '__array__'):
                return np.array(obj)
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                return np.array(obj)
        
        # Convert states to numpy arrays
        state_np = to_numpy(state)
        next_state_np = to_numpy(next_state)
        
        # Create TensorDict without artificial batch dimensions - let TorchRL handle batching
        tensordict = TensorDict({
            "state": torch.from_numpy(state_np).float(),           # [8, 128, 128]
            "action": torch.tensor(action, dtype=torch.long),      # scalar
            "reward": torch.tensor(reward, dtype=torch.float),     # scalar  
            "next_state": torch.from_numpy(next_state_np).float(), # [8, 128, 128]
            "done": torch.tensor(done, dtype=torch.bool),          # scalar
            "td_error": torch.tensor(1.0, dtype=torch.float)       # scalar
        }, batch_size=[], device=self.cpu_device)  # No batch dimension - let TorchRL handle it
        
        # Add to buffer
        self._replay_buffer.add(tensordict)
        self._current_size = min(self._current_size + 1, self.capacity)

    def push_batch(self, experiences):
        """
        Add multiple experiences to the buffer at once - stored on CPU.
        Add each experience individually to ensure TorchRL compatibility.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
        """
        if not experiences:
            return
        
        # Add each experience individually to ensure TorchRL consistency
        for experience in experiences:
            state, action, reward, next_state, done = experience
            self.push(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        """Sample a batch from the buffer, move to GPU only when sampling."""
        if self._current_size < batch_size:
            return None
        
        # Update beta for importance sampling
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        self._sampler.beta = self.beta
        
        try:
            # Create a new buffer instance with the desired batch size
            temp_buffer = TensorDictReplayBuffer(
                storage=self._storage,
                sampler=self._sampler,
                batch_size=batch_size,
                priority_key="td_error"
            )
            
            # Sample from TorchRL buffer (still on CPU)
            sampled_tensordict = temp_buffer.sample()
            
            # TorchRL should now naturally create proper batch dimensions:
            # state: [batch_size, 8, 128, 128], action: [batch_size], etc.
            
            # NOW move only the sampled batch to GPU for training
            sampled_tensordict = sampled_tensordict.to(DEVICE)
            
            return sampled_tensordict
            
        except Exception as e:
            print(f"Error sampling from TorchRL buffer: {e}")
            return None
    
    def update_priorities(self, indices, new_priorities):
        """Update priorities for sampled experiences."""
        if indices is None:
            return
        
        try:
            # Ensure priorities are positive - keep on CPU for buffer operations
            if isinstance(new_priorities, np.ndarray):
                new_priorities = np.abs(new_priorities) + 1e-6
                priority_tensor = torch.from_numpy(new_priorities).float()
            else:
                # If it's already a GPU tensor, move to CPU for buffer operations
                if new_priorities.is_cuda:
                    priority_tensor = torch.abs(new_priorities).cpu() + 1e-6
                else:
                    priority_tensor = torch.abs(new_priorities) + 1e-6
                
            if isinstance(indices, np.ndarray):
                index_tensor = torch.from_numpy(indices).long()
            else:
                # If indices are on GPU, move to CPU for buffer operations
                if indices.is_cuda:
                    index_tensor = indices.cpu().long()
                else:
                    index_tensor = indices.long()
            
            # Update priorities in the sampler (CPU-based operations)
            self._sampler.update_priority(index_tensor, priority_tensor)
            
        except Exception as e:
            print(f"Error updating priorities: {e}")
    
    def __len__(self):
        return self._current_size


class RankBasedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity

        # Priority parameters
        self.alpha = alpha  # Priority exponent (how much prioritization to use)
        self.beta = beta  # Importance sampling exponent (to correct for bias)
        self.beta_increment = beta_increment
        self.max_beta = 1.0

        # Storage
        self.buffer = []
        # Use a numpy array for priorities for efficient vectorized operations
        self.priorities = np.array([], dtype=np.float64)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer. New experiences are given the highest
        known priority to ensure they are sampled at least once.
        """
        priority = self.max_priority

        # If the buffer is full, remove the oldest experience (FIFO).
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            # Correctly remove the first element from the numpy array
            self.priorities = np.delete(self.priorities, 0)

        # Add new experience
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities = np.append(self.priorities, priority)

    def _get_rank_based_probs(self):
        """
        Calculates sampling probabilities based on the rank of experiences.
        The experience with the highest priority gets rank 1.
        Probabilities are calculated as P(i) = (1 / rank(i))^alpha.
        """
        if len(self.priorities) == 0:
            return np.array([])

        # Get the indices that would sort priorities in descending order
        sorted_indices = np.argsort(self.priorities)[::-1]

        # Create an array of ranks (1, 2, 3, ...)
        ranks = np.arange(1, len(self.priorities) + 1)

        # Calculate probabilities based on rank
        rank_probs = (1 / ranks) ** self.alpha

        # Create an array to map these probabilities back to their original positions
        probs = np.zeros_like(rank_probs)
        probs[sorted_indices] = rank_probs
        probs /= probs.sum()

        return probs

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None, None, None

        # Get rank-based sampling probabilities
        probs = self._get_rank_based_probs()
        if len(probs) == 0:
            return None, None, None, None, None, None, None

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        total_experiences = len(self.buffer)
        weights = (total_experiences * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Anneal beta towards 1.0
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        # Get sampled experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, new_priorities):
        """
        Updates the priorities of sampled experiences.
        """
        # Ensure priorities are positive
        new_priorities = np.abs(new_priorities) + 1e-6
        self.priorities[indices] = new_priorities
        # Update the max priority seen so far
        self.max_priority = max(self.max_priority, np.max(new_priorities))

    def __len__(self):
        return len(self.buffer)


class MarioAgent:
    def __init__(self, state_shape, n_actions, experience_queue, lr=0.00025, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.02, epsilon_decay=0.001, memory_size=200000, batch_size=32):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # tau describes the percentage the target network gets nudged to the q-network each step
        self.tau = AGENT_TAU

        # Neural networks
        self.q_network = DQN(state_shape, n_actions).to(DEVICE)
        self.target_network = DQN(state_shape, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.current_epoch = 0

                # Experience replay - use TorchRL PrioritizedReplayBuffer if available
        if TORCHRL_AVAILABLE:
            print("Using TorchRL PrioritizedReplayBuffer")
            self.memory = TorchRLPrioritizedReplayBuffer(memory_size)
        else:
            print("Using StandardReplayBuffer (TorchRL not available)")
            self.memory = StandardReplayBuffer(memory_size)

    def act(self, state, epsilon_override=None):
        epsilon = self.epsilon
        if epsilon_override is not None:
            epsilon = epsilon_override
        if np.random.random() <= epsilon:
            return np.random.randint(0, self.n_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size=32, episodes=1):
        # If we don't have enough collected memories for a single batch, we skip training
        if len(self.memory) < batch_size:
            return 0.0, 0.0, 0.0, 0.0  # Return default values instead of None

        total_reward = 0  # Track total reward for this replay session
        returned_loss = 0
        returned_td_error = 0

        for _ in range(episodes):
            # Sample batch - now returns TensorDict or tuple depending on buffer type
            batch = self.memory.sample(batch_size)
            if batch is None:
                continue

            # Handle different buffer types - try TensorDict first, fallback to tuple
            try:
                # Try TorchRL TensorDict format - everything already on GPU
                states = batch["state"]  # type: ignore
                actions = batch["action"]  # type: ignore
                rewards = batch["reward"]  # type: ignore
                next_states = batch["next_state"]  # type: ignore
                dones = batch["done"]  # type: ignore
                
                # Get indices and weights for priority updates
                indices = batch.get("index", None)  # type: ignore
                weights = batch.get("_weight", None)  # type: ignore
                if weights is None:
                    weights = torch.ones(batch_size, device=DEVICE)
                
                total_reward += rewards.mean().item()
                
            except (TypeError, AttributeError, KeyError):
                # Fallback to legacy tuple format (for compatibility with other buffer types)
                states, actions, rewards, next_states, dones, indices, weights = batch
                
                # Check if any critical components are None
                if states is None or actions is None or rewards is None or next_states is None or dones is None:
                    continue
                
                total_reward += rewards.mean()

                # Move to GPU if not already there
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.BoolTensor(dones).to(DEVICE)
                
                # Handle weights - use ones if weights is None
                if weights is None:
                    weights = torch.ones(batch_size, device=DEVICE)
                else:
                    weights = torch.FloatTensor(weights).to(DEVICE)

            # Calculate the q-value predictions for the current state
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # First we get the action we would take in the next state
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # Then we get the target q values, for the taken action
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1).detach()

            # Calculate the target q-values using the Bellman equation
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

            # Calculate TD errors for priority updates
            td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach()

            returned_td_error = td_errors.mean().item()

            # Calculate weighted loss
            loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()

            returned_loss = loss.item()

            # Train the q-network
            self.optimizer.zero_grad()
            loss.backward()

            total_norm = 0
            for p in self.q_network.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Log this value to see how it changes over time
            print(f"[AGENT] Gradient Norm: {total_norm}")

            self.optimizer.step()

            # Update priorities based on TD errors only if using prioritized replay
            if indices is not None:
                self.memory.update_priorities(indices, td_errors + 1e-6)  # Add small constant to avoid zero priorities

            # Instead of doing hard target network updates we use this polyac averaging at each step to get more stable results
            q_network_params = self.q_network.parameters()
            target_network_params = self.target_network.parameters()
            for target_param, q_param in zip(target_network_params, q_network_params):
                target_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_param.data
                )

        # Update epsilon after all batches have been processed
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        avg_reward = total_reward / episodes

        if self.current_epoch != 0 and self.current_epoch % LR_DECAY_RATE == 0:
            self.optimizer.param_groups[0]['lr'] *= LR_DECAY_FACTOR

        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[AGENT] Current learning rate: {current_lr}")

        self.current_epoch += 1
        return current_lr, avg_reward, returned_loss, returned_td_error
