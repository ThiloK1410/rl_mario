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

from config import LR_DECAY_RATE, LR_DECAY_FACTOR, AGENT_TAU, PER_ALPHA, PER_BETA, PER_BETA_INCREMENT, USE_DUELING_NETWORK, STACKED_FRAMES, DOWNSCALE_RESOLUTION, EPSILON_FINE_TUNE_THRESHOLD, EPSILON_FINE_TUNE_DECAY, EPSILON_FINE_TUNE_MIN

# Global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        # Use config parameters directly
        input_channels = STACKED_FRAMES
        input_height = DOWNSCALE_RESOLUTION
        input_width = DOWNSCALE_RESOLUTION
        
        self.input_shape = (input_height, input_width)
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            # The input is a stack of frames with configurable resolution
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(self.input_shape, input_channels)

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

# by thilo
class DuelingDQN(nn.Module):
    def __init__(self, n_actions):
        super(DuelingDQN, self).__init__()
        
        # Use config parameters directly
        input_channels = STACKED_FRAMES
        input_height = DOWNSCALE_RESOLUTION
        input_width = DOWNSCALE_RESOLUTION
        
        self.input_shape = (input_height, input_width)
        self.input_channels = input_channels

        self.conv = nn.Sequential(
            # The input is a stack of frames with configurable resolution
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(self.input_shape, input_channels)

        # Value stream - outputs a single value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream - outputs advantage values A(s,a) for each action
        self.advantage_stream = nn.Sequential(
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

        # Separate value and advantage streams
        value = self.value_stream(conv_out)  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(conv_out)  # Shape: (batch_size, n_actions)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Subtracting the mean helps with identifiability and stability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class StandardReplayBuffer:
    """
    Memory-efficient standard replay buffer that stores uint8 data
    and only converts to float32 when sampling for training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Convert LazyFrames to numpy arrays and ensure uint8 for memory efficiency
        def to_numpy_uint8(obj):
            if hasattr(obj, '__array__'):
                arr = np.array(obj)
            elif isinstance(obj, np.ndarray):
                arr = obj
            else:
                arr = np.array(obj)
            
            # Ensure uint8 dtype for states to save memory
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return arr
        
        state_np = to_numpy_uint8(state)
        next_state_np = to_numpy_uint8(next_state)
        
        self.buffer.append((state_np, action, reward, next_state_np, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None, None, None
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # Convert uint8 states to float32 for training
        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        
        return state, action, reward, next_state, done, None, None

    def push_batch(self, experiences):
        """
        Add multiple experiences to the buffer at once.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
        """
        if not experiences:
            return
        
        # Add each experience individually
        for experience in experiences:
            state, action, reward, next_state, done = experience
            self.push(state, action, reward, next_state, done)

    def update_priorities(self, indices, new_priorities):
        """No-op for compatibility with prioritized replay buffers."""
        pass

    def __len__(self):
        return len(self.buffer)

# by thilo
class TorchRLPrioritizedReplayBuffer:
    """
    Memory-efficient TorchRL-based PrioritizedReplayBuffer that stores uint8 data on CPU
    and only converts to float32 when sampling for training.
    This reduces memory usage by ~75% compared to storing float32 data.
    """
    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None):
        if not TORCHRL_AVAILABLE:
            raise ImportError("TorchRL is not available. Please install it with: pip install torchrl")
        
        self.capacity = capacity
        self.alpha = alpha if alpha is not None else PER_ALPHA
        self.beta = beta if beta is not None else PER_BETA
        self.beta_increment = beta_increment if beta_increment is not None else PER_BETA_INCREMENT
        self.max_beta = 1.0
        self._current_size = 0
        
        # Store buffer data on CPU to save GPU memory
        self.cpu_device = torch.device('cpu')
        
        # Use ListStorage instead of LazyTensorStorage to avoid memory allocation issues
        self._storage = ListStorage(capacity)
        self._sampler = PrioritizedSampler(
            max_capacity=capacity,
            alpha=self.alpha,
            beta=self.beta
        )
        
        self._replay_buffer = TensorDictReplayBuffer(
            storage=self._storage,
            sampler=self._sampler,
            batch_size=32,  # Default batch size
            priority_key="td_error"
        )
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer - stored as uint8 on CPU for memory efficiency."""
        # Convert LazyFrames to numpy arrays if needed
        def to_numpy(obj):
            if hasattr(obj, '__array__'):
                return np.array(obj)
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                return np.array(obj)
        
        # Convert states to numpy arrays and keep as uint8 for memory efficiency
        state_np = to_numpy(state)
        next_state_np = to_numpy(next_state)
        
        # Ensure uint8 dtype for maximum memory efficiency
        if state_np.dtype != np.uint8:
            state_np = state_np.astype(np.uint8)
        if next_state_np.dtype != np.uint8:
            next_state_np = next_state_np.astype(np.uint8)
        
        # Create TensorDict with uint8 data - let TorchRL handle batching
        tensordict = TensorDict({
            "state": torch.from_numpy(state_np).byte(),           # [STACKED_FRAMES, DOWNSCALE_RESOLUTION, DOWNSCALE_RESOLUTION] uint8
            "action": torch.tensor(action, dtype=torch.long),      # scalar
            "reward": torch.tensor(reward, dtype=torch.float),     # scalar  
            "next_state": torch.from_numpy(next_state_np).byte(), # [STACKED_FRAMES, DOWNSCALE_RESOLUTION, DOWNSCALE_RESOLUTION] uint8
            "done": torch.tensor(done, dtype=torch.bool),          # scalar
            "td_error": torch.tensor(1.0, dtype=torch.float)       # scalar
        }, batch_size=[], device=self.cpu_device)  # No batch dimension - let TorchRL handle it
        
        # Add to buffer
        self._replay_buffer.add(tensordict)
        self._current_size = min(self._current_size + 1, self.capacity)

    def push_batch(self, experiences):
        """
        Add multiple experiences to the buffer at once - stored as uint8 on CPU.
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
        """Sample a batch from the buffer, convert uint8 to float32 only when sampling."""
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
            
            # Sample from TorchRL buffer (still on CPU, still uint8)
            sampled_tensordict = temp_buffer.sample()
            
            # Convert uint8 states to float32 for training (on CPU first, then move to GPU)
            sampled_tensordict["state"] = sampled_tensordict["state"].float()
            sampled_tensordict["next_state"] = sampled_tensordict["next_state"].float()
            
            # NOW move the converted batch to GPU for training
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

# by thilo (deprecated)
class RankBasedPrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=None, beta=None, beta_increment=None):
        self.capacity = capacity

        # Priority parameters
        self.alpha = alpha if alpha is not None else PER_ALPHA  # Priority exponent (how much prioritization to use)
        self.beta = beta if beta is not None else PER_BETA  # Importance sampling exponent (to correct for bias)
        self.beta_increment = beta_increment if beta_increment is not None else PER_BETA_INCREMENT
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

# by thilo
class EpsilonScheduler:
    """
    Epsilon scheduler with two phases:
    1. Regular phase: Normal epsilon decay from start to min
    2. Fine-tuning phase: Activated after n flag completions from level start,
       uses slower decay from first min to even lower min for fine-tuning
    """
    
    def __init__(self, epsilon_start=1.0, epsilon_min=0.02, epsilon_decay=0.001,
                 fine_tune_threshold=5, fine_tune_decay=0.0001, fine_tune_min=0.01):
        # Phase 1 parameters (regular training)
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Phase 2 parameters (fine-tuning)
        self.fine_tune_threshold = fine_tune_threshold
        self.fine_tune_decay = fine_tune_decay
        self.fine_tune_min = fine_tune_min
        
        # Current state
        self.current_epsilon = epsilon_start
        self.phase = 1  # 1 = regular, 2 = fine-tuning
        self.flag_completions = 0
        
        print(f"[EPSILON] Initialized scheduler:")
        print(f"[EPSILON] Phase 1: {epsilon_start:.3f} → {epsilon_min:.3f} (decay: {epsilon_decay:.4f})")
        print(f"[EPSILON] Phase 2: {epsilon_min:.3f} → {fine_tune_min:.3f} (decay: {fine_tune_decay:.4f})")
        print(f"[EPSILON] Fine-tuning threshold: {fine_tune_threshold} flag completions")
    
    def update_flag_completions(self, flag_completions):
        """Update the number of flag completions from level start."""
        self.flag_completions = flag_completions
        
        # Check if we should transition to fine-tuning phase
        # Only transition if we're in phase 1, have enough flags, AND have reached the first minimum
        if (self.phase == 1 and 
            self.flag_completions >= self.fine_tune_threshold and 
            self.current_epsilon <= self.epsilon_min):
            self.phase = 2
            # Start fine-tuning phase from the first phase minimum (should already be there)
            self.current_epsilon = self.epsilon_min
            print(f"[EPSILON] 🎯 FINE-TUNING PHASE ACTIVATED! ({self.flag_completions} flags completed)")
            print(f"[EPSILON] Switching to fine-tuning: {self.epsilon_min:.3f} → {self.fine_tune_min:.3f}")
    
    def get_epsilon(self):
        """Get the current epsilon value."""
        return self.current_epsilon
    
    def step(self):
        """Update epsilon for one training step."""
        if self.phase == 1:
            # Regular phase: decay from start to min
            if self.current_epsilon > self.epsilon_min:
                self.current_epsilon -= self.epsilon_decay
                # Clamp to minimum to prevent going below
                if self.current_epsilon < self.epsilon_min:
                    self.current_epsilon = self.epsilon_min
            
            # Check if we can transition to fine-tuning phase (now that epsilon is properly clamped)
            if (self.flag_completions >= self.fine_tune_threshold and 
                self.current_epsilon <= self.epsilon_min):
                self.phase = 2
                print(f"[EPSILON] 🎯 FINE-TUNING PHASE ACTIVATED! ({self.flag_completions} flags completed)")
                print(f"[EPSILON] Switching to fine-tuning: {self.epsilon_min:.3f} → {self.fine_tune_min:.3f}")
        else:
            # Fine-tuning phase: decay from first min to second min
            if self.current_epsilon > self.fine_tune_min:
                self.current_epsilon -= self.fine_tune_decay
                # Clamp to minimum to prevent going below
                if self.current_epsilon < self.fine_tune_min:
                    self.current_epsilon = self.fine_tune_min
    
    def get_phase_info(self):
        """Get information about the current phase."""
        if self.phase == 1:
            # Check if ready for fine-tuning but waiting for epsilon to reach minimum
            flags_ready = self.flag_completions >= self.fine_tune_threshold
            epsilon_ready = self.current_epsilon <= self.epsilon_min
            
            phase_name = 'Regular'
            if flags_ready and not epsilon_ready:
                phase_name = 'Regular (Ready for Fine-tuning)'
            
            return {
                'phase': phase_name,
                'epsilon': self.current_epsilon,
                'target_min': self.epsilon_min,
                'decay_rate': self.epsilon_decay,
                'flags_to_fine_tune': max(0, self.fine_tune_threshold - self.flag_completions),
                'ready_for_fine_tune': flags_ready and epsilon_ready
            }
        else:
            return {
                'phase': 'Fine-tuning',
                'epsilon': self.current_epsilon,
                'target_min': self.fine_tune_min,
                'decay_rate': self.fine_tune_decay,
                'flags_completed': self.flag_completions
            }


class MarioAgent:
    def __init__(self, n_actions, lr=0.00025, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.02, epsilon_decay=0.001, memory_size=200000, batch_size=32):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize epsilon scheduler with config parameters
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            fine_tune_threshold=EPSILON_FINE_TUNE_THRESHOLD,
            fine_tune_decay=EPSILON_FINE_TUNE_DECAY,
            fine_tune_min=EPSILON_FINE_TUNE_MIN
        )

        # tau describes the percentage the target network gets nudged to the q-network each step
        self.tau = AGENT_TAU

        # Neural networks - choose between DQN and DuelingDQN based on config
        # Networks will automatically use STACKED_FRAMES and DOWNSCALE_RESOLUTION from config
        if USE_DUELING_NETWORK:
            print(f"Using Dueling Network architecture ({STACKED_FRAMES} channels, {DOWNSCALE_RESOLUTION}x{DOWNSCALE_RESOLUTION} resolution)")
            self.q_network = DuelingDQN(n_actions).to(DEVICE)
            self.target_network = DuelingDQN(n_actions).to(DEVICE)
        else:
            print(f"Using standard DQN architecture ({STACKED_FRAMES} channels, {DOWNSCALE_RESOLUTION}x{DOWNSCALE_RESOLUTION} resolution)")
            self.q_network = DQN(n_actions).to(DEVICE)
            self.target_network = DQN(n_actions).to(DEVICE)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.current_epoch = 0

                # Experience replay - use TorchRL PrioritizedReplayBuffer if available
        if TORCHRL_AVAILABLE:
            print("Using TorchRL PrioritizedReplayBuffer (Memory-Efficient uint8 storage)")
            self.memory = TorchRLPrioritizedReplayBuffer(memory_size)
            # Calculate memory savings using config parameters
            uint8_gb = memory_size * (STACKED_FRAMES * DOWNSCALE_RESOLUTION * DOWNSCALE_RESOLUTION * 2) / (1024**3)  # 2 states per experience
            float32_gb = uint8_gb * 4  # float32 is 4x larger than uint8
            print(f"[MEMORY] Buffer will use ~{uint8_gb:.1f}GB instead of {float32_gb:.1f}GB")
            print(f"[MEMORY] Saving {float32_gb - uint8_gb:.1f}GB (~{((float32_gb - uint8_gb) / float32_gb * 100):.0f}% reduction)")
        else:
            print("Using StandardReplayBuffer (Memory-Efficient uint8 storage)")
            self.memory = StandardReplayBuffer(memory_size)
            # Calculate memory savings using config parameters
            uint8_gb = memory_size * (STACKED_FRAMES * DOWNSCALE_RESOLUTION * DOWNSCALE_RESOLUTION * 2) / (1024**3)  # 2 states per experience
            float32_gb = uint8_gb * 4  # float32 is 4x larger than uint8
            print(f"[MEMORY] Buffer will use ~{uint8_gb:.1f}GB instead of {float32_gb:.1f}GB")
            print(f"[MEMORY] Saving {float32_gb - uint8_gb:.1f}GB (~{((float32_gb - uint8_gb) / float32_gb * 100):.0f}% reduction)")

    def act(self, state, epsilon_override=None):
        epsilon = self.epsilon_scheduler.get_epsilon()
        if epsilon_override is not None:
            epsilon = epsilon_override
        if np.random.random() <= epsilon:
            return np.random.randint(0, self.n_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update_flag_completions(self, flag_completions):
        """Update the epsilon scheduler with the current number of flag completions."""
        self.epsilon_scheduler.update_flag_completions(flag_completions)
    
    @property
    def epsilon(self):
        """Get current epsilon value for compatibility."""
        return self.epsilon_scheduler.get_epsilon()
    
    def get_epsilon_info(self):
        """Get detailed information about epsilon scheduler state."""
        return self.epsilon_scheduler.get_phase_info()

    def replay(self, batch_size=32, episodes=1):
        # If we don't have enough collected memories for a single batch, we skip training
        if len(self.memory) < batch_size:
            return 0.0, 0.0, 0.0, 0.0  # Return default values instead of None

        total_reward = 0  # Track total reward for this replay session
        returned_loss = 0
        returned_td_error = 0
        total_gradient_norm = 0  # Track total gradient norm across all batches
        processed_batches = 0  # Count of successfully processed batches

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

            # Accumulate gradient norm for averaging
            total_gradient_norm += total_norm
            processed_batches += 1

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

        # Update epsilon using the scheduler
        self.epsilon_scheduler.step()

        avg_reward = total_reward / episodes

        if self.current_epoch != 0 and self.current_epoch % LR_DECAY_RATE == 0:
            self.optimizer.param_groups[0]['lr'] *= LR_DECAY_FACTOR

        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[AGENT] Current learning rate: {current_lr}")

        # Print average gradient norm across all batches
        if processed_batches > 0:
            avg_gradient_norm = total_gradient_norm / processed_batches
            print(f"[AGENT] Average Gradient Norm: {avg_gradient_norm}")

        self.current_epoch += 1
        return current_lr, avg_reward, returned_loss, returned_td_error
