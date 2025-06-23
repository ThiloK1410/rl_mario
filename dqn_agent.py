import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from multiprocessing import Lock, Event
import bisect

from config import LR_DECAY_RATE, LR_DECAY_FACTOR

# Global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, input_channels=4):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            # The input is a stack of 4 grayscale 84x84 frames
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32), # Added BatchNorm
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64), # Added BatchNorm
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64), # Added BatchNorm
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape, input_channels)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Added Dropout to regularize the dense layer
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
class SharedReplayBuffer:
    def __init__(self, capacity, experience_queue):
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()
        self.experience_queue = experience_queue

    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done


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
        self.tau = 0.002

        # Neural networks
        self.q_network = DQN(state_shape, n_actions).to(DEVICE)
        self.target_network = DQN(state_shape, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.current_epoch = 0

        # Experience replay
        self.memory = RankBasedPrioritizedReplayBuffer(memory_size)

        # Training metrics
        self.update_target_frequency = 5
        self.steps = 0
        self.best_reward = float('-inf')  # Track best reward for scheduler

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
        if len(self.memory.buffer) < batch_size:
            return

        # Process in smaller chunks to reduce memory usage
        chunk_size = 10  # Process 10 batches at a time
        total_reward = 0  # Track total reward for this replay session

        returned_loss = 0
        returned_td_error = 0

        for chunk_start in range(0, episodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, episodes)

            for _ in range(chunk_start, chunk_end):
                # Sample batch with priorities
                batch = self.memory.sample(batch_size)
                if batch is None:
                    continue

                states, actions, rewards, next_states, dones, indices, weights = batch
                total_reward += rewards.mean()  # Add mean reward for this batch

                # Move to GPU if possible
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.BoolTensor(dones).to(DEVICE)
                weights = torch.FloatTensor(weights).to(DEVICE)

                # Calculate the q-value predictions for the current state
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

                # Predict the highest q-value action reachable from the next state
                next_q_values = self.target_network(next_states).max(1)[0].detach()

                # Calculate the target q-values using the Bellman equation
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)

                # Calculate TD errors for priority updates
                td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()

                returned_td_error = td_errors.mean()

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

                # Update priorities based on TD errors
                self.memory.update_priorities(indices, td_errors + 1e-6)  # Add small constant to avoid zero priorities

                # instead of doing hard target network updates we use this polyac averaging at each step to get more stable results
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

        # Update target network after all batches have been processed and epsilon is updated
        self.steps += 1  # Increment steps by the number of episodes trained

        avg_reward = total_reward / episodes

        if self.current_epoch != 0 and self.current_epoch % LR_DECAY_RATE == 0:
            self.optimizer.param_groups[0]['lr'] *= LR_DECAY_FACTOR

        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[AGENT] Current learning rate: {current_lr}")

        self.current_epoch += 1
        return current_lr, avg_reward, returned_loss, returned_td_error
