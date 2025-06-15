import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from multiprocessing import Lock, Event

# Global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, input_channels=4):
        super(DQN, self).__init__()

        self.input_channels = input_channels

        self.conv = nn.Sequential(
            # one grayscale input channel
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(self.input_channels, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # transform input to [0,1) float
        fx = x.float() / 256.0
        # calculate convolutional output and transform it into 1d vector
        conv_out = self.conv(fx).flatten(start_dim=1).float()
        # return the results of the fully connected layer
        return self.fc(conv_out)


class SharedReplayBuffer:
    def __init__(self, capacity, experience_queue):
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()
        self.experience_queue = experience_queue
        self.stop_event = Event()

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

        # Neural networks
        self.q_network = DQN(state_shape, n_actions).to(DEVICE)
        self.target_network = DQN(state_shape, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay
        self.memory = SharedReplayBuffer(memory_size, experience_queue)

        # Training metrics
        self.update_target_frequency = 120
        self.steps = 0

    def act(self, state, epsilon_override=None):
        epsilon = self.epsilon
        if epsilon_override is not None:
            epsilon = epsilon_override
        if np.random.random() <= epsilon:
            return random.choice(np.arange(self.n_actions))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size=32, episodes=1):
        # If we don't have enough collected memories for a single batch, we skip training
        if len(self.memory.buffer) < batch_size:
            return

        # Sample all batches at once
        with self.memory.lock:
            all_samples = list(self.memory.buffer)
            random.shuffle(all_samples)
            
        # Calculate how many complete batches we can make
        num_batches = min(len(all_samples) // batch_size, episodes)
        
        # Process in smaller chunks to reduce memory usage
        chunk_size = 10  # Process 10 batches at a time
        for chunk_start in range(0, num_batches, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_batches)
            
            for i in range(chunk_start, chunk_end):
                # Get the next batch from our shuffled samples
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch = all_samples[start_idx:end_idx]
                
                # Unzip the batch
                states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

            # Move to GPU if possible
            states = torch.FloatTensor(states).to(DEVICE)
            actions = torch.LongTensor(actions).to(DEVICE)
            rewards = torch.FloatTensor(rewards).to(DEVICE)
            next_states = torch.FloatTensor(next_states).to(DEVICE)
            dones = torch.BoolTensor(dones).to(DEVICE)

            # Calculate the q-value predictions for the current state from the up-to-date q-network
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Predict the highest q-value action reachable from the next state,
            # calculated from semi-static target network. Use .detach() to remove gradient calculation.
            next_q_values = self.target_network(next_states).max(1)[0].detach()

            # Calculate the target q-values we want to train for, using the Bellman equation
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

            # Train the q-network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clear CUDA cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update epsilon after all batches have been processed
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        # Update target network after all batches have been processed and epsilon is updated
        self.steps += num_batches  # Increment steps by the number of batches trained
        if self.steps % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict()) 