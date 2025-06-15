# Reinforcement Learning Mario Agent

This project implements a Deep Q-Learning (DQN) agent to play Super Mario Bros using reinforcement learning. The agent learns to navigate through the game environment by maximizing its rewards through trial and error.

## Setup

### Prerequisites
- Python 3.11
- pip (Python package installer)
- NVIDIA GPU (recommended for faster training)

### PyTorch with CUDA Installation

For optimal performance, it's recommended to install PyTorch with CUDA support:

1. Visit the official PyTorch website: https://pytorch.org/get-started/locally/
2. Select your preferences:
   - PyTorch Build: Stable
   - Your OS: Windows/Linux/Mac
   - Package: Pip
   - Language: Python
   - Compute Platform: CUDA (select your CUDA version)
3. Copy and run the provided installation command. For example:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Note: Make sure to install PyTorch with CUDA before installing other requirements.

### Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Agent

To start training the DQN agent, run:
```bash
python mario_rl.py
```

The training process will:
- Initialize the Mario environment
- Create a DQN agent
- Train the agent through episodes
- Save the trained model periodically

### Evaluating the Trained Model

To test the trained model's performance, run:
```bash
python model_test.py
```

This will load the saved model and demonstrate the agent's gameplay in the Mario environment.

## Agent Architecture

The project consists of several key components:

1. **DQN Agent** (`dqn_agent.py`):
   - Implements the Deep Q-Network algorithm
   - Uses experience replay for stable learning
   - Implements epsilon-greedy exploration strategy
   - Maintains target and policy networks

2. **Environment** (`environment.py`):
   - Wraps the Super Mario Bros environment
   - Handles state preprocessing
   - Manages action space and rewards
   - Provides observation space normalization

3. **Training Loop** (`mario_rl.py`):
   - Manages the training process
   - Handles episode execution
   - Implements model saving and loading
   - Tracks training metrics

4. **Model Testing** (`model_test.py`):
   - Loads trained models
   - Visualizes agent performance
   - Runs evaluation episodes

The agent uses a convolutional neural network to process game frames and make action decisions. The architecture includes:
- Input layer for processed game frames
- Convolutional layers for feature extraction
- Fully connected layers for action value estimation
- Experience replay buffer for stable learning
- Target network for stable Q-value estimation 