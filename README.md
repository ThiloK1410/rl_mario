# Reinforcement Learning Mario Agent

This project implements a Deep Q-Learning (DQN) agent to play Super Mario Bros using reinforcement learning. The agent learns to navigate through the game environment by maximizing its rewards through trial and error.

## ✨ Features

- **TorchRL Prioritized Experience Replay (PER)** - Advanced memory management for efficient learning
- **TensorBoard Integration** - Real-time visualization of training metrics and hyperparameters
- **Curriculum Learning** - Gradual increase in recorded start positions for better training
- **Recorded Gameplay System** - Training from diverse level positions for improved state coverage
- **EpsilonScheduler** - Sophisticated two-phase epsilon decay with performance-based fine-tuning
- **GPU Acceleration** - CUDA support for faster training

## Setup

### Prerequisites
- Python 3.11
- pip (Python package installer)

### Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

2. Install PyTorch with CUDA support (recommended):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install other requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training Scripts

#### `mario_rl_simple.py` (Recommended)
Main training script with TensorBoard integration and curriculum learning:

```bash
python mario_rl_simple.py
```

**Features:**
- Interactive experiment naming
- TensorBoard logging with hyperparameter tracking
- Curriculum learning for recorded start positions
- Progress metrics comparing level start vs recorded start performance
- EpsilonScheduler with two-phase training

#### `mario_rl.py` (Work in Progress)
**⚠️ DO NOT USE - This is a second multithreaded approach that is still work in progress and should not be used.**

This script is being developed as an alternative multithreaded training approach but is not yet ready for production use.

### Testing and Evaluation

#### `model_test.py`
Evaluate trained models:

```bash
python model_test.py
```

Loads saved model checkpoints and demonstrates the agent's gameplay performance.

### Interactive Gameplay

#### `play_mario.py`
Record human gameplay for the training system:

```bash
python play_mario.py
```

**Controls:**
- Arrow keys: Movement
- A: Jump
- S: Run/fireball
- R/F5: Reset
- Play through levels and save recordings when prompted

### TensorBoard Monitoring

#### Starting TensorBoard
Monitor training progress in real-time:

```bash
tensorboard --logdir=runs --port=6006
```

Or use the TensorBoard manager:
```bash
python tb_manager.py start --port 6006
```

Access at: `http://localhost:6006`

#### TensorBoard Manager
Manage experiments and cleanup old runs:

```bash
python tb_manager.py
```

**Commands:**
- `python tb_manager.py list` - List all experiments
- `python tb_manager.py interactive` - Interactive experiment management
- `python tb_manager.py remove <name>` - Remove specific experiment
- `python tb_manager.py cleanup` - Clean up empty experiments

## Key Features

### Curriculum Learning
The system implements curriculum learning for recorded start positions:
- **Early training**: Low probability (20%) of using recorded starts
- **Late training**: High probability (80%) of using recorded starts
- **Linear increase**: Over 1000 epochs for gradual difficulty progression

### Progress Metrics
Tracks meaningful progress metrics:
- Average distances from level start vs recorded start episodes
- Best distance achieved from level start
- Episode counts for each type
- Provides practical training insights since flag completion is rare

### EpsilonScheduler
Sophisticated two-phase epsilon decay:
- **Regular training phase**: Normal epsilon decay
- **Performance-based fine-tuning**: Activates after Mario reaches flag from level start
- **Smooth transitions**: No epsilon jumps between phases

### TensorBoard Integration
Comprehensive logging system:
- Real-time metrics visualization
- Hyperparameter tracking
- Experiment comparison
- Old CSV logging system has been replaced

## Configuration

Edit `config.py` to customize training parameters:

```python
# Recorded gameplay settings
USE_RECORDED_GAMEPLAY = True
RECORDED_START_PROBABILITY_MIN = 0.2  # 20% at start
RECORDED_START_PROBABILITY_MAX = 0.8  # 80% at end
RECORDED_START_PROBABILITY_INCREASE_EPOCHS = 1000

# Training settings
LEARNING_RATE = 0.0001
EPSILON_DECAY = 0.001
BUFFER_SIZE = 100000
USE_PRIORITIZED_REPLAY = True
```

## Project Structure

**Core Training:**
- `mario_rl_simple.py` - Main training script (recommended)
- `mario_rl.py` - Multithreaded approach (work in progress, do not use)
- `dqn_agent.py` - DQN agent implementation
- `environment.py` - Mario environment wrapper

**Utilities:**
- `model_test.py` - Model evaluation
- `play_mario.py` - Record gameplay
- `tb_manager.py` - TensorBoard experiment management
- `tensorboard_logger.py` - TensorBoard integration
- `config.py` - Configuration settings

**Data:**
- `checkpoints/` - Saved model checkpoints
- `recorded_gameplay/` - Recorded gameplay sessions
- `runs/` - TensorBoard logs

## Troubleshooting

**Common Issues:**
- **CUDA not available**: Install PyTorch with CUDA support
- **Module not found**: Ensure virtual environment is activated
- **Training too slow**: Verify GPU usage, reduce batch size
- **Agent not learning**: Increase exploration (epsilon), check reward shaping

**Performance Tips:**
- Use GPU acceleration for faster training
- Monitor TensorBoard for training issues
- Use PER buffer for better learning efficiency
- Record diverse gameplay for better state coverage

---

*This README is auto-generated and then adjusted and reflects the current state of the project.* 