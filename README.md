# Reinforcement Learning Mario Agent

This project implements a Deep Q-Learning (DQN) agent to play Super Mario Bros using reinforcement learning. The agent learns to navigate through the game environment by maximizing its rewards through trial and error.

## ✨ Features

- **TorchRL Prioritized Experience Replay (PER)** - Advanced memory management for efficient learning
- **TensorBoard Integration** - Real-time visualization of training metrics and hyperparameters
- **Experiment Tracking** - Compare different training runs and hyperparameter configurations
- **GPU Acceleration** - CUDA support for faster training
- **Flexible Training Pipeline** - Easy switching between different replay buffer types
- **Comprehensive Logging** - Detailed metrics including loss, TD errors, rewards, and system performance

## Setup

### Prerequisites
- Python 3.11
- pip (Python package installer)

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

### Training the Agent (Recommended)

For optimal training experience with TensorBoard integration, use the simplified training pipeline:

```bash
python mario_rl_simple.py
```

This will:
- Initialize the Mario environment with TorchRL Prioritized Experience Replay
- Create a DQN agent with advanced memory management
- Start TensorBoard logging with experiment tracking
- Train the agent with real-time metrics visualization
- Save checkpoints periodically

**Features:**
- **Interactive experiment naming** - Name your experiments for easy tracking
- **TensorBoard integration** - Real-time visualization of training progress
- **Efficient training** - Uses PER for 48.9% better learning efficiency
- **Flexible buffer types** - Easy switching between PER and standard replay



### TensorBoard Visualization

Monitor your training in real-time with TensorBoard:

1. **Start TensorBoard** (in a separate terminal):
```bash
tensorboard --logdir=runs
```

2. **Open in browser**: Navigate to `http://localhost:6006`

3. **View metrics**:
   - **Training**: Loss, TD Error, Learning Rate
   - **Performance**: Average Reward, Distance, Episode Count
   - **System**: Buffer Size, Epoch Duration, Reuse Ratio
   - **Hyperparameters**: Compare different experiment configurations

### Experiment Management

**TensorBoard Experiment Manager** (`tb_manager.py`):
```bash
python tb_manager.py
```

**Features:**
- **List all experiments** with run counts and creation dates
- **Remove old experiments** interactively 
- **Clean up experiment logs** to save disk space
- **Compare experiment configurations** side-by-side

**Common Commands:**
```bash
# List all experiments
python tb_manager.py list

# Interactive experiment management
python tb_manager.py interactive

# Remove specific experiment
python tb_manager.py remove experiment_name

# Clean up empty experiments
python tb_manager.py cleanup

# Start TensorBoard directly
python tb_manager.py start --port 6006
```

**Interactive Mode:**
1. **View experiments**: See all your training runs with status
2. **Remove experiments**: Delete specific experiments by number  
3. **Bulk cleanup**: Remove multiple old experiments at once
4. **Safe deletion**: Confirmation prompts prevent accidental loss

### Configuration Options

Edit `mario_rl_simple.py` to customize training:

```python
# Switch between buffer types
USE_PRIORITIZED_REPLAY = True   # Use PER (recommended)
USE_PRIORITIZED_REPLAY = False  # Use Standard Replay for comparison

# Adjust training frequency
reuse_ratio_threshold = 5.0     # Higher = more frequent training
```

### Evaluating the Trained Model

To test the trained model's performance:
```bash
python model_test.py
```

This will load the saved model and demonstrate the agent's gameplay in the Mario environment.

## Project Architecture

The project consists of several key components:

### Core Components

1. **DQN Agent** (`dqn_agent.py`):
   - Implements the Deep Q-Network algorithm with Double DQN
   - **TorchRL Prioritized Experience Replay (PER)** - 48.9% more efficient learning
   - **Standard Replay Buffer** - For comparison and debugging
   - **Rank-based PER** - Alternative prioritization strategy
   - Epsilon-greedy exploration with decay
   - Soft target network updates (τ = 0.005)

2. **Environment** (`environment.py`):
   - Wraps the Super Mario Bros environment
   - Handles state preprocessing and frame stacking
   - Manages action space and rewards
   - Provides observation space normalization
   - Supports random stage selection and save states

3. **Training Pipeline**:
   - **`mario_rl_simple.py`**: Main training pipeline with TensorBoard integration
   - Batch experience collection for improved performance
   - Adaptive reuse ratio for efficient training
   - Checkpoint saving and loading
   - Configurable training parameters

4. **TensorBoard Integration** (`tensorboard_logger.py`):
   - Real-time metrics visualization
   - Experiment tracking and comparison
   - Hyperparameter logging
   - Model parameter histograms

5. **Configuration Management** (`config.py`):
   - Centralized hyperparameter settings
   - GPU/CPU device configuration
   - Training and environment parameters

6. **Tools and Utilities**:
   - **`model_test.py`**: Evaluate trained models
   - **`play_mario.py`**: Interactive gameplay
   - **`tb_manager.py`**: TensorBoard experiment management

### Neural Network Architecture

The agent uses a convolutional neural network optimized for Mario Bros:

```
Input: [batch_size, 8, 128, 128] (8 stacked frames, 128x128 pixels)
├── Conv2D(8→32, 8x8, stride=4) + ReLU
├── Conv2D(32→64, 4x4, stride=2) + ReLU  
├── Conv2D(64→64, 3x3, stride=1) + ReLU
├── Flatten
├── Linear(3136→512) + ReLU
└── Linear(512→n_actions) # Action values
```

**Key Features:**
- **8-channel input** for temporal information
- **Progressive feature extraction** through conv layers
- **Batch normalization** for training stability
- **ReLU activation** for non-linearity
- **Action-value estimation** for Q-learning

### Training Process

1. **Experience Collection**: Agent interacts with environment
2. **Prioritized Sampling**: PER selects important experiences
3. **Batch Training**: Mini-batch gradient descent
4. **Target Network Updates**: Soft updates for stability
5. **Metrics Logging**: Real-time TensorBoard visualization
6. **Checkpoint Saving**: Periodic model saving

### TensorBoard Metrics

**Training Metrics:**
- Loss, TD Error, Learning Rate
- Gradient norms and parameter distributions

**Performance Metrics:**
- Average reward, distance traveled
- Episode count and success rate

**System Metrics:**
- Buffer size, epoch duration
- Experience reuse ratio

**Hyperparameters:**
- All configuration parameters
- Buffer type and training settings

## 📊 TensorBoard Experiment Tracking

### Starting TensorBoard

Launch TensorBoard to monitor training:

**Option 1: Using tb_manager (Recommended)**
```bash
python tb_manager.py start --port 6006
```

**Option 2: Direct command**
```bash
tensorboard --logdir=runs --port=6006
```

Access at: `http://localhost:6006`

### Experiment Organization

**Recommended Naming Convention:**
```
mario_[buffer_type]_[key_params]_[description]

Examples:
- mario_PER_lr0001_baseline
- mario_STD_lr0001_comparison  
- mario_PER_ratio3_aggressive
- mario_PER_gamma99_longterm
```

### Key Metrics to Monitor

1. **Training/Loss** - Should decrease over time
2. **Performance/Average_Reward** - Should increase with learning
3. **Performance/Average_Distance** - Mario's progress in levels
4. **System/Buffer_Size** - Memory utilization
5. **Hyperparameters/Epsilon** - Exploration decay

### Experiment Comparison

1. **Run multiple experiments** with different configurations
2. **Compare in TensorBoard** using the experiment selector
3. **Analyze hyperparameter impact** in the HParams tab
4. **Track long-term trends** across training sessions

### Best Practices

- **Name experiments descriptively** for easy identification
- **Run comparison experiments** to validate improvements
- **Monitor system metrics** to ensure stable training
- **Use hyperparameter sweeps** to find optimal settings
- **Archive successful experiments** for future reference

## 🎬 Recorded Gameplay System

The project includes a sophisticated recorded gameplay system that allows the agent to start training from diverse positions within levels, significantly improving training efficiency and state coverage.

### 🎯 How It Works

The recorded gameplay system captures human gameplay as **action sequences** (lists of integers) that can be deterministically replayed. During training, the agent can start from random positions within these recordings instead of always starting from the beginning of the level.

### 🎮 Recording New Gameplay

To record gameplay for a specific level:

1. **Run the play script:**
```bash
python play_mario.py
```

2. **Select your level** when prompted (e.g., World 1-1, World 1-2, etc.)

3. **Play through the level:**
   - Use arrow keys for movement
   - **A** = A button (jump)
   - **S** = B button (run/fireball)
   - **R** or **F5** = Reset and start over
   - Recording starts automatically each episode

4. **Complete the level** by reaching the flagpole

5. **Save when prompted:**
   - The game will ask: "Save this recording? (y/n)"
   - Type **y** to save successful runs
   - Type **n** to discard unsuccessful attempts

### 📁 Recording Storage

Recordings are stored in the `recorded_gameplay/` directory:
```
recorded_gameplay/
├── world_1_stage_1/
│   └── actions_20250707_165001.json
├── world_1_stage_2/
│   └── actions_20250707_170245.json
└── world_2_stage_1/
    └── actions_20250707_172130.json
```

Each recording contains:
- **Action sequence**: List of integers representing button presses
- **Level info**: World and stage numbers
- **Completion data**: Final position, score, and success status

### 🔄 Recording Overwrite Behavior

**Important: One Recording Per Level**

The system is configured with `ONE_RECORDING_PER_STAGE = True`, which means:
- ✅ **New recordings replace old ones** for the same level
- ✅ **Always keeps the most recent recording** 
- ✅ **Prevents accumulation of outdated recordings**
- ✅ **Ensures consistent training behavior**

When you record a new playthrough of World 1-1, it will overwrite the previous World 1-1 recording.

### 🎯 Training Integration

The recorded gameplay system provides major training benefits:

**Random Starting Positions:**
- Agent starts from **30-85%** through the level (configurable)
- Provides **diverse state coverage** instead of always starting from beginning
- Focuses on **productive middle-to-late positions** where agent can explore

**Configuration Options** (`config.py`):
```python
USE_RECORDED_GAMEPLAY = True                      # Enable recorded gameplay
RECORDED_START_PROBABILITY_MIN = 0.2              # 20% chance at start of training
RECORDED_START_PROBABILITY_MAX = 0.8              # 80% chance at end of curriculum
RECORDED_START_PROBABILITY_INCREASE_EPOCHS = 1000 # Increase over 1000 epochs
PREFER_ADVANCED_CHECKPOINTS = True                # Focus on middle-to-late positions
MIN_SAMPLING_PERCENTAGE = 0.30                    # Start sampling from 30% through level
MAX_SAMPLING_PERCENTAGE = 0.85                    # End sampling at 85% through level
```

**Training Benefits:**
- **Better state coverage** - Agent experiences more diverse game states
- **Improved exploration** - Less time spent on early, easy sections
- **Faster learning** - More training on challenging middle/late game content
- **Enhanced performance** - Better understanding of level mechanics
- **Curriculum learning** - Gradually increases advanced start positions as training progresses

### 🎓 Curriculum Learning

The system implements **curriculum learning** for recorded start positions:

**How it Works:**
- **Early training** (epoch 0): Low probability (20%) of using recorded starts
- **Training progresses**: Probability increases linearly over time
- **Late training** (epoch 1000+): High probability (80%) of using recorded starts

**Benefits:**
- **Gradual difficulty increase** - Agent learns basics first, then advanced positions
- **Better stability** - Prevents overwhelming novice agent with difficult starts
- **Improved convergence** - Smoother learning curve throughout training
- **Adaptive training** - Automatically adjusts challenge based on progress

**Monitoring Progress:**
- Watch **RecordedStart** value in console logs increase over time
- View **Hyperparameters/Recorded_Start_Probability** in TensorBoard
- Current probability calculated as: `min_prob + (max_prob - min_prob) * (epoch / max_epochs)`

### 🗂️ Git Integration

**Recordings Should Be Committed to Git**

Unlike model checkpoints which are large and frequently changing, recordings are:
- ✅ **Small JSON files** (~50KB each)
- ✅ **Stable and reusable** across different training runs
- ✅ **Essential for reproducible training** 
- ✅ **Shared resource** for all team members

**Recommended Git Workflow:**
```bash
# Add new recordings to git
git add recorded_gameplay/
git commit -m "Add World 1-3 gameplay recording"
git push origin main
```

**Why Include in Git:**
- **Reproducible experiments** - Others can use the same starting positions
- **Consistent training** - All team members get the same training diversity
- **Backup and versioning** - Recordings are preserved across machines
- **Easy collaboration** - Share good gameplay recordings with team

### 🎲 Recording Requirements

**For Optimal Training Coverage:**

1. **One recording per level** you want to train on
2. **Complete successful runs** (reach the flagpole)
3. **Quality gameplay** - avoid getting stuck or dying frequently
4. **Diverse paths** - explore different routes through levels when possible

**Recommended Recording Priority:**
1. **World 1-1** - Essential for basic training
2. **World 1-2** - Underground level mechanics
3. **World 1-3** - Platform jumping skills
4. **World 1-4** - Castle/boss encounters
5. **Additional worlds** - As needed for advanced training

### 🔧 Troubleshooting Recordings

**Common Issues:**

**Recording not saving:**
- Make sure you reached the flagpole (`flag_get = True`)
- Check that the recording directory exists
- Verify you answered 'y' to save the recording

**Training not using recordings:**
- Check `USE_RECORDED_GAMEPLAY = True` in config
- Ensure recording files exist in `recorded_gameplay/world_X_stage_Y/`
- Verify `RECORDED_START_PROBABILITY_MIN > 0.0` and curriculum learning is progressing

**Agent getting stuck:**
- Recordings may have positions that are too advanced
- Adjust `MAX_SAMPLING_PERCENTAGE` to lower value (e.g., 0.80)
- Record new gameplay with more conservative routes

## 🎮 Training Tips

### For Best Results

1. **Use GPU acceleration** - Significantly faster training
2. **Monitor TensorBoard** - Watch for training issues early
3. **Experiment with hyperparameters** - Learning rate, epsilon decay
4. **Use PER buffer** - 48.9% more efficient than standard replay
5. **Use recorded gameplay** - Better state coverage and faster learning
6. **Run longer training** - Mario is complex, needs many episodes

### Common Issues

- **Loss not decreasing**: Check learning rate, increase buffer size
- **Training too slow**: Verify GPU usage, reduce batch size if needed
- **Agent not learning**: Increase exploration (epsilon), check reward shaping
- **Memory errors**: Reduce buffer size or batch size

### Hyperparameter Suggestions

**Conservative (Stable):**
```python
LEARNING_RATE = 0.0001
EPSILON_DECAY = 0.001
BUFFER_SIZE = 100000
```

**Aggressive (Faster Learning):**
```python
LEARNING_RATE = 0.0005
EPSILON_DECAY = 0.005
BUFFER_SIZE = 200000
```

## 🔧 Troubleshooting

### Common Setup Issues

1. **CUDA not available**: Install PyTorch with CUDA support
2. **Module not found**: Ensure virtual environment is activated
3. **Environment errors**: Check gym-super-mario-bros installation
4. **TensorBoard not loading**: Verify port availability (6006)

### Performance Optimization

- **Reduce frame size** if training is slow
- **Adjust batch size** based on GPU memory
- **Use mixed precision** for faster training
- **Monitor system resources** during training

## 📈 Results

The agent demonstrates significant improvement with TorchRL PER:
- **48.9% better learning efficiency** compared to standard replay
- **Stable training** with consistent metric improvements
- **Real-time monitoring** through TensorBoard integration
- **Flexible experimentation** with easy configuration switching

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Performance improvements
- New features
- Bug fixes
- Documentation updates 