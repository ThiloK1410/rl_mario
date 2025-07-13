# Configuration constants for Mario RL training
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Movesets from gym_super_mario_bros.actions can be imported:
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# Define expanded action space to match recording environment
# This matches the 19-action space used in play_mario.py
EXPANDED_COMPLEX_MOVEMENT = [
    ['NOOP'],           # 0
    ['right'],          # 1
    ['right', 'A'],     # 2
    ['right', 'B'],     # 3
    ['right', 'A', 'B'], # 4
    ['A'],              # 5
    ['left'],           # 6
    ['left', 'A'],      # 7
    ['left', 'B'],      # 8
    ['left', 'A', 'B'], # 9
    ['down'],           # 10
    ['up'],             # 11
    ['A', 'B'],         # 12 - NEW
    ['down', 'A'],      # 13 - NEW
    ['down', 'B'],      # 14 - NEW
    ['down', 'A', 'B'], # 15 - NEW
    ['up', 'A'],        # 16 - NEW
    ['up', 'B'],        # 17 - NEW
    ['up', 'A', 'B'],   # 18 - NEW
]

# folder where agent gets saved to and loaded from
AGENT_FOLDER = "checkpoints"

# specifies if we want to train on all stages or just the first
RANDOM_STAGES = False

# Interval at which the model will be saved
SAVE_INTERVAL = 100

# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------------------------------------------------

STACKED_FRAMES = 8

DOWNSCALE_RESOLUTION = 64

SKIPPED_FRAMES = 8

USED_MOVESET = SIMPLE_MOVEMENT

# ----------------------------------------------------------------------------------------------------------------------
# SAMPLE CONTROL
# ----------------------------------------------------------------------------------------------------------------------

# The size of the replay buffer, where the agent stores its memories,
# bigger memory -> old replays stay longer in memory -> more stable gradient updates
BUFFER_SIZE = 200000

# The batch size for the agents policy training
BATCH_SIZE = 64

# Minimum number of experiences to collect before starting training
# Must be >= BATCH_SIZE to ensure we can sample batches
MIN_BUFFER_SIZE = 20000

# Validate that MIN_BUFFER_SIZE is at least BATCH_SIZE
if MIN_BUFFER_SIZE < BATCH_SIZE:
    raise ValueError(f"MIN_BUFFER_SIZE ({MIN_BUFFER_SIZE}) must be >= BATCH_SIZE ({BATCH_SIZE})")

# controls how much experiences needs to be collected before we can start the next epoch
# exp_collected = (BATCH_SIZE * EPISODES_PER_EPOCH) / REUSE_FACTOR
REUSE_FACTOR = 5.0

# The amount of batches we train per epoch
EPISODES_PER_EPOCH = 8

# On how many epochs we want to train, this is basically forever
NUM_EPOCHS = 20000

# ----------------------------------------------------------------------------------------------------------------------
# LEARNING PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# The amount of steps a run can last at max (0 for unlimited)
MAX_STEPS_PER_RUN = 0

# starting learning rate for the neural network
LEARNING_RATE = 0.0001

# Learning rate decay factor
LR_DECAY_FACTOR = 0.9

# Learning rate decay rate
LR_DECAY_RATE = 200

# Initial epsilon value for epsilon-greedy exploration
EPSILON_START = 0.9

# How much epsilon decays each training epoch, high epsilon means high chance to randomly explore the environment
EPSILON_DECAY = 0.0005

# Minimum epsilon value
EPSILON_MIN = 0.2

# Performance-based epsilon scheduler configuration
# Number of flag completions from level start required to enter fine-tuning phase
EPSILON_FINE_TUNE_THRESHOLD = 5

# Second phase epsilon parameters (fine-tuning phase)
EPSILON_FINE_TUNE_DECAY = 0.0001  # Slower decay for fine-tuning
EPSILON_FINE_TUNE_MIN = 0.01      # Lower minimum for fine-tuning

# Gamma describes how much the agent should look for future rewards vs immediate ones.
# gamma = 1 future rewards are as valuable as immediate ones
# gamma = 0 only immediate rewards matter
GAMMA = 0.95


# ----------------------------------------------------------------------------------------------------------------------
# REWARD SHAPING
# ----------------------------------------------------------------------------------------------------------------------

# If an agent does not improve (x-position) for this amount of steps, the run gets canceled
DEADLOCK_STEPS = 20

# how much the reward for moving should be factored in, moving backwards is half that
MOVE_REWARD = 1.0 / 12.0

# cap for move rewards to prevent them from overwhelming other reward signals
# movement rewards will be clamped between -MOVE_REWARD_CAP and +MOVE_REWARD_CAP
MOVE_REWARD_CAP = 2.0

# reward penalty for getting stuck (absolute value)
DEADLOCK_PENALTY = 0.5

# the penalty for dying
DEATH_PENALTY = 1.0

# the reward when mario reaches a flag
COMPLETION_REWARD = 2.0

# factors the amount mario gets rewarded for gaining item effects
ITEM_REWARD_FACTOR = 0.0

# how much gaining score should be factored in the reward function,
# score is very high so keep this factor low (ca. 0.01)
# Enabled to provide intermediate feedback for collecting coins, defeating enemies, etc.
SCORE_REWARD_FACTOR = 0.00

# tau describes the percentage of how much the target networks aligns with the dqn each step
AGENT_TAU = 0.005

# Whether to use Dueling Network architecture instead of standard DQN
# Dueling Network separates value and advantage estimation for better performance
USE_DUELING_NETWORK = True

# ----------------------------------------------------------------------------------------------------------------------
# PRIORITIZED EXPERIENCE REPLAY (PER) PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# PER alpha parameter - controls how much prioritization is used
# alpha = 0: uniform random sampling (no prioritization)
# alpha = 1: full prioritization based on TD error
# Typical values: 0.6-0.7 for good balance between exploration and exploitation
PER_ALPHA = 0.8

# PER beta parameter - controls importance sampling correction
# beta = 0: no correction for bias introduced by prioritization
# beta = 1: full correction for bias
# Starts low and increases during training to gradually correct bias
PER_BETA = 0.4

# PER beta increment - how much beta increases per sampling step
# beta gradually increases to 1.0 during training for full bias correction
# Typical values: 0.001-0.0001 depending on training length
PER_BETA_INCREMENT = 0.001


# ----------------------------------------------------------------------------------------------------------------------
# RECORDED GAMEPLAY RANDOM START PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# Enable using recorded gameplay for random start locations
USE_RECORDED_GAMEPLAY = False

# Directory where recorded gameplay sessions are stored
RECORDED_GAMEPLAY_DIR = "recorded_gameplay"

# Global variable to track current training epoch (used by environment for curriculum learning)
CURRENT_TRAINING_EPOCH = 0

# Curriculum learning parameters for recorded start probability
# The probability starts at RECORDED_START_PROBABILITY_MIN and increases linearly
# to RECORDED_START_PROBABILITY_MAX over RECORDED_START_PROBABILITY_INCREASE_EPOCHS epochs

# Minimum probability of using recorded start positions (early training)
RECORDED_START_PROBABILITY_MIN = 0.1  # 20% chance at start of training

# Maximum probability of using recorded start positions (late training)
RECORDED_START_PROBABILITY_MAX = 0.6  # 80% chance at end of curriculum

# Number of epochs over which to increase the probability linearly
RECORDED_START_PROBABILITY_INCREASE_EPOCHS = 500  # Increase over 1000 epochs

# Function to calculate current recorded start probability based on epoch
def get_recorded_start_probability(current_epoch=None):
    """
    Calculate the current recorded start probability based on training progress.
    
    Args:
        current_epoch: Current training epoch (uses global CURRENT_TRAINING_EPOCH if None)
        
    Returns:
        float: Current probability (0.0 to 1.0) of using recorded start positions
    """
    if current_epoch is None:
        current_epoch = CURRENT_TRAINING_EPOCH
    
    if current_epoch >= RECORDED_START_PROBABILITY_INCREASE_EPOCHS:
        return RECORDED_START_PROBABILITY_MAX
    
    # Linear interpolation between min and max
    progress = current_epoch / RECORDED_START_PROBABILITY_INCREASE_EPOCHS
    return RECORDED_START_PROBABILITY_MIN + (RECORDED_START_PROBABILITY_MAX - RECORDED_START_PROBABILITY_MIN) * progress

# Function to update the global epoch counter (called by training loop)
def update_training_epoch(epoch):
    """Update the global training epoch counter."""
    global CURRENT_TRAINING_EPOCH
    CURRENT_TRAINING_EPOCH = epoch

# Whether to prefer checkpoints that are further in the level
# True = weight checkpoints by x_pos, False = equal probability for all checkpoints
PREFER_ADVANCED_CHECKPOINTS = False

# Minimum x_pos to consider for random starts (avoid very early checkpoints)
MIN_CHECKPOINT_X_POS = 0

# Whether to keep only one recording per stage (new recordings overwrite old ones)
ONE_RECORDING_PER_STAGE = True

# Sampling range for recorded gameplay (to avoid getting stuck in late stages)
MIN_SAMPLING_PERCENTAGE = 0.20  # Start sampling from 30% through the action sequence
MAX_SAMPLING_PERCENTAGE = 0.750  # End sampling at 85% through the action sequence

#-----------------------------------------------------------------------------------------------------------------------





