# Configuration constants for Mario RL training

# Data file for logging training progress
DATA_FILE = "training_log.csv"

# folder where agent gets saved to and loaded from
AGENT_FOLDER = "checkpoints"

# specifies if we want to train on all stages or just the first
RANDOM_STAGES = True

# number of processes collecting experiences
# ( this is CPU expensive and the amount of collected experiences is capped by REP_Q_SIZE => finetuning for machine necessary)
# OPTIMIZATION: Reduced from 4 to 2 processes to reduce CPU contention
NUM_PROCESSES = 2

# Interval at which the model will be saved
SAVE_INTERVAL = 100


# ----------------------------------------------------------------------------------------------------------------------
# SAMPLE CONTROL ( ratio recommendation: (BATCH_SIZE * EPISODES_PER_EPOCH) / REP_Q_SIZE ≈ 8 )
# ----------------------------------------------------------------------------------------------------------------------

# The size of the replay buffer, where the agent stores its memories,
# bigger memory -> old replays stay longer in memory -> more stable gradient updates
BUFFER_SIZE = 200000

# Maximum size of the queue where collector processes store replays,
# the limit is for when the collector threads outpace the main thread
REP_Q_SIZE = 2000

# The batch size for the agents policy training
BATCH_SIZE = 4096

# The amount of batches we train per epoch
EPISODES_PER_EPOCH = 10

# On how many epochs we want to train, this is basically forever
NUM_EPOCHS = 20000


# ----------------------------------------------------------------------------------------------------------------------
# LEARNING PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# The amount of steps a run can last at max (0 for unlimited)
MAX_STEPS_PER_RUN = 0

# starting learning rate for the neural network
LEARNING_RATE = 0.0002

# Learning rate decay factor
LR_DECAY_FACTOR = 0.8

# Learning rate decay rate
LR_DECAY_RATE = 100

# Initial epsilon value for epsilon-greedy exploration
EPSILON_START = 1

# How much epsilon decays each training epoch, high epsilon means high chance to randomly explore the environment
EPSILON_DECAY = 0.001

# Minimum epsilon value
EPSILON_MIN = 0.1

# Gamma describes how much the agent should look for future rewards vs immediate ones.
# gamma = 1 future rewards are as valuable as immediate ones
# gamma = 0 only immediate rewards matter
GAMMA = 0.95


# ----------------------------------------------------------------------------------------------------------------------
# PERFORMANCE OPTIMIZATIONS
# ----------------------------------------------------------------------------------------------------------------------

# Model update frequency - only update collector models every N epochs to reduce overhead
MODEL_UPDATE_FREQUENCY = 3  # Update every 3 epochs instead of every epoch

# Enable mixed precision training for faster GPU operations
USE_MIXED_PRECISION = True

# Reduce model synchronization overhead by updating collectors less frequently
ASYNC_MODEL_UPDATES = True


# ----------------------------------------------------------------------------------------------------------------------
# REWARD SHAPING
# ----------------------------------------------------------------------------------------------------------------------

# If an agent does not improve (x-position) for this amount of steps, the run gets canceled
DEADLOCK_STEPS = 40

# how much the reward for moving should be factored in, moving backwards is half that
# 1/12 roughly normalizes to 1
MOVE_REWARD = 1.0 / 6.0

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
SCORE_REWARD_FACTOR = 0.0

# tau describes the percentage of how much the target networks aligns with the dqn each step
AGENT_TAU = 0.01

# ----------------------------------------------------------------------------------------------------------------------
# PRIORITIZED EXPERIENCE REPLAY (PER) PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# PER alpha parameter - controls how much prioritization is used
# alpha = 0: uniform random sampling (no prioritization)
# alpha = 1: full prioritization based on TD error
# Typical values: 0.6-0.7 for good balance between exploration and exploitation
PER_ALPHA = 0.6

# PER beta parameter - controls importance sampling correction
# beta = 0: no correction for bias introduced by prioritization
# beta = 1: full correction for bias
# Starts low and increases during training to gradually correct bias
PER_BETA = 0.4

# PER beta increment - how much beta increases per sampling step
# beta gradually increases to 1.0 during training for full bias correction
# Typical values: 0.001-0.0001 depending on training length
PER_BETA_INCREMENT = 0.001
