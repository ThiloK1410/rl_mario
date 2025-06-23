# Configuration constants for Mario RL training

# Data file for logging training progress
DATA_FILE = "training_log.csv"

# folder where agent gets saved to and loaded from
AGENT_FOLDER = "checkpoints"

# specifies if we want to train on all stages or just the first
RANDOM_STAGES = False

# number of processes collecting experiences
# ( this is CPU expensive and the amount of collected experiences is capped by REP_Q_SIZE => finetuning for machine necessary)
NUM_PROCESSES = 2

# Interval at which the model will be saved
SAVE_INTERVAL = 100


# ----------------------------------------------------------------------------------------------------------------------
# SAMPLE CONTROL ( ratio recommendation: (BATCH_SIZE * EPISODES_PER_EPOCH) / REP_Q_SIZE ≈ 8 )
# ----------------------------------------------------------------------------------------------------------------------

# The size of the replay buffer, where the agent stores its memories,
# bigger memory -> old replays stay longer in memory -> more stable gradient updates
BUFFER_SIZE = 20000

# Maximum size of the queue where collector processes store replays,
# the limit is for when the collector threads outpace the main thread
REP_Q_SIZE = 1000

# The batch size for the agents policy training
BATCH_SIZE = 1024

# The amount of batches we train per epoch
EPISODES_PER_EPOCH = 4

# On how many epochs we want to train, this is basically forever
NUM_EPOCHS = 20000



# ----------------------------------------------------------------------------------------------------------------------
# LEARNING PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# The amount of steps a run can last at max (0 for unlimited)
MAX_STEPS_PER_RUN = 0

# starting learning rate for the neural network
LEARNING_RATE = 0.001

# Initial epsilon value for epsilon-greedy exploration
EPSILON_START = 0.9

# How much epsilon decays each training epoch, high epsilon means high chance to randomly explore the environment
EPSILON_DECAY = 0.005

# Minimum epsilon value
EPSILON_MIN = 0.1

# Learning rate decay factor
LR_DECAY_FACTOR = 0.9

# Learning rate decay rate
LR_DECAY_RATE = 100

# Gamma describes how much the agent should look for future rewards vs immediate ones.
# gamma = 1 future rewards are as valuable as immediate ones
# gamma = 0 only immediate rewards matter
GAMMA = 0.95



# ----------------------------------------------------------------------------------------------------------------------
# REWARD SHAPING
# ----------------------------------------------------------------------------------------------------------------------

# If an agent does not improve (x-position) for this amount of steps, the run gets canceled
DEADLOCK_STEPS = 20

# reward penalty for getting stuck (absolute value)
DEADLOCK_PENALTY = 0.5

# the penalty for dying
DEATH_PENALTY = 1.0

# the reward when mario reaches a flag
COMPLETION_REWARD = 2.0

# factors the amount mario gets rewarded for gaining item effects
ITEM_REWARD_FACTOR = 1.0