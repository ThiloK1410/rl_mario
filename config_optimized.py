# Optimized configuration for maximum training performance

# Import base configuration
from config import *

# PERFORMANCE OPTIMIZATIONS
# ----------------------------------------------------------------------------------------------------------------------

# Increase number of collector processes for better CPU utilization
NUM_PROCESSES = 4  # Increased from 2

# Larger replay buffer for more stable training
BUFFER_SIZE = 50000  # Increased from 10000

# Larger batch size for GPU efficiency
BATCH_SIZE = 512  # Increased from 256

# More episodes per epoch for better GPU utilization
EPISODES_PER_EPOCH = 16  # Increased from 8

# Reduce model update frequency to minimize overhead
MODEL_UPDATE_FREQUENCY = 5  # Update every 5 epochs

# Enable all performance optimizations
USE_MIXED_PRECISION = True
ASYNC_MODEL_UPDATES = True

# Larger experience queue for reduced contention
REP_Q_SIZE = 10000  # Increased from 2000

# More aggressive learning rate for faster convergence
LEARNING_RATE = 0.001  # Increased from 0.0005

# Faster epsilon decay for quicker exploitation
EPSILON_DECAY = 0.002  # Increased from 0.001

# Higher minimum epsilon for continued exploration
EPSILON_MIN = 0.05  # Reduced from 0.1

# More frequent checkpointing
SAVE_INTERVAL = 50  # Reduced from 100

# THREAD OPTIMIZATION SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Batch processing size for experience processor
EXPERIENCE_BATCH_SIZE = 1000  # Process experiences in larger batches

# Reduced sleep intervals for lower latency
PROCESSOR_INTERVAL = 0.001  # 1ms polling interval

# Queue sizes for lock-free architecture
LOCAL_QUEUE_SIZE = 10000  # Per-collector queue size
CONSOLIDATED_QUEUE_SIZE = REP_Q_SIZE * 2  # Central queue size

# GRADIENT ACCUMULATION SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Number of gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS = 4

# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# 512 * 4 = 2048 effective batch size for stable training

# OPTIMIZER SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Weight decay for regularization
WEIGHT_DECAY = 1e-5

# Gradient clipping for stability
GRADIENT_CLIP_NORM = 10.0

# Learning rate scheduler settings
LR_SCHEDULER_T0 = 100  # Initial period for cosine annealing
LR_SCHEDULER_T_MULT = 2  # Period multiplier
LR_SCHEDULER_ETA_MIN = LEARNING_RATE * 0.01  # Minimum learning rate

# CUDA OPTIMIZATION FLAGS
# ----------------------------------------------------------------------------------------------------------------------

# Enable TF32 for faster computation (requires Ampere GPUs)
ENABLE_TF32 = True

# Enable cudnn benchmarking for optimal convolution algorithms
CUDNN_BENCHMARK = True

# Memory allocation strategy
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# ADVANCED REPLAY BUFFER SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Prioritized Experience Replay with optimized parameters
PER_ALPHA = 0.7  # Slightly increased for more prioritization
PER_BETA = 0.5  # Increased starting beta
PER_BETA_INCREMENT = 0.0001  # Slower beta increment for longer training

# PRE-ALLOCATION SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Pre-allocate tensors for common batch sizes
PREALLOCATE_BATCH_SIZES = [BATCH_SIZE, BATCH_SIZE * 2, BATCH_SIZE * 4]

# MONITORING AND LOGGING
# ----------------------------------------------------------------------------------------------------------------------

# Reduced logging frequency for less overhead
STATS_PRINT_INTERVAL = 30  # Print stats every 30 seconds
EPISODE_LOG_INTERVAL = 20  # Log episode stats every 20 episodes

# Performance profiling settings
ENABLE_PROFILING = False  # Set to True for detailed performance analysis
PROFILE_WAIT_STEPS = 10
PROFILE_WARMUP_STEPS = 10
PROFILE_ACTIVE_STEPS = 20

# GPU MEMORY OPTIMIZATION
# ----------------------------------------------------------------------------------------------------------------------

# Clear CUDA cache frequency
CUDA_CACHE_CLEAR_INTERVAL = 100  # Clear every 100 epochs

# Maximum GPU memory fraction (0.0 - 1.0)
MAX_GPU_MEMORY_FRACTION = 0.95  # Use up to 95% of GPU memory

# NETWORK ARCHITECTURE OPTIMIZATIONS
# ----------------------------------------------------------------------------------------------------------------------

# Use batch normalization for faster convergence
USE_BATCH_NORM = True

# Use layer normalization in fully connected layers
USE_LAYER_NORM = True

# Dropout rate for regularization (0.0 = no dropout)
DROPOUT_RATE = 0.0  # Disabled for speed

# PARALLEL TRAINING SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

# Enable data parallel training if multiple GPUs available
USE_DATA_PARALLEL = False  # Set to True if multiple GPUs

# Number of worker threads for data loading
NUM_WORKERS = 0  # Set to 0 for main thread loading (faster for small data)

print("[CONFIG] Loaded optimized configuration for maximum performance")
print(f"[CONFIG] Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"[CONFIG] Collector processes: {NUM_PROCESSES}")
print(f"[CONFIG] Buffer size: {BUFFER_SIZE:,}")
print(f"[CONFIG] Model update frequency: every {MODEL_UPDATE_FREQUENCY} epochs")