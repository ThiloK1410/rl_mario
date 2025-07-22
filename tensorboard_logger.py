import os
import threading
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TensorBoardLogger:
    """Simple, process-safe TensorBoard logger for multiprocessing environments."""
    
    def __init__(self, experiment_name=None, process_name=None):
        # Auto-generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"mario_rl_{timestamp}"
        
        # Create log directory (separate for each process)
        if process_name:
            log_dir = os.path.join("runs", experiment_name, process_name)
        else:
            log_dir = os.path.join("runs", experiment_name)
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize writer and thread lock
        self.writer = SummaryWriter(log_dir=log_dir)
        self.lock = threading.Lock()
        
        print(f"[TB] Logging to: {log_dir}")
    
    def log(self, key, value, epoch):
        """Log single metric safely."""
        if not isinstance(value, (int, float)):
            print(f"[TB] Warning: {key} value must be numeric, got {type(value)}")
            return
        
        with self.lock:
            try:
                self.writer.add_scalar(key, value, epoch)
            except Exception as e:
                print(f"[TB] Error logging {key}: {e}")
    
    def close(self):
        """Close the logger."""
        with self.lock:
            self.writer.close()


# For easy process-specific logger creation
def create_logger(process_name, experiment_name=None):
    """Create a logger for a specific process."""
    return TensorBoardLogger(experiment_name=experiment_name, process_name=process_name)
