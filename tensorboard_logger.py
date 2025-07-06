import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class TensorBoardLogger:
    def __init__(self, experiment_name=None, log_dir="runs"):
        """
        Initialize TensorBoard logger with unique experiment name
        
        Args:
            experiment_name: Name for this experiment (auto-generated if None)
            log_dir: Directory to store TensorBoard logs
        """
        if experiment_name is None:
            # Auto-generate experiment name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"mario_rl_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        
        # Ensure the directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create the SummaryWriter with explicit log_dir to prevent nested directories
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Track if we've already logged hyperparameters to prevent duplicates
        self._hparams_logged = False
        
        print(f"[TENSORBOARD] Logging to: {self.log_dir}")
        print(f"[TENSORBOARD] Start with: tensorboard --logdir={log_dir}")
    
    def log_hyperparameters(self, hparams_dict, metrics_dict=None):
        """Log hyperparameters as scalars to avoid nested directory creation"""
        if self._hparams_logged:
            print("[TENSORBOARD] Hyperparameters already logged, skipping to prevent duplicates")
            return
        
        # Log hyperparameters as scalars at step 0 to avoid nested directories
        # This prevents TensorBoard from creating additional subdirectories
        for key, value in hparams_dict.items():
            if isinstance(value, (int, float)):
                # Log numeric hyperparameters as scalars
                self.writer.add_scalar(f"Hyperparameters/{key}", value, 0)
            elif isinstance(value, bool):
                # Log boolean hyperparameters as 0/1
                self.writer.add_scalar(f"Hyperparameters/{key}", 1.0 if value else 0.0, 0)
            else:
                # Log string/other hyperparameters as text
                self.writer.add_text(f"Hyperparameters/{key}", str(value), 0)
        
        # Also log a summary text with all hyperparameters
        hparam_summary = "\n".join([f"{k}: {v}" for k, v in hparams_dict.items()])
        self.writer.add_text("Hyperparameters/Summary", hparam_summary, 0)
        
        self._hparams_logged = True
        print(f"[TENSORBOARD] Logged {len(hparams_dict)} hyperparameters as scalars/text (no nested directories)")
        
        # Log final metrics if provided
        if metrics_dict:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"Final_Results/{key}", value, 0)
    
    def log_metrics(self, metrics_dict, step):
        """Log training metrics"""
        for key, value in metrics_dict.items():
            if value is not None:
                try:
                    self.writer.add_scalar(key, value, step)
                except Exception as e:
                    print(f"[TENSORBOARD] Warning: Could not log metric {key}: {e}")
    
    def log_model_graph(self, model, input_tensor):
        """Log model architecture"""
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"[TENSORBOARD] Could not log model graph: {e}")
    
    def log_histogram(self, tag, values, step):
        """Log parameter histograms"""
        if torch.is_tensor(values):
            self.writer.add_histogram(tag, values, step)
    
    def log_model_parameters(self, model, step):
        """Log model parameter histograms"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"parameters/{name}", param, step)
                self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def close(self):
        """Close the logger"""
        self.writer.close()
        print(f"[TENSORBOARD] Closed logger for {self.experiment_name}")


def create_experiment_config(config_module):
    """Extract hyperparameters from config module for logging"""
    hparams = {}
    
    # Extract relevant hyperparameters
    for attr_name in dir(config_module):
        if not attr_name.startswith('_'):
            attr_value = getattr(config_module, attr_name)
            if isinstance(attr_value, (int, float, str, bool)):
                hparams[attr_name] = attr_value
    
    return hparams 