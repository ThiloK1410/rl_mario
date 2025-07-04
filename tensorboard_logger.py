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
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)
        self.experiment_name = experiment_name
        
        print(f"[TENSORBOARD] Logging to: {self.log_dir}")
        print(f"[TENSORBOARD] Start with: tensorboard --logdir={log_dir}")
    
    def log_hyperparameters(self, hparams_dict, metrics_dict=None):
        """Log hyperparameters for this experiment"""
        if metrics_dict is None:
            metrics_dict = {}
        self.writer.add_hparams(hparams_dict, metrics_dict)
        
    def log_metrics(self, metrics_dict, step):
        """Log training metrics"""
        for key, value in metrics_dict.items():
            if value is not None:
                self.writer.add_scalar(key, value, step)
    
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