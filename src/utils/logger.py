"""
Logging utilities
"""

import logging
import os
from datetime import datetime


def setup_logger(name='autonomous_driving', log_dir='logs', level=logging.INFO):
    """
    Setup logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class TrainingLogger:
    """
    Custom logger for tracking training progress.
    """
    
    def __init__(self, log_file='training_log.txt'):
        self.log_file = log_file
        self.metrics = []
        
    def log_epoch(self, epoch, metrics_dict):
        """
        Log metrics for an epoch.
        
        Args:
            epoch (int): Epoch number
            metrics_dict (dict): Dictionary of metrics
        """
        log_entry = f"Epoch {epoch}: " + ", ".join([f"{k}={v:.6f}" for k, v in metrics_dict.items()])
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
        
        # Store in memory
        self.metrics.append({'epoch': epoch, **metrics_dict})
        
        print(log_entry)
    
    def get_metrics(self):
        """
        Get all logged metrics.
        
        Returns:
            list: List of metric dictionaries
        """
        return self.metrics
    
    def save_summary(self, summary_file='training_summary.txt'):
        """
        Save training summary.
        
        Args:
            summary_file (str): Path to save summary
        """
        with open(summary_file, 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if self.metrics:
                f.write(f"Total Epochs: {len(self.metrics)}\n")
                
                # Get best metrics
                if 'val_loss' in self.metrics[0]:
                    best_epoch = min(self.metrics, key=lambda x: x.get('val_loss', float('inf')))
                    f.write(f"Best Epoch: {best_epoch['epoch']}\n")
                    f.write(f"Best Val Loss: {best_epoch.get('val_loss', 'N/A'):.6f}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("Epoch-wise Metrics:\n")
                f.write("=" * 50 + "\n")
                
                for metric in self.metrics:
                    f.write(f"Epoch {metric['epoch']}: ")
                    f.write(", ".join([f"{k}={v:.6f}" for k, v in metric.items() if k != 'epoch']))
                    f.write("\n")
