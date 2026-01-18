"""
Checkpoint utilities for saving and loading models
"""

import torch
import os
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save the checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to the checkpoint file
        model (nn.Module): The model to load weights into
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into
        device (str): Device to load the model on
        
    Returns:
        dict: Checkpoint dictionary with epoch, loss, and other saved items
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def save_model_weights(model, filepath):
    """
    Save only model weights (not optimizer state).
    
    Args:
        model (nn.Module): The model to save
        filepath (str): Path to save the weights
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath, device='cpu'):
    """
    Load only model weights.
    
    Args:
        model (nn.Module): The model to load weights into
        filepath (str): Path to the weights file
        device (str): Device to load the model on
        
    Returns:
        nn.Module: Model with loaded weights
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model weights loaded from {filepath}")
    
    return model


def get_latest_checkpoint(checkpoint_dir):
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        
    Returns:
        str: Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    
    latest = checkpoints[-1]
    return os.path.join(checkpoint_dir, latest)


class CheckpointManager:
    """
    Manager for handling multiple checkpoints.
    """
    
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save(self, model, optimizer, epoch, loss, name=None, **kwargs):
        """
        Save checkpoint and manage the number of stored checkpoints.
        
        Args:
            model (nn.Module): The model to save
            optimizer (torch.optim.Optimizer): The optimizer
            epoch (int): Current epoch
            loss (float): Current loss
            name (str, optional): Custom name for the checkpoint
            **kwargs: Additional items to save
        """
        if name is None:
            name = f"checkpoint_epoch_{epoch}.pth"
        
        filepath = os.path.join(self.checkpoint_dir, name)
        save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs)
        
        # Clean old checkpoints
        self._clean_old_checkpoints()
        
    def _clean_old_checkpoints(self):
        """Remove old checkpoints to keep only max_checkpoints most recent ones."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda x: os.path.getmtime(
                os.path.join(self.checkpoint_dir, x)))
            
            # Remove oldest checkpoints
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_latest(self, model, optimizer=None, device='cpu'):
        """
        Load the latest checkpoint.
        
        Args:
            model (nn.Module): The model to load weights into
            optimizer (torch.optim.Optimizer, optional): The optimizer
            device (str): Device to load on
            
        Returns:
            dict: Checkpoint dictionary, or None if no checkpoints found
        """
        latest = get_latest_checkpoint(self.checkpoint_dir)
        
        if latest is None:
            print("No checkpoints found")
            return None
        
        return load_checkpoint(latest, model, optimizer, device)
