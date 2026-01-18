"""
Base model interface for autonomous driving models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for autonomous driving models.
    
    All models should inherit from this class and implement the required methods.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output predictions
        """
        pass
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self):
        """
        Get a summary of the model architecture.
        
        Returns:
            str: Model summary string
        """
        summary = []
        summary.append(f"Model: {self.__class__.__name__}")
        summary.append(f"Total parameters: {self.count_parameters():,}")
        summary.append("\nArchitecture:")
        summary.append(str(self))
        return "\n".join(summary)
    
    def save(self, filepath):
        """
        Save model weights to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath, device='cpu'):
        """
        Load model weights from a file.
        
        Args:
            filepath (str): Path to the saved model
            device (str): Device to load the model on ('cpu' or 'cuda')
        """
        self.load_state_dict(torch.load(filepath, map_location=device))
        self.to(device)
