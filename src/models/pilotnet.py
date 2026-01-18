"""
PilotNet - NVIDIA's End-to-End Learning for Self-Driving Cars
Based on: https://arxiv.org/abs/1604.07316
"""

import torch
import torch.nn as nn
from .base_model import BaseModel


class PilotNet(BaseModel):
    """
    PilotNet model for end-to-end learning of driving commands from images.
    
    This is based on NVIDIA's architecture that learns to map raw pixels 
    from a front-facing camera to steering commands.
    
    Architecture:
    - 5 Convolutional layers (3 with 5x5 kernel, 2 with 3x3 kernel)
    - 3 Fully connected layers
    - Input: 66x200 RGB image
    - Output: Steering angle (or other driving commands)
    
    Args:
        input_channels (int): Number of input channels (default: 3 for RGB)
        output_size (int): Number of output values (default: 1 for steering angle)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, input_channels=3, output_size=1, dropout_rate=0.5):
        super(PilotNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_size = output_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Calculate the size of the flattened features
        # For input size 66x200, after convolutions: 64 channels of 1x18
        self.flatten_size = 64 * 1 * 18
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_size)
        
    def forward(self, x):
        """
        Forward pass of PilotNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 66, 200)
            
        Returns:
            torch.Tensor: Predicted steering angle(s) of shape (batch_size, output_size)
        """
        # Convolutional layers with batch normalization
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class SimplePilotNet(BaseModel):
    """
    Simplified version of PilotNet with fewer parameters.
    Useful for faster training and testing.
    
    Args:
        input_channels (int): Number of input channels (default: 3 for RGB)
        output_size (int): Number of output values (default: 1 for steering angle)
    """
    
    def __init__(self, input_channels=3, output_size=1):
        super(SimplePilotNet, self).__init__()
        
        # Simpler convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size (approximate)
        self.flatten_size = 64 * 2 * 5
        
        # Simpler fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, output_size)
        
    def forward(self, x):
        """Forward pass of SimplePilotNet"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
