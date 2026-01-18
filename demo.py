"""
Demo script showing how to use the autonomous driving pipeline
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.pilotnet import PilotNet
from data.transforms import get_val_transforms


def demo_model_creation():
    """Demonstrate model creation and summary"""
    print("="*60)
    print("Demo 1: Model Creation")
    print("="*60)
    
    # Create PilotNet model
    model = PilotNet(input_channels=3, output_size=1, dropout_rate=0.5)
    print(model.get_model_summary())
    print()


def demo_data_preprocessing():
    """Demonstrate data preprocessing"""
    print("="*60)
    print("Demo 2: Data Preprocessing")
    print("="*60)
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (200, 66), color='blue')
    print(f"Original image size: {dummy_image.size}")
    
    # Apply transforms
    transform = get_val_transforms(input_size=(66, 200))
    transformed = transform(dummy_image)
    
    print(f"Transformed tensor shape: {transformed.shape}")
    print(f"Transformed tensor range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    print()


def demo_inference():
    """Demonstrate inference with random data"""
    print("="*60)
    print("Demo 3: Model Inference")
    print("="*60)
    
    # Create model
    model = PilotNet(input_channels=3, output_size=1)
    model.eval()
    
    # Create random input (batch of 4 images)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 66, 200)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print(f"Output shape: {predictions.shape}")
    print(f"Predictions: {predictions.squeeze().numpy()}")
    print()


def demo_training_setup():
    """Demonstrate training setup"""
    print("="*60)
    print("Demo 4: Training Setup")
    print("="*60)
    
    # Create model
    model = PilotNet(input_channels=3, output_size=1)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = torch.nn.MSELoss()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print()


def demo_single_training_step():
    """Demonstrate a single training step"""
    print("="*60)
    print("Demo 5: Single Training Step")
    print("="*60)
    
    # Create model
    model = PilotNet(input_channels=3, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Dummy data
    images = torch.randn(8, 3, 66, 200)
    targets = torch.randn(8, 1)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    outputs = model(images)
    loss = criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()
    
    print(f"Input shape: {images.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Loss: {loss.item():.6f}")
    print()


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("Vision-based Autonomous Driving - Demo")
    print("="*60 + "\n")
    
    demo_model_creation()
    demo_data_preprocessing()
    demo_inference()
    demo_training_setup()
    demo_single_training_step()
    
    print("="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your dataset (images and CSV with labels)")
    print("2. Update config/config.yaml with your data paths")
    print("3. Run: python train.py --config config/config.yaml")
    print("4. Evaluate: python inference.py --model_path models/best_model.pth --visualize")
    print()


if __name__ == '__main__':
    main()
