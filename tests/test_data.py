"""
Unit tests for data loading and transforms
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest
from PIL import Image
from src.data.transforms import get_train_transforms, get_val_transforms, get_augmented_transforms


def test_train_transforms():
    """Test training transforms"""
    transform = get_train_transforms(input_size=(66, 200))
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='blue')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 66, 200)


def test_val_transforms():
    """Test validation transforms"""
    transform = get_val_transforms(input_size=(66, 200))
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='red')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 66, 200)


def test_augmented_transforms():
    """Test augmented transforms"""
    transform = get_augmented_transforms(input_size=(66, 200))
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='green')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 66, 200)


def test_transform_normalization():
    """Test that transforms normalize the image correctly"""
    transform = get_val_transforms(input_size=(66, 200))
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='white')
    
    # Apply transform
    transformed = transform(dummy_image)
    
    # Check that values are normalized (should not be 0-255)
    assert transformed.min() < 1.0
    assert transformed.max() > -1.0


if __name__ == '__main__':
    pytest.main([__file__])
