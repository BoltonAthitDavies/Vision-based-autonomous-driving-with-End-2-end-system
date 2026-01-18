"""
Data transforms and augmentation for driving images
"""

import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image


def get_train_transforms(input_size=(66, 200)):
    """
    Get training data transforms with augmentation.
    
    Args:
        input_size (tuple): Target image size (height, width)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(input_size=(66, 200)):
    """
    Get validation/test data transforms without augmentation.
    
    Args:
        input_size (tuple): Target image size (height, width)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class RandomBrightness:
    """Custom transform to randomly adjust brightness"""
    
    def __init__(self, brightness_range=(0.5, 1.5)):
        self.brightness_range = brightness_range
    
    def __call__(self, img):
        brightness_factor = np.random.uniform(*self.brightness_range)
        enhancer = Image.Enhance.Brightness(img)
        return enhancer.enhance(brightness_factor)


class RandomShadow:
    """Custom transform to add random shadows (simulates changing lighting conditions)"""
    
    def __init__(self, shadow_prob=0.5):
        self.shadow_prob = shadow_prob
    
    def __call__(self, img):
        if np.random.random() < self.shadow_prob:
            # Convert to numpy for shadow manipulation
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Create random shadow region
            x1, y1 = np.random.randint(0, w), 0
            x2, y2 = np.random.randint(0, w), h
            
            # Apply shadow darkening
            shadow_mask = np.zeros_like(img_array)
            shadow_mask = np.where(
                (img_array[:, :, 0] > x1) if x2 > x1 else (img_array[:, :, 0] < x1),
                img_array * 0.5,
                img_array
            )
            img = Image.fromarray(shadow_mask.astype('uint8'))
        
        return img


def get_augmented_transforms(input_size=(66, 200)):
    """
    Get transforms with advanced augmentation techniques.
    
    Args:
        input_size (tuple): Target image size (height, width)
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        RandomBrightness(brightness_range=(0.7, 1.3)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
