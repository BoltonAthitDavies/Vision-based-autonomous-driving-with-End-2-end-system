"""
Dataset class for loading driving images and corresponding steering angles
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DrivingDataset(Dataset):
    """
    PyTorch Dataset for autonomous driving data.
    
    Expected data format:
    - CSV file with columns: ['image_path', 'steering_angle', 'throttle', 'brake', 'speed']
    - Images in a directory structure
    
    Args:
        csv_file (str): Path to the CSV file with driving data
        root_dir (str): Root directory containing images
        transform (callable, optional): Optional transform to be applied on images
        target_columns (list): List of column names to use as targets (default: ['steering_angle'])
    """
    
    def __init__(self, csv_file, root_dir, transform=None, target_columns=None):
        self.driving_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_columns = target_columns or ['steering_angle']
        
    def __len__(self):
        return len(self.driving_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_name = os.path.join(self.root_dir, self.driving_data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get target values (steering angle, throttle, etc.)
        targets = self.driving_data.iloc[idx][self.target_columns].values.astype('float32')
        targets = torch.from_numpy(targets)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, targets


class DrivingDatasetFromFolder(Dataset):
    """
    Alternative dataset class that loads images from a folder structure.
    Useful when you don't have a CSV file.
    
    Args:
        data_dir (str): Directory containing images
        labels_dict (dict): Dictionary mapping image filenames to labels
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, data_dir, labels_dict, transform=None):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.labels_dict = labels_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get label from dictionary
        label = self.labels_dict.get(img_name, 0.0)
        label = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
