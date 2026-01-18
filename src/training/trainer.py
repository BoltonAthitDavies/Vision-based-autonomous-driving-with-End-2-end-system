"""
Trainer class for training autonomous driving models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os


class Trainer:
    """
    Trainer class to handle model training.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to train on ('cpu' or 'cuda')
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler
    """
    
    def __init__(self, model, train_loader, val_loader=None, criterion=None, 
                 optimizer=None, device='cpu', scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.MSELoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.device = device
        self.scheduler = scheduler
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """
        Validate the model on validation data.
        
        Returns:
            float: Average validation loss
        """
        if self.val_loader is None:
            return None
            
        self.model.eval()
        epoch_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.val_loader)
        return avg_loss
    
    def train(self, num_epochs, save_dir='models', save_best=True):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save model checkpoints
            save_best (bool): Whether to save the best model based on validation loss
            
        Returns:
            dict: Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
                print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model with val loss: {val_loss:.6f}")
            else:
                print(f"Train Loss: {train_loss:.6f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
