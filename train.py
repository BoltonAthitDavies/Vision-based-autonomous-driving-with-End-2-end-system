"""
Training script for autonomous driving model
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import DrivingDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.pilotnet import PilotNet, SimplePilotNet
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager
from utils.visualization import plot_training_history


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    """Create model based on configuration"""
    model_type = config['model']['type']
    
    if model_type == 'PilotNet':
        model = PilotNet(
            input_channels=config['model']['input_channels'],
            output_size=config['model']['output_size'],
            dropout_rate=config['model']['dropout_rate']
        )
    elif model_type == 'SimplePilotNet':
        model = SimplePilotNet(
            input_channels=config['model']['input_channels'],
            output_size=config['model']['output_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_optimizer(model, config):
    """Create optimizer based on configuration"""
    opt_type = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def get_loss_function(config):
    """Create loss function based on configuration"""
    loss_type = config['training']['loss_function'].lower()
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")


def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    if 'scheduler' not in config['training']:
        return None
    
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config.get('patience', 5),
            factor=scheduler_config.get('factor', 0.5),
            min_lr=scheduler_config.get('min_lr', 0.00001)
        )
    elif scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 50),
            eta_min=scheduler_config.get('min_lr', 0.00001)
        )
    else:
        return None
    
    return scheduler


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Setup logger
    logger = setup_logger(log_dir=config['paths']['log_dir'])
    logger.info("Starting training...")
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create data transforms
    input_size = tuple(config['data']['image_size'])
    train_transform = get_train_transforms(input_size)
    val_transform = get_val_transforms(input_size)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = DrivingDataset(
        csv_file=config['data']['train_csv'],
        root_dir=config['data']['root_dir'],
        transform=train_transform,
        target_columns=config['data']['target_columns']
    )
    
    val_dataset = DrivingDataset(
        csv_file=config['data']['val_csv'],
        root_dir=config['data']['root_dir'],
        transform=val_transform,
        target_columns=config['data']['target_columns']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = get_model(config)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {model.count_parameters():,}")
    
    # Create optimizer, loss function, and scheduler
    optimizer = get_optimizer(model, config)
    criterion = get_loss_function(config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    # Train model
    logger.info("Starting training loop...")
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=config['paths']['checkpoint_dir'],
        save_best=config['save_best']
    )
    
    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history, save_path=os.path.join(config['paths']['log_dir'], 'training_history.png'))
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autonomous driving model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)
