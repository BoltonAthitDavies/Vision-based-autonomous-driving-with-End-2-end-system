"""
Inference script for autonomous driving model
"""

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import DrivingDataset
from data.transforms import get_val_transforms
from models.pilotnet import PilotNet, SimplePilotNet
from training.evaluator import Evaluator
from utils.visualization import plot_predictions, plot_error_distribution


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


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data transforms
    input_size = tuple(config['data']['image_size'])
    test_transform = get_val_transforms(input_size)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = DrivingDataset(
        csv_file=config['data']['test_csv'],
        root_dir=config['data']['root_dir'],
        transform=test_transform,
        target_columns=config['data']['target_columns']
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model = get_model(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {args.model_path}")
    
    # Create evaluator
    evaluator = Evaluator(model, test_loader, device)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate()
    
    # Print metrics
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    for key, value in results['metrics'].items():
        print(f"{key}: {value:.6f}")
    
    # Get error statistics
    error_stats = evaluator.get_prediction_errors(
        results['predictions'],
        results['ground_truth']
    )
    
    print("\n" + "="*50)
    print("Error Statistics")
    print("="*50)
    print(f"Mean Absolute Error: {error_stats['mean_abs_error']}")
    print(f"Max Error: {error_stats['max_error']}")
    print(f"Std Error: {error_stats['std_error']}")
    
    # Visualize results
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Plot predictions
        plot_predictions(
            results['predictions'],
            results['ground_truth'],
            num_samples=min(200, len(results['predictions'])),
            save_path=os.path.join(config['paths']['log_dir'], 'predictions.png')
        )
        
        # Plot error distribution
        plot_error_distribution(
            results['predictions'],
            results['ground_truth'],
            save_path=os.path.join(config['paths']['log_dir'], 'error_distribution.png')
        )
        
        print("Visualizations saved!")
    
    # Save predictions
    if args.save_predictions:
        output_file = os.path.join(config['paths']['log_dir'], 'predictions.npy')
        np.save(output_file, {
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth'],
            'metrics': results['metrics']
        })
        print(f"\nPredictions saved to {output_file}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate autonomous driving model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    args = parser.parse_args()
    
    main(args)
