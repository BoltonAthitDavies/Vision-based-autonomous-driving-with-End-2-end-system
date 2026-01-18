"""
Visualization utilities for training and predictions
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss history.
    
    Args:
        history (dict): Dictionary with 'train_losses' and 'val_losses' keys
        save_path (str, optional): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    
    if 'val_losses' in history and history['val_losses']:
        ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_predictions(predictions, ground_truth, num_samples=100, save_path=None):
    """
    Plot predictions vs ground truth.
    
    Args:
        predictions (np.ndarray): Model predictions
        ground_truth (np.ndarray): Ground truth values
        num_samples (int): Number of samples to plot
        save_path (str, optional): Path to save the plot
    """
    # Limit number of samples
    num_samples = min(num_samples, len(predictions))
    pred_sample = predictions[:num_samples]
    truth_sample = ground_truth[:num_samples]
    
    num_outputs = predictions.shape[1] if len(predictions.shape) > 1 else 1
    
    if num_outputs == 1:
        # Single output (e.g., steering angle)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predictions vs ground truth
        x = np.arange(num_samples)
        axes[0].plot(x, truth_sample, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        axes[0].plot(x, pred_sample, 'r--', label='Predictions', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Predictions vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot scatter
        axes[1].scatter(truth_sample, pred_sample, alpha=0.5, s=10)
        axes[1].plot([truth_sample.min(), truth_sample.max()], 
                     [truth_sample.min(), truth_sample.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Ground Truth')
        axes[1].set_ylabel('Predictions')
        axes[1].set_title('Prediction Scatter Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    else:
        # Multiple outputs
        fig, axes = plt.subplots(num_outputs, 2, figsize=(14, 4 * num_outputs))
        
        for i in range(num_outputs):
            x = np.arange(num_samples)
            pred_i = pred_sample[:, i]
            truth_i = truth_sample[:, i]
            
            # Time series plot
            axes[i, 0].plot(x, truth_i, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.7)
            axes[i, 0].plot(x, pred_i, 'r--', label='Predictions', linewidth=1.5, alpha=0.7)
            axes[i, 0].set_xlabel('Sample Index')
            axes[i, 0].set_ylabel(f'Output {i}')
            axes[i, 0].set_title(f'Output {i}: Predictions vs Ground Truth')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Scatter plot
            axes[i, 1].scatter(truth_i, pred_i, alpha=0.5, s=10)
            axes[i, 1].plot([truth_i.min(), truth_i.max()], 
                           [truth_i.min(), truth_i.max()], 
                           'r--', linewidth=2, label='Perfect Prediction')
            axes[i, 1].set_xlabel('Ground Truth')
            axes[i, 1].set_ylabel('Predictions')
            axes[i, 1].set_title(f'Output {i}: Scatter Plot')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_error_distribution(predictions, ground_truth, save_path=None):
    """
    Plot distribution of prediction errors.
    
    Args:
        predictions (np.ndarray): Model predictions
        ground_truth (np.ndarray): Ground truth values
        save_path (str, optional): Path to save the plot
    """
    errors = predictions - ground_truth
    
    num_outputs = errors.shape[1] if len(errors.shape) > 1 else 1
    
    fig, axes = plt.subplots(1, num_outputs, figsize=(6 * num_outputs, 5))
    
    if num_outputs == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        error_i = errors if num_outputs == 1 else errors[:, i]
        
        ax.hist(error_i, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution - Output {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error_i)
        std_error = np.std(error_i)
        ax.text(0.02, 0.98, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_sample_predictions(model, dataset, num_samples=5, device='cpu'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model (nn.Module): The trained model
        dataset (Dataset): Dataset to sample from
        num_samples (int): Number of samples to visualize
        device (str): Device to run inference on
    """
    model.eval()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    
    if num_samples == 1:
        axes = [axes]
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            image, target = dataset[idx]
            
            # Make prediction
            if len(image.shape) == 3:
                image_input = image.unsqueeze(0).to(device)
            else:
                image_input = image.to(device)
            
            prediction = model(image_input).cpu().numpy()[0]
            
            # Denormalize image for visualization
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            ax.imshow(img_display)
            ax.axis('off')
            ax.set_title(f'True: {target.numpy()[0]:.3f}\nPred: {prediction[0]:.3f}')
    
    plt.tight_layout()
    plt.show()
