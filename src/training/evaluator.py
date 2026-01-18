"""
Evaluator class for model evaluation and metrics
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Evaluator:
    """
    Evaluator class to evaluate model performance.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): DataLoader for test data
        device (str): Device to run evaluation on ('cpu' or 'cuda')
    """
    
    def __init__(self, model, test_loader, device='cpu'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        
    def evaluate(self):
        """
        Evaluate the model on test data.
        
        Returns:
            dict: Dictionary containing predictions, ground truth, and metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, ground_truth)
        
        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'metrics': metrics
        }
    
    def calculate_metrics(self, predictions, ground_truth):
        """
        Calculate evaluation metrics.
        
        Args:
            predictions (np.ndarray): Model predictions
            ground_truth (np.ndarray): Ground truth values
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # For each output dimension (e.g., steering, throttle)
        for i in range(predictions.shape[1]):
            pred_i = predictions[:, i]
            true_i = ground_truth[:, i]
            
            metrics[f'mse_{i}'] = mean_squared_error(true_i, pred_i)
            metrics[f'rmse_{i}'] = np.sqrt(mean_squared_error(true_i, pred_i))
            metrics[f'mae_{i}'] = mean_absolute_error(true_i, pred_i)
            metrics[f'r2_{i}'] = r2_score(true_i, pred_i)
        
        # Overall metrics
        metrics['overall_mse'] = mean_squared_error(ground_truth.flatten(), 
                                                     predictions.flatten())
        metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        metrics['overall_mae'] = mean_absolute_error(ground_truth.flatten(), 
                                                      predictions.flatten())
        
        return metrics
    
    def predict(self, image):
        """
        Make a prediction on a single image.
        
        Args:
            image (torch.Tensor): Input image tensor
            
        Returns:
            np.ndarray: Prediction
        """
        self.model.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            output = self.model(image)
            
            return output.cpu().numpy()
    
    def get_prediction_errors(self, predictions, ground_truth):
        """
        Calculate prediction errors.
        
        Args:
            predictions (np.ndarray): Model predictions
            ground_truth (np.ndarray): Ground truth values
            
        Returns:
            dict: Dictionary containing error statistics
        """
        errors = predictions - ground_truth
        
        error_stats = {
            'mean_error': np.mean(errors, axis=0),
            'std_error': np.std(errors, axis=0),
            'abs_errors': np.abs(errors),
            'mean_abs_error': np.mean(np.abs(errors), axis=0),
            'max_error': np.max(np.abs(errors), axis=0),
            'min_error': np.min(np.abs(errors), axis=0)
        }
        
        return error_stats
