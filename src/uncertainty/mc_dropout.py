import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Union

class MCDropoutEstimator:
    """
    Estimates epistemic uncertainty using Monte Carlo Dropout.
    """
    def __init__(self, model: nn.Module, n_samples: int = 20, device: str = 'cpu'):
        self.model = model
        self.n_samples = n_samples
        self.device = device
        self.model.to(self.device)

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Runs N forward passes with dropout enabled.
        Returns mean prediction, variance, and entropy.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        X = X.to(self.device)
        self.model.eval()
        self.enable_dropout() # Force dropout

        # Check if model has dropout
        has_dropout = any(m.__class__.__name__.startswith('Dropout') for m in self.model.modules())
        if not has_dropout:
            print("Warning: Model does not have Dropout layers. MC Dropout will output zero variance.")

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model(X)
                probs = torch.sigmoid(outputs)
                predictions.append(probs.cpu().numpy())

        # Shape: (n_samples, n_instances, 1) or (n_samples, n_instances)
        predictions = np.array(predictions)
        if predictions.ndim == 3:
            predictions = predictions.squeeze(-1)
            
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0) # Predictive variance
        
        # Entropy: -p*log(p) - (1-p)*log(1-p) for binary
        epsilon = 1e-10
        entropy = -(mean_pred * np.log(mean_pred + epsilon) + 
                    (1 - mean_pred) * np.log(1 - mean_pred + epsilon))
                    
        return {
            "mean": mean_pred,
            "variance": variance,
            "entropy": entropy,
            "predictions": predictions # Optional: return all samples
        }
