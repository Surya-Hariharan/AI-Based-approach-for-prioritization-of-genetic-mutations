import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Union

def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], y_pred_probs: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Args:
        y_true: True labels (0 or 1).
        y_pred_probs: Predicted probabilities (between 0 and 1).
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, and auc.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
        
    # Binarize predictions for classification metrics
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
    }
    
    # AUC requires at least one positive and one negative sample
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred_probs))
    except ValueError:
        metrics["auc"] = 0.0
        
    return metrics
