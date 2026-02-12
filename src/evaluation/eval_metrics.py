import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Union, List

def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], y_pred_probs: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    
    Args:
        y_true: True labels (0 or 1).
        y_pred_probs: Predicted probabilities (between 0 and 1).
        threshold: Classification threshold.
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
        
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_binary)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
    }
    
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred_probs))
    except ValueError:
        metrics["auc"] = 0.0
        
    return metrics

def calculate_top_k_recall(y_true: np.ndarray, y_pred_probs: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """
    Calculates Recall@K.
    """
    # Combine into DataFrame for sorting
    data = list(zip(y_true, y_pred_probs))
    data.sort(key=lambda x: x[1], reverse=True)
    
    sorted_labels = np.array([x[0] for x in data])
    total_positives = np.sum(sorted_labels)
    
    metrics = {}
    for k in k_values:
        if k > len(sorted_labels):
            k = len(sorted_labels)
            
        top_k_labels = sorted_labels[:k]
        found_positives = np.sum(top_k_labels)
        
        recall_at_k = found_positives / total_positives if total_positives > 0 else 0.0
        metrics[f"recall_at_{k}"] = float(recall_at_k)
        
    return metrics

def confusion_matrix_stats(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    """Returns TN, FP, FN, TP."""
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
