import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional, List

class ModelInterpreter:
    """
    Wraps pyTorch model for SHAP analysis.
    """
    def __init__(self, model, background_data: torch.Tensor, feature_names: List[str], output_dir: str = "reports/figures"):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use KernelExplainer for broad compatibility (or DeepExplainer for gradients)
        # Using a small background sample for speed
        self.explainer = shap.KernelExplainer(self._predict_wrapper, background_data.cpu().numpy())

    def _predict_wrapper(self, x):
        tensor_x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(tensor_x)
            return torch.sigmoid(outputs).numpy()

    def explain_summary(self, X_evaluate: np.ndarray):
        """
        Generates and saves summary plot.
        """
        shap_values = self.explainer.shap_values(X_evaluate)
        
        # For binary classification, shap_values is list, take index 0 (or correct class)
        # Verify shape: if model output is 1D (logit), shap_values might be array.
        # KernelExplainer typically returns a list for classification output even if single output node?
        # Let's check based on wrapper output (sigmoid -> 1D array per sample).
        # Wrapper returns (N, 1) or (N,).
        
        if isinstance(shap_values, list):
            vals = shap_values[0]
        else:
            vals = shap_values
            
        # Summary Plot
        plt.figure()
        shap.summary_plot(vals, X_evaluate, feature_names=self.feature_names, show=False)
        plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))
        plt.close()
        
        return vals

    def explain_local(self, x_instance: np.ndarray, index: int):
        """
        Explains single instance. (Note: SHAP force plots are JS based, might be hard to save static)
        We can save a waterfall plot instead for matplotlib.
        """
        shap_values = self.explainer.shap_values(x_instance.reshape(1, -1))
        
        # Depending on SHAP version, this might vary.
        # We will skip complex interactive plots for static report.
        pass
