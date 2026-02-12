import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Dict, Any, Optional
import os
import joblib
from tqdm.auto import tqdm

# Check for XGBoost/LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to make PyTorch models compatible with scikit-learn interface.
    """
    def __init__(self, model_class, input_dim, model_params=None, epochs=20, batch_size=32, lr=0.001, device='cpu'):
        self.model_class = model_class
        self.input_dim = input_dim
        self.model_params = model_params if model_params else {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(self, X, y):
        # Initialize model
        self.model = self.model_class(self.input_dim, **self.model_params)
        self.model.to(self.device)
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in tqdm(range(self.epochs), desc="Training epochs", leave=False):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()
        return np.hstack([1-probs, probs]) # Return (N, 2) for sklearn compatibility

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

class StackingEnsemble:
    def __init__(self, config: Dict[str, Any], input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.models = []
        self.meta_learner = LogisticRegression()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()

    def _init_models(self):
        base_models_config = self.config.get('base_models', [])
        for model_cfg in base_models_config:
            m_type = model_cfg['type']
            params = model_cfg.get('params', {})
            
            if m_type == 'mlp':
                from src.models.mlp import MLP
                # Pass input_dim and params from config.model.mlp if not in ensemble params
                # Assuming simple mapping for now
                wrapper = PyTorchClassifierWrapper(
                    model_class=MLP, 
                    input_dim=self.input_dim,
                    model_params={'hidden_layers': [64, 32], 'dropout': 0.2}, # Default or from config
                    device=self.device
                )
                self.models.append(('mlp', wrapper))
            
            elif m_type == 'xgboost':
                if XGB_AVAILABLE:
                    clf = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
                else:
                    print("XGBoost not available, falling back to sklearn GradientBoosting")
                    clf = GradientBoostingClassifier(**params)
                self.models.append(('xgboost', clf))
                
            elif m_type == 'lightgbm':
                if LGB_AVAILABLE:
                    clf = lgb.LGBMClassifier(**params)
                else:
                    print("LightGBM not available, falling back to sklearn GradientBoosting")
                    clf = GradientBoostingClassifier(**params)
                self.models.append(('lightgbm', clf))

    def fit(self, X, y):
        n_folds = self.config.get('n_folds', 5)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        meta_features = np.zeros((X.shape[0], len(self.models)))
        
        print(f"Training Stacking Ensemble with {len(self.models)} base models...")
        
        # OOF Predictions
        for i, (name, model) in enumerate(tqdm(self.models, desc="Base models")):
            print(f"  Training {name}...")
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X, y), total=n_folds, desc=f"{name} folds", leave=False)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train) # Note: clones model? No, sklearn fit modifies in place. 
                # This logic is flawed for standard sklearn models if we want to keep them.
                # Standard stacking: 
                # 1. Train on (K-1) folds, predict on Kth. Repeat K times. Combine predictions.
                # 2. Train meta-learner on Combined predictions.
                # 3. Retrain base models on FULL data for inference.
                
                # We need to clone the model for each fold or just use cross_val_predict? 
                # cross_val_predict is cleaner but wrapper needs to be cloneable.
                
                preds = model.predict_proba(X_val)[:, 1]
                meta_features[val_idx, i] = preds
                
            # After K-Fold, retrain on full data
            print(f"  Retraining {name} on full data...")
            model.fit(X, y)
            
        print("  Training Meta-Learner...")
        self.meta_learner.fit(meta_features, y)
        print("Ensemble Training Complete.")

    def predict_proba(self, X):
        meta_features = np.zeros((X.shape[0], len(self.models)))
        for i, (name, model) in enumerate(self.models):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
        
        # Meta learner prediction
        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def save(self, path: str):
        # We need to save all base models + meta learner
        # PyTorch models inside wrappers might be tricky with joblib
        # Solution: Save wrapper state (including model state dict)
        
        # For simplicity, we'll try joblib first, but PyTorch objects might require torch.save.
        # Custom save logic:
        state = {
            'meta_learner': self.meta_learner,
            'models': []
        }
        
        import copy
        
        for name, model in self.models:
            if isinstance(model, PyTorchClassifierWrapper):
                # Save state dict
                model_state = {
                    'type': 'pytorch_wrapper',
                    'name': name,
                    'params': model.get_params(), # Sklearn method
                    'state_dict': model.model.state_dict(),
                    'model_class_name': 'MLP' # Hardcoded for now
                }
                state['models'].append(model_state)
            else:
                state['models'].append({'type': 'sklearn', 'name': name, 'model': model})
                
        # Actually, pickling local classes/wrappers might be fragile. 
        # Ideally, we save weights independently.
        
        # Simplified: Just torch.save dictionary?
        # Let's rely on joblib for sklearn and torch.save for pytorch components?
        # Or just use joblib for everything and hope PyTorch serialization works (it usually does for simple wrappers if not across devices/versions)
        # But we must ensure device is CPU before saving.
        
        for name, model in self.models:
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                model.model.cpu()
                
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
