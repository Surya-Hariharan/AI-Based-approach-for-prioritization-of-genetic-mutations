import argparse
import torch
import pandas as pd
import numpy as np
import os

from src.config.loader import Config
from src.utils.data_loader import get_data_loaders
from src.ensemble.stacking import StackingEnsemble

def main():
    parser = argparse.ArgumentParser(description="Train Ensemble Stacking Model")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--output_path", type=str, default="models/ensemble.joblib", help="Path to save trained ensemble")
    args = parser.parse_args()

    config = Config(args.config)
    
    # Check if ensemble enabled
    if not config.ensemble.get('enabled', False):
        print("Ensemble is not enabled in config. Skipping training.")
        return

    # Load Data (Train/Val combined for CV usually, but here we can just use train_loader data)
    # StackingEnsemble expects raw X, y arrays for sklearn compatibility mostly.
    # PyTorch loader returns batches. We need full arrays.
    
    print("Loading Data...")
    train_loader, val_loader, _, input_dim = get_data_loaders(config)
    
    # Collect all training data
    X_train_list = []
    y_train_list = []
    
    for X, y in train_loader:
        X_train_list.append(X.numpy())
        y_train_list.append(y.numpy())
        
    # Also include validation data for robust training? Usually stacking uses CV on Train.
    # Let's stick to Train data for CV stacking, and validate on Val/Test later.
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    print(f"Data Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    
    # Initialize Ensemble
    ensemble = StackingEnsemble(config.ensemble, input_dim=input_dim)
    
    # Fit
    print("Training Ensemble (this may take time)...")
    ensemble.fit(X_train, y_train.flatten())
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    ensemble.save(args.output_path)
    print(f"Ensemble saved to {args.output_path}")

if __name__ == "__main__":
    main()
