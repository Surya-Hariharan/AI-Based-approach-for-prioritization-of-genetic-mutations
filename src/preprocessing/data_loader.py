import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.dataset import MutationDataset
from src.preprocessing.preprocessing import Preprocessor
from src.utils.config import Config
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Go up from data_loader.py -> preprocessing -> src -> project root
    return current_file.parent.parent.parent.resolve()


def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Loads data, preprocesses it, and returns DataLoaders for train, val, and test sets.
    
    Args:
        config (Config): Configuration object.
        
    Returns:
        train_loader, val_loader, test_loader, input_dim
    """
    # Get project root and resolve path
    project_root = get_project_root()
    processed_path = (project_root / config.data['processed_data_path']).resolve()
    
    df = pd.read_csv(processed_path)
    
    # The processed data is already transformed (scaled, one-hot encoded)
    # Separate features and target
    target_col = config.data['target_col']
    
    # All columns except target are features
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values if target_col in df.columns else None
    
    # Split data (Train/Test first, then Train/Val)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.data['test_size'], random_state=config.data['random_seed'], stratify=y
    )
    
    # Normalize validation size relative to the remaining training data
    val_size_adjusted = config.data['val_size'] / (1 - config.data['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=config.data['random_seed'], stratify=y_train_val
    )
    
    # Create PyTorch datasets
    train_dataset = MutationDataset(X_train, y_train)
    val_dataset = MutationDataset(X_val, y_val)
    test_dataset = MutationDataset(X_test, y_test)
    
    # Create DataLoaders
    batch_size = config.training['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim

