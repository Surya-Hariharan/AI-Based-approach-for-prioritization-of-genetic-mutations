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
    
    # Handle missing values and scaling
    target_col = config.data['target_col']
    features = config.data['numerical_features'] + config.data['categorical_features']
    
    X = df[features]
    y = df[target_col] if target_col in df.columns else None
    
    # Split data (Train/Test first, then Train/Val)
    # Using sklearn for splitting to ensure easy stratification if needed and reproducibility
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.data['test_size'], random_state=config.random_seed
    )
    
    # Normalize validation size relative to the remaining training data
    val_size_adjusted = config.data['val_size'] / (1 - config.data['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=config.random_seed
    )
    
    # Preprocessing
    # Fit on training data only
    preprocessor = Preprocessor(
        numerical_features=config.data['numerical_features'],
        categorical_features=config.data['categorical_features']
    )
    
    # Fit transform training data
    # Note: Preprocessor expects a DataFrame with both num & cat columns
    # We fit the internal transformer on X_train
    preprocessor.numeric_transformer.fit(X_train[config.data['numerical_features']])
    preprocessor.categorical_transformer.fit(X_train[config.data['categorical_features']])

    # Transform all splits
    # Helper to transform
    def transform_data(subset_df):
        # We need to manually apply the parts since our preprocessor class wraps a ColumnTransformer 
        # but the fit method was slightly different in the provided snippet. 
        # Let's adjust to use the preprocessor's fit method which expects the full dataframe.
        return subset_df
    
    # Re-initialize preprocessor to use the unified fit
    preprocessor = Preprocessor(
        numerical_features=config.data['numerical_features'],
        categorical_features=config.data['categorical_features']
    )
    
    # Helper function to get transformed dataframe
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor for inference later
    preprocessor_path = (project_root / config.data.get('preprocessor_path', 'data/processed/preprocessor.joblib')).resolve()
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(str(preprocessor_path))

    # Create Datasets
    train_dataset = MutationDataset(X_train_processed, y_train)
    val_dataset = MutationDataset(X_val_processed, y_val)
    test_dataset = MutationDataset(X_test_processed, y_test)
    
    # Create DataLoaders
    batch_size = config.training['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_processed.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim
