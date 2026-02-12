"""
Configuration management utilities.

This module provides a clean interface for loading and accessing
experiment configurations from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class Config:
    """
    Configuration loader and accessor for experiment settings.
    
    Loads configuration from YAML file and provides property-based
    access to different configuration sections.
    
    Example:
        >>> config = Config("configs/config.yaml")
        >>> print(config.model['type'])
        >>> print(config.training['epochs'])
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If YAML file is invalid
        """
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {path}\n"
                f"Current working directory: {Path.cwd()}"
            )
        
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")
                
        return config

    @property
    def data(self) -> Dict[str, Any]:
        """Data configuration (paths, features, splits)."""
        return self.config.get('data', {})

    @property
    def model(self) -> Dict[str, Any]:
        """Model architecture configuration."""
        return self.config.get('model', {})

    @property
    def training(self) -> Dict[str, Any]:
        """Training hyperparameters configuration."""
        return self.config.get('training', {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Evaluation metrics and thresholds configuration."""
        return self.config.get('evaluation', {})

    @property
    def interpretation(self) -> Dict[str, Any]:
        """Model interpretation configuration."""
        return self.config.get('interpretation', {})

    @property
    def ensemble(self) -> Dict[str, Any]:
        """Ensemble learning configuration."""
        return self.config.get('ensemble', {})

    @property
    def uncertainty(self) -> Dict[str, Any]:
        """Uncertainty estimation configuration."""
        return self.config.get('uncertainty', {})

    @property
    def graph(self) -> Dict[str, Any]:
        """Graph neural network configuration."""
        return self.config.get('graph', {})

    @property
    def aggregation(self) -> Dict[str, Any]:
        """Gene-level aggregation configuration."""
        return self.config.get('aggregation', {})
    
    @property
    def numerical_features(self) -> List[str]:
        """List of numerical feature column names."""
        return self.data.get('numerical_features', [])

    @property
    def categorical_features(self) -> List[str]:
        """List of categorical feature column names."""
        return self.data.get('categorical_features', [])

    @property
    def target_col(self) -> str:
        """Target column name."""
        return self.data.get('target_col', 'pathogenicity_label')
    
    @property
    def random_seed(self) -> int:
        """Random seed for reproducibility."""
        return self.data.get('random_seed', 42)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get('training.learning_rate', 0.001)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"
