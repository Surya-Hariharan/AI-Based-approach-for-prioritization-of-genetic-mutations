"""
Reproducibility utilities for ensuring deterministic behavior across experiments.

This module provides functions to set random seeds for all major libraries
used in the project (PyTorch, NumPy, Python random) and configure PyTorch
for deterministic operations.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enables deterministic operations in PyTorch
                      (may impact performance)
    
    Example:
        >>> from src.utils.seed import set_seed
        >>> set_seed(42)
        >>> # All random operations are now reproducible
    
    Note:
        Deterministic mode may reduce performance but ensures exact reproducibility.
        Set deterministic=False for faster training if exact reproduction is not critical.
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for even more deterministic behavior
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms (PyTorch >= 1.8)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # Allow non-deterministic for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> dict:
    """
    Get current random number generator states for all libraries.
    
    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch
    
    Example:
        >>> state = get_rng_state()
        >>> # Do some random operations
        >>> restore_rng_state(state)  # Restore to saved state
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng_state(state: dict) -> None:
    """
    Restore random number generator states from a saved state dictionary.
    
    Args:
        state: Dictionary containing RNG states (from get_rng_state())
    
    Example:
        >>> state = get_rng_state()
        >>> # Do some random operations
        >>> restore_rng_state(state)  # Restore to saved state
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


class SeedContext:
    """
    Context manager for temporary seed setting.
    
    Example:
        >>> with SeedContext(123):
        ...     # All operations here use seed 123
        ...     x = torch.rand(10)
        >>> # Original seed state is restored
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.saved_state = None
    
    def __enter__(self):
        self.saved_state = get_rng_state()
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_state is not None:
            restore_rng_state(self.saved_state)
        return False
