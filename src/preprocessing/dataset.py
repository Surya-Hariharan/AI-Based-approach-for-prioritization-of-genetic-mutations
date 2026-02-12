import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, List

class MutationDataset(Dataset):
    """
    PyTorch Dataset for Genetic Mutation Prioritization.
    """
    def __init__(self, features, targets=None):
        """
        Args:
            features: Processed feature matrix (pd.DataFrame or np.ndarray).
            targets: Target labels (pd.Series or np.ndarray, optional).
        """
        # Handle both pandas and numpy inputs
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.Series):
            targets = targets.values
            
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets is not None:
            # Keep targets as 1D - DataLoader will handle batching
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            # Return features and target - DataLoader will batch them
            return self.features[idx], self.targets[idx]
        return self.features[idx]
