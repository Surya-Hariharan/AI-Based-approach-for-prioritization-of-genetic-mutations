import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Optional, List

class MutationDataset(Dataset):
    """
    PyTorch Dataset for Genetic Mutation Prioritization.
    """
    def __init__(self, features: pd.DataFrame, targets: Optional[pd.Series] = None):
        """
        Args:
            features (pd.DataFrame): Processed feature matrix.
            targets (pd.Series, optional): Target labels.
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]
