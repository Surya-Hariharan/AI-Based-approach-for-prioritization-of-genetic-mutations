import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) for genetic mutation prioritization.
    """
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.5):
        super(MLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
