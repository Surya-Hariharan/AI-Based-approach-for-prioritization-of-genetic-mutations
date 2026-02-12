import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.uncertainty.mc_dropout import MCDropoutEstimator

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Dropout(0.5), # Crucial for MC Dropout
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def test_mc_dropout():
    print("Testing MCDropoutEstimator...")
    
    # Dummy Data
    N = 10
    F = 5
    X = torch.randn(N, F)
    
    model = SimpleMLP(F)
    estimator = MCDropoutEstimator(model, n_samples=50)
    
    results = estimator.predict(X)
    
    mean = results['mean']
    variance = results['variance']
    
    print(f"Mean shape: {mean.shape}")
    print(f"Variance shape: {variance.shape}")
    print(f"Sample Mean: {mean[:3]}")
    print(f"Sample Variance: {variance[:3]}")
    
    # Assert correctness
    assert mean.shape == (N,)
    assert variance.shape == (N,)
    assert (variance > 0).any(), "Variance should be positive with dropout enabled"
    
    print("MCDropout Test Passed!")

if __name__ == "__main__":
    test_mc_dropout()
