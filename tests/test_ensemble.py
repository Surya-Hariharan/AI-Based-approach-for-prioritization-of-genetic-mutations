import sys
import os
import joblib
# Set env var for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import pandas as pd
from src.ensemble.stacking import StackingEnsemble
from src.config.loader import Config

def test_stacking_ensemble():
    print("Testing StackingEnsemble...")
    
    # Mock Config
    config_dict = {
        'ensemble': {
            'enabled': True,
            'n_folds': 3,
            'base_models': [
                {'type': 'mlp', 'params': {'hidden_layers': [16, 8], 'dropout': 0.1}},
                {'type': 'xgboost', 'params': {'n_estimators': 10, 'max_depth': 2, 'learning_rate': 0.1}}
            ]
        }
    }
    
    # Generate Dummy Data
    N = 100
    F = 20
    X = np.random.rand(N, F).astype(np.float32)
    y = np.random.randint(0, 2, N).astype(np.float32) # Stacking expects float for some reason or int? Base models handle it.
    
    # Init Ensemble
    ensemble = StackingEnsemble(config_dict['ensemble'], input_dim=F)
    
    # Fit
    print("Fitting...")
    ensemble.fit(X, y)
    
    # Predict
    print("Predicting...")
    probs = ensemble.predict_proba(X)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probs: {probs[:5]}")
    
    assert probs.shape == (N,)
    assert (probs >= 0).all() and (probs <= 1).all()
    
    # Save/Load
    print("Saving...")
    os.makedirs("tests/models", exist_ok=True)
    ensemble.save("tests/models/ensemble.joblib")
    
    print("Loading...")
    loaded_ensemble = StackingEnsemble.load("tests/models/ensemble.joblib")
    
    # Verify loaded prediction
    probs_loaded = loaded_ensemble.predict_proba(X)
    np.testing.assert_allclose(probs, probs_loaded, atol=1e-5)
    print("Save/Load verification passed.")
    
    print("Ensemble Test Passed!")

if __name__ == "__main__":
    test_stacking_ensemble()
