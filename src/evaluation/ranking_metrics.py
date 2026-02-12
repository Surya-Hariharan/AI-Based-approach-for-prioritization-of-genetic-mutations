import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

class Ranker:
    """
    Ranks mutations based on predicted pathogenicity scores.
    """
    def __init__(self, threshold: float = 0.5, output_dir: str = "reports/results"):
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def rank_mutations(
        self, 
        probs: np.ndarray, 
        features: pd.DataFrame, 
        variant_ids: Optional[List[str]] = None,
        target_labels: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Ranks mutations and filters based on threshold.

        Args:
            probs: Predicted probabilities of pathogenicity.
            features: DataFrame of input features (for context).
            variant_ids: List of variant identifiers (optional).
            target_labels: True labels (optional, for validation).

        Returns:
            DataFrame containing ranked mutations with scores and features.
        """
        n_samples = len(probs)
        
        # Create base DataFrame
        df = features.copy()
        df['score'] = probs
        
        if variant_ids is not None:
            df['variant_id'] = variant_ids
        else:
            # Generate dummy IDs if not provided
            df['variant_id'] = [f"var_{i}" for i in range(n_samples)]
            
        if target_labels is not None:
            df['true_label'] = target_labels

        # Sort by score descending
        df = df.sort_values(by='score', ascending=False)
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['rank', 'variant_id', 'score']
        if target_labels is not None:
            cols.append('true_label')
        
        # Append feature columns
        feature_cols = [c for c in features.columns if c not in cols]
        cols.extend(feature_cols)
        
        df = df[cols]
        return df

    def save_ranked_list(self, ranked_df: pd.DataFrame, filename: str = "ranked_mutations.csv"):
        """Saves values to CSV."""
        path = os.path.join(self.output_dir, filename)
        ranked_df.to_csv(path, index=False)
        print(f"Ranked list saved to {path}")
