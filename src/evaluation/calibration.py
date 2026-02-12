import numpy as np
import pandas as pd
from typing import List, Set, Dict

class BiologicalEvaluator:
    """
    Evaluates rankings against known biological ground truth.
    """
    def __init__(self, pathogenic_variants: Set[str]):
        """
        Args:
            pathogenic_variants: Set of variant IDs known to be pathogenic.
        """
        self.pathogenic_variants = pathogenic_variants

    def enrichment_analysis(self, ranked_df: pd.DataFrame, top_k: int = 100) -> float:
        """
        Calculates enrichment factor at top K.
        Enrichment = (Observed Fraction) / (Expected Fraction at Random)
        """
        top_variants = set(ranked_df.iloc[:top_k]['variant_id'].values)
        total_variants = len(ranked_df)
        total_pathogenic = len(self.pathogenic_variants.intersection(set(ranked_df['variant_id'].values)))
        
        if total_pathogenic == 0:
            return 0.0
            
        observed_pathogenic = len(top_variants.intersection(self.pathogenic_variants))
        observed_fraction = observed_pathogenic / top_k
        expected_fraction = total_pathogenic / total_variants
        
        return observed_fraction / expected_fraction if expected_fraction > 0 else 0.0

    def permutation_test(self, scores: np.ndarray, labels: np.ndarray, n_permutations: int = 1000) -> float:
        """
        Performs permutation test to assess significance of the AUC.
        Returns p-value.
        """
        from sklearn.metrics import roc_auc_score
        
        observed_auc = roc_auc_score(labels, scores)
        permuted_aucs = []
        
        for _ in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            permuted_aucs.append(roc_auc_score(permuted_labels, scores))
            
        permuted_aucs = np.array(permuted_aucs)
        p_value = np.mean(permuted_aucs >= observed_auc)
        
        return p_value
