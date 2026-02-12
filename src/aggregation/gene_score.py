import pandas as pd
import numpy as np
from typing import Dict, Any, List

class GeneAggregator:
    def __init__(self, method: str = 'bayesian', top_k: int = 3):
        self.method = method
        self.top_k = top_k
        
    def aggregate(self, variant_df: pd.DataFrame, score_col: str, gene_col: str) -> pd.DataFrame:
        """
        Aggregates variant scores to gene level.
        Args:
            variant_df: DataFrame containing variants.
            score_col: Column name for variant scores (probabilities).
            gene_col: Column name for gene IDs.
        Returns:
            DataFrame with [gene_col, 'gene_score', 'n_variants']
        """
        if gene_col not in variant_df.columns:
            raise ValueError(f"Column {gene_col} not found in DataFrame.")
            
        grouped = variant_df.groupby(gene_col)
        
        gene_scores = []
        for gene, group in grouped:
            scores = group[score_col].values
            
            # Simple handling of NaNs
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                final_score = 0.0
            else:
                if self.method == 'mean':
                    final_score = np.mean(scores)
                    
                elif self.method == 'max':
                    final_score = np.max(scores)
                    
                elif self.method == 'top_k':
                    # Average of top K scores
                    k = min(len(scores), self.top_k)
                    top_scores = np.sort(scores)[::-1][:k]
                    final_score = np.mean(top_scores)
                    
                elif self.method == 'bayesian':
                    # Noisy OR: Probability that at least one variant is pathogenic
                    # P(Gene) = 1 - product(1 - P(Variant_i))
                    # Clipping to avoid exactly 1.0 or 0.0 issues if needed, but logic holds.
                    # P_neg = product(1 - p)
                    # P_pos = 1 - P_neg
                    
                    # Log-sum-exp trick for stability?
                    # log(P_neg) = sum(log(1-p))
                    # P_pos = 1 - exp(sum(log(1-p)))
                    
                    # scores assumed 0..1
                    neg_probs = 1.0 - scores
                    # Avoid log(0)
                    epsilon = 1e-9
                    neg_probs = np.clip(neg_probs, epsilon, 1.0)
                    
                    log_neg = np.sum(np.log(neg_probs))
                    final_score = 1.0 - np.exp(log_neg)
                    
                else:
                    raise ValueError(f"Unknown aggregation method: {self.method}")
                
            gene_scores.append({
                gene_col: gene, 
                'gene_score': final_score,
                'n_variants': len(scores)
            })
            
        result_df = pd.DataFrame(gene_scores)
        return result_df.sort_values('gene_score', ascending=False).reset_index(drop=True)
