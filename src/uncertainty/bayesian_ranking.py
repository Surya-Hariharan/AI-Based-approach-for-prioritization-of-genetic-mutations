import numpy as np
from scipy.stats import beta
from typing import Tuple, Dict, Any, Union

class BayesianRanker:
    """
    Ranks mutations based on Bayesian credible intervals.
    """
    
    def __init__(self, confidence_strength: float = 10.0):
        self.confidence_strength = confidence_strength

    def compute_posterior(self, mean: np.ndarray, variance: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        epsilon = 1e-6
        mean = np.clip(mean, epsilon, 1-epsilon)
        
        if variance is not None:
            max_var = mean * (1 - mean)
            variance = np.clip(variance, epsilon, max_var - epsilon)
            
            nu = (mean * (1 - mean) / variance) - 1
            nu = np.maximum(nu, epsilon)
            
            alpha = mean * nu
            beta_param = (1 - mean) * nu
        else:
            n = self.confidence_strength
            alpha = 1 + n * mean
            beta_param = 1 + n * (1 - mean)
            
        post_mean = alpha / (alpha + beta_param)
        lower = beta.ppf(0.025, alpha, beta_param)
        upper = beta.ppf(0.975, alpha, beta_param)
        
        return post_mean, lower, upper
        
    def rank(self, mean: np.ndarray, variance: np.ndarray = None) -> Dict[str, Any]:
        """
        Returns ranking scores.
        'ranked_indices': Indices of items sorted by Lower Bound (descending). Use this to sort your DataFrame.
        'ranks': The rank of each item (1-based). ranks[i] is the rank of item i.
        """
        post_mean, lower, upper = self.compute_posterior(mean, variance)
        
        # Sort indices by Lower Bound (descending)
        ranked_indices = np.argsort(lower)[::-1]
        
        # Calculate rank for each item (1 = best)
        ranks = np.empty_like(ranked_indices)
        ranks[ranked_indices] = np.arange(len(lower)) + 1
        
        return {
            "posterior_mean": post_mean,
            "posterior_lower": lower,
            "posterior_upper": upper,
            "ranked_indices": ranked_indices,
            "ranks": ranks
        }
