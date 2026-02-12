import numpy as np
from scipy.stats import beta
from typing import Tuple, Dict, Any, Union

class BayesianRanker:
    """
    Ranks mutations based on Bayesian credible intervals.
    Treats predicted probability as a success/failure outcome or direct likelihood proxy.
    Here, we assume the model output p is the underlying parameter of a Bernoulli distribution.
    But since we only have one observation (the prediction p), strictly Bayesian update is tricky without more data.
    
    Alternative interpretation:
    The model output p is the mean of a Beta distribution Beta(alpha, beta).
    We can construct a Beta distribution such that mean = p and variance is derived from MC Dropout or assumed.
    
    If we have uncertainty (variance) from MC Dropout:
    Mean mu = p
    Var sigma^2 = variance
    
    We can solve for alpha and beta:
    mu = alpha / (alpha + beta)
    sigma^2 = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    
    If no variance provided, we can assume a prior (e.g., uniform Beta(1,1)) and treat the prediction p as pseudo-count evidence.
    E.g. p ~ k/n. If we assume n (confidence strength), we can get k = p*n.
    Then posterior is Beta(1+k, 1+(n-k)).
    
    Let's use the second approach (pseudo-counts) or standard Beta-Binomial if appropriate.
    However, using MC Dropout variance is more robust.
    """
    
    def __init__(self, confidence_strength: float = 10.0):
        """
        Args:
            confidence_strength: Only used if variance is NOT provided. Represents effective sample size 'n'.
        """
        self.confidence_strength = confidence_strength

    def compute_posterior(self, mean: np.ndarray, variance: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes posterior stats (Mean, Lower Bound, Upper Bound).
        Args:
            mean: Predicted probabilities (0 to 1).
            variance: Predictive variance (optional).
        
        Returns:
            posterior_mean, lower_bound (2.5%), upper_bound (97.5%)
        """
        epsilon = 1e-6
        mean = np.clip(mean, epsilon, 1-epsilon)
        
        if variance is not None:
            # Method of Moments to find alpha, beta
            # sigma^2 < mu(1-mu) must hold
            max_var = mean * (1 - mean)
            variance = np.clip(variance, epsilon, max_var - epsilon)
            
            # Common reparameterization:
            # nu = alpha + beta = mu(1-mu)/sigma^2 - 1
            nu = (mean * (1 - mean) / variance) - 1
            nu = np.maximum(nu, epsilon) # Avoid negative nu
            
            alpha = mean * nu
            beta_param = (1 - mean) * nu
        else:
            # Heuristic: treat mean as observing n * mean successes out of n trials
            n = self.confidence_strength
            alpha = 1 + n * mean
            beta_param = 1 + n * (1 - mean)
            
        # Compute stats from Beta distribution
        post_mean = alpha / (alpha + beta_param)
        lower = beta.ppf(0.025, alpha, beta_param)
        upper = beta.ppf(0.975, alpha, beta_param)
        
        return post_mean, lower, upper
        
    def rank(self, mean: np.ndarray, variance: np.ndarray = None) -> Dict[str, Any]:
        """
        Returns ranking scores.
        Primary ranking by Lower Bound (pessimistic / risk-averse).
        """
        post_mean, lower, upper = self.compute_posterior(mean, variance)
        
        # Rank by Lower Bound (descending)
        # Higher lower bound = consistently high probability of pathogenicity
        ranks = np.argsort(lower)[::-1] + 1
        
        return {
            "posterior_mean": post_mean,
            "posterior_lower": lower,
            "posterior_upper": upper,
            "rank_by_lower": ranks
        }
