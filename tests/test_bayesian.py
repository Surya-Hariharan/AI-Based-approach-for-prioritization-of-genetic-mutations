import sys
import os
import numpy as np

# Set env var for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.ranking.bayesian import BayesianRanker

def test_bayesian_ranker():
    print("Testing BayesianRanker...")
    
    # Case 1: No variance (Heuristic)
    ranker = BayesianRanker(confidence_strength=10)
    means = np.array([0.9, 0.6, 0.9])
    # Case: item 0 and 2 have same mean.
    
    post_mean, lower, upper = ranker.compute_posterior(means)
    print(f"Heuristic Post Mean: {post_mean}")
    print(f"Heuristic Lower: {lower}")
    
    assert (lower[0] > lower[1])
    
    # Case 2: With Variance (Accurate)
    # Item 0: Mean 0.9, Low Var
    # Item 1: Mean 0.9, High Var (Should have lower "Lower Bound")
    
    means = np.array([0.9, 0.9])
    variances = np.array([0.01, 0.05]) 
    
    # Check bounds
    # For item 1, high variance -> smaller alpha+beta (less confidence) -> wider interval -> lower LB
    
    _, lower_v, _ = ranker.compute_posterior(means, variances)
    print(f"Variance-aware Lower Bounds: {lower_v}")
    
    assert lower_v[0] > lower_v[1], "Higher variance should penalize lower bound"
    
    # ranks[i] is the rank of item i.
    # ranked_indices[0] is the index of the #1 item.
    
    result = ranker.rank(means, variances)
    ranks = result['ranks']
    ranked_indices = result['ranked_indices']
    
    print(f"Ranks (per item): {ranks}")
    print(f"Ranked Indices (sorted order): {ranked_indices}")
    
    # We expect Item 0 (LB higher) to be Rank 1
    # We expect Item 1 (LB lower) to be Rank 2
    
    assert ranks[0] == 1, f"Item 0 should be Rank 1, got {ranks[0]}"
    assert ranks[1] == 2, f"Item 1 should be Rank 2, got {ranks[1]}"
    
    assert ranked_indices[0] == 0, "First item in sorted list should be index 0"
    
    print("Bayesian Ranking Test Passed!")

if __name__ == "__main__":
    test_bayesian_ranker()
