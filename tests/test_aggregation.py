import sys
import os
import pandas as pd
import numpy as np

# Set env var for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.aggregation.gene_score import GeneAggregator

def test_aggregation():
    print("Testing Gene Aggregation...")
    
    data = {
        'gene_id': ['g1', 'g1', 'g2', 'g2', 'g3'],
        'score': [0.1, 0.9, 0.5, 0.5, 0.2]
    } # g1: one very high. g2: two medium. g3: low.
    
    df = pd.DataFrame(data)
    
    # 1. Max
    agg = GeneAggregator(method='max')
    res = agg.aggregate(df, 'score', 'gene_id')
    print("Max Aggregation:\n", res)
    # g1 -> 0.9, g2 -> 0.5, g3 -> 0.2
    assert res.iloc[0]['gene_id'] == 'g1' and res.iloc[0]['gene_score'] == 0.9
    
    # 2. Mean
    agg = GeneAggregator(method='mean')
    res = agg.aggregate(df, 'score', 'gene_id')
    print("Mean Aggregation:\n", res)
    # g1 -> 0.5, g2 -> 0.5
    
    # 3. Bayesian (Noisy OR)
    # g1: 1 - (0.9 * 0.1) = 1 - 0.09 = 0.91
    # g2: 1 - (0.5 * 0.5) = 1 - 0.25 = 0.75
    agg = GeneAggregator(method='bayesian')
    res = agg.aggregate(df, 'score', 'gene_id')
    print("Bayesian Aggregation:\n", res)
    
    assert res.iloc[0]['gene_id'] == 'g1' 
    # Check value roughly
    score_g1 = res[res['gene_id'] == 'g1']['gene_score'].values[0]
    expected_g1 = 1 - ( (1-0.1)*(1-0.9) ) # 1 - (0.9 * 0.1) = 0.91
    assert abs(score_g1 - expected_g1) < 1e-5
    
    print("Aggregation Test Passed!")

if __name__ == "__main__":
    test_aggregation()
