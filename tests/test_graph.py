import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.graph.construct import GraphBuilder
from src.models.gnn import VariantGNN

def test_graph_module():
    print("Testing Graph Module...")
    
    # 1. Test Graph Construction
    # Dummy DF: 5 variants, 2 genes.
    # V0, V1 -> G0
    # V2, V3 -> G1
    # V4 -> G0
    
    data = {
        'variant_id': ['v0', 'v1', 'v2', 'v3', 'v4'],
        'gene_id': ['g0', 'g0', 'g1', 'g1', 'g0'],
        'feat1': np.random.rand(5)
    }
    df = pd.DataFrame(data)
    
    builder = GraphBuilder(gene_col='gene_id')
    adj = builder.build_adjacency(df)
    
    print(f"Adjacency Matrix Shape: {adj.shape}")
    assert adj.shape == (5, 5)
    
    # Check connections
    # V0 should be connected to V1 and V4 (same gene G0)
    # Convert sparse to dense for check
    adj_dense = adj.to_dense()
    print("Dense Adj:\n", adj_dense)
    
    # V0 (idx 0) and V1 (idx 1) should have non-zero entry
    assert adj_dense[0, 1] > 0
    assert adj_dense[0, 4] > 0
    # V0 and V2 (diff genes) should be 0
    assert adj_dense[0, 2] == 0
    
    print("Graph Construction Verified.")
    
    # 2. Test GNN Forward Pass
    input_dim = 16
    hidden_dim = 8
    output_dim = 1 # Logits
    
    model = VariantGNN(input_dim, hidden_dim, output_dim)
    
    # Fake Features (N=5, F=16)
    x = torch.randn(5, input_dim)
    
    # Forward
    out = model(x, adj)
    print(f"GNN Output Shape: {out.shape}")
    
    assert out.shape == (5, 1)
    
    print("GNN Forward Pass Verified.")
    print("Graph Module Test Passed!")

if __name__ == "__main__":
    test_graph_module()
