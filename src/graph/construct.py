import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix

class GraphBuilder:
    def __init__(self, variant_col='variant_id', gene_col='gene_id'):
        self.variant_col = variant_col
        self.gene_col = gene_col
        
    def build_adjacency(self, df: pd.DataFrame):
        """
        Constructs a Variant-Variant adjacency matrix based on shared genes.
        Two variants are connected if they share the same gene.
        Warning: This is dense if a gene has many variants (clique).
        
        Alternative: Node set = Variants + Genes.
        Edges: Variant <-> Gene.
        
        Let's do the Variant-Variant Graph via Gene Clique expansion for simplicity in GCN,
        OR bipartite.
        
        For the GNN model implemented (N x N adj), we need variant-variant connections.
        
        Output:
            adj: torch.sparse_coo_tensor (N, N) normalized.
        """
        N = len(df)
        if self.gene_col not in df.columns:
            print(f"Warning: {self.gene_col} not found. Returning Identity adjacency.")
            return self._identity_adj(N)
        
        # Map genes to indices
        genes = df[self.gene_col].unique()
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        
        # Build incidence matrix H (Variants x Genes)
        # Rows: Variants, Cols: Genes
        row_indices = np.arange(N)
        col_indices = df[self.gene_col].map(gene_to_idx).values
        
        # Filter NaNs
        mask = ~pd.isna(col_indices)
        row_indices = row_indices[mask]
        col_indices = col_indices[mask].astype(int)
        
        data = np.ones(len(row_indices))
        
        H = coo_matrix((data, (row_indices, col_indices)), shape=(N, len(genes)))
        
        # A = H * H.T (Common Gene)
        # H is V x G.
        # A is V x V. A[i, j] = number of shared genes (0 or 1 usually).
        # This operation can be expensive for large N.
        
        # For prototype, we use sparse multiplication.
        A = (H @ H.T).tocoo()
        
        # A contains self-loops (diagonal).
        # We need to normalize: D^-0.5 A D^-0.5
        
        # Convert to torch sparse
        values = A.data
        indices = np.vstack((A.row, A.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = A.shape
        
        adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        # Normalize
        # Compute Degree
        # Degree of node i = sum of row i
        # Since A is symmetric, row sum = col sum
        rowsum = np.array(A.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        
        # D^-0.5 A D^-0.5
        # In sparse land: multiplying diagonal matrices.
        # Equivalent to scaling value[k] by D[row[k]]^-0.5 * D[col[k]]^-0.5
        
        val_scaled = v * torch.from_numpy(r_inv_sqrt[indices[0]]).float() * torch.from_numpy(r_inv_sqrt[indices[1]]).float()
        
        adj_normalized = torch.sparse.FloatTensor(i, val_scaled, torch.Size(shape))
        
        return adj_normalized

    def _identity_adj(self, N):
        i = torch.arange(N)
        index = torch.stack([i, i])
        value = torch.ones(N)
        return torch.sparse.FloatTensor(index, value, torch.Size((N, N)))
