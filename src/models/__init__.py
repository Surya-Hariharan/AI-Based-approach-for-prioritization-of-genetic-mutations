"""Machine Learning Models for Genetic Mutation Classification"""

from .baseline import LogisticRegression
from .mlp import MLP

# Optional imports
try:
    from .gnn import VariantGNN
    __all__ = ['LogisticRegression', 'MLP', 'VariantGNN']
except ImportError:
    __all__ = ['LogisticRegression', 'MLP']
