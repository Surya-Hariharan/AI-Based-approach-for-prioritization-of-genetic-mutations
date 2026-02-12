from src.models.baseline import LogisticRegression
from src.models.mlp import MLP
try:
    from src.models.gnn import VariantGNN
except ImportError:
    pass
