"""AI-Based Genetic Mutation Prioritization

A comprehensive machine learning platform for genetic mutation classification,
uncertainty quantification, and gene-level prioritization.
"""

__version__ = '1.0.0'
__author__ = 'Surya Hariharan'

# Core modules
from . import models
from . import preprocessing
from . import evaluation
from . import uncertainty
from . import ensemble
from . import aggregation
from . import utils

# Main exports
__all__ = [
    'models',
    'preprocessing', 
    'evaluation',
    'uncertainty',
    'ensemble',
    'aggregation',
    'utils'
]