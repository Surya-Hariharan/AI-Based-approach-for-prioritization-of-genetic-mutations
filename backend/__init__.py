# Flask Backend for Genetic Mutation Prioritization
# Production-ready REST API for mutation pathogenicity prediction

__version__ = '1.0.0'
__author__ = 'Surya Hariharan'
__email__ = 'suryahariharan2006@gmail.com'
__github__ = 'https://github.com/Surya-Hariharan'
__linkedin__ = 'https://linkedin.com/in/surya-ha'

from .app import create_app

__all__ = ['create_app']