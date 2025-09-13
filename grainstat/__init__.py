
"""
GrainStat: Professional grain size analysis for materials science
"""

__version__ = "1.0.0"
__author__ = "Pranav Kokati"

# Import main analyzer class
from .main import GrainAnalyzer

# Import core modules
from . import core
from . import export
from . import plugins
from . import processing
from . import visualization

# Import plugin decorator
from .plugins.base import feature

__all__ = [
    'GrainAnalyzer', 
    'feature',
    'core',
    'export', 
    'plugins',
    'processing',
    'visualization'
]