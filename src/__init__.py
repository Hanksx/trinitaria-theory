"""
Trinitária Theory - Multi-Scale Scalar Field Dynamics
====================================================

A comprehensive alternative to dark matter using three interacting scalar fields.

Modules:
    core: Core theory implementation and model classes
    validation: Multi-scale validation scripts  
    optimization: Parameter optimization algorithms
    visualization: Plotting and analysis tools

Example:
    >>> from src.core.trinitaria_theory import TrinitariaModel
    >>> model = TrinitariaModel(scale='galaxy')
    >>> import numpy as np
    >>> r = np.linspace(0.1, 30, 100)
    >>> v = model.velocity_profile(r)
    >>> print(f"Flat rotation curve: {np.std(v[50:]) < 10}")
"""

__version__ = "1.0.0"
__author__ = "Research Collaboration"
__license__ = "MIT"

# Core exports
from .core.trinitaria_theory import (
    TrinitariaModel,
    trinitaria_velocity,
    create_galaxy_model,
    create_cluster_model,
    create_cosmic_model
)

__all__ = [
    'TrinitariaModel',
    'trinitaria_velocity', 
    'create_galaxy_model',
    'create_cluster_model',
    'create_cosmic_model'
]