"""Adjoint sensitivity method for memory-efficient backpropagation."""

from .params import flatten_params, unflatten_params
from .adjoint import (
    create_augmented_dynamics,
    AugmentedDynamics,
    adjoint_integrate,
)

__all__ = [
    'flatten_params',
    'unflatten_params',
    'create_augmented_dynamics',
    'AugmentedDynamics',
    'adjoint_integrate',
]
