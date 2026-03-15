"""Utility functions for visualization, benchmarking, and validation."""

from .validation import (
    check_finite,
    validate_ode_inputs,
    validate_tolerances,
    IntegrationError,
    MaxStepsExceeded,
    StepSizeTooSmall,
)
from .trace import (
    hutchinson_trace,
    exact_trace,
)

__all__ = [
    'check_finite',
    'validate_ode_inputs',
    'validate_tolerances',
    'IntegrationError',
    'MaxStepsExceeded',
    'StepSizeTooSmall',
    'hutchinson_trace',
    'exact_trace',
]
