"""Input validation utilities for ODE integration."""

import torch
from torch import Tensor


class IntegrationError(Exception):
    """Raised when ODE integration fails."""
    
    def __init__(self, message: str, t_current: float = None, state: Tensor = None):
        """Initialize integration error with diagnostic information.
        
        Args:
            message: Error description
            t_current: Time at which integration failed (optional)
            state: State at failure point (optional)
        """
        self.t_current = t_current
        self.state = state
        
        error_msg = message
        if t_current is not None:
            error_msg += f"\nFailed at t={t_current:.6f}"
        if state is not None:
            error_msg += f"\nState norm: {torch.norm(state).item():.6f}"
            error_msg += f"\nState contains NaN: {torch.isnan(state).any().item()}"
        
        super().__init__(error_msg)


class MaxStepsExceeded(IntegrationError):
    """Raised when adaptive solver exceeds maximum steps."""
    pass


class StepSizeTooSmall(IntegrationError):
    """Raised when adaptive solver step size becomes too small."""
    pass


def check_finite(tensor: Tensor, name: str) -> None:
    """Validate tensor contains finite values.
    
    Args:
        tensor: Tensor to validate
        name: Name of tensor for error message
        
    Raises:
        ValueError: If tensor contains NaN or Inf values
    """
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains NaN or Inf values. "
            f"This often indicates numerical instability. "
            f"Try: (1) reducing learning rate, "
            f"(2) tightening solver tolerances, "
            f"(3) gradient clipping, or "
            f"(4) checking dynamics function for instabilities."
        )


def validate_ode_inputs(y0: Tensor, t: Tensor) -> None:
    """Validate inputs to ODE solver.
    
    Args:
        y0: Initial state tensor
        t: Time points tensor
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Check dimensions
    if y0.dim() != 2:
        raise ValueError(
            f"Initial state must be 2D (batch_size, state_dim), "
            f"got shape {y0.shape}"
        )
    
    if t.dim() != 1:
        raise ValueError(
            f"Time must be 1D tensor, got shape {t.shape}"
        )
    
    # Check minimum time points
    if len(t) < 2:
        raise ValueError(
            f"Need at least 2 time points, got {len(t)}"
        )
    
    # Check finite values (before monotonicity check)
    check_finite(y0, "Initial state")
    check_finite(t, "Time points")
    
    # Check time ordering
    if not torch.all(t[1:] >= t[:-1]):
        raise ValueError(
            "Time points must be monotonically increasing"
        )


def validate_tolerances(rtol: float, atol: float) -> None:
    """Validate tolerance parameters for adaptive solvers.
    
    Args:
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Raises:
        ValueError: If tolerances are not positive
    """
    if rtol <= 0:
        raise ValueError(f"Relative tolerance must be positive, got {rtol}")
    if atol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {atol}")
