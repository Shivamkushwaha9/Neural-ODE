"""
Trace estimation utilities for Continuous Normalizing Flows.

This module provides functions for computing the trace of the Jacobian matrix,
which is required for tracking the instantaneous change of variables in CNF.
"""

import torch
from torch import Tensor
from typing import Optional


def hutchinson_trace(
    dy_dt: Tensor,
    y: Tensor,
    num_samples: int = 1,
    noise: Optional[Tensor] = None
) -> Tensor:
    """
    Estimate trace using Hutchinson's stochastic estimator.
    
    The Hutchinson estimator approximates the trace of the Jacobian using:
    tr(∂f/∂y) ≈ E[ε^T (∂f/∂y) ε] where ε ~ N(0, I)
    
    This is more efficient than exact trace computation for high-dimensional
    spaces, requiring only O(num_samples) vector-Jacobian products instead
    of O(dim) backward passes.
    
    Args:
        dy_dt: Function output, shape (batch, dim)
        y: Function input, shape (batch, dim). Must have requires_grad=True.
        num_samples: Number of random vectors for estimation. Higher values
                    give more accurate estimates but increase computation.
                    Default: 1
        noise: Optional pre-generated noise tensor, shape (num_samples, batch, dim).
              If None, will sample from N(0, I). Useful for reproducibility.
    
    Returns:
        Trace estimate, shape (batch,)
    
    Example:
        >>> y = torch.randn(32, 64, requires_grad=True)
        >>> dy_dt = model(y)
        >>> trace = hutchinson_trace(dy_dt, y, num_samples=1)
        >>> trace.shape
        torch.Size([32])
    
    References:
        Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
        influence matrix for laplacian smoothing splines.
    """
    if not y.requires_grad:
        raise ValueError(
            "Input tensor y must have requires_grad=True for trace computation"
        )
    
    batch_size, dim = y.shape
    trace_estimate = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
    
    for i in range(num_samples):
        # Sample random vector from N(0, I)
        if noise is not None:
            epsilon = noise[i]
        else:
            epsilon = torch.randn_like(y)
        
        # Compute vector-Jacobian product: ε^T (∂f/∂y)
        # This is equivalent to computing (∂f/∂y) ε
        vjp = torch.autograd.grad(
            outputs=dy_dt,
            inputs=y,
            grad_outputs=epsilon,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        # Handle case where gradient is None (no dependency)
        if vjp is not None:
            # Compute ε^T (∂f/∂y) ε = sum(ε * vjp)
            trace_estimate += (epsilon * vjp).sum(dim=1)
        # If vjp is None, the contribution is zero (no dependency)
    
    # Average over samples
    return trace_estimate / num_samples


def exact_trace(dy_dt: Tensor, y: Tensor) -> Tensor:
    """
    Compute exact trace of the Jacobian matrix.
    
    This function computes tr(∂f/∂y) by explicitly computing the diagonal
    elements of the Jacobian. This requires O(dim) backward passes, making
    it more expensive than Hutchinson estimation for high-dimensional spaces,
    but it provides the exact trace without stochastic approximation.
    
    Args:
        dy_dt: Function output, shape (batch, dim)
        y: Function input, shape (batch, dim). Must have requires_grad=True.
    
    Returns:
        Exact trace, shape (batch,)
    
    Example:
        >>> y = torch.randn(32, 64, requires_grad=True)
        >>> dy_dt = model(y)
        >>> trace = exact_trace(dy_dt, y)
        >>> trace.shape
        torch.Size([32])
    
    Note:
        For high-dimensional spaces (dim > 100), consider using hutchinson_trace
        instead for better computational efficiency.
    """
    if not y.requires_grad:
        raise ValueError(
            "Input tensor y must have requires_grad=True for trace computation"
        )
    
    batch_size, dim = y.shape
    trace = torch.zeros(batch_size, device=y.device, dtype=y.dtype)
    
    # Compute each diagonal element ∂f_i/∂y_i
    for i in range(dim):
        # Compute gradient of i-th output w.r.t. all inputs
        grad = torch.autograd.grad(
            outputs=dy_dt[:, i],
            inputs=y,
            grad_outputs=torch.ones_like(dy_dt[:, i]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        # Handle case where gradient is None (no dependency)
        if grad is not None:
            # Add diagonal element to trace
            trace += grad[:, i]
    
    return trace
