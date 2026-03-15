"""Adjoint sensitivity method for memory-efficient backpropagation through ODEs.

This module implements the adjoint sensitivity method from Chen et al. (2018),
which enables O(1) memory cost for backpropagation through ODE solutions,
regardless of the number of solver steps.

The key idea is to solve an augmented ODE system backward in time to compute
gradients, rather than storing all intermediate states during the forward pass.
"""

import torch
from torch import Tensor
from typing import Tuple, Callable

from ..solvers.base import ODESolver
from .params import flatten_params


def create_augmented_dynamics(func: Callable,
                              flat_params: Tensor,
                              t: float,
                              y: Tensor,
                              adj_y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Create augmented dynamics for adjoint computation.
    
    This function computes the dynamics of the augmented ODE system:
    1. State dynamics: dy/dt = f(y, t, θ)
    2. Adjoint dynamics: da/dt = -a^T ∂f/∂y
    3. Parameter gradient dynamics: d(∂L/∂θ)/dt = a^T ∂f/∂θ
    
    The adjoint method solves these equations backward in time to compute
    gradients without storing intermediate forward states.
    
    Args:
        func: Dynamics function f(t, y) that computes dy/dt
        flat_params: Flattened parameter tensor, shape (num_params,)
        t: Current time (scalar)
        y: Current state, shape (batch_size, state_dim)
        adj_y: Current adjoint state, shape (batch_size, state_dim)
        
    Returns:
        Tuple of (dy_dt, dadj_dt, dparam_dt) where:
            dy_dt: State derivative, shape (batch_size, state_dim)
            dadj_dt: Adjoint derivative, shape (batch_size, state_dim)
            dparam_dt: Parameter gradient derivative, shape (num_params,)
    
    Mathematical Background:
        The adjoint state a(t) satisfies: da/dt = -a^T ∂f/∂y
        The parameter gradients satisfy: dL/dθ = ∫ a^T ∂f/∂θ dt
        
        We compute these using vector-Jacobian products (VJPs):
        - VJP for state: a^T ∂f/∂y (adjoint dynamics)
        - VJP for parameters: a^T ∂f/∂θ (parameter gradients)
    """
    # Ensure y requires gradients for autograd
    y = y.detach().requires_grad_(True)
    
    # Compute forward dynamics: dy/dt = f(y, t, θ)
    with torch.enable_grad():
        dy_dt = func(t, y)
        
        # Compute adjoint dynamics: da/dt = -a^T ∂f/∂y
        # This is a vector-Jacobian product (VJP)
        vjp_y = torch.autograd.grad(
            outputs=dy_dt,
            inputs=y,
            grad_outputs=adj_y,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # Handle case where gradient is None (no dependency)
        if vjp_y is None:
            vjp_y = torch.zeros_like(y)
        
        # Adjoint equation: da/dt = -a^T ∂f/∂y
        dadj_dt = -vjp_y
        
        # Compute parameter gradients: dL/dθ = a^T ∂f/∂θ
        # Get parameters from the function
        params = list(func.parameters())
        
        if len(params) > 0:
            # Compute VJP for parameters
            vjp_params = torch.autograd.grad(
                outputs=dy_dt,
                inputs=params,
                grad_outputs=adj_y,
                retain_graph=True,
                allow_unused=True
            )
            
            # Flatten parameter gradients
            dparam_dt_list = []
            for grad in vjp_params:
                if grad is not None:
                    dparam_dt_list.append(grad.reshape(-1))
                else:
                    # If gradient is None, use zeros
                    # This shouldn't happen for parameters that affect the output
                    dparam_dt_list.append(torch.zeros_like(flat_params[:0]))
            
            if len(dparam_dt_list) > 0:
                dparam_dt = torch.cat(dparam_dt_list)
            else:
                dparam_dt = torch.zeros_like(flat_params)
        else:
            # No parameters to compute gradients for
            dparam_dt = torch.zeros_like(flat_params)
    
    return dy_dt, dadj_dt, dparam_dt


class AugmentedDynamics:
    """Wrapper for augmented ODE dynamics used in adjoint computation.
    
    This class encapsulates the augmented system that combines:
    - Original state dynamics
    - Adjoint state dynamics
    - Parameter gradient accumulation
    
    The augmented state is a tuple: (y, adj_y, adj_params)
    
    During backward integration, we solve:
        dy/dt = -f(y, t, θ)  (negative because we integrate backward)
        da/dt = -a^T ∂f/∂y
        d(∂L/∂θ)/dt = a^T ∂f/∂θ
    
    Args:
        func: Original dynamics function f(t, y)
        flat_params: Flattened parameters
    """
    
    def __init__(self, func: Callable, flat_params: Tensor):
        self.func = func
        self.flat_params = flat_params
        
    def __call__(self, t: float, augmented_state: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute derivatives for augmented system.
        
        Args:
            t: Current time
            augmented_state: Tuple of (y, adj_y, adj_params)
            
        Returns:
            Tuple of (dy_dt, dadj_dt, dparam_dt)
        """
        y, adj_y, adj_params = augmented_state
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            self.func, self.flat_params, t, y, adj_y
        )
        
        # For backward integration, negate the state dynamics
        # (we're integrating from t1 to t0, so time flows backward)
        dy_dt = -dy_dt
        
        return dy_dt, dadj_dt, dparam_dt


class AdjointODE(torch.autograd.Function):
    """Custom autograd function implementing the adjoint sensitivity method.
    
    This class provides memory-efficient backpropagation through ODE solutions
    by solving an augmented ODE system backward in time, rather than storing
    all intermediate states from the forward pass.
    
    Forward pass: Integrate y' = f(y, t, θ) from t[0] to t[-1]
    Backward pass: Solve augmented system for gradients w.r.t. y0, θ, and t
    
    Memory cost: O(1) in the number of solver steps (vs O(N) for standard autograd)
    
    Reference:
        Chen et al. (2018), "Neural Ordinary Differential Equations"
        Section 3: "Reverse-mode automatic differentiation of ODE solutions"
    """
    
    @staticmethod
    def forward(ctx, y0: Tensor, t: Tensor, flat_params: Tensor, 
                solver: ODESolver, func: Callable) -> Tensor:
        """Forward pass: integrate ODE from t[0] to t[-1].
        
        Args:
            ctx: Context object for saving tensors for backward pass
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,)
            flat_params: Flattened parameters, shape (num_params,)
            solver: ODE solver instance
            func: Dynamics function f(t, y)
            
        Returns:
            Final state y(t[-1]), shape (batch_size, state_dim)
        """
        # Integrate forward without tracking gradients
        with torch.no_grad():
            y1 = solver.integrate(func, y0, t)
        
        # Save tensors for backward pass
        ctx.save_for_backward(y0, y1, t, flat_params)
        ctx.solver = solver
        ctx.func = func
        
        return y1
    
    @staticmethod
    def backward(ctx, grad_y1: Tensor) -> Tuple[Tensor, None, Tensor, None, None]:
        """Backward pass: solve augmented ODE system for gradients.
        
        Solves the augmented system backward in time:
            dy/dt = -f(y, t, θ)  (negative for backward integration)
            da/dt = -a^T ∂f/∂y   (adjoint equation)
            d(∂L/∂θ)/dt = a^T ∂f/∂θ  (parameter gradients)
            d(∂L/∂t)/dt = a^T f  (time gradients)
        
        Args:
            ctx: Context object with saved tensors
            grad_y1: Gradient of loss w.r.t. final state, shape (batch_size, state_dim)
            
        Returns:
            Tuple of (grad_y0, grad_t, grad_params, None, None) where:
                grad_y0: Gradient w.r.t. initial state
                grad_t: Gradient w.r.t. time points (None for now)
                grad_params: Gradient w.r.t. parameters
                None, None: Placeholders for solver and func (non-differentiable)
        """
        y0, y1, t, flat_params = ctx.saved_tensors
        solver = ctx.solver
        func = ctx.func
        
        # Initialize adjoint state with gradient from loss
        adj_y = grad_y1
        
        # Initialize parameter gradients to zero
        adj_params = torch.zeros_like(flat_params)
        
        # Initialize time gradients to zero
        adj_t = torch.zeros_like(t)
        
        # Define augmented dynamics for backward integration
        def augmented_dynamics(t_val: float, state: Tensor) -> Tensor:
            """Compute derivatives for augmented system.
            
            The augmented state contains:
                - y: current state (batch_size, state_dim)
                - adj_y: adjoint state (batch_size, state_dim)
                - adj_params: accumulated parameter gradients (num_params,) - shared across batch
                - adj_t: accumulated time gradients (1,) - shared across batch
            
            Args:
                t_val: Current time
                state: Augmented state tensor, shape (batch_size, state_dim + state_dim + num_params + 1)
                
            Returns:
                Augmented derivatives, same shape as state
            """
            batch_size = y0.shape[0]
            state_dim = y0.shape[1]
            num_params = flat_params.shape[0]
            
            # Unpack augmented state
            y = state[:, :state_dim]
            adj_y_curr = state[:, state_dim:2*state_dim]
            # Note: adj_params and adj_t are replicated across batch, but we only use first row
            
            # Ensure y requires gradients
            y = y.detach().requires_grad_(True)
            
            # Compute forward dynamics
            with torch.enable_grad():
                dy_dt = func(t_val, y)
                
                # Compute adjoint dynamics: da/dt = -a^T ∂f/∂y
                # When integrating backward in time (with time reversal), this becomes positive
                vjp_y = torch.autograd.grad(
                    outputs=dy_dt,
                    inputs=y,
                    grad_outputs=adj_y_curr,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                
                if vjp_y is None:
                    vjp_y = torch.zeros_like(y)
                
                # For backward integration with time reversal (τ = t1 - t):
                # da/dτ = +a^T ∂f/∂y (sign flips due to chain rule)
                dadj_dt = vjp_y
                
                # Compute parameter gradients: d(∂L/∂θ)/dt = sum over batch of a^T ∂f/∂θ
                params = list(func.parameters())
                if len(params) > 0:
                    # Use batched VJP - sum over batch dimension
                    vjp_params = torch.autograd.grad(
                        outputs=dy_dt,
                        inputs=params,
                        grad_outputs=adj_y_curr,
                        retain_graph=True,
                        allow_unused=True
                    )
                    
                    # Flatten parameter gradients
                    dparam_dt_list = []
                    for grad in vjp_params:
                        if grad is not None:
                            dparam_dt_list.append(grad.reshape(-1))
                    
                    if len(dparam_dt_list) > 0:
                        dparam_dt = torch.cat(dparam_dt_list)
                        # Replicate to batch dimension (all rows are the same)
                        dparam_dt = dparam_dt.unsqueeze(0).expand(batch_size, -1)
                    else:
                        dparam_dt = torch.zeros(batch_size, num_params, device=y.device, dtype=y.dtype)
                else:
                    dparam_dt = torch.zeros(batch_size, num_params, device=y.device, dtype=y.dtype)
                
                # Compute time gradients: d(∂L/∂t)/dt = sum over batch of a^T f
                dt_dt = torch.sum(adj_y_curr * dy_dt)
                # Replicate to batch dimension
                dt_dt = dt_dt.unsqueeze(0).expand(batch_size, 1)
            
            # For backward integration, negate state dynamics
            dy_dt_backward = -dy_dt
            
            # Concatenate all derivatives
            augmented_deriv = torch.cat([
                dy_dt_backward,
                dadj_dt,
                dparam_dt,
                dt_dt
            ], dim=1)
            
            return augmented_deriv
        
        # Prepare augmented initial state at t[-1]
        batch_size = y1.shape[0]
        state_dim = y1.shape[1]
        num_params = flat_params.shape[0]
        
        # Expand adj_params and adj_t to batch dimension
        adj_params_expanded = adj_params.unsqueeze(0).expand(batch_size, -1)
        adj_t_expanded = torch.zeros(batch_size, 1, device=y1.device, dtype=y1.dtype)
        
        augmented_state = torch.cat([
            y1,
            adj_y,
            adj_params_expanded,
            adj_t_expanded
        ], dim=1)
        
        # For backward integration, we need to integrate from t[-1] to t[0]
        # But solvers expect monotonically increasing time, so we'll integrate
        # from 0 to (t[-1] - t[0]) and adjust the time values in the dynamics
        t_span = t[-1] - t[0]
        t_backward = torch.tensor([0.0, t_span], device=t.device, dtype=t.dtype)
        
        # Wrapper to adjust time for backward integration
        def augmented_dynamics_wrapper(t_val: float, state: Tensor) -> Tensor:
            # Convert forward time to backward time: t_backward = t[-1] - t_val
            t_actual = t[-1].item() - t_val
            return augmented_dynamics(t_actual, state)
        
        # Integrate augmented system backward
        with torch.no_grad():
            augmented_final = solver.integrate(augmented_dynamics_wrapper, augmented_state, t_backward)
        
        # Extract gradients from final augmented state
        grad_y0 = augmented_final[:, state_dim:2*state_dim]
        # Parameter and time gradients are replicated across batch, so take first row
        grad_params = augmented_final[0, 2*state_dim:2*state_dim+num_params]
        grad_t_val = augmented_final[0, -1]
        
        # Time gradients: distribute to all time points (simplified)
        # For now, we don't compute per-time-point gradients
        grad_t = None
        
        return grad_y0, grad_t, grad_params, None, None


def adjoint_integrate(solver: ODESolver,
                     func: Callable,
                     y0: Tensor,
                     t: Tensor) -> Tensor:
    """Integrate ODE using adjoint method for memory-efficient backpropagation.
    
    This function wraps the AdjointODE autograd function to provide a simple
    interface for Neural ODE layers. It automatically handles parameter
    flattening and gradient computation.
    
    Args:
        solver: ODE solver to use for integration
        func: Dynamics function f(t, y) that computes dy/dt
        y0: Initial state, shape (batch_size, state_dim)
        t: Time points, shape (num_times,)
        
    Returns:
        Final state y(t[-1]), shape (batch_size, state_dim)
        
    Example:
        >>> solver = Dopri5Solver()
        >>> func = ODEFunc(nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2)))
        >>> y0 = torch.randn(32, 2)
        >>> t = torch.tensor([0.0, 1.0])
        >>> y1 = adjoint_integrate(solver, func, y0, t)
        >>> loss = y1.sum()
        >>> loss.backward()  # Uses adjoint method automatically
    
    Mathematical Background:
        Forward pass: y(t1) = y(t0) + ∫[t0,t1] f(y(t), t, θ) dt
        
        Backward pass (adjoint method):
            a(t1) = ∂L/∂y(t1)  (initial condition for adjoint)
            da/dt = -a^T ∂f/∂y  (adjoint equation)
            ∂L/∂y(t0) = a(t0)  (gradient w.r.t. initial state)
            ∂L/∂θ = ∫[t0,t1] a^T ∂f/∂θ dt  (gradient w.r.t. parameters)
    
    Reference:
        Chen et al. (2018), "Neural Ordinary Differential Equations"
        Section 3: "Reverse-mode automatic differentiation of ODE solutions"
    """
    # Get parameters and flatten them
    params = list(func.parameters())
    if len(params) > 0:
        flat_params, param_info = flatten_params(params)
    else:
        flat_params = torch.tensor([], device=y0.device, dtype=y0.dtype)
    
    # Apply adjoint ODE autograd function
    return AdjointODE.apply(y0, t, flat_params, solver, func)
