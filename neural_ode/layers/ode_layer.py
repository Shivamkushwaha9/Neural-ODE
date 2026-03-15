"""Neural ODE layer for continuous-depth transformations."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union

from ..solvers.base import ODESolver
from ..solvers.adaptive import Dopri5Solver
from .ode_func import ODEFunc


class NeuralODE(nn.Module):
    """Neural ODE layer for continuous-depth transformations.
    
    This layer computes transformations through ODE integration rather than
    discrete operations. Given an initial state h(t0), it computes the final
    state h(t1) by integrating the dynamics function:
        dh/dt = f(h(t), t, θ)
    
    The layer can be used like any other PyTorch layer and supports standard
    autograd for gradient computation. For memory-efficient training, use the
    adjoint sensitivity method (implemented separately).
    
    Args:
        func: Dynamics function (typically an ODEFunc wrapping a neural network)
              that defines dh/dt = f(h, t)
        solver: ODE solver instance for numerical integration.
                Default: Dopri5Solver with rtol=1e-3, atol=1e-4
        sensitivity: Gradient computation mode. Options:
                    - 'autograd': Standard PyTorch backpropagation through solver
                    - 'adjoint': Memory-efficient adjoint sensitivity method
                    Default: 'autograd' (adjoint not yet implemented)
        rtol: Relative tolerance for adaptive solvers (default: 1e-3)
        atol: Absolute tolerance for adaptive solvers (default: 1e-4)
    
    Attributes:
        func: The dynamics function
        solver: The ODE solver
        sensitivity: Gradient computation mode
    
    Example:
        >>> # Create a simple time-independent dynamics network
        >>> net = nn.Sequential(
        ...     nn.Linear(2, 64),
        ...     nn.Tanh(),
        ...     nn.Linear(64, 2)
        ... )
        >>> func = ODEFunc(net, time_dependent=False)
        >>> 
        >>> # Create Neural ODE layer
        >>> ode_layer = NeuralODE(func)
        >>> 
        >>> # Forward pass
        >>> x = torch.randn(10, 2)  # batch_size=10, state_dim=2
        >>> y = ode_layer(x)  # Integrate from t=0 to t=1
        >>> print(y.shape)  # torch.Size([10, 2])
        >>> 
        >>> # Custom time span
        >>> t = torch.tensor([0.0, 2.0])
        >>> y = ode_layer(x, t)  # Integrate from t=0 to t=2
        >>> 
        >>> # Use in a larger model
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 2),
        ...     ode_layer,
        ...     nn.Linear(2, 1)
        ... )
    """
    
    def __init__(self,
                 func: Union[nn.Module, ODEFunc],
                 solver: Optional[ODESolver] = None,
                 sensitivity: str = 'autograd',
                 rtol: float = 1e-3,
                 atol: float = 1e-4):
        """Initialize Neural ODE layer.
        
        Args:
            func: Dynamics function (nn.Module or ODEFunc)
            solver: ODE solver (default: Dopri5Solver)
            sensitivity: 'autograd' or 'adjoint' (default: 'autograd')
            rtol: Relative tolerance (default: 1e-3)
            atol: Absolute tolerance (default: 1e-4)
            
        Raises:
            ValueError: If sensitivity mode is invalid
        """
        super().__init__()
        
        # Wrap func in ODEFunc if it's not already
        if isinstance(func, ODEFunc):
            self.func = func
        else:
            # Assume it's a regular nn.Module, wrap it
            self.func = ODEFunc(func, time_dependent=False)
        
        # Set up solver
        if solver is None:
            self.solver = Dopri5Solver(rtol=rtol, atol=atol)
        else:
            self.solver = solver
        
        # Validate and set sensitivity mode
        if sensitivity not in ['autograd', 'adjoint']:
            raise ValueError(
                f"sensitivity must be 'autograd' or 'adjoint', got '{sensitivity}'"
            )
        
        self.sensitivity = sensitivity
        
    def forward(self, 
                x: Tensor,
                t: Optional[Tensor] = None) -> Tensor:
        """Integrate from t[0] to t[-1].
        
        Args:
            x: Initial state, shape (batch_size, state_dim)
            t: Integration times, shape (num_times,)
               Must be monotonically increasing.
               Default: torch.tensor([0.0, 1.0])
               
        Returns:
            Final state h(t[-1]), shape (batch_size, state_dim)
            
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration fails
        """
        # Set default time span if not provided
        if t is None:
            t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        
        # Ensure t is on the same device as x
        if t.device != x.device:
            t = t.to(x.device)
        
        # Ensure t has the same dtype as x
        if t.dtype != x.dtype:
            t = t.to(dtype=x.dtype)
        
        # Use appropriate gradient computation method
        if self.sensitivity == 'autograd':
            return self.solver.integrate(self.func, x, t)
        else:  # adjoint
            from ..adjoint.adjoint import adjoint_integrate
            return adjoint_integrate(self.solver, self.func, x, t)
    
    def forward_with_trajectory(self,
                               x: Tensor,
                               t: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Integrate and return full trajectory.
        
        This method is useful for visualization and debugging, as it returns
        the state at all requested time points rather than just the final state.
        
        Args:
            x: Initial state, shape (batch_size, state_dim)
            t: Time points for trajectory, shape (num_times,)
               Default: torch.linspace(0, 1, 10)
               
        Returns:
            Tuple of (times, states) where:
                times: Time points, shape (num_times,)
                states: States at each time, shape (num_times, batch_size, state_dim)
        """
        # Set default time points if not provided
        if t is None:
            t = torch.linspace(0, 1, 10, dtype=x.dtype, device=x.device)
        
        # Ensure t is on the same device and dtype as x
        if t.device != x.device:
            t = t.to(x.device)
        if t.dtype != x.dtype:
            t = t.to(dtype=x.dtype)
        
        return self.solver.integrate_with_trajectory(self.func, x, t)
    
    def reset_nfe(self):
        """Reset function evaluation counters.
        
        Resets the NFE counter in both the solver and the dynamics function.
        Useful for benchmarking and tracking computational cost.
        """
        if hasattr(self.solver, 'nfe'):
            self.solver.nfe = 0
        if hasattr(self.func, 'nfe'):
            self.func.reset_nfe()
    
    def get_nfe(self) -> dict:
        """Get function evaluation counts.
        
        Returns:
            Dictionary with 'solver_nfe' and 'func_nfe' counts
        """
        return {
            'solver_nfe': getattr(self.solver, 'nfe', 0),
            'func_nfe': getattr(self.func, 'nfe', 0)
        }
