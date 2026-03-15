"""Abstract base class for ODE solvers."""

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import torch
from torch import Tensor


class ODESolver(ABC):
    """Base class for ODE solvers.
    
    All ODE solvers must implement the integrate() and integrate_with_trajectory()
    methods. Solvers compute solutions to initial value problems of the form:
        dy/dt = f(t, y)
        y(t0) = y0
    
    where f is the dynamics function, y is the state, and t is time.
    """
    
    @abstractmethod
    def integrate(self, 
                  func: Callable[[float, Tensor], Tensor],
                  y0: Tensor,
                  t: Tensor,
                  **kwargs) -> Tensor:
        """Integrate ODE from t[0] to t[-1].
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
                  Takes scalar time t and state tensor y, returns time derivative
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,), must be monotonically increasing
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Final state y(t[-1]), shape (batch_size, state_dim)
            
        Raises:
            ValueError: If inputs are invalid (wrong shapes, non-monotonic times, etc.)
            IntegrationError: If integration fails (NaN, convergence failure, etc.)
        """
        pass
    
    @abstractmethod
    def integrate_with_trajectory(self,
                                   func: Callable[[float, Tensor], Tensor],
                                   y0: Tensor,
                                   t: Tensor,
                                   **kwargs) -> Tuple[Tensor, Tensor]:
        """Integrate and return full trajectory.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,)
            **kwargs: Additional solver-specific parameters
            
        Returns:
            Tuple of (times, states) where:
                times: Time points, shape (num_times,)
                states: States at each time point, shape (num_times, batch_size, state_dim)
                
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration fails
        """
        pass
