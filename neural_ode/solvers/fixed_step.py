"""Fixed-step ODE solvers (Euler and RK4)."""

from typing import Callable, Tuple
import torch
from torch import Tensor

from .base import ODESolver
from ..utils.validation import validate_ode_inputs, check_finite, IntegrationError


class EulerSolver(ODESolver):
    """First-order Euler method for ODE integration.
    
    Implements the forward Euler method:
        y_{n+1} = y_n + h * f(t_n, y_n)
    
    where h is the step size. This is the simplest ODE solver but has
    first-order accuracy O(h).
    
    Attributes:
        step_size: Fixed step size for integration
        nfe: Number of function evaluations (cumulative across all integrations)
    """
    
    def __init__(self, step_size: float = 0.1):
        """Initialize Euler solver.
        
        Args:
            step_size: Fixed step size for integration (default: 0.1)
            
        Raises:
            ValueError: If step_size is not positive
        """
        if step_size <= 0:
            raise ValueError(f"Step size must be positive, got {step_size}")
        
        self.step_size = step_size
        self.nfe = 0
    
    def integrate(self, 
                  func: Callable[[float, Tensor], Tensor],
                  y0: Tensor,
                  t: Tensor,
                  **kwargs) -> Tensor:
        """Integrate ODE from t[0] to t[-1] using forward Euler method.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,), must be monotonically increasing
            **kwargs: Additional parameters (ignored for fixed-step solver)
            
        Returns:
            Final state y(t[-1]), shape (batch_size, state_dim)
            
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration produces NaN or Inf values
        """
        # Validate inputs
        validate_ode_inputs(y0, t)
        
        # Extract start and end times
        t0, t1 = t[0].item(), t[-1].item()
        
        # Initialize state
        y = y0.clone()
        t_current = t0
        
        # Integration loop
        while t_current < t1:
            # Determine step size (don't overshoot t1)
            h = min(self.step_size, t1 - t_current)
            
            # Evaluate dynamics function
            dy_dt = func(t_current, y)
            self.nfe += 1
            
            # Check for numerical issues
            check_finite(dy_dt, f"Dynamics at t={t_current:.6f}")
            
            # Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
            y = y + h * dy_dt
            
            # Check state after update
            if not torch.isfinite(y).all():
                raise IntegrationError(
                    "Integration produced NaN or Inf values",
                    t_current=t_current + h,
                    state=y
                )
            
            # Advance time
            t_current += h
        
        return y
    
    def integrate_with_trajectory(self,
                                   func: Callable[[float, Tensor], Tensor],
                                   y0: Tensor,
                                   t: Tensor,
                                   **kwargs) -> Tuple[Tensor, Tensor]:
        """Integrate and return full trajectory at specified time points.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points where trajectory should be recorded, shape (num_times,)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Tuple of (times, states) where:
                times: Time points, shape (num_times,)
                states: States at each time point, shape (num_times, batch_size, state_dim)
                
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration fails
        """
        # Validate inputs
        validate_ode_inputs(y0, t)
        
        num_times = len(t)
        batch_size, state_dim = y0.shape
        
        # Allocate storage for trajectory
        states = torch.zeros(num_times, batch_size, state_dim, 
                            dtype=y0.dtype, device=y0.device)
        states[0] = y0
        
        # Integrate to each requested time point
        y = y0.clone()
        for i in range(1, num_times):
            t_start = t[i-1].item()
            t_end = t[i].item()
            t_current = t_start
            
            # Integrate from t[i-1] to t[i]
            while t_current < t_end:
                h = min(self.step_size, t_end - t_current)
                
                # Euler step
                dy_dt = func(t_current, y)
                self.nfe += 1
                
                check_finite(dy_dt, f"Dynamics at t={t_current:.6f}")
                
                y = y + h * dy_dt
                
                if not torch.isfinite(y).all():
                    raise IntegrationError(
                        "Integration produced NaN or Inf values",
                        t_current=t_current + h,
                        state=y
                    )
                
                t_current += h
            
            # Store state at requested time point
            states[i] = y
        
        return t, states


class RK4Solver(ODESolver):
    """Fourth-order Runge-Kutta method for ODE integration.
    
    Implements the classic RK4 method with four stages:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h/2, y_n + h*k2/2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    This method has fourth-order accuracy O(h^4), providing much better
    accuracy than Euler for the same step size.
    
    Attributes:
        step_size: Fixed step size for integration
        nfe: Number of function evaluations (cumulative across all integrations)
    """
    
    def __init__(self, step_size: float = 0.1):
        """Initialize RK4 solver.
        
        Args:
            step_size: Fixed step size for integration (default: 0.1)
            
        Raises:
            ValueError: If step_size is not positive
        """
        if step_size <= 0:
            raise ValueError(f"Step size must be positive, got {step_size}")
        
        self.step_size = step_size
        self.nfe = 0
    
    def integrate(self, 
                  func: Callable[[float, Tensor], Tensor],
                  y0: Tensor,
                  t: Tensor,
                  **kwargs) -> Tensor:
        """Integrate ODE from t[0] to t[-1] using RK4 method.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,), must be monotonically increasing
            **kwargs: Additional parameters (ignored for fixed-step solver)
            
        Returns:
            Final state y(t[-1]), shape (batch_size, state_dim)
            
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration produces NaN or Inf values
        """
        # Validate inputs
        validate_ode_inputs(y0, t)
        
        # Extract start and end times
        t0, t1 = t[0].item(), t[-1].item()
        
        # Initialize state
        y = y0.clone()
        t_current = t0
        
        # Integration loop
        while t_current < t1:
            # Determine step size (don't overshoot t1)
            h = min(self.step_size, t1 - t_current)
            
            # RK4 stage 1: k1 = f(t_n, y_n)
            k1 = func(t_current, y)
            self.nfe += 1
            check_finite(k1, f"k1 at t={t_current:.6f}")
            
            # RK4 stage 2: k2 = f(t_n + h/2, y_n + h*k1/2)
            k2 = func(t_current + h/2, y + h * k1 / 2)
            self.nfe += 1
            check_finite(k2, f"k2 at t={t_current + h/2:.6f}")
            
            # RK4 stage 3: k3 = f(t_n + h/2, y_n + h*k2/2)
            k3 = func(t_current + h/2, y + h * k2 / 2)
            self.nfe += 1
            check_finite(k3, f"k3 at t={t_current + h/2:.6f}")
            
            # RK4 stage 4: k4 = f(t_n + h, y_n + h*k3)
            k4 = func(t_current + h, y + h * k3)
            self.nfe += 1
            check_finite(k4, f"k4 at t={t_current + h:.6f}")
            
            # Combine stages: y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            y = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Check state after update
            if not torch.isfinite(y).all():
                raise IntegrationError(
                    "Integration produced NaN or Inf values",
                    t_current=t_current + h,
                    state=y
                )
            
            # Advance time
            t_current += h
        
        return y
    
    def integrate_with_trajectory(self,
                                   func: Callable[[float, Tensor], Tensor],
                                   y0: Tensor,
                                   t: Tensor,
                                   **kwargs) -> Tuple[Tensor, Tensor]:
        """Integrate and return full trajectory at specified time points.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points where trajectory should be recorded, shape (num_times,)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Tuple of (times, states) where:
                times: Time points, shape (num_times,)
                states: States at each time point, shape (num_times, batch_size, state_dim)
                
        Raises:
            ValueError: If inputs are invalid
            IntegrationError: If integration fails
        """
        # Validate inputs
        validate_ode_inputs(y0, t)
        
        num_times = len(t)
        batch_size, state_dim = y0.shape
        
        # Allocate storage for trajectory
        states = torch.zeros(num_times, batch_size, state_dim, 
                            dtype=y0.dtype, device=y0.device)
        states[0] = y0
        
        # Integrate to each requested time point
        y = y0.clone()
        for i in range(1, num_times):
            t_start = t[i-1].item()
            t_end = t[i].item()
            t_current = t_start
            
            # Integrate from t[i-1] to t[i]
            while t_current < t_end:
                h = min(self.step_size, t_end - t_current)
                
                # RK4 stages
                k1 = func(t_current, y)
                self.nfe += 1
                check_finite(k1, f"k1 at t={t_current:.6f}")
                
                k2 = func(t_current + h/2, y + h * k1 / 2)
                self.nfe += 1
                check_finite(k2, f"k2 at t={t_current + h/2:.6f}")
                
                k3 = func(t_current + h/2, y + h * k2 / 2)
                self.nfe += 1
                check_finite(k3, f"k3 at t={t_current + h/2:.6f}")
                
                k4 = func(t_current + h, y + h * k3)
                self.nfe += 1
                check_finite(k4, f"k4 at t={t_current + h:.6f}")
                
                # Combine stages
                y = y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
                
                if not torch.isfinite(y).all():
                    raise IntegrationError(
                        "Integration produced NaN or Inf values",
                        t_current=t_current + h,
                        state=y
                    )
                
                t_current += h
            
            # Store state at requested time point
            states[i] = y
        
        return t, states
