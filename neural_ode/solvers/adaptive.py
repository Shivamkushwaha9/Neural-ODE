"""Adaptive ODE solvers (Dormand-Prince)."""

from typing import Callable, Tuple, List
import torch
from torch import Tensor

from .base import ODESolver
from ..utils.validation import (
    validate_ode_inputs, 
    check_finite, 
    validate_tolerances,
    IntegrationError,
    MaxStepsExceeded,
    StepSizeTooSmall
)


class Dopri5Solver(ODESolver):
    """Dormand-Prince adaptive solver (RK45).
    
    Implements the Dormand-Prince method, a 5th-order Runge-Kutta method
    with embedded 4th-order error estimation for adaptive step size control.
    
    The method uses 7 stages to compute both a 5th-order solution and a
    4th-order solution, then uses the difference to estimate the local error
    and adjust the step size accordingly.
    
    Butcher tableau coefficients for DOPRI5:
        c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
        a = [[0],
             [1/5],
             [3/40, 9/40],
             [44/45, -56/15, 32/9],
             [19372/6561, -25360/2187, 64448/6561, -212/729],
             [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
             [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
        b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    Attributes:
        rtol: Relative tolerance for error control
        atol: Absolute tolerance for error control
        max_steps: Maximum number of integration steps
        nfe: Number of function evaluations (cumulative)
        safety: Safety factor for step size adjustment (default: 0.9)
        min_step: Minimum allowed step size (default: 1e-10)
        max_step: Maximum allowed step size (default: inf)
    """
    
    # Butcher tableau coefficients for DOPRI5
    C = torch.tensor([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])
    
    A = [
        torch.tensor([]),
        torch.tensor([1/5]),
        torch.tensor([3/40, 9/40]),
        torch.tensor([44/45, -56/15, 32/9]),
        torch.tensor([19372/6561, -25360/2187, 64448/6561, -212/729]),
        torch.tensor([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
        torch.tensor([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84])
    ]
    
    # 5th order solution coefficients
    B5 = torch.tensor([35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
    
    # 4th order solution coefficients (for error estimation)
    B4 = torch.tensor([5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    def __init__(self, 
                 rtol: float = 1e-3,
                 atol: float = 1e-4,
                 max_steps: int = 1000,
                 safety: float = 0.9,
                 min_step: float = 1e-10,
                 max_step: float = float('inf')):
        """Initialize Dormand-Prince solver.
        
        Args:
            rtol: Relative tolerance for error control (default: 1e-3)
            atol: Absolute tolerance for error control (default: 1e-4)
            max_steps: Maximum number of integration steps (default: 1000)
            safety: Safety factor for step size adjustment (default: 0.9)
            min_step: Minimum allowed step size (default: 1e-10)
            max_step: Maximum allowed step size (default: inf)
            
        Raises:
            ValueError: If tolerances are not positive
        """
        validate_tolerances(rtol, atol)
        
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        
        if not (0 < safety < 1):
            raise ValueError(f"safety factor must be in (0, 1), got {safety}")
        
        if min_step <= 0:
            raise ValueError(f"min_step must be positive, got {min_step}")
        
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.safety = safety
        self.min_step = min_step
        self.max_step = max_step
        self.nfe = 0
    
    def _compute_error_ratio(self, error: Tensor, y: Tensor, y_new: Tensor) -> float:
        """Compute error ratio for step size control.
        
        The error ratio is computed as:
            error_ratio = ||error|| / tolerance
        where tolerance = atol + rtol * max(||y||, ||y_new||)
        
        Args:
            error: Local error estimate, shape (batch_size, state_dim)
            y: Current state, shape (batch_size, state_dim)
            y_new: Proposed new state, shape (batch_size, state_dim)
            
        Returns:
            Error ratio (scalar)
        """
        # Compute scale for error tolerance
        scale = self.atol + self.rtol * torch.maximum(
            torch.abs(y), torch.abs(y_new)
        )
        
        # Compute normalized error
        error_norm = torch.sqrt(torch.mean((error / scale) ** 2))
        
        return error_norm.item()
    
    def _dopri5_step(self, 
                     func: Callable[[float, Tensor], Tensor],
                     t: float,
                     y: Tensor,
                     h: float) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform one Dormand-Prince step.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            t: Current time
            y: Current state, shape (batch_size, state_dim)
            h: Step size
            
        Returns:
            Tuple of (y5, y4, k_stages) where:
                y5: 5th order solution, shape (batch_size, state_dim)
                y4: 4th order solution, shape (batch_size, state_dim)
                k_stages: List of 7 stage derivatives for FSAL
        """
        device = y.device
        dtype = y.dtype
        
        # Move coefficients to correct device and dtype
        c = self.C.to(device=device, dtype=dtype)
        b5 = self.B5.to(device=device, dtype=dtype)
        b4 = self.B4.to(device=device, dtype=dtype)
        
        # Compute 7 stages
        k = []
        
        # Stage 1: k1 = f(t, y)
        k1 = func(t, y)
        self.nfe += 1
        k.append(k1)
        
        # Stage 2: k2 = f(t + c[1]*h, y + h*a[1][0]*k1)
        a1 = self.A[1].to(device=device, dtype=dtype)
        k2 = func(t + c[1]*h, y + h * a1[0] * k1)
        self.nfe += 1
        k.append(k2)
        
        # Stage 3: k3 = f(t + c[2]*h, y + h*(a[2][0]*k1 + a[2][1]*k2))
        a2 = self.A[2].to(device=device, dtype=dtype)
        k3 = func(t + c[2]*h, y + h * (a2[0]*k1 + a2[1]*k2))
        self.nfe += 1
        k.append(k3)
        
        # Stage 4
        a3 = self.A[3].to(device=device, dtype=dtype)
        k4 = func(t + c[3]*h, y + h * (a3[0]*k1 + a3[1]*k2 + a3[2]*k3))
        self.nfe += 1
        k.append(k4)
        
        # Stage 5
        a4 = self.A[4].to(device=device, dtype=dtype)
        k5 = func(t + c[4]*h, y + h * (a4[0]*k1 + a4[1]*k2 + a4[2]*k3 + a4[3]*k4))
        self.nfe += 1
        k.append(k5)
        
        # Stage 6
        a5 = self.A[5].to(device=device, dtype=dtype)
        k6 = func(t + c[5]*h, y + h * (a5[0]*k1 + a5[1]*k2 + a5[2]*k3 + a5[3]*k4 + a5[4]*k5))
        self.nfe += 1
        k.append(k6)
        
        # Stage 7 (same as next step's k1 for FSAL - First Same As Last)
        a6 = self.A[6].to(device=device, dtype=dtype)
        k7 = func(t + c[6]*h, y + h * (a6[0]*k1 + a6[1]*k2 + a6[2]*k3 + a6[3]*k4 + a6[4]*k5 + a6[5]*k6))
        self.nfe += 1
        k.append(k7)
        
        # Compute 5th order solution
        y5 = y + h * (b5[0]*k1 + b5[1]*k2 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6 + b5[6]*k7)
        
        # Compute 4th order solution
        y4 = y + h * (b4[0]*k1 + b4[1]*k2 + b4[2]*k3 + b4[3]*k4 + b4[4]*k5 + b4[5]*k6 + b4[6]*k7)
        
        return y5, y4, k
    
    def integrate(self, 
                  func: Callable[[float, Tensor], Tensor],
                  y0: Tensor,
                  t: Tensor,
                  **kwargs) -> Tensor:
        """Integrate ODE from t[0] to t[-1] using adaptive Dormand-Prince method.
        
        Args:
            func: Dynamics function f(t, y) -> dy/dt
            y0: Initial state, shape (batch_size, state_dim)
            t: Time points, shape (num_times,), must be monotonically increasing
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Final state y(t[-1]), shape (batch_size, state_dim)
            
        Raises:
            ValueError: If inputs are invalid
            MaxStepsExceeded: If integration exceeds max_steps
            StepSizeTooSmall: If step size becomes smaller than min_step
            IntegrationError: If integration produces NaN or Inf values
        """
        # Validate inputs
        validate_ode_inputs(y0, t)
        
        # Extract start and end times
        t0, t1 = t[0].item(), t[-1].item()
        
        # Initialize state and step size
        y = y0.clone()
        t_current = t0
        h = min(0.1, t1 - t0)  # Initial step size guess
        
        # Integration loop
        step_count = 0
        
        while t_current < t1:
            # Check max steps
            if step_count >= self.max_steps:
                raise MaxStepsExceeded(
                    f"Integration exceeded maximum steps ({self.max_steps})",
                    t_current=t_current,
                    state=y
                )
            
            # Don't overshoot final time
            h = min(h, t1 - t_current)
            h = min(h, self.max_step)
            
            # Check minimum step size
            if h < self.min_step:
                raise StepSizeTooSmall(
                    f"Step size ({h:.2e}) below minimum ({self.min_step:.2e})",
                    t_current=t_current,
                    state=y
                )
            
            # Perform Dormand-Prince step
            y5, y4, k_stages = self._dopri5_step(func, t_current, y, h)
            
            # Check for NaN/Inf
            if not torch.isfinite(y5).all():
                raise IntegrationError(
                    "Integration produced NaN or Inf values",
                    t_current=t_current + h,
                    state=y5
                )
            
            # Compute error estimate
            error = y5 - y4
            error_ratio = self._compute_error_ratio(error, y, y5)
            
            # Accept or reject step
            if error_ratio <= 1.0:
                # Accept step
                y = y5
                t_current += h
                step_count += 1
            
            # Adjust step size for next iteration
            if error_ratio > 0:
                # Standard step size control formula
                h_new = h * self.safety * (1.0 / error_ratio) ** 0.2
                h = max(min(h_new, self.max_step), self.min_step)
            else:
                # Error is zero, increase step size
                h = min(h * 2.0, self.max_step)
        
        return y
    
    def integrate_with_trajectory(self,
                                   func: Callable[[float, Tensor], Tensor],
                                   y0: Tensor,
                                   t: Tensor,
                                   **kwargs) -> Tuple[Tensor, Tensor]:
        """Integrate and return trajectory at specified time points.
        
        This method integrates the ODE and records the state at each requested
        time point. The actual integration uses adaptive stepping, but the
        output is interpolated to match the requested times.
        
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
            MaxStepsExceeded: If integration exceeds max_steps
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
            
            # Create temporary time span for this segment
            t_segment = torch.tensor([t_start, t_end], dtype=t.dtype, device=t.device)
            
            # Integrate from t[i-1] to t[i]
            y = self.integrate(func, y, t_segment)
            
            # Store state at requested time point
            states[i] = y
        
        return t, states

