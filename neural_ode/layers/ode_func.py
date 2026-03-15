"""Wrapper for dynamics functions with time input."""

import torch
import torch.nn as nn
from torch import Tensor


class ODEFunc(nn.Module):
    """Wrapper for dynamics function with time input.
    
    This class wraps a user-provided neural network to create a dynamics function
    suitable for ODE integration. It handles:
    - Time-dependent and time-independent dynamics
    - Time concatenation for time-dependent functions
    - Function evaluation counting (NFE)
    
    The wrapped function defines the time derivative: dh/dt = f(h, t)
    
    Args:
        net: Neural network that computes the dynamics. For time-dependent mode,
             the network should accept input of shape (batch, state_dim + 1) where
             the last dimension is time. For time-independent mode, it accepts
             input of shape (batch, state_dim).
        time_dependent: If True, concatenate time as an additional input feature.
                       If False, ignore time and only pass state to the network.
                       Default: True
    
    Attributes:
        net: The wrapped neural network
        time_dependent: Whether the dynamics depend on time
        nfe: Number of function evaluations (incremented on each forward call)
    
    Example:
        >>> # Time-dependent dynamics
        >>> net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 2))
        >>> func = ODEFunc(net, time_dependent=True)
        >>> h = torch.randn(10, 2)  # batch_size=10, state_dim=2
        >>> t = 0.5
        >>> dh_dt = func(t, h)  # Returns shape (10, 2)
        >>> print(func.nfe)  # Prints 1
        
        >>> # Time-independent dynamics
        >>> net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        >>> func = ODEFunc(net, time_dependent=False)
        >>> dh_dt = func(t, h)  # Time is ignored
    """
    
    def __init__(self, net: nn.Module, time_dependent: bool = True):
        super().__init__()
        self.net = net
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        
    def forward(self, t: float, h: Tensor) -> Tensor:
        """Compute dh/dt = f(h, t).
        
        Args:
            t: Current time (scalar float)
            h: Current state, shape (batch_size, state_dim)
            
        Returns:
            Time derivative dh/dt, shape (batch_size, state_dim)
        """
        self.nfe += 1
        
        if self.time_dependent:
            # Concatenate time as additional input feature
            # Create time vector with same batch size as h
            t_vec = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype) * t
            h_with_t = torch.cat([h, t_vec], dim=1)
            return self.net(h_with_t)
        else:
            # Time-independent: just pass state to network
            return self.net(h)
    
    def reset_nfe(self):
        """Reset the function evaluation counter to zero.
        
        Useful for tracking NFE separately for different operations
        (e.g., forward pass vs backward pass).
        """
        self.nfe = 0
