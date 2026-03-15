"""Continuous Normalizing Flow (CNF) for density estimation.

This module implements Continuous Normalizing Flows, which transform samples
from a base distribution to a target distribution through continuous-time
dynamics while tracking the change in log-probability.

The key innovation is computing the instantaneous change of variables using
the trace of the Jacobian: d(log p)/dt = -tr(∂f/∂z)

Reference:
    Chen et al. (2018), "Neural Ordinary Differential Equations"
    Grathwohl et al. (2018), "FINT: Scalable and Flexible Normalizing Flows"
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union

from ..solvers.base import ODESolver
from ..solvers.adaptive import Dopri5Solver
from ..utils.trace import hutchinson_trace, exact_trace
from .ode_func import ODEFunc


class CNF(nn.Module):
    """Continuous Normalizing Flow for density estimation.
    
    CNF transforms samples from a base distribution (e.g., Gaussian) to a
    target distribution through continuous-time dynamics. Unlike discrete
    normalizing flows, CNF does not require invertible architectures and
    can compute exact log-likelihoods using the instantaneous change of
    variables formula.
    
    The transformation is defined by integrating an ODE:
        dz/dt = f(z, t, θ)
    
    The log-probability changes according to:
        d(log p)/dt = -tr(∂f/∂z)
    
    Args:
        net: Neural network defining the dynamics function f(z, t)
        solver: ODE solver for integration (default: Dopri5Solver)
        trace_estimator: Method for computing trace of Jacobian:
                        - 'hutchinson': Stochastic estimation (faster, O(1) in dim)
                        - 'exact': Exact computation (slower, O(dim) in dim)
                        Default: 'hutchinson'
        hutchinson_samples: Number of samples for Hutchinson estimator (default: 1)
        rtol: Relative tolerance for adaptive solver (default: 1e-3)
        atol: Absolute tolerance for adaptive solver (default: 1e-4)
    
    Attributes:
        func: ODEFunc wrapper around the dynamics network
        solver: ODE solver instance
        trace_estimator: Trace computation method
        hutchinson_samples: Number of samples for stochastic trace estimation
    
    Example:
        >>> # Create dynamics network
        >>> net = nn.Sequential(
        ...     nn.Linear(2, 64),
        ...     nn.Tanh(),
        ...     nn.Linear(64, 2)
        ... )
        >>> 
        >>> # Create CNF
        >>> cnf = CNF(net, trace_estimator='hutchinson')
        >>> 
        >>> # Transform samples and compute log-determinant
        >>> x = torch.randn(100, 2)
        >>> z, log_det = cnf(x, reverse=False)
        >>> 
        >>> # Compute log-probability
        >>> base_dist = torch.distributions.Normal(0, 1)
        >>> log_prob = cnf.log_prob(x, base_dist)
        >>> 
        >>> # Generate samples
        >>> samples = cnf.sample(100, base_dist)
    """
    
    def __init__(self,
                 net: nn.Module,
                 solver: Optional[ODESolver] = None,
                 trace_estimator: str = 'hutchinson',
                 hutchinson_samples: int = 1,
                 rtol: float = 1e-3,
                 atol: float = 1e-4):
        """Initialize CNF layer.
        
        Args:
            net: Neural network for dynamics
            solver: ODE solver (default: Dopri5Solver)
            trace_estimator: 'hutchinson' or 'exact'
            hutchinson_samples: Number of samples for Hutchinson estimator
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Raises:
            ValueError: If trace_estimator is invalid
        """
        super().__init__()
        
        # Store the network directly (we'll handle time concatenation in augmented_dynamics)
        self.net = net
        
        # Set up solver
        if solver is None:
            self.solver = Dopri5Solver(rtol=rtol, atol=atol)
        else:
            self.solver = solver
        
        # Validate and set trace estimator
        if trace_estimator not in ['hutchinson', 'exact']:
            raise ValueError(
                f"trace_estimator must be 'hutchinson' or 'exact', "
                f"got '{trace_estimator}'"
            )
        self.trace_estimator = trace_estimator
        self.hutchinson_samples = hutchinson_samples
    
    def forward(self, 
                x: Tensor,
                reverse: bool = False,
                t: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Transform x and compute log-determinant of Jacobian.
        
        This method integrates the augmented ODE system that combines the
        state dynamics with the log-determinant dynamics:
            dz/dt = f(z, t)
            d(log p)/dt = -tr(∂f/∂z)
        
        Args:
            x: Input samples, shape (batch_size, dim)
            reverse: If True, integrate backward in time (for sampling)
                    If False, integrate forward (for likelihood computation)
            t: Time span for integration, shape (2,)
               Default: [0.0, 1.0] for forward, [1.0, 0.0] for reverse
               
        Returns:
            Tuple of (z, log_det) where:
                z: Transformed samples, shape (batch_size, dim)
                log_det: Log-determinant of Jacobian, shape (batch_size,)
                        This represents the change in log-probability
        
        Example:
            >>> cnf = CNF(net)
            >>> x = torch.randn(32, 2)
            >>> 
            >>> # Forward transformation (data to latent)
            >>> z, log_det = cnf(x, reverse=False)
            >>> 
            >>> # Reverse transformation (latent to data)
            >>> x_reconstructed, log_det_inv = cnf(z, reverse=True)
        """
        batch_size, dim = x.shape
        
        # Initialize log-determinant to zero
        logp = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # Set time span
        if t is None:
            if reverse:
                # For reverse, we integrate forward in time but negate dynamics
                t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
            else:
                t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        
        # Ensure t is on correct device and dtype
        if t.device != x.device:
            t = t.to(x.device)
        if t.dtype != x.dtype:
            t = t.to(dtype=x.dtype)
        
        # Define augmented dynamics that combines state and log-determinant
        def augmented_dynamics(t_val: float, state: Tensor) -> Tensor:
            """Compute derivatives for augmented system.
            
            Args:
                t_val: Current time
                state: Augmented state [z, log_p], shape (batch_size, dim + 1)
                
            Returns:
                Augmented derivatives [dz/dt, d(log_p)/dt], shape (batch_size, dim + 1)
            """
            # Split state into z and log_p
            z = state[:, :-1]  # shape: (batch_size, dim)
            logp_curr = state[:, -1:]  # shape: (batch_size, 1)
            
            # Ensure z requires gradients for trace computation
            z = z.requires_grad_(True)
            
            # Compute state dynamics: dz/dt = f(z, t)
            # Note: We don't use ODEFunc here to avoid double time concatenation
            with torch.enable_grad():
                dz_dt = self.net(z)
                
                # For reverse integration, negate the dynamics
                if reverse:
                    dz_dt = -dz_dt
                
                # Compute trace of Jacobian: tr(∂f/∂z)
                if self.trace_estimator == 'exact':
                    trace = exact_trace(dz_dt, z)
                else:  # hutchinson
                    trace = hutchinson_trace(
                        dz_dt, z, 
                        num_samples=self.hutchinson_samples
                    )
                
                # Log-determinant dynamics: d(log p)/dt = -tr(∂f/∂z)
                dlogp_dt = -trace.view(batch_size, 1)
            
            # Concatenate derivatives
            return torch.cat([dz_dt, dlogp_dt], dim=1)
        
        # Create augmented initial state
        state0 = torch.cat([x, logp], dim=1)
        
        # Integrate augmented system
        state1 = self.solver.integrate(augmented_dynamics, state0, t)
        
        # Extract final state and log-determinant
        z = state1[:, :-1]
        log_det = state1[:, -1]
        
        return z, log_det
    
    def log_prob(self, 
                 x: Tensor,
                 base_dist: torch.distributions.Distribution) -> Tensor:
        """Compute log-probability of data samples under the model.
        
        The log-probability is computed using the change of variables formula:
            log p(x) = log p(z) + log |det(∂z/∂x)|
        
        where z = f(x) is the transformation to the base distribution.
        
        Args:
            x: Data samples, shape (batch_size, dim)
            base_dist: Base distribution (e.g., torch.distributions.Normal)
                      Must support log_prob() method
            
        Returns:
            Log-probabilities, shape (batch_size,)
        
        Example:
            >>> cnf = CNF(net)
            >>> x = torch.randn(100, 2)
            >>> base_dist = torch.distributions.Normal(
            ...     torch.zeros(2), torch.ones(2)
            ... )
            >>> log_prob = cnf.log_prob(x, base_dist)
            >>> print(log_prob.shape)  # torch.Size([100])
            >>> 
            >>> # Compute negative log-likelihood loss
            >>> nll_loss = -log_prob.mean()
        """
        # Transform to base distribution (forward pass)
        z, log_det = self.forward(x, reverse=False)
        
        # Compute log-probability in base distribution
        # Handle both univariate and multivariate distributions
        log_pz = base_dist.log_prob(z)
        
        # If log_pz has shape (batch, dim), sum over dimensions
        if log_pz.dim() > 1:
            log_pz = log_pz.sum(dim=1)
        
        # Apply change of variables: log p(x) = log p(z) + log |det(∂z/∂x)|
        return log_pz + log_det
    
    def sample(self,
               num_samples: int,
               base_dist: torch.distributions.Distribution,
               device: Optional[torch.device] = None) -> Tensor:
        """Generate samples from the model.
        
        Samples are generated by:
        1. Sampling z from the base distribution
        2. Transforming z to x by integrating backward in time
        
        Args:
            num_samples: Number of samples to generate
            base_dist: Base distribution to sample from
            device: Device to generate samples on (default: CPU)
            
        Returns:
            Generated samples, shape (num_samples, dim)
        
        Example:
            >>> cnf = CNF(net)
            >>> base_dist = torch.distributions.Normal(
            ...     torch.zeros(2), torch.ones(2)
            ... )
            >>> samples = cnf.sample(1000, base_dist)
            >>> print(samples.shape)  # torch.Size([1000, 2])
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from base distribution
        with torch.no_grad():
            z = base_dist.sample((num_samples,)).to(device)
            
            # Handle case where sample returns shape (num_samples, 1, dim)
            if z.dim() > 2:
                z = z.squeeze(1)
            
            # Transform to data space (reverse integration)
            x, _ = self.forward(z, reverse=True)
        
        return x
    
    def reset_nfe(self):
        """Reset function evaluation counter.
        
        Useful for benchmarking and tracking computational cost.
        """
        if hasattr(self.solver, 'nfe'):
            self.solver.nfe = 0
    
    def get_nfe(self) -> dict:
        """Get function evaluation counts.
        
        Returns:
            Dictionary with 'solver_nfe' count
        """
        return {
            'solver_nfe': getattr(self.solver, 'nfe', 0)
        }
