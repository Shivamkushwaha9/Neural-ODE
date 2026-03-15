"""Tests for AdjointODE autograd function."""

import torch
import torch.nn as nn
import pytest

from neural_ode.adjoint.adjoint import AdjointODE, adjoint_integrate
from neural_ode.adjoint.params import flatten_params
from neural_ode.layers.ode_func import ODEFunc
from neural_ode.solvers import Dopri5Solver, RK4Solver


class TestAdjointODE:
    """Test AdjointODE autograd function."""
    
    def test_adjoint_ode_forward(self):
        """Test that AdjointODE forward pass works."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Flatten parameters
        flat_params, _ = flatten_params(func.parameters())
        
        # Create test inputs
        batch_size = 5
        state_dim = 2
        y0 = torch.randn(batch_size, state_dim)
        t = torch.tensor([0.0, 1.0])
        
        # Create solver
        solver = Dopri5Solver(rtol=1e-3, atol=1e-4)
        
        # Apply AdjointODE forward
        y1 = AdjointODE.apply(y0, t, flat_params, solver, func)
        
        # Check shape
        assert y1.shape == (batch_size, state_dim)
        assert torch.isfinite(y1).all()
    
    def test_adjoint_ode_backward(self):
        """Test that AdjointODE backward pass computes gradients."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Flatten parameters
        flat_params, _ = flatten_params(func.parameters())
        
        # Create test inputs
        batch_size = 5
        state_dim = 2
        y0 = torch.randn(batch_size, state_dim, requires_grad=True)
        t = torch.tensor([0.0, 1.0])
        
        # Create solver
        solver = Dopri5Solver(rtol=1e-3, atol=1e-4)
        
        # Forward pass
        y1 = AdjointODE.apply(y0, t, flat_params, solver, func)
        
        # Backward pass
        loss = y1.sum()
        loss.backward()
        
        # Check that gradients exist
        assert y0.grad is not None
        assert torch.isfinite(y0.grad).all()
    
    def test_adjoint_integrate_wrapper(self):
        """Test adjoint_integrate wrapper function."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Create test inputs
        batch_size = 5
        state_dim = 2
        y0 = torch.randn(batch_size, state_dim)
        t = torch.tensor([0.0, 1.0])
        
        # Create solver
        solver = Dopri5Solver(rtol=1e-3, atol=1e-4)
        
        # Use adjoint_integrate
        y1 = adjoint_integrate(solver, func, y0, t)
        
        # Check shape
        assert y1.shape == (batch_size, state_dim)
        assert torch.isfinite(y1).all()
    
    def test_adjoint_integrate_gradients(self):
        """Test that adjoint_integrate computes gradients correctly."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Create test inputs
        batch_size = 5
        state_dim = 2
        y0 = torch.randn(batch_size, state_dim, requires_grad=True)
        t = torch.tensor([0.0, 1.0])
        
        # Create solver
        solver = Dopri5Solver(rtol=1e-3, atol=1e-4)
        
        # Forward pass
        y1 = adjoint_integrate(solver, func, y0, t)
        
        # Backward pass
        loss = (y1 ** 2).sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert y0.grad is not None
        assert torch.isfinite(y0.grad).all()
        
        # Check that parameter gradients exist
        for param in func.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
