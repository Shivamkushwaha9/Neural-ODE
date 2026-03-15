"""Tests for augmented ODE dynamics in adjoint method."""

import torch
import torch.nn as nn
import pytest

from neural_ode.adjoint.adjoint import create_augmented_dynamics, AugmentedDynamics
from neural_ode.adjoint.params import flatten_params
from neural_ode.layers.ode_func import ODEFunc


class TestAugmentedDynamics:
    """Test augmented dynamics for adjoint computation."""
    
    def test_create_augmented_dynamics_shapes(self):
        """Test that augmented dynamics returns correct shapes."""
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
        t = 0.5
        y = torch.randn(batch_size, state_dim)
        adj_y = torch.randn(batch_size, state_dim)
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # Check shapes
        assert dy_dt.shape == (batch_size, state_dim), \
            f"Expected dy_dt shape {(batch_size, state_dim)}, got {dy_dt.shape}"
        assert dadj_dt.shape == (batch_size, state_dim), \
            f"Expected dadj_dt shape {(batch_size, state_dim)}, got {dadj_dt.shape}"
        assert dparam_dt.shape == flat_params.shape, \
            f"Expected dparam_dt shape {flat_params.shape}, got {dparam_dt.shape}"
    
    def test_create_augmented_dynamics_finite(self):
        """Test that augmented dynamics produces finite values."""
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
        t = 0.5
        y = torch.randn(batch_size, state_dim)
        adj_y = torch.randn(batch_size, state_dim)
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # Check all values are finite
        assert torch.isfinite(dy_dt).all(), "dy_dt contains non-finite values"
        assert torch.isfinite(dadj_dt).all(), "dadj_dt contains non-finite values"
        assert torch.isfinite(dparam_dt).all(), "dparam_dt contains non-finite values"
    
    def test_augmented_dynamics_wrapper(self):
        """Test AugmentedDynamics wrapper class."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Flatten parameters
        flat_params, _ = flatten_params(func.parameters())
        
        # Create augmented dynamics wrapper
        aug_dynamics = AugmentedDynamics(func, flat_params)
        
        # Create test inputs
        batch_size = 5
        state_dim = 2
        t = 0.5
        y = torch.randn(batch_size, state_dim)
        adj_y = torch.randn(batch_size, state_dim)
        adj_params = torch.zeros_like(flat_params)
        
        augmented_state = (y, adj_y, adj_params)
        
        # Call augmented dynamics
        dy_dt, dadj_dt, dparam_dt = aug_dynamics(t, augmented_state)
        
        # Check shapes
        assert dy_dt.shape == (batch_size, state_dim)
        assert dadj_dt.shape == (batch_size, state_dim)
        assert dparam_dt.shape == flat_params.shape
        
        # Check all values are finite
        assert torch.isfinite(dy_dt).all()
        assert torch.isfinite(dadj_dt).all()
        assert torch.isfinite(dparam_dt).all()
    
    def test_augmented_dynamics_backward_negation(self):
        """Test that AugmentedDynamics negates state dynamics for backward integration."""
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
        t = 0.5
        y = torch.randn(batch_size, state_dim)
        adj_y = torch.randn(batch_size, state_dim)
        adj_params = torch.zeros_like(flat_params)
        
        # Compute forward dynamics directly
        dy_dt_forward, _, _ = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # Compute using AugmentedDynamics wrapper
        aug_dynamics = AugmentedDynamics(func, flat_params)
        augmented_state = (y, adj_y, adj_params)
        dy_dt_backward, _, _ = aug_dynamics(t, augmented_state)
        
        # Check that backward dynamics is negated
        assert torch.allclose(dy_dt_backward, -dy_dt_forward, rtol=1e-5), \
            "AugmentedDynamics should negate state dynamics for backward integration"
    
    def test_vjp_computation_linear_function(self):
        """Test VJP computation with a simple linear function."""
        # Create a linear dynamics function: dy/dt = A * y
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        class LinearFunc(nn.Module):
            def __init__(self):
                super().__init__()
                self.A = nn.Parameter(A.clone())
            
            def forward(self, y):
                return y @ self.A.t()
        
        net = LinearFunc()
        func = ODEFunc(net, time_dependent=False)
        
        # Flatten parameters
        flat_params, _ = flatten_params(func.parameters())
        
        # Create test inputs
        y = torch.tensor([[1.0, 0.0]])  # Single sample
        adj_y = torch.tensor([[1.0, 0.0]])  # Adjoint state
        t = 0.0
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # For linear system dy/dt = y @ A^T, we have:
        # - dy/dt = [1, 0] @ [[1, 3], [2, 4]] = [1, 3]
        # - ∂f/∂y = A (Jacobian of y @ A^T w.r.t. y is A)
        # - da/dt = -adj_y @ A = -[1, 0] @ [[1, 2], [3, 4]] = -[1, 2]
        
        expected_dy_dt = torch.tensor([[1.0, 3.0]])
        expected_dadj_dt = -torch.tensor([[1.0, 2.0]])
        
        assert torch.allclose(dy_dt, expected_dy_dt, rtol=1e-5), \
            f"Expected dy_dt={expected_dy_dt}, got {dy_dt}"
        assert torch.allclose(dadj_dt, expected_dadj_dt, rtol=1e-5), \
            f"Expected dadj_dt={expected_dadj_dt}, got {dadj_dt}"
    
    def test_parameter_gradients_nonzero(self):
        """Test that parameter gradients are non-zero when they should be."""
        # Create a simple dynamics function
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 2)
        )
        func = ODEFunc(net, time_dependent=False)
        
        # Flatten parameters
        flat_params, _ = flatten_params(func.parameters())
        
        # Create test inputs with non-zero adjoint
        batch_size = 5
        state_dim = 2
        t = 0.5
        y = torch.randn(batch_size, state_dim)
        adj_y = torch.ones(batch_size, state_dim)  # Non-zero adjoint
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # Parameter gradients should be non-zero (with high probability)
        # since the network has parameters and adjoint is non-zero
        assert not torch.allclose(dparam_dt, torch.zeros_like(dparam_dt)), \
            "Parameter gradients should be non-zero for non-zero adjoint"
    
    def test_no_parameters_case(self):
        """Test augmented dynamics with a function that has no parameters."""
        # Create a function with no learnable parameters
        class NoParamFunc(nn.Module):
            def forward(self, y):
                return y * 2.0  # Simple scaling, no parameters
        
        func = ODEFunc(NoParamFunc(), time_dependent=False)
        
        # Flatten parameters (should be empty)
        flat_params, _ = flatten_params(func.parameters())
        assert flat_params.numel() == 0, "Should have no parameters"
        
        # Create test inputs
        y = torch.randn(5, 2)
        adj_y = torch.randn(5, 2)
        t = 0.5
        
        # Compute augmented dynamics
        dy_dt, dadj_dt, dparam_dt = create_augmented_dynamics(
            func, flat_params, t, y, adj_y
        )
        
        # Should still work, with empty parameter gradients
        assert dy_dt.shape == y.shape
        assert dadj_dt.shape == y.shape
        assert dparam_dt.shape == flat_params.shape
        assert dparam_dt.numel() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
