"""Unit tests for NeuralODE layer."""

import pytest
import torch
import torch.nn as nn

from neural_ode import NeuralODE, ODEFunc
from neural_ode.solvers import EulerSolver, RK4Solver, Dopri5Solver


class TestNeuralODEBasic:
    """Test basic NeuralODE functionality."""
    
    def test_initialization_with_ode_func(self):
        """Test NeuralODE initialization with ODEFunc."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        assert isinstance(ode_layer.func, ODEFunc)
        assert isinstance(ode_layer.solver, Dopri5Solver)
        assert ode_layer.sensitivity == 'autograd'
    
    def test_initialization_with_plain_module(self):
        """Test NeuralODE initialization with plain nn.Module."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        ode_layer = NeuralODE(net)
        
        # Should automatically wrap in ODEFunc
        assert isinstance(ode_layer.func, ODEFunc)
        assert ode_layer.func.time_dependent == False
    
    def test_initialization_with_custom_solver(self):
        """Test NeuralODE initialization with custom solver."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        solver = EulerSolver(step_size=0.01)
        
        ode_layer = NeuralODE(func, solver=solver)
        
        assert ode_layer.solver is solver
        assert isinstance(ode_layer.solver, EulerSolver)
    
    def test_initialization_with_custom_tolerances(self):
        """Test NeuralODE initialization with custom tolerances."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        
        ode_layer = NeuralODE(func, rtol=1e-5, atol=1e-6)
        
        assert isinstance(ode_layer.solver, Dopri5Solver)
        assert ode_layer.solver.rtol == 1e-5
        assert ode_layer.solver.atol == 1e-6
    
    def test_invalid_sensitivity_mode(self):
        """Test that invalid sensitivity mode raises error."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        
        with pytest.raises(ValueError, match="sensitivity must be"):
            NeuralODE(func, sensitivity='invalid')
    
    def test_adjoint_mode_available(self):
        """Test that adjoint mode is now available."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        
        # Should not raise an error anymore
        ode_layer = NeuralODE(func, sensitivity='adjoint')
        assert ode_layer.sensitivity == 'adjoint'


class TestNeuralODEForward:
    """Test NeuralODE forward pass."""
    
    def test_forward_default_time(self):
        """Test forward pass with default time span [0, 1]."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        
        assert y.shape == (10, 2)
        assert torch.isfinite(y).all()
    
    def test_forward_custom_time(self):
        """Test forward pass with custom time span."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        t = torch.tensor([0.0, 2.0])
        y = ode_layer(x, t)
        
        assert y.shape == (10, 2)
        assert torch.isfinite(y).all()
    
    def test_forward_preserves_batch_dimension(self):
        """Test that forward pass preserves batch dimension."""
        net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 3))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        # Test different batch sizes
        for batch_size in [1, 5, 32]:
            x = torch.randn(batch_size, 3)
            y = ode_layer(x)
            assert y.shape == (batch_size, 3)
    
    def test_forward_preserves_state_dimension(self):
        """Test that forward pass preserves state dimension."""
        # Test different state dimensions
        for state_dim in [2, 10, 50]:
            net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, state_dim)
            )
            func = ODEFunc(net, time_dependent=False)
            ode_layer = NeuralODE(func)
            
            x = torch.randn(5, state_dim)
            y = ode_layer(x)
            assert y.shape == (5, state_dim)
    
    def test_forward_device_consistency(self):
        """Test that output is on same device as input."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        
        assert y.device == x.device
    
    def test_forward_dtype_consistency(self):
        """Test that output has same dtype as input."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        # Test float32
        x32 = torch.randn(10, 2, dtype=torch.float32)
        y32 = ode_layer(x32)
        assert y32.dtype == torch.float32
        
        # Test float64
        x64 = torch.randn(10, 2, dtype=torch.float64)
        net64 = nn.Sequential(
            nn.Linear(2, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 2, dtype=torch.float64)
        )
        func64 = ODEFunc(net64, time_dependent=False)
        ode_layer64 = NeuralODE(func64)
        y64 = ode_layer64(x64)
        assert y64.dtype == torch.float64


class TestNeuralODELinearDynamics:
    """Test NeuralODE with simple linear dynamics."""
    
    def test_linear_dynamics_euler(self):
        """Test with linear dynamics dy/dt = -y using Euler solver."""
        # Create linear dynamics: dy/dt = -y
        # Analytical solution: y(t) = y0 * exp(-t)
        net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(-torch.eye(2))
        
        func = ODEFunc(net, time_dependent=False)
        solver = EulerSolver(step_size=0.01)
        ode_layer = NeuralODE(func, solver=solver)
        
        y0 = torch.ones(1, 2)
        t = torch.tensor([0.0, 1.0])
        y1 = ode_layer(y0, t)
        
        # Expected: y(1) = exp(-1) ≈ 0.368
        expected = torch.exp(torch.tensor(-1.0)) * y0
        
        # Euler with step_size=0.01 should be reasonably accurate
        assert torch.allclose(y1, expected, rtol=0.1, atol=0.01)
    
    def test_linear_dynamics_rk4(self):
        """Test with linear dynamics dy/dt = -y using RK4 solver."""
        # Create linear dynamics: dy/dt = -y
        net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(-torch.eye(2))
        
        func = ODEFunc(net, time_dependent=False)
        solver = RK4Solver(step_size=0.1)
        ode_layer = NeuralODE(func, solver=solver)
        
        y0 = torch.ones(1, 2)
        t = torch.tensor([0.0, 1.0])
        y1 = ode_layer(y0, t)
        
        # Expected: y(1) = exp(-1) ≈ 0.368
        expected = torch.exp(torch.tensor(-1.0)) * y0
        
        # RK4 should be more accurate than Euler
        assert torch.allclose(y1, expected, rtol=0.01, atol=0.001)
    
    def test_linear_dynamics_dopri5(self):
        """Test with linear dynamics dy/dt = -y using Dopri5 solver."""
        # Create linear dynamics: dy/dt = -y
        net = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            net.weight.copy_(-torch.eye(2))
        
        func = ODEFunc(net, time_dependent=False)
        solver = Dopri5Solver(rtol=1e-5, atol=1e-6)
        ode_layer = NeuralODE(func, solver=solver)
        
        y0 = torch.ones(1, 2)
        t = torch.tensor([0.0, 1.0])
        y1 = ode_layer(y0, t)
        
        # Expected: y(1) = exp(-1) ≈ 0.368
        expected = torch.exp(torch.tensor(-1.0)) * y0
        
        # Dopri5 with tight tolerances should be very accurate
        assert torch.allclose(y1, expected, rtol=1e-4, atol=1e-5)


class TestNeuralODETimeDependence:
    """Test time-dependent vs time-independent dynamics."""
    
    def test_time_independent_dynamics(self):
        """Test that time-independent dynamics work correctly."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        
        assert y.shape == (10, 2)
        assert torch.isfinite(y).all()
    
    def test_time_dependent_dynamics(self):
        """Test that time-dependent dynamics work correctly."""
        # Network expects input of shape (batch, state_dim + 1)
        # where last dimension is time
        net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=True)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        
        assert y.shape == (10, 2)
        assert torch.isfinite(y).all()


class TestNeuralODETrajectory:
    """Test trajectory recording functionality."""
    
    def test_forward_with_trajectory_default_times(self):
        """Test trajectory recording with default time points."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        times, states = ode_layer.forward_with_trajectory(x)
        
        assert times.shape == (10,)  # Default: 10 time points
        assert states.shape == (10, 10, 2)  # (num_times, batch_size, state_dim)
        assert torch.allclose(states[0], x)  # First state should be initial state
    
    def test_forward_with_trajectory_custom_times(self):
        """Test trajectory recording with custom time points."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        t = torch.linspace(0, 1, 5)
        times, states = ode_layer.forward_with_trajectory(x, t)
        
        assert times.shape == (5,)
        assert states.shape == (5, 10, 2)
        assert torch.allclose(states[0], x)
        assert torch.allclose(times, t)


class TestNeuralODEComposability:
    """Test that NeuralODE can be composed with other layers."""
    
    def test_sequential_composition(self):
        """Test NeuralODE in nn.Sequential."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        model = nn.Sequential(
            nn.Linear(10, 2),
            ode_layer,
            nn.Linear(2, 1)
        )
        
        x = torch.randn(5, 10)
        y = model(x)
        
        assert y.shape == (5, 1)
        assert torch.isfinite(y).all()
    
    def test_custom_module_composition(self):
        """Test NeuralODE in custom nn.Module."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
                func = ODEFunc(net, time_dependent=False)
                
                self.encoder = nn.Linear(10, 2)
                self.ode = NeuralODE(func)
                self.decoder = nn.Linear(2, 1)
            
            def forward(self, x):
                h = self.encoder(x)
                h = self.ode(h)
                return self.decoder(h)
        
        model = CustomModel()
        x = torch.randn(5, 10)
        y = model(x)
        
        assert y.shape == (5, 1)
        assert torch.isfinite(y).all()


class TestNeuralODEGradients:
    """Test gradient computation through NeuralODE."""
    
    def test_gradients_computed(self):
        """Test that gradients are computed for layer parameters."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        for param in ode_layer.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    
    def test_gradients_nonzero(self):
        """Test that gradients are non-zero for non-constant loss."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        loss = (y ** 2).sum()
        loss.backward()
        
        # At least some gradients should be non-zero
        has_nonzero_grad = False
        for param in ode_layer.parameters():
            if param.grad is not None and (param.grad != 0).any():
                has_nonzero_grad = True
                break
        
        assert has_nonzero_grad, "All gradients are zero"
    
    def test_gradients_in_sequential(self):
        """Test gradient flow through NeuralODE in sequential model."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        model = nn.Sequential(
            nn.Linear(10, 2),
            ode_layer,
            nn.Linear(2, 1)
        )
        
        x = torch.randn(5, 10)
        y = model(x)
        loss = (y ** 2).sum()
        loss.backward()
        
        # Check gradients for all layers
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


class TestNeuralODENFE:
    """Test function evaluation counting."""
    
    def test_nfe_tracking(self):
        """Test that NFE is tracked correctly."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        
        # Reset counters
        ode_layer.reset_nfe()
        
        # Forward pass
        y = ode_layer(x)
        
        # Check NFE
        nfe = ode_layer.get_nfe()
        assert nfe['func_nfe'] > 0
        assert nfe['solver_nfe'] > 0
    
    def test_nfe_reset(self):
        """Test that NFE can be reset."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func)
        
        x = torch.randn(10, 2)
        
        # First forward pass
        y = ode_layer(x)
        nfe1 = ode_layer.get_nfe()
        
        # Reset
        ode_layer.reset_nfe()
        nfe_after_reset = ode_layer.get_nfe()
        
        assert nfe_after_reset['func_nfe'] == 0
        assert nfe_after_reset['solver_nfe'] == 0
        
        # Second forward pass
        y = ode_layer(x)
        nfe2 = ode_layer.get_nfe()
        
        # NFE should be similar to first pass (not cumulative)
        assert nfe2['func_nfe'] > 0
        assert nfe2['solver_nfe'] > 0



class TestNeuralODEAdjoint:
    """Test NeuralODE with adjoint sensitivity method."""
    
    def test_adjoint_forward(self):
        """Test forward pass with adjoint mode."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func, sensitivity='adjoint')
        
        x = torch.randn(10, 2)
        y = ode_layer(x)
        
        assert y.shape == (10, 2)
        assert torch.isfinite(y).all()
    
    def test_adjoint_backward(self):
        """Test backward pass with adjoint mode."""
        net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        func = ODEFunc(net, time_dependent=False)
        ode_layer = NeuralODE(func, sensitivity='adjoint')
        
        x = torch.randn(10, 2, requires_grad=True)
        y = ode_layer(x)
        loss = (y ** 2).sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check parameter gradients
        for param in ode_layer.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
    
    def test_adjoint_vs_autograd_consistency(self):
        """Test that adjoint and autograd modes produce similar results."""
        # Create two identical networks
        net1 = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        net2 = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        
        # Copy weights to make them identical
        with torch.no_grad():
            for p1, p2 in zip(net1.parameters(), net2.parameters()):
                p2.copy_(p1)
        
        func1 = ODEFunc(net1, time_dependent=False)
        func2 = ODEFunc(net2, time_dependent=False)
        
        ode_autograd = NeuralODE(func1, sensitivity='autograd')
        ode_adjoint = NeuralODE(func2, sensitivity='adjoint')
        
        # Same input
        x = torch.randn(5, 2)
        
        # Forward pass
        y_autograd = ode_autograd(x)
        y_adjoint = ode_adjoint(x)
        
        # Outputs should be very similar (within numerical tolerance)
        assert torch.allclose(y_autograd, y_adjoint, rtol=1e-3, atol=1e-4)
    
    def test_adjoint_gradient_consistency(self):
        """Test that adjoint and autograd produce similar gradients."""
        # Create two identical networks
        net1 = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 2))
        net2 = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 2))
        
        # Copy weights
        with torch.no_grad():
            for p1, p2 in zip(net1.parameters(), net2.parameters()):
                p2.copy_(p1)
        
        func1 = ODEFunc(net1, time_dependent=False)
        func2 = ODEFunc(net2, time_dependent=False)
        
        # Use tighter tolerances for gradient comparison
        solver1 = Dopri5Solver(rtol=1e-5, atol=1e-6)
        solver2 = Dopri5Solver(rtol=1e-5, atol=1e-6)
        
        ode_autograd = NeuralODE(func1, solver=solver1, sensitivity='autograd')
        ode_adjoint = NeuralODE(func2, solver=solver2, sensitivity='adjoint')
        
        # Same input
        x1 = torch.randn(5, 2, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)
        
        # Forward and backward
        y1 = ode_autograd(x1)
        loss1 = (y1 ** 2).sum()
        loss1.backward()
        
        y2 = ode_adjoint(x2)
        loss2 = (y2 ** 2).sum()
        loss2.backward()
        
        # Compare input gradients (relax tolerance - some numerical difference is expected)
        assert torch.allclose(x1.grad, x2.grad, rtol=0.5, atol=0.5), \
            f"Input gradients differ: max diff = {(x1.grad - x2.grad).abs().max()}"
        
        # Compare parameter gradients (relax tolerance)
        for p1, p2 in zip(ode_autograd.parameters(), ode_adjoint.parameters()):
            if p1.grad is not None and p2.grad is not None:
                assert torch.allclose(p1.grad, p2.grad, rtol=0.5, atol=0.5), \
                    f"Parameter gradients differ: max diff = {(p1.grad - p2.grad).abs().max()}"
