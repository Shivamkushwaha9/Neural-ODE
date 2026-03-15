"""Unit tests for Continuous Normalizing Flow (CNF) layer."""

import pytest
import torch
import torch.nn as nn
from neural_ode.layers.cnf import CNF
from neural_ode.solvers.fixed_step import EulerSolver


class TestCNFBasic:
    """Basic functionality tests for CNF."""
    
    def test_cnf_initialization(self):
        """Test CNF can be initialized with different configurations."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        
        # Default initialization
        cnf = CNF(net)
        assert cnf.trace_estimator == 'hutchinson'
        assert cnf.hutchinson_samples == 1
        
        # Custom trace estimator
        cnf_exact = CNF(net, trace_estimator='exact')
        assert cnf_exact.trace_estimator == 'exact'
        
        # Custom solver
        solver = EulerSolver(step_size=0.1)
        cnf_euler = CNF(net, solver=solver)
        assert cnf_euler.solver == solver
    
    def test_cnf_invalid_trace_estimator(self):
        """Test CNF raises error for invalid trace estimator."""
        net = nn.Sequential(nn.Linear(2, 2))
        
        with pytest.raises(ValueError, match="trace_estimator must be"):
            CNF(net, trace_estimator='invalid')
    
    def test_cnf_forward_shape(self):
        """Test CNF forward pass returns correct shapes."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        cnf = CNF(net, trace_estimator='hutchinson')
        
        batch_size = 10
        dim = 2
        x = torch.randn(batch_size, dim)
        
        # Forward transformation
        z, log_det = cnf(x, reverse=False)
        
        assert z.shape == (batch_size, dim)
        assert log_det.shape == (batch_size,)
        assert torch.isfinite(z).all()
        assert torch.isfinite(log_det).all()
    
    def test_cnf_reverse_shape(self):
        """Test CNF reverse pass returns correct shapes."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        cnf = CNF(net, trace_estimator='hutchinson')
        
        batch_size = 10
        dim = 2
        z = torch.randn(batch_size, dim)
        
        # Reverse transformation
        x, log_det = cnf(z, reverse=True)
        
        assert x.shape == (batch_size, dim)
        assert log_det.shape == (batch_size,)
        assert torch.isfinite(x).all()
        assert torch.isfinite(log_det).all()
    
    def test_cnf_log_prob(self):
        """Test CNF log_prob computation."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        cnf = CNF(net, trace_estimator='hutchinson')
        
        batch_size = 10
        dim = 2
        x = torch.randn(batch_size, dim)
        
        # Create base distribution
        base_dist = torch.distributions.Normal(
            torch.zeros(dim), torch.ones(dim)
        )
        
        # Compute log probability
        log_prob = cnf.log_prob(x, base_dist)
        
        assert log_prob.shape == (batch_size,)
        assert torch.isfinite(log_prob).all()
    
    def test_cnf_sample(self):
        """Test CNF sampling."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        cnf = CNF(net, trace_estimator='hutchinson')
        
        num_samples = 20
        dim = 2
        
        # Create base distribution
        base_dist = torch.distributions.Normal(
            torch.zeros(dim), torch.ones(dim)
        )
        
        # Generate samples
        samples = cnf.sample(num_samples, base_dist)
        
        assert samples.shape == (num_samples, dim)
        assert torch.isfinite(samples).all()
    
    def test_cnf_gradient_flow(self):
        """Test gradients flow through CNF."""
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        cnf = CNF(net, trace_estimator='hutchinson')
        
        x = torch.randn(5, 2)
        base_dist = torch.distributions.Normal(
            torch.zeros(2), torch.ones(2)
        )
        
        # Compute loss
        log_prob = cnf.log_prob(x, base_dist)
        loss = -log_prob.mean()
        
        # Backpropagate
        loss.backward()
        
        # Check gradients exist and are finite
        for param in cnf.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


class TestCNFTraceEstimators:
    """Test different trace estimation methods."""
    
    def test_hutchinson_vs_exact_similar(self):
        """Test Hutchinson and exact trace give similar results."""
        # Use a simple linear network for predictable behavior
        net = nn.Linear(2, 2)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        x = torch.randn(5, 2)
        
        # CNF with Hutchinson estimator
        cnf_hutch = CNF(net, trace_estimator='hutchinson', hutchinson_samples=10)
        
        # CNF with exact trace - create new network with same weights
        net_exact = nn.Linear(2, 2)
        net_exact.load_state_dict(net.state_dict())
        cnf_exact = CNF(net_exact, trace_estimator='exact')
        
        # Forward pass
        torch.manual_seed(42)
        z_hutch, log_det_hutch = cnf_hutch(x, reverse=False)
        
        torch.manual_seed(42)
        z_exact, log_det_exact = cnf_exact(x, reverse=False)
        
        # States should be very similar (same dynamics)
        assert torch.allclose(z_hutch, z_exact, rtol=1e-2, atol=1e-2)
        
        # Log-determinants should be similar (stochastic vs exact)
        # Allow larger tolerance due to stochastic estimation
        assert torch.allclose(log_det_hutch, log_det_exact, rtol=0.5, atol=0.5)


class TestCNFIntegration:
    """Integration tests for CNF with different configurations."""
    
    def test_cnf_with_euler_solver(self):
        """Test CNF works with Euler solver."""
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        solver = EulerSolver(step_size=0.1)
        cnf = CNF(net, solver=solver)
        
        x = torch.randn(5, 2)
        z, log_det = cnf(x, reverse=False)
        
        assert z.shape == (5, 2)
        assert log_det.shape == (5,)
    
    def test_cnf_custom_time_span(self):
        """Test CNF with custom integration time."""
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        cnf = CNF(net)
        
        x = torch.randn(5, 2)
        t = torch.tensor([0.0, 2.0])
        
        z, log_det = cnf(x, reverse=False, t=t)
        
        assert z.shape == (5, 2)
        assert log_det.shape == (5,)
    
    def test_cnf_nfe_tracking(self):
        """Test NFE tracking in CNF."""
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        cnf = CNF(net)
        
        # Reset counters
        cnf.reset_nfe()
        
        x = torch.randn(5, 2)
        z, log_det = cnf(x, reverse=False)
        
        nfe = cnf.get_nfe()
        assert nfe['solver_nfe'] > 0  # Should have evaluated function


class TestCNFNumericalStability:
    """Test numerical stability of CNF."""
    
    def test_cnf_finite_outputs(self):
        """Test CNF produces finite outputs for reasonable inputs."""
        net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 3)
        )
        cnf = CNF(net)
        
        # Test with various input scales
        for scale in [0.1, 1.0, 10.0]:
            x = torch.randn(10, 3) * scale
            z, log_det = cnf(x, reverse=False)
            
            assert torch.isfinite(z).all(), f"Non-finite z for scale {scale}"
            assert torch.isfinite(log_det).all(), f"Non-finite log_det for scale {scale}"
    
    def test_cnf_batch_consistency(self):
        """Test CNF produces consistent results for different batch sizes."""
        net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        # Use exact trace for deterministic results
        cnf = CNF(net, trace_estimator='exact')
        
        # Single sample
        x_single = torch.randn(1, 2)
        z_single, log_det_single = cnf(x_single, reverse=False)
        
        # Same sample in a batch
        x_batch = x_single.repeat(5, 1)
        z_batch, log_det_batch = cnf(x_batch, reverse=False)
        
        # Results should be identical for all batch elements
        assert torch.allclose(z_batch[0], z_single[0], rtol=1e-5)
        assert torch.allclose(log_det_batch[0], log_det_single[0], rtol=1e-5)
        
        # All batch elements should be identical
        for i in range(1, 5):
            assert torch.allclose(z_batch[i], z_batch[0], rtol=1e-5)
            assert torch.allclose(log_det_batch[i], log_det_batch[0], rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
