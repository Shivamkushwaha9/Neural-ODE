"""Unit tests for trace estimation utilities."""

import pytest
import torch
import torch.nn as nn
from neural_ode.utils.trace import hutchinson_trace, exact_trace


class TestHutchinsonTrace:
    """Tests for hutchinson_trace function."""
    
    def test_linear_function_trace(self):
        """Test Hutchinson estimator on linear function with known trace."""
        # For linear function f(y) = Ay, trace(∂f/∂y) = trace(A)
        batch_size, dim = 4, 3
        
        # Create a simple linear transformation
        A = torch.randn(dim, dim)
        expected_trace = torch.trace(A)
        
        # Create input
        y = torch.randn(batch_size, dim, requires_grad=True)
        
        # Compute output
        dy_dt = y @ A.t()
        
        # Estimate trace with many samples for accuracy
        trace_est = hutchinson_trace(dy_dt, y, num_samples=100)
        
        # All batch elements should have same trace (function is same for all)
        assert trace_est.shape == (batch_size,)
        
        # Check that estimate is close to expected (with some tolerance due to randomness)
        mean_trace = trace_est.mean().item()
        assert abs(mean_trace - expected_trace.item()) < 0.5
    
    def test_identity_function_trace(self):
        """Test on identity function f(y) = y, which has trace = dim."""
        batch_size, dim = 8, 5
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y  # Identity function
        
        trace_est = hutchinson_trace(dy_dt, y, num_samples=50)
        
        # Trace of identity matrix is the dimension
        expected_trace = float(dim)
        mean_trace = trace_est.mean().item()
        
        assert abs(mean_trace - expected_trace) < 0.5
    
    def test_zero_function_trace(self):
        """Test on zero function f(y) = 0, which has trace = 0."""
        batch_size, dim = 4, 6
        
        # Create a function that returns zeros but has a grad_fn
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y * 0.0  # This creates a grad_fn unlike torch.zeros_like
        
        trace_est = hutchinson_trace(dy_dt, y, num_samples=10)
        
        # Trace should be zero
        assert torch.allclose(trace_est, torch.zeros(batch_size), atol=1e-6)
    
    def test_diagonal_matrix_trace(self):
        """Test on diagonal matrix, where trace equals sum of diagonal elements."""
        batch_size, dim = 4, 5
        
        # Create diagonal matrix
        diag_values = torch.randn(dim)
        A = torch.diag(diag_values)
        expected_trace = diag_values.sum()
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y @ A.t()
        
        trace_est = hutchinson_trace(dy_dt, y, num_samples=100)
        mean_trace = trace_est.mean().item()
        
        assert abs(mean_trace - expected_trace.item()) < 0.5
    
    def test_single_sample(self):
        """Test with single sample (default behavior)."""
        batch_size, dim = 4, 3
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y
        
        trace_est = hutchinson_trace(dy_dt, y, num_samples=1)
        
        assert trace_est.shape == (batch_size,)
        assert torch.isfinite(trace_est).all()
    
    def test_multiple_samples_reduces_variance(self):
        """Test that more samples give more consistent estimates."""
        batch_size, dim = 4, 10
        
        A = torch.randn(dim, dim)
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y @ A.t()
        
        # Run multiple times with few samples
        estimates_few = []
        for _ in range(10):
            y_new = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt_new = y_new @ A.t()
            trace = hutchinson_trace(dy_dt_new, y_new, num_samples=1)
            estimates_few.append(trace.mean().item())
        
        # Run multiple times with many samples
        estimates_many = []
        for _ in range(10):
            y_new = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt_new = y_new @ A.t()
            trace = hutchinson_trace(dy_dt_new, y_new, num_samples=50)
            estimates_many.append(trace.mean().item())
        
        # Variance should be lower with more samples
        var_few = torch.tensor(estimates_few).var().item()
        var_many = torch.tensor(estimates_many).var().item()
        
        assert var_many < var_few
    
    def test_custom_noise(self):
        """Test with pre-generated noise for reproducibility."""
        batch_size, dim = 4, 3
        num_samples = 5
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y
        
        # Generate custom noise
        noise = torch.randn(num_samples, batch_size, dim)
        
        # Should give same result with same noise
        trace1 = hutchinson_trace(dy_dt, y, num_samples=num_samples, noise=noise)
        
        # Need to recompute with fresh graph
        y2 = y.detach().clone().requires_grad_(True)
        dy_dt2 = y2
        trace2 = hutchinson_trace(dy_dt2, y2, num_samples=num_samples, noise=noise)
        
        assert torch.allclose(trace1, trace2, rtol=1e-5)
    
    def test_requires_grad_error(self):
        """Should raise error if input doesn't require gradients."""
        y = torch.randn(4, 3, requires_grad=False)
        dy_dt = y
        
        with pytest.raises(ValueError, match="requires_grad=True"):
            hutchinson_trace(dy_dt, y)
    
    def test_batch_dimension_preserved(self):
        """Test that batch dimension is preserved in output."""
        for batch_size in [1, 4, 16]:
            dim = 5
            y = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt = y
            
            trace = hutchinson_trace(dy_dt, y, num_samples=10)
            assert trace.shape == (batch_size,)
    
    def test_gradient_flow(self):
        """Test that gradients flow through trace computation."""
        batch_size, dim = 4, 3
        
        # Create a simple network
        net = nn.Linear(dim, dim)
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = net(y)
        
        trace = hutchinson_trace(dy_dt, y, num_samples=10)
        loss = trace.sum()
        
        # Check that the computation graph is properly constructed
        assert loss.requires_grad
        assert loss.grad_fn is not None
        
        # Perform backward pass
        loss.backward()
        
        # The trace computation creates a complex graph, so we just verify
        # that backward() completes without error


class TestExactTrace:
    """Tests for exact_trace function."""
    
    def test_linear_function_trace(self):
        """Test exact trace on linear function with known trace."""
        batch_size, dim = 4, 3
        
        # Create a simple linear transformation
        A = torch.randn(dim, dim)
        expected_trace = torch.trace(A)
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y @ A.t()
        
        trace = exact_trace(dy_dt, y)
        
        # All batch elements should have same trace
        assert trace.shape == (batch_size,)
        
        # Check exact match (no stochastic approximation)
        assert torch.allclose(trace, torch.full((batch_size,), expected_trace.item()), rtol=1e-5)
    
    def test_identity_function_trace(self):
        """Test on identity function f(y) = y, which has trace = dim."""
        batch_size, dim = 8, 5
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y
        
        trace = exact_trace(dy_dt, y)
        
        # Trace of identity matrix is the dimension
        expected_trace = float(dim)
        
        assert torch.allclose(trace, torch.full((batch_size,), expected_trace), rtol=1e-5)
    
    def test_zero_function_trace(self):
        """Test on zero function f(y) = 0, which has trace = 0."""
        batch_size, dim = 4, 6
        
        # Create a function that returns zeros but has a grad_fn
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y * 0.0  # This creates a grad_fn unlike torch.zeros_like
        
        trace = exact_trace(dy_dt, y)
        
        # Trace should be exactly zero
        assert torch.allclose(trace, torch.zeros(batch_size), atol=1e-6)
    
    def test_diagonal_matrix_trace(self):
        """Test on diagonal matrix, where trace equals sum of diagonal elements."""
        batch_size, dim = 4, 5
        
        # Create diagonal matrix
        diag_values = torch.randn(dim)
        A = torch.diag(diag_values)
        expected_trace = diag_values.sum()
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y @ A.t()
        
        trace = exact_trace(dy_dt, y)
        
        assert torch.allclose(trace, torch.full((batch_size,), expected_trace.item()), rtol=1e-5)
    
    def test_requires_grad_error(self):
        """Should raise error if input doesn't require gradients."""
        y = torch.randn(4, 3, requires_grad=False)
        dy_dt = y
        
        with pytest.raises(ValueError, match="requires_grad=True"):
            exact_trace(dy_dt, y)
    
    def test_batch_dimension_preserved(self):
        """Test that batch dimension is preserved in output."""
        for batch_size in [1, 4, 16]:
            dim = 5
            y = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt = y
            
            trace = exact_trace(dy_dt, y)
            assert trace.shape == (batch_size,)
    
    def test_gradient_flow(self):
        """Test that gradients flow through trace computation."""
        batch_size, dim = 4, 3
        
        # Create a simple network
        net = nn.Linear(dim, dim)
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = net(y)
        
        trace = exact_trace(dy_dt, y)
        loss = trace.sum()
        
        # Check that the computation graph is properly constructed
        assert loss.requires_grad
        assert loss.grad_fn is not None
        
        # Perform backward pass
        loss.backward()
        
        # The trace computation creates a complex graph, so we just verify
        # that backward() completes without error
    
    def test_nonlinear_function(self):
        """Test on nonlinear function."""
        batch_size, dim = 4, 3
        
        # Create nonlinear function: f(y) = y^2 (element-wise)
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y ** 2
        
        # For f(y) = y^2, ∂f_i/∂y_i = 2*y_i
        # So trace = sum(2*y_i) for each batch element
        expected_trace = 2 * y.sum(dim=1)
        
        trace = exact_trace(dy_dt, y)
        
        assert torch.allclose(trace, expected_trace, rtol=1e-5)


class TestTraceComparison:
    """Tests comparing Hutchinson and exact trace methods."""
    
    def test_hutchinson_vs_exact_linear(self):
        """Compare Hutchinson and exact methods on linear function."""
        batch_size, dim = 8, 10
        
        A = torch.randn(dim, dim)
        expected_trace = torch.trace(A).item()
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y @ A.t()
        
        # Compute exact trace
        trace_exact = exact_trace(dy_dt, y)
        
        # All batch elements should have the same trace (linear function)
        assert torch.allclose(trace_exact, torch.full((batch_size,), expected_trace), rtol=1e-4)
        
        # Compute Hutchinson estimate with many samples
        y2 = y.detach().clone().requires_grad_(True)
        dy_dt2 = y2 @ A.t()
        trace_hutchinson = hutchinson_trace(dy_dt2, y2, num_samples=200)
        
        # Should be close with many samples
        mean_hutchinson = trace_hutchinson.mean().item()
        assert abs(mean_hutchinson - expected_trace) < 0.5
    
    def test_hutchinson_vs_exact_identity(self):
        """Compare methods on identity function."""
        batch_size, dim = 4, 8
        
        y = torch.randn(batch_size, dim, requires_grad=True)
        dy_dt = y
        
        trace_exact = exact_trace(dy_dt, y)
        
        y2 = y.detach().clone().requires_grad_(True)
        dy_dt2 = y2
        trace_hutchinson = hutchinson_trace(dy_dt2, y2, num_samples=100)
        
        # Should be very close for identity
        assert torch.allclose(trace_hutchinson, trace_exact, rtol=0.1, atol=0.5)
    
    def test_exact_is_deterministic(self):
        """Test that exact trace gives same result every time."""
        batch_size, dim = 4, 5
        
        A = torch.randn(dim, dim)
        
        traces = []
        for _ in range(5):
            y = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt = y @ A.t()
            trace = exact_trace(dy_dt, y)
            traces.append(trace.mean().item())
        
        # All should be the same (deterministic)
        traces_tensor = torch.tensor(traces)
        assert traces_tensor.std().item() < 1e-5
    
    def test_hutchinson_has_variance(self):
        """Test that Hutchinson trace has variance across runs."""
        batch_size, dim = 4, 10
        
        A = torch.randn(dim, dim)
        
        traces = []
        for _ in range(10):
            y = torch.randn(batch_size, dim, requires_grad=True)
            dy_dt = y @ A.t()
            trace = hutchinson_trace(dy_dt, y, num_samples=1)
            traces.append(trace.mean().item())
        
        # Should have some variance (stochastic)
        traces_tensor = torch.tensor(traces)
        assert traces_tensor.std().item() > 0.01
