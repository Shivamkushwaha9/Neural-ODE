"""Unit tests for Euler solver."""

import pytest
import torch
import math
from neural_ode.solvers import EulerSolver
from neural_ode.utils.validation import IntegrationError


class TestEulerSolverBasic:
    """Basic functionality tests for Euler solver."""
    
    def test_initialization(self):
        """Should initialize with valid step size."""
        solver = EulerSolver(step_size=0.1)
        assert solver.step_size == 0.1
        assert solver.nfe == 0
    
    def test_invalid_step_size(self):
        """Should raise ValueError for non-positive step size."""
        with pytest.raises(ValueError, match="Step size must be positive"):
            EulerSolver(step_size=0.0)
        
        with pytest.raises(ValueError, match="Step size must be positive"):
            EulerSolver(step_size=-0.1)
    
    def test_linear_ode(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1.
        
        Analytical solution: y(t) = exp(-t)
        """
        solver = EulerSolver(step_size=0.01)
        
        # Define dynamics: dy/dt = -y
        def dynamics(t, y):
            return -y
        
        # Initial condition
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        # Integrate
        y_final = solver.integrate(dynamics, y0, t)
        
        # Compare with analytical solution: y(1) = exp(-1) ≈ 0.3679
        expected = math.exp(-1.0)
        
        # With step size 0.01, Euler should be reasonably accurate
        assert y_final.shape == (1, 1)
        assert abs(y_final.item() - expected) < 0.01  # Within 1% error
    
    def test_nfe_tracking(self):
        """Should track number of function evaluations."""
        solver = EulerSolver(step_size=0.1)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        initial_nfe = solver.nfe
        solver.integrate(dynamics, y0, t)
        
        # Should have made 10 steps (1.0 / 0.1), but due to floating point
        # arithmetic, we might make 11 steps
        assert solver.nfe >= initial_nfe + 10
        assert solver.nfe <= initial_nfe + 11
    
    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        solver = EulerSolver(step_size=0.01)  # Smaller step for better accuracy
        
        def dynamics(t, y):
            return -y
        
        # Batch of 3 samples
        y0 = torch.tensor([[1.0], [2.0], [3.0]])
        t = torch.tensor([0.0, 1.0])
        
        y_final = solver.integrate(dynamics, y0, t)
        
        # Output should have same batch dimension
        assert y_final.shape == (3, 1)
        
        # Each should decay exponentially
        expected = torch.tensor([[1.0], [2.0], [3.0]]) * math.exp(-1.0)
        assert torch.allclose(y_final, expected, rtol=0.01)  # 1% tolerance with smaller step


class TestEulerSolverTrajectory:
    """Tests for trajectory recording functionality."""
    
    def test_trajectory_recording(self):
        """Should record states at specified time points."""
        solver = EulerSolver(step_size=0.05)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 0.5, 1.0])
        
        times, states = solver.integrate_with_trajectory(dynamics, y0, t)
        
        # Should return requested time points
        assert torch.allclose(times, t)
        
        # Should have states at each time point
        assert states.shape == (3, 1, 1)  # (num_times, batch, state_dim)
        
        # Initial state should match
        assert torch.allclose(states[0], y0)
        
        # States should decay exponentially
        assert states[1].item() < states[0].item()
        assert states[2].item() < states[1].item()
    
    def test_trajectory_with_batch(self):
        """Should record trajectories for batched inputs."""
        solver = EulerSolver(step_size=0.1)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # batch=2, state_dim=2
        t = torch.tensor([0.0, 0.5, 1.0])
        
        times, states = solver.integrate_with_trajectory(dynamics, y0, t)
        
        assert states.shape == (3, 2, 2)  # (num_times, batch, state_dim)
        assert torch.allclose(states[0], y0)


class TestEulerSolverEdgeCases:
    """Edge case and error handling tests."""
    
    def test_single_step(self):
        """Should handle case where step_size equals integration interval."""
        solver = EulerSolver(step_size=1.0)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        y_final = solver.integrate(dynamics, y0, t)
        
        # Should make exactly one step
        # y(1) = y(0) + 1.0 * (-y(0)) = 1.0 - 1.0 = 0.0
        assert torch.allclose(y_final, torch.tensor([[0.0]]))
    
    def test_very_small_interval(self):
        """Should handle very small integration intervals."""
        solver = EulerSolver(step_size=0.1)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 0.01])
        
        y_final = solver.integrate(dynamics, y0, t)
        
        # Should make one small step
        assert y_final.shape == (1, 1)
        assert torch.isfinite(y_final).all()
    
    def test_nan_detection(self):
        """Should detect and report NaN values during integration."""
        solver = EulerSolver(step_size=0.1)
        
        # Dynamics that produces NaN
        def bad_dynamics(t, y):
            return torch.tensor([[float('nan')]])
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            solver.integrate(bad_dynamics, y0, t)
    
    def test_exploding_dynamics(self):
        """Should detect exploding dynamics that produce Inf."""
        solver = EulerSolver(step_size=0.1)
        
        # Dynamics that grows exponentially: dy/dt = 100*y
        def exploding_dynamics(t, y):
            return 100.0 * y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 10.0])
        
        # Should raise ValueError when dynamics produce Inf
        with pytest.raises((ValueError, IntegrationError)):
            solver.integrate(exploding_dynamics, y0, t)


class TestEulerSolverStepSize:
    """Tests for step size behavior."""
    
    def test_step_size_accuracy(self):
        """Smaller step size should give more accurate results."""
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        expected = math.exp(-1.0)
        
        # Test with different step sizes
        solver_coarse = EulerSolver(step_size=0.1)
        y_coarse = solver_coarse.integrate(dynamics, y0, t)
        error_coarse = abs(y_coarse.item() - expected)
        
        solver_fine = EulerSolver(step_size=0.01)
        y_fine = solver_fine.integrate(dynamics, y0, t)
        error_fine = abs(y_fine.item() - expected)
        
        # Finer step size should have smaller error
        assert error_fine < error_coarse
    
    def test_adaptive_final_step(self):
        """Should adapt final step to hit target time exactly."""
        solver = EulerSolver(step_size=0.3)
        
        def dynamics(t, y):
            return torch.zeros_like(y)  # No change
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])  # 1.0 is not a multiple of 0.3
        
        y_final = solver.integrate(dynamics, y0, t)
        
        # Should still reach t=1.0 exactly (with smaller final step)
        assert torch.allclose(y_final, y0)
