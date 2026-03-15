"""Unit tests for RK4 solver."""

import pytest
import torch
import math
from neural_ode.solvers import RK4Solver
from neural_ode.utils.validation import IntegrationError


class TestRK4SolverBasic:
    """Basic functionality tests for RK4 solver."""
    
    def test_initialization(self):
        """Should initialize with valid step size."""
        solver = RK4Solver(step_size=0.1)
        assert solver.step_size == 0.1
        assert solver.nfe == 0
    
    def test_invalid_step_size(self):
        """Should raise ValueError for non-positive step size."""
        with pytest.raises(ValueError, match="Step size must be positive"):
            RK4Solver(step_size=0.0)
        
        with pytest.raises(ValueError, match="Step size must be positive"):
            RK4Solver(step_size=-0.1)
    
    def test_linear_ode(self):
        """Test against analytical solution: dy/dt = -y, y(0) = 1.
        
        Analytical solution: y(t) = exp(-t)
        """
        solver = RK4Solver(step_size=0.1)
        
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
        
        # RK4 should be very accurate even with step size 0.1
        assert y_final.shape == (1, 1)
        assert abs(y_final.item() - expected) < 0.0001  # Within 0.01% error
    
    def test_nfe_tracking(self):
        """Should track number of function evaluations.
        
        RK4 makes 4 function evaluations per step.
        """
        solver = RK4Solver(step_size=0.1)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        initial_nfe = solver.nfe
        solver.integrate(dynamics, y0, t)
        
        # Should have made 10 steps (1.0 / 0.1), each with 4 function evaluations
        # Total: 40 evaluations (or 44 if we make 11 steps due to floating point)
        assert solver.nfe >= initial_nfe + 40
        assert solver.nfe <= initial_nfe + 44
    
    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        solver = RK4Solver(step_size=0.1)
        
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
        assert torch.allclose(y_final, expected, rtol=0.0001)  # Very tight tolerance


class TestRK4SolverAccuracy:
    """Tests comparing RK4 accuracy with Euler."""
    
    def test_rk4_more_accurate_than_euler(self):
        """RK4 should be significantly more accurate than Euler for same step size."""
        from neural_ode.solvers import EulerSolver
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        expected = math.exp(-1.0)
        
        # Use same step size for both
        step_size = 0.1
        
        euler_solver = EulerSolver(step_size=step_size)
        y_euler = euler_solver.integrate(dynamics, y0, t)
        error_euler = abs(y_euler.item() - expected)
        
        rk4_solver = RK4Solver(step_size=step_size)
        y_rk4 = rk4_solver.integrate(dynamics, y0, t)
        error_rk4 = abs(y_rk4.item() - expected)
        
        # RK4 should have much smaller error
        assert error_rk4 < error_euler
        assert error_rk4 < 0.0001  # RK4 should be very accurate


class TestRK4SolverTrajectory:
    """Tests for trajectory recording functionality."""
    
    def test_trajectory_recording(self):
        """Should record states at specified time points."""
        solver = RK4Solver(step_size=0.1)
        
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
        
        # Check accuracy at final time
        expected = math.exp(-1.0)
        assert abs(states[2].item() - expected) < 0.0001


class TestRK4SolverEdgeCases:
    """Edge case and error handling tests."""
    
    def test_nan_detection(self):
        """Should detect and report NaN values during integration."""
        solver = RK4Solver(step_size=0.1)
        
        # Dynamics that produces NaN
        def bad_dynamics(t, y):
            return torch.tensor([[float('nan')]])
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            solver.integrate(bad_dynamics, y0, t)
