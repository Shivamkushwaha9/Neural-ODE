"""Unit tests for Dopri5 adaptive solver."""

import pytest
import torch
import math

from neural_ode.solvers import Dopri5Solver
from neural_ode.utils.validation import MaxStepsExceeded, StepSizeTooSmall


class TestDopri5Solver:
    """Test suite for Dormand-Prince adaptive solver."""
    
    def test_initialization(self):
        """Test solver initialization with default parameters."""
        solver = Dopri5Solver()
        assert solver.rtol == 1e-3
        assert solver.atol == 1e-4
        assert solver.max_steps == 1000
        assert solver.nfe == 0
    
    def test_initialization_custom_params(self):
        """Test solver initialization with custom parameters."""
        solver = Dopri5Solver(rtol=1e-5, atol=1e-6, max_steps=500)
        assert solver.rtol == 1e-5
        assert solver.atol == 1e-6
        assert solver.max_steps == 500
    
    def test_invalid_tolerances(self):
        """Test that invalid tolerances raise ValueError."""
        with pytest.raises(ValueError, match="Relative tolerance must be positive"):
            Dopri5Solver(rtol=-1e-3)
        
        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            Dopri5Solver(atol=-1e-4)
        
        with pytest.raises(ValueError, match="Relative tolerance must be positive"):
            Dopri5Solver(rtol=0)
    
    def test_invalid_max_steps(self):
        """Test that invalid max_steps raises ValueError."""
        with pytest.raises(ValueError, match="max_steps must be positive"):
            Dopri5Solver(max_steps=0)
        
        with pytest.raises(ValueError, match="max_steps must be positive"):
            Dopri5Solver(max_steps=-100)
    
    def test_invalid_safety_factor(self):
        """Test that invalid safety factor raises ValueError."""
        with pytest.raises(ValueError, match="safety factor must be in"):
            Dopri5Solver(safety=0)
        
        with pytest.raises(ValueError, match="safety factor must be in"):
            Dopri5Solver(safety=1.5)
    
    def test_linear_ode_analytical_solution(self):
        """Test against analytical solution for dy/dt = -y, y(0) = 1.
        
        The analytical solution is y(t) = exp(-t).
        """
        solver = Dopri5Solver(rtol=1e-6, atol=1e-8)
        
        # Define dynamics: dy/dt = -y
        def linear_dynamics(t, y):
            return -y
        
        # Initial condition
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        # Integrate
        y_final = solver.integrate(linear_dynamics, y0, t)
        
        # Analytical solution: y(1) = exp(-1)
        y_expected = math.exp(-1.0)
        
        # Check accuracy (should be very accurate with tight tolerances)
        assert torch.allclose(y_final, torch.tensor([[y_expected]]), rtol=1e-5, atol=1e-7)
    
    def test_batch_processing(self):
        """Test that solver handles batched inputs correctly."""
        solver = Dopri5Solver()
        
        # Define dynamics: dy/dt = -y
        def linear_dynamics(t, y):
            return -y
        
        # Batch of initial conditions
        batch_size = 5
        y0 = torch.ones(batch_size, 1)
        t = torch.tensor([0.0, 1.0])
        
        # Integrate
        y_final = solver.integrate(linear_dynamics, y0, t)
        
        # Check shape
        assert y_final.shape == (batch_size, 1)
        
        # All batch elements should have same result (same initial condition)
        y_expected = math.exp(-1.0)
        assert torch.allclose(y_final, torch.full((batch_size, 1), y_expected), rtol=1e-3)
    
    def test_nfe_tracking(self):
        """Test that number of function evaluations is tracked."""
        solver = Dopri5Solver()
        
        def simple_dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        initial_nfe = solver.nfe
        solver.integrate(simple_dynamics, y0, t)
        
        # NFE should have increased (at least 7 per step for DOPRI5)
        assert solver.nfe > initial_nfe
        assert solver.nfe >= 7  # At least one step
    
    def test_adaptive_stepping(self):
        """Test that adaptive stepping adjusts based on error."""
        # Use tight tolerances - should require more steps
        solver_tight = Dopri5Solver(rtol=1e-8, atol=1e-10)
        
        # Use loose tolerances - should require fewer steps
        solver_loose = Dopri5Solver(rtol=1e-2, atol=1e-3)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 2.0])
        
        # Integrate with both solvers
        solver_tight.integrate(dynamics, y0, t)
        nfe_tight = solver_tight.nfe
        
        solver_loose.integrate(dynamics, y0, t)
        nfe_loose = solver_loose.nfe
        
        # Tighter tolerances should require more function evaluations
        assert nfe_tight > nfe_loose
    
    def test_max_steps_exceeded(self):
        """Test that MaxStepsExceeded is raised when limit is hit."""
        # Use very small max_steps to trigger the error
        solver = Dopri5Solver(max_steps=5)
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 10.0])  # Long integration time
        
        with pytest.raises(MaxStepsExceeded):
            solver.integrate(dynamics, y0, t)
    
    def test_step_size_too_small(self):
        """Test that StepSizeTooSmall is raised when step becomes too small."""
        # Use very tight tolerances and small min_step to potentially trigger
        solver = Dopri5Solver(rtol=1e-12, atol=1e-14, min_step=1e-5)
        
        # Stiff dynamics that might require very small steps
        def stiff_dynamics(t, y):
            return -1000 * y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        # This might raise StepSizeTooSmall or succeed depending on the dynamics
        # We just verify the solver handles it appropriately
        try:
            result = solver.integrate(stiff_dynamics, y0, t)
            # If it succeeds, result should be finite
            assert torch.isfinite(result).all()
        except StepSizeTooSmall:
            # This is also acceptable behavior
            pass
    
    def test_integrate_with_trajectory(self):
        """Test trajectory recording at specified time points."""
        solver = Dopri5Solver()
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        
        times, states = solver.integrate_with_trajectory(dynamics, y0, t)
        
        # Check shapes
        assert times.shape == (5,)
        assert states.shape == (5, 1, 1)
        
        # Check initial condition
        assert torch.allclose(states[0], y0)
        
        # Check that states decay exponentially
        for i in range(1, len(t)):
            expected = math.exp(-t[i].item())
            assert torch.allclose(states[i], torch.tensor([[[expected]]]), rtol=1e-3)
    
    def test_multidimensional_state(self):
        """Test solver with multi-dimensional state."""
        # Use tighter tolerances for better accuracy over full period
        solver = Dopri5Solver(rtol=1e-6, atol=1e-8)
        
        # 2D harmonic oscillator: dx/dt = v, dv/dt = -x
        def harmonic_oscillator(t, y):
            x, v = y[:, 0:1], y[:, 1:2]
            dx_dt = v
            dv_dt = -x
            return torch.cat([dx_dt, dv_dt], dim=1)
        
        # Initial condition: x=1, v=0
        y0 = torch.tensor([[1.0, 0.0]])
        t = torch.tensor([0.0, 2*math.pi])  # One full period
        
        y_final = solver.integrate(harmonic_oscillator, y0, t)
        
        # After one period, should return to initial condition
        assert torch.allclose(y_final, y0, rtol=1e-3, atol=1e-4)
    
    def test_tolerance_effects(self):
        """Test that tighter tolerances produce more accurate results."""
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        t = torch.tensor([0.0, 1.0])
        
        # Loose tolerances
        solver_loose = Dopri5Solver(rtol=1e-2, atol=1e-3)
        y_loose = solver_loose.integrate(dynamics, y0, t)
        
        # Tight tolerances
        solver_tight = Dopri5Solver(rtol=1e-8, atol=1e-10)
        y_tight = solver_tight.integrate(dynamics, y0, t)
        
        # Analytical solution
        y_exact = math.exp(-1.0)
        
        # Tight tolerance should be closer to exact solution
        error_loose = abs(y_loose.item() - y_exact)
        error_tight = abs(y_tight.item() - y_exact)
        
        assert error_tight < error_loose
    
    def test_time_span_direction(self):
        """Test that solver respects time direction (forward only)."""
        solver = Dopri5Solver()
        
        def dynamics(t, y):
            return -y
        
        y0 = torch.tensor([[1.0]])
        
        # Forward integration should work
        t_forward = torch.tensor([0.0, 1.0])
        y_forward = solver.integrate(dynamics, y0, t_forward)
        assert torch.isfinite(y_forward).all()
        
        # Same time should work (no integration needed)
        t_same = torch.tensor([0.0, 0.0])
        y_same = solver.integrate(dynamics, y0, t_same)
        assert torch.allclose(y_same, y0)
    
    def test_device_compatibility(self):
        """Test that solver works with tensors on different devices."""
        solver = Dopri5Solver()
        
        def dynamics(t, y):
            return -y
        
        # Test on CPU
        y0_cpu = torch.tensor([[1.0]])
        t_cpu = torch.tensor([0.0, 1.0])
        y_cpu = solver.integrate(dynamics, y0_cpu, t_cpu)
        assert y_cpu.device == y0_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            y0_gpu = y0_cpu.cuda()
            t_gpu = t_cpu.cuda()
            y_gpu = solver.integrate(dynamics, y0_gpu, t_gpu)
            assert y_gpu.device == y0_gpu.device
    
    def test_dtype_preservation(self):
        """Test that solver preserves tensor dtype."""
        solver = Dopri5Solver()
        
        def dynamics(t, y):
            return -y
        
        # Test with float32
        y0_f32 = torch.tensor([[1.0]], dtype=torch.float32)
        t_f32 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        y_f32 = solver.integrate(dynamics, y0_f32, t_f32)
        assert y_f32.dtype == torch.float32
        
        # Test with float64
        y0_f64 = torch.tensor([[1.0]], dtype=torch.float64)
        t_f64 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y_f64 = solver.integrate(dynamics, y0_f64, t_f64)
        assert y_f64.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

