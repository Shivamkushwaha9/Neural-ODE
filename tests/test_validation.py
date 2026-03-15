"""Unit tests for input validation utilities."""

import pytest
import torch
from neural_ode.utils.validation import (
    check_finite,
    validate_ode_inputs,
    validate_tolerances,
    IntegrationError,
)


class TestCheckFinite:
    """Tests for check_finite function."""
    
    def test_valid_tensor(self):
        """Should not raise for valid finite tensor."""
        tensor = torch.randn(10, 5)
        check_finite(tensor, "test_tensor")  # Should not raise
    
    def test_nan_tensor(self):
        """Should raise ValueError for tensor with NaN."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            check_finite(tensor, "test_tensor")
    
    def test_inf_tensor(self):
        """Should raise ValueError for tensor with Inf."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            check_finite(tensor, "test_tensor")


class TestValidateODEInputs:
    """Tests for validate_ode_inputs function."""
    
    def test_valid_inputs(self):
        """Should not raise for valid inputs."""
        y0 = torch.randn(4, 3)  # batch_size=4, state_dim=3
        t = torch.tensor([0.0, 0.5, 1.0])
        validate_ode_inputs(y0, t)  # Should not raise
    
    def test_invalid_y0_dimensions(self):
        """Should raise for non-2D initial state."""
        y0 = torch.randn(4)  # 1D instead of 2D
        t = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="must be 2D"):
            validate_ode_inputs(y0, t)
    
    def test_invalid_t_dimensions(self):
        """Should raise for non-1D time tensor."""
        y0 = torch.randn(4, 3)
        t = torch.randn(5, 2)  # 2D instead of 1D
        with pytest.raises(ValueError, match="must be 1D"):
            validate_ode_inputs(y0, t)
    
    def test_non_monotonic_time(self):
        """Should raise for non-monotonic time points."""
        y0 = torch.randn(4, 3)
        t = torch.tensor([0.0, 1.0, 0.5])  # Not monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            validate_ode_inputs(y0, t)
    
    def test_too_few_time_points(self):
        """Should raise for less than 2 time points."""
        y0 = torch.randn(4, 3)
        t = torch.tensor([0.0])
        with pytest.raises(ValueError, match="at least 2 time points"):
            validate_ode_inputs(y0, t)
    
    def test_nan_in_y0(self):
        """Should raise for NaN in initial state."""
        y0 = torch.tensor([[1.0, float('nan')], [3.0, 4.0]])
        t = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="Initial state.*NaN"):
            validate_ode_inputs(y0, t)
    
    def test_nan_in_t(self):
        """Should raise for NaN in time points."""
        y0 = torch.randn(4, 3)
        t = torch.tensor([0.0, float('nan'), 1.0])
        with pytest.raises(ValueError, match="Time points.*NaN"):
            validate_ode_inputs(y0, t)


class TestValidateTolerances:
    """Tests for validate_tolerances function."""
    
    def test_valid_tolerances(self):
        """Should not raise for positive tolerances."""
        validate_tolerances(1e-3, 1e-4)  # Should not raise
    
    def test_zero_rtol(self):
        """Should raise for zero relative tolerance."""
        with pytest.raises(ValueError, match="Relative tolerance must be positive"):
            validate_tolerances(0.0, 1e-4)
    
    def test_negative_rtol(self):
        """Should raise for negative relative tolerance."""
        with pytest.raises(ValueError, match="Relative tolerance must be positive"):
            validate_tolerances(-1e-3, 1e-4)
    
    def test_zero_atol(self):
        """Should raise for zero absolute tolerance."""
        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            validate_tolerances(1e-3, 0.0)
    
    def test_negative_atol(self):
        """Should raise for negative absolute tolerance."""
        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            validate_tolerances(1e-3, -1e-4)


class TestIntegrationError:
    """Tests for IntegrationError exception."""
    
    def test_basic_error(self):
        """Should create error with basic message."""
        error = IntegrationError("Test error")
        assert "Test error" in str(error)
    
    def test_error_with_time(self):
        """Should include time in error message."""
        error = IntegrationError("Test error", t_current=0.5)
        assert "t=0.500000" in str(error)
    
    def test_error_with_state(self):
        """Should include state diagnostics in error message."""
        state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        error = IntegrationError("Test error", state=state)
        assert "State norm:" in str(error)
        assert "State contains NaN:" in str(error)
