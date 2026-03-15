"""Unit tests for ODESolver base class."""

import pytest
import torch
from neural_ode.solvers import ODESolver


class TestODESolverInterface:
    """Tests for ODESolver abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Should not be able to instantiate abstract ODESolver."""
        with pytest.raises(TypeError):
            ODESolver()
    
    def test_subclass_must_implement_integrate(self):
        """Subclass must implement integrate method."""
        
        class IncompleteSolver(ODESolver):
            def integrate_with_trajectory(self, func, y0, t, **kwargs):
                pass
        
        with pytest.raises(TypeError):
            IncompleteSolver()
    
    def test_subclass_must_implement_integrate_with_trajectory(self):
        """Subclass must implement integrate_with_trajectory method."""
        
        class IncompleteSolver(ODESolver):
            def integrate(self, func, y0, t, **kwargs):
                pass
        
        with pytest.raises(TypeError):
            IncompleteSolver()
    
    def test_complete_subclass_can_be_instantiated(self):
        """Complete subclass with both methods can be instantiated."""
        
        class CompleteSolver(ODESolver):
            def integrate(self, func, y0, t, **kwargs):
                return y0
            
            def integrate_with_trajectory(self, func, y0, t, **kwargs):
                return t, y0.unsqueeze(0).expand(len(t), -1, -1)
        
        solver = CompleteSolver()
        assert isinstance(solver, ODESolver)
        
        # Test that methods work
        y0 = torch.randn(2, 3)
        t = torch.tensor([0.0, 1.0])
        
        result = solver.integrate(lambda t, y: y, y0, t)
        assert result.shape == y0.shape
        
        times, states = solver.integrate_with_trajectory(lambda t, y: y, y0, t)
        assert times.shape == t.shape
        assert states.shape == (len(t), *y0.shape)
