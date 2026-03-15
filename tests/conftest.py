"""Shared pytest fixtures and configuration."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return available device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_linear_dynamics():
    """Simple linear dynamics: dy/dt = -y.
    
    Analytical solution: y(t) = y0 * exp(-t)
    """
    def dynamics(t, y):
        return -y
    
    def analytical_solution(y0, t):
        return y0 * torch.exp(-t)
    
    return dynamics, analytical_solution


@pytest.fixture
def simple_nn_dynamics():
    """Simple neural network dynamics for testing."""
    import torch.nn as nn
    
    class SimpleDynamics(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 32),
                nn.Tanh(),
                nn.Linear(32, dim)
            )
        
        def forward(self, t, y):
            return self.net(y)
    
    return SimpleDynamics


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
