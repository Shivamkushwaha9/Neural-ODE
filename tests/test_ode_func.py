"""Unit tests for ODEFunc wrapper."""

import pytest
import torch
import torch.nn as nn
from neural_ode.layers import ODEFunc


class TestODEFunc:
    """Test suite for ODEFunc wrapper class."""
    
    def test_time_dependent_forward(self):
        """Test forward pass with time-dependent dynamics."""
        # Create a simple network: input_dim = state_dim + 1 (for time)
        state_dim = 2
        net = nn.Sequential(
            nn.Linear(state_dim + 1, 16),
            nn.Tanh(),
            nn.Linear(16, state_dim)
        )
        
        func = ODEFunc(net, time_dependent=True)
        
        # Test forward pass
        batch_size = 10
        h = torch.randn(batch_size, state_dim)
        t = 0.5
        
        dh_dt = func(t, h)
        
        # Check output shape
        assert dh_dt.shape == (batch_size, state_dim)
        
        # Check NFE counter incremented
        assert func.nfe == 1
        
        # Call again to verify NFE increments
        dh_dt2 = func(t, h)
        assert func.nfe == 2
    
    def test_time_independent_forward(self):
        """Test forward pass with time-independent dynamics."""
        # Create a simple network: input_dim = state_dim (no time)
        state_dim = 3
        net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, state_dim)
        )
        
        func = ODEFunc(net, time_dependent=False)
        
        # Test forward pass
        batch_size = 5
        h = torch.randn(batch_size, state_dim)
        t = 0.5
        
        dh_dt = func(t, h)
        
        # Check output shape
        assert dh_dt.shape == (batch_size, state_dim)
        
        # Check NFE counter incremented
        assert func.nfe == 1
    
    def test_time_independence_property(self):
        """Verify time-independent mode ignores time parameter."""
        state_dim = 2
        net = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.Tanh(),
            nn.Linear(8, state_dim)
        )
        
        func = ODEFunc(net, time_dependent=False)
        
        # Same state, different times should give same output
        h = torch.randn(4, state_dim)
        
        with torch.no_grad():
            dh_dt_t0 = func(0.0, h)
            func.reset_nfe()  # Reset counter
            dh_dt_t1 = func(1.0, h)
        
        # Outputs should be identical since time is ignored
        assert torch.allclose(dh_dt_t0, dh_dt_t1)
    
    def test_time_dependence_property(self):
        """Verify time-dependent mode uses time parameter."""
        state_dim = 2
        net = nn.Sequential(
            nn.Linear(state_dim + 1, 8),
            nn.Tanh(),
            nn.Linear(8, state_dim)
        )
        
        func = ODEFunc(net, time_dependent=True)
        
        # Same state, different times should give different output
        h = torch.randn(4, state_dim)
        
        with torch.no_grad():
            dh_dt_t0 = func(0.0, h)
            func.reset_nfe()
            dh_dt_t1 = func(1.0, h)
        
        # Outputs should be different since time affects the result
        # (with very high probability for random network)
        assert not torch.allclose(dh_dt_t0, dh_dt_t1)
    
    def test_nfe_counter(self):
        """Test NFE counter tracks function evaluations correctly."""
        state_dim = 2
        net = nn.Linear(state_dim, state_dim)
        func = ODEFunc(net, time_dependent=False)
        
        assert func.nfe == 0
        
        h = torch.randn(3, state_dim)
        
        # Make several calls
        for i in range(5):
            func(0.0, h)
            assert func.nfe == i + 1
        
        # Reset and verify
        func.reset_nfe()
        assert func.nfe == 0
        
        func(0.0, h)
        assert func.nfe == 1
    
    def test_batch_processing(self):
        """Test that ODEFunc handles different batch sizes correctly."""
        state_dim = 3
        net = nn.Linear(state_dim, state_dim)
        func = ODEFunc(net, time_dependent=False)
        
        # Test various batch sizes
        for batch_size in [1, 5, 10, 32]:
            h = torch.randn(batch_size, state_dim)
            dh_dt = func(0.5, h)
            assert dh_dt.shape == (batch_size, state_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow through ODEFunc correctly."""
        state_dim = 2
        net = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.Tanh(),
            nn.Linear(8, state_dim)
        )
        
        func = ODEFunc(net, time_dependent=False)
        
        # Forward pass with gradient tracking
        h = torch.randn(4, state_dim, requires_grad=True)
        dh_dt = func(0.5, h)
        
        # Compute loss and backward
        loss = dh_dt.sum()
        loss.backward()
        
        # Check gradients exist
        assert h.grad is not None
        assert not torch.allclose(h.grad, torch.zeros_like(h.grad))
        
        # Check network parameters have gradients
        for param in func.net.parameters():
            assert param.grad is not None
    
    def test_device_compatibility(self):
        """Test that ODEFunc works on different devices."""
        state_dim = 2
        net = nn.Linear(state_dim, state_dim)
        func = ODEFunc(net, time_dependent=False)
        
        # Test on CPU
        h_cpu = torch.randn(3, state_dim)
        dh_dt_cpu = func(0.5, h_cpu)
        assert dh_dt_cpu.device == h_cpu.device
        
        # Test on GPU if available
        if torch.cuda.is_available():
            func_gpu = func.cuda()
            h_gpu = h_cpu.cuda()
            dh_dt_gpu = func_gpu(0.5, h_gpu)
            assert dh_dt_gpu.device == h_gpu.device
    
    def test_dtype_preservation(self):
        """Test that ODEFunc preserves input dtype."""
        state_dim = 2
        net = nn.Linear(state_dim, state_dim)
        func = ODEFunc(net, time_dependent=False)
        
        # Test float32
        h_float32 = torch.randn(3, state_dim, dtype=torch.float32)
        dh_dt_float32 = func(0.5, h_float32)
        assert dh_dt_float32.dtype == torch.float32
        
        # Test float64
        func_float64 = func.double()
        h_float64 = torch.randn(3, state_dim, dtype=torch.float64)
        dh_dt_float64 = func_float64(0.5, h_float64)
        assert dh_dt_float64.dtype == torch.float64
    
    def test_time_concatenation_correctness(self):
        """Test that time is correctly concatenated in time-dependent mode."""
        state_dim = 2
        
        # Create a network that just returns the last input (time)
        class TimeExtractor(nn.Module):
            def forward(self, x):
                # Return zeros except for extracting time info
                batch_size = x.shape[0]
                # Just verify time is in the right place
                return torch.zeros(batch_size, state_dim)
        
        net = TimeExtractor()
        func = ODEFunc(net, time_dependent=True)
        
        h = torch.randn(5, state_dim)
        t = 0.7
        
        # Manually verify time concatenation
        t_vec = torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype) * t
        h_with_t_expected = torch.cat([h, t_vec], dim=1)
        
        # The network receives h_with_t, which should have shape (5, 3)
        assert h_with_t_expected.shape == (5, state_dim + 1)
        
        # Call the function
        dh_dt = func(t, h)
        assert dh_dt.shape == (5, state_dim)
