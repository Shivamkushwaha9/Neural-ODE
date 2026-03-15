"""Unit tests for parameter flattening utilities."""

import pytest
import torch
import torch.nn as nn
from neural_ode.adjoint import flatten_params, unflatten_params


class TestFlattenParams:
    """Test flatten_params function."""
    
    def test_flatten_simple_linear(self):
        """Test flattening a simple linear layer."""
        model = nn.Linear(10, 5, bias=True)
        flat, info = flatten_params(model.parameters())
        
        # Check shape: 10*5 + 5 = 55 parameters
        assert flat.shape == (55,)
        
        # Check info structure
        assert len(info) == 2  # weight and bias
        assert info[0] == (torch.Size([5, 10]), True)  # weight
        assert info[1] == (torch.Size([5]), True)  # bias
    
    def test_flatten_linear_no_bias(self):
        """Test flattening a linear layer without bias."""
        model = nn.Linear(10, 5, bias=False)
        flat, info = flatten_params(model.parameters())
        
        # Check shape: 10*5 = 50 parameters
        assert flat.shape == (50,)
        
        # Check info structure
        assert len(info) == 1  # only weight
        assert info[0] == (torch.Size([5, 10]), True)
    
    def test_flatten_sequential(self):
        """Test flattening a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        flat, info = flatten_params(model.parameters())
        
        # Check shape: (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325 parameters
        assert flat.shape == (325,)
        
        # Check info structure: 4 parameters (2 weights, 2 biases)
        assert len(info) == 4
    
    def test_flatten_conv2d(self):
        """Test flattening a convolutional layer."""
        model = nn.Conv2d(3, 16, kernel_size=3, bias=True)
        flat, info = flatten_params(model.parameters())
        
        # Check shape: 3*16*3*3 + 16 = 448 parameters
        assert flat.shape == (448,)
        
        # Check info structure
        assert len(info) == 2
        assert info[0] == (torch.Size([16, 3, 3, 3]), True)  # weight
        assert info[1] == (torch.Size([16]), True)  # bias
    
    def test_flatten_empty_parameters(self):
        """Test flattening with no parameters."""
        # Create empty parameter list
        params = []
        flat, info = flatten_params(iter(params))
        
        # Should return empty tensor and empty info
        assert flat.shape == (0,)
        assert len(info) == 0
    
    def test_flatten_preserves_values(self):
        """Test that flattening preserves parameter values."""
        model = nn.Linear(3, 2, bias=True)
        
        # Set known values
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            model.bias.copy_(torch.tensor([7.0, 8.0]))
        
        flat, info = flatten_params(model.parameters())
        
        # Check values are preserved
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert torch.allclose(flat, expected)
    
    def test_flatten_mixed_requires_grad(self):
        """Test flattening with mixed requires_grad settings."""
        model = nn.Linear(5, 3)
        
        # Disable gradient for bias
        model.bias.requires_grad = False
        
        flat, info = flatten_params(model.parameters())
        
        # Check info captures requires_grad correctly
        assert info[0][1] == True  # weight requires grad
        assert info[1][1] == False  # bias doesn't require grad


class TestUnflattenParams:
    """Test unflatten_params function."""
    
    def test_unflatten_simple_linear(self):
        """Test unflattening a simple linear layer."""
        model = nn.Linear(10, 5, bias=True)
        flat, info = flatten_params(model.parameters())
        
        # Unflatten
        params = unflatten_params(flat, info)
        
        # Check structure
        assert len(params) == 2
        assert params[0].shape == torch.Size([5, 10])  # weight
        assert params[1].shape == torch.Size([5])  # bias
    
    def test_unflatten_preserves_values(self):
        """Test that unflattening preserves values."""
        model = nn.Linear(3, 2, bias=True)
        
        # Set known values
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            model.bias.copy_(torch.tensor([7.0, 8.0]))
        
        flat, info = flatten_params(model.parameters())
        params = unflatten_params(flat, info)
        
        # Check values match original
        assert torch.allclose(params[0], model.weight)
        assert torch.allclose(params[1], model.bias)
    
    def test_unflatten_requires_grad(self):
        """Test that unflattening preserves requires_grad."""
        model = nn.Linear(5, 3)
        model.bias.requires_grad = False
        
        flat, info = flatten_params(model.parameters())
        params = unflatten_params(flat, info)
        
        # Check requires_grad is preserved
        assert params[0].requires_grad == True  # weight
        assert params[1].requires_grad == False  # bias
    
    def test_unflatten_empty(self):
        """Test unflattening with no parameters."""
        flat = torch.tensor([])
        info = []
        
        params = unflatten_params(flat, info)
        
        assert len(params) == 0
    
    def test_unflatten_sequential(self):
        """Test unflattening a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        
        flat, info = flatten_params(model.parameters())
        params = unflatten_params(flat, info)
        
        # Check structure: 4 parameters
        assert len(params) == 4
        assert params[0].shape == torch.Size([20, 10])  # first weight
        assert params[1].shape == torch.Size([20])  # first bias
        assert params[2].shape == torch.Size([5, 20])  # second weight
        assert params[3].shape == torch.Size([5])  # second bias


class TestRoundTrip:
    """Test round-trip flatten -> unflatten."""
    
    def test_roundtrip_linear(self):
        """Test round-trip with linear layer."""
        model = nn.Linear(10, 5)
        
        # Get original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Flatten and unflatten
        flat, info = flatten_params(model.parameters())
        restored_params = unflatten_params(flat, info)
        
        # Check all parameters match
        for orig, restored in zip(original_params, restored_params):
            assert torch.allclose(orig, restored)
            assert orig.shape == restored.shape
    
    def test_roundtrip_conv(self):
        """Test round-trip with convolutional layer."""
        model = nn.Conv2d(3, 16, kernel_size=3)
        
        # Get original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Flatten and unflatten
        flat, info = flatten_params(model.parameters())
        restored_params = unflatten_params(flat, info)
        
        # Check all parameters match
        for orig, restored in zip(original_params, restored_params):
            assert torch.allclose(orig, restored)
            assert orig.shape == restored.shape
    
    def test_roundtrip_complex_model(self):
        """Test round-trip with complex model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.Linear(100, 50),
            nn.Linear(50, 10)
        )
        
        # Get original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Flatten and unflatten
        flat, info = flatten_params(model.parameters())
        restored_params = unflatten_params(flat, info)
        
        # Check all parameters match
        assert len(original_params) == len(restored_params)
        for orig, restored in zip(original_params, restored_params):
            assert torch.allclose(orig, restored)
            assert orig.shape == restored.shape
    
    def test_roundtrip_preserves_requires_grad(self):
        """Test round-trip preserves requires_grad settings."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        
        # Disable gradient for some parameters
        model[0].bias.requires_grad = False
        model[1].weight.requires_grad = False
        
        # Get original requires_grad settings
        original_grad = [p.requires_grad for p in model.parameters()]
        
        # Flatten and unflatten
        flat, info = flatten_params(model.parameters())
        restored_params = unflatten_params(flat, info)
        
        # Check requires_grad is preserved
        restored_grad = [p.requires_grad for p in restored_params]
        assert original_grad == restored_grad
    
    def test_roundtrip_with_gradient_computation(self):
        """Test round-trip maintains gradient computation capability."""
        model = nn.Linear(5, 3)
        
        # Flatten and unflatten
        flat, info = flatten_params(model.parameters())
        
        # Modify flat params (simulate gradient update)
        flat_modified = flat + 0.1
        
        # Unflatten
        restored_params = unflatten_params(flat_modified, info)
        
        # Check we can compute gradients through restored params
        x = torch.randn(2, 5)
        output = torch.nn.functional.linear(x, restored_params[0], restored_params[1])
        loss = output.sum()
        
        # This should not raise an error
        loss.backward()
        
        # Check that parameters have requires_grad=True
        # Note: .grad will be None for non-leaf tensors, but we can verify
        # that gradients flow through by checking requires_grad
        assert restored_params[0].requires_grad == True
        assert restored_params[1].requires_grad == True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_flatten_single_parameter(self):
        """Test flattening a single parameter."""
        param = nn.Parameter(torch.randn(10, 5))
        flat, info = flatten_params([param])
        
        assert flat.shape == (50,)
        assert len(info) == 1
        assert info[0] == (torch.Size([10, 5]), True)
    
    def test_flatten_1d_parameter(self):
        """Test flattening a 1D parameter."""
        param = nn.Parameter(torch.randn(10))
        flat, info = flatten_params([param])
        
        assert flat.shape == (10,)
        assert len(info) == 1
        assert info[0] == (torch.Size([10]), True)
    
    def test_flatten_scalar_parameter(self):
        """Test flattening a scalar parameter."""
        param = nn.Parameter(torch.tensor(5.0))
        flat, info = flatten_params([param])
        
        assert flat.shape == (1,)
        assert len(info) == 1
        assert info[0] == (torch.Size([]), True)
    
    def test_unflatten_scalar_parameter(self):
        """Test unflattening a scalar parameter."""
        param = nn.Parameter(torch.tensor(5.0))
        flat, info = flatten_params([param])
        restored = unflatten_params(flat, info)
        
        assert restored[0].shape == torch.Size([])
        assert torch.allclose(restored[0], param)
    
    def test_flatten_high_dimensional_tensor(self):
        """Test flattening high-dimensional tensors."""
        # 4D tensor (like conv weights)
        param = nn.Parameter(torch.randn(16, 3, 5, 5))
        flat, info = flatten_params([param])
        
        assert flat.shape == (16 * 3 * 5 * 5,)
        assert info[0] == (torch.Size([16, 3, 5, 5]), True)
        
        # Unflatten and check
        restored = unflatten_params(flat, info)
        assert restored[0].shape == torch.Size([16, 3, 5, 5])
        assert torch.allclose(restored[0], param)
