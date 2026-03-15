"""Parameter flattening utilities for adjoint sensitivity method."""

from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn


def flatten_params(parameters) -> Tuple[Tensor, List[Tuple[torch.Size, bool]]]:
    """Convert parameter list to single flattened tensor.
    
    This function flattens all parameters into a single 1D tensor for use in
    the adjoint sensitivity method. It preserves information about parameter
    shapes and gradient requirements to enable reconstruction.
    
    Args:
        parameters: Iterator of parameters (e.g., from model.parameters())
        
    Returns:
        Tuple containing:
            - flat_params: Flattened parameter tensor, shape (total_params,)
            - param_info: List of (shape, requires_grad) tuples for reconstruction
            
    Example:
        >>> model = nn.Linear(10, 5)
        >>> flat, info = flatten_params(model.parameters())
        >>> flat.shape
        torch.Size([55])  # 10*5 + 5 = 55 parameters
    """
    param_list = []
    param_info = []
    
    for param in parameters:
        # Store shape and gradient requirement
        param_info.append((param.shape, param.requires_grad))
        
        # Flatten and add to list
        param_list.append(param.reshape(-1))
    
    # Concatenate all parameters into single tensor
    if len(param_list) == 0:
        # Handle case with no parameters
        return torch.tensor([]), param_info
    
    flat_params = torch.cat(param_list)
    
    return flat_params, param_info


def unflatten_params(flat_params: Tensor, 
                     param_info: List[Tuple[torch.Size, bool]]) -> List[Tensor]:
    """Restore parameter structure from flattened tensor.
    
    This function reconstructs the original parameter structure from a flattened
    tensor using the shape and gradient information stored during flattening.
    
    Args:
        flat_params: Flattened parameter tensor, shape (total_params,)
        param_info: List of (shape, requires_grad) tuples from flatten_params()
        
    Returns:
        List of parameter tensors with original shapes and gradient requirements
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> flat, info = flatten_params(model.parameters())
        >>> params = unflatten_params(flat, info)
        >>> len(params)
        2  # weight and bias
        >>> params[0].shape
        torch.Size([5, 10])
    """
    if len(param_info) == 0:
        return []
    
    params = []
    offset = 0
    
    for shape, requires_grad in param_info:
        # Calculate number of elements for this parameter
        numel = torch.Size(shape).numel()
        
        # Extract slice and reshape
        param = flat_params[offset:offset + numel].reshape(shape)
        
        # Detach to create a new tensor, then set gradient requirement
        # This ensures requires_grad is properly set on the new tensor
        param = param.detach().requires_grad_(requires_grad)
        
        params.append(param)
        offset += numel
    
    return params
