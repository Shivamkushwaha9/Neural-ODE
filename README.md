# Neural ODE

A PyTorch implementation of Neural Ordinary Differential Equations (Neural ODEs) based on the paper ["Neural Ordinary Differential Equations"](https://arxiv.org/abs/1806.07366) by Chen et al. (NeurIPS 2018, Best Paper Award).

Neural ODEs parameterize continuous-depth neural networks using ordinary differential equations, enabling memory-efficient training through the adjoint sensitivity method and providing adaptive computation based on problem complexity.

## Features

- **Multiple ODE Solvers**: Fixed-step (Euler, RK4) and adaptive (Dopri5) solvers with configurable tolerances
- **Memory-Efficient Training**: Adjoint sensitivity method with O(1) memory cost regardless of integration steps
- **Continuous Normalizing Flows**: Tractable density estimation with exact log-likelihood computation
- **Flexible Architecture**: Drop-in replacement for standard PyTorch layers with full autograd support
- **Visualization Tools**: Trajectory plotting and vector field visualization utilities
- **Production-Ready**: Comprehensive test suite with >80% code coverage

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0
- Matplotlib ≥ 3.5.0 (for visualization)

### Install from source

```bash
git clone https://github.com/shivamkushwaha9/neural-ode.git
cd neural-ode
pip install -e .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

This includes pytest, hypothesis, and pytest-cov for running tests.

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from neural_ode import NeuralODE, ODEFunc

# Define dynamics function
net = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 2)
)

# Wrap in ODEFunc
func = ODEFunc(net, time_dependent=False)

# Create Neural ODE layer
ode_layer = NeuralODE(func)

# Forward pass
x = torch.randn(10, 2)
y = ode_layer(x)  # Integrates from t=0 to t=1

print(f"Input shape: {x.shape}")   # torch.Size([10, 2])
print(f"Output shape: {y.shape}")  # torch.Size([10, 2])
```

### Custom Time Spans

```python
# Integrate from t=0 to t=5
t = torch.tensor([0.0, 5.0])
y = ode_layer(x, t)
```

### Using Different Solvers

```python
from neural_ode.solvers import Dopri5Solver, RK4Solver

# Adaptive solver with custom tolerances
solver = Dopri5Solver(rtol=1e-5, atol=1e-6)
ode_layer = NeuralODE(func, solver=solver)

# Fixed-step solver
solver = RK4Solver(step_size=0.1)
ode_layer = NeuralODE(func, solver=solver)
```

### Adjoint Method for Memory Efficiency

```python
# Use adjoint method for O(1) memory backpropagation
ode_layer = NeuralODE(func, sensitivity='adjoint')

# Standard autograd (for debugging)
ode_layer = NeuralODE(func, sensitivity='autograd')
```

### Continuous Normalizing Flows

```python
from neural_ode.layers import CNF
import torch.distributions as dist

# Define dynamics network
net = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 2)
)

# Create CNF layer
cnf = CNF(net)

# Compute log-likelihood
base_dist = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
x = torch.randn(100, 2)
log_prob = cnf.log_prob(x, base_dist)

# Generate samples
samples = cnf.sample(100, base_dist)
```

## Examples

The `examples/` directory contains complete working examples:

- **`simple_neural_ode.py`**: Basic usage patterns, solver comparison, and trajectory visualization
- **`cnf_example.py`**: Continuous normalizing flows for density estimation

Run examples:

```bash
python examples/simple_neural_ode.py
python examples/cnf_example.py
```

## Architecture

The library is organized into modular components:

```
neural_ode/
├── solvers/          # ODE integration methods
│   ├── base.py       # Abstract solver interface
│   ├── fixed_step.py # Euler and RK4 solvers
│   └── adaptive.py   # Dopri5 adaptive solver
├── adjoint/          # Adjoint sensitivity method
│   └── adjoint.py    # Memory-efficient backpropagation
├── layers/           # Neural network layers
│   ├── ode_layer.py  # Main NeuralODE layer
│   └── cnf.py        # Continuous normalizing flows
├── models/           # Example architectures
│   └── resnet.py     # Continuous-depth ResNet
└── utils/            # Utilities
    ├── validation.py # Input validation
    └── params.py     # Parameter manipulation
```

## Key Concepts

### Neural ODE Layer

Replaces discrete layer transformations `h_{t+1} = h_t + f(h_t)` with continuous dynamics:

```
dh/dt = f(h(t), t, θ)
```

The transformation is computed by integrating from `t0` to `t1` using numerical ODE solvers.

### Adjoint Sensitivity Method

Computes gradients through ODE solutions with O(1) memory cost by solving an augmented ODE system backward in time:

```
da/dt = -a^T ∂f/∂h
dL/dθ = ∫ a^T ∂f/∂θ dt
```

This enables training of very deep continuous networks without storing intermediate activations.

### Continuous Normalizing Flows

Transforms distributions through continuous-time dynamics while tracking the change of variables:

```
d(log p)/dt = -tr(∂f/∂z)
```

Provides exact log-likelihood computation without requiring invertible architectures.

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neural_ode --cov-report=html

# Run specific test file
pytest tests/test_ode_layer.py

# Run property-based tests
pytest tests/property_tests/
```

The library includes:
- **Unit tests**: Specific examples and edge cases
- **Property-based tests**: Universal correctness properties using Hypothesis
- **Integration tests**: End-to-end workflows and PyTorch ecosystem compatibility

## Performance

Number of function evaluations (NFE) for different solvers on a simple 2D system:

| Solver | Step Size / Tolerance | NFE | Error |
|--------|----------------------|-----|-------|
| Euler | h=0.01 | 100 | 1e-3 |
| RK4 | h=0.1 | 40 | 1e-6 |
| Dopri5 | rtol=1e-5 | 25 | 1e-7 |

Adaptive solvers automatically adjust computation based on problem difficulty.

## Citation

If you use this library in your research, please cite the original Neural ODE paper:

```bibtex
@inproceedings{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6571--6583},
  year={2018}
}
```

Paper: [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

## Compatibility

- **Python**: 3.8, 3.9, 3.10, 3.11
- **PyTorch**: 2.0.0 and later
- **Operating Systems**: Linux, macOS, Windows
- **Hardware**: CPU and CUDA-enabled GPUs

## Documentation

- **API Reference**: See docstrings in source code for detailed API documentation
- **Examples**: Check `examples/` directory for usage patterns
- **Tests**: Review `tests/` directory for additional usage examples

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest`
2. Code coverage remains >80%: `pytest --cov=neural_ode`
3. Code follows existing style conventions
4. New features include tests and documentation

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation is based on the Neural ODE paper by Chen et al. (2018). The adjoint sensitivity method follows the approach described in the original paper and subsequent work by the authors.

## Related Work

- [torchdiffeq](https://github.com/rtqichen/torchdiffeq): Official implementation by the paper authors
- [diffrax](https://github.com/patrick-kidger/diffrax): JAX-based differential equation solvers
- [torchsde](https://github.com/google-research/torchsde): Stochastic differential equations in PyTorch
