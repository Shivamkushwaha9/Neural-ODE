"""Simple example demonstrating NeuralODE layer usage."""

import torch
import torch.nn as nn

from neural_ode import NeuralODE, ODEFunc
from neural_ode.solvers import Dopri5Solver

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Skipping visualization examples.")


def example_basic_usage():
    """Basic usage of NeuralODE layer."""
    print("=" * 60)
    print("Example 1: Basic NeuralODE Usage")
    print("=" * 60)
    
    # Create a simple dynamics network
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )
    
    # Wrap in ODEFunc
    func = ODEFunc(net, time_dependent=False)
    
    # Create NeuralODE layer
    ode_layer = NeuralODE(func)
    
    # Forward pass
    x = torch.randn(10, 2)
    print(f"Input shape: {x.shape}")
    
    y = ode_layer(x)
    print(f"Output shape: {y.shape}")
    print(f"NFE: {ode_layer.get_nfe()}")
    print()


def example_custom_time_span():
    """Using custom time spans."""
    print("=" * 60)
    print("Example 2: Custom Time Span")
    print("=" * 60)
    
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )
    func = ODEFunc(net, time_dependent=False)
    ode_layer = NeuralODE(func)
    
    x = torch.randn(5, 2)
    
    # Integrate from t=0 to t=2
    t = torch.tensor([0.0, 2.0])
    y = ode_layer(x, t)
    
    print(f"Integrated from t={t[0].item()} to t={t[-1].item()}")
    print(f"Output shape: {y.shape}")
    print()


def example_in_sequential():
    """Using NeuralODE in nn.Sequential."""
    print("=" * 60)
    print("Example 3: NeuralODE in Sequential Model")
    print("=" * 60)
    
    # Create dynamics network
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )
    func = ODEFunc(net, time_dependent=False)
    ode_layer = NeuralODE(func)
    
    # Build full model
    model = nn.Sequential(
        nn.Linear(10, 2),
        ode_layer,
        nn.Linear(2, 1)
    )
    
    # Forward pass
    x = torch.randn(5, 10)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print()


def example_trajectory_visualization():
    """Visualizing ODE trajectories."""
    print("=" * 60)
    print("Example 4: Trajectory Visualization")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("Skipping: matplotlib not installed")
        print()
        return
    
    # Create simple linear dynamics: dy/dt = -y
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(-torch.eye(2))
    
    func = ODEFunc(net, time_dependent=False)
    ode_layer = NeuralODE(func)
    
    # Initial states
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    # Get trajectory
    t = torch.linspace(0, 2, 20)
    times, states = ode_layer.forward_with_trajectory(x, t)
    
    print(f"Recorded {len(times)} time points")
    print(f"Trajectory shape: {states.shape}")
    
    # Plot trajectories
    plt.figure(figsize=(10, 5))
    
    # Plot in state space
    plt.subplot(1, 2, 1)
    for i in range(3):
        traj = states[:, i, :].detach().numpy()
        plt.plot(traj[:, 0], traj[:, 1], 'o-', label=f'Trajectory {i+1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Trajectories in State Space')
    plt.legend()
    plt.grid(True)
    
    # Plot over time
    plt.subplot(1, 2, 2)
    for i in range(3):
        traj = states[:, i, :].detach().numpy()
        plt.plot(times.numpy(), traj[:, 0], 'o-', label=f'x1 (traj {i+1})')
        plt.plot(times.numpy(), traj[:, 1], 's--', label=f'x2 (traj {i+1})')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Evolution Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_ode_trajectories.png', dpi=150)
    print("Saved trajectory plot to 'neural_ode_trajectories.png'")
    print()


def example_gradient_computation():
    """Computing gradients through NeuralODE."""
    print("=" * 60)
    print("Example 5: Gradient Computation")
    print("=" * 60)
    
    # Create model
    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2)
    )
    func = ODEFunc(net, time_dependent=False)
    ode_layer = NeuralODE(func)
    
    # Forward pass
    x = torch.randn(10, 2, requires_grad=True)
    y = ode_layer(x)
    
    # Compute loss and gradients
    loss = (y ** 2).sum()
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    
    # Check parameter gradients
    num_params_with_grad = sum(1 for p in ode_layer.parameters() if p.grad is not None)
    total_params = sum(1 for _ in ode_layer.parameters())
    print(f"Parameters with gradients: {num_params_with_grad}/{total_params}")
    print()


def example_different_solvers():
    """Comparing different ODE solvers."""
    print("=" * 60)
    print("Example 6: Comparing Different Solvers")
    print("=" * 60)
    
    from neural_ode.solvers import EulerSolver, RK4Solver
    
    # Create dynamics
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(-torch.eye(2))
    
    x = torch.ones(1, 2)
    t = torch.tensor([0.0, 1.0])
    
    # Expected solution: y(1) = exp(-1) * y(0)
    expected = torch.exp(torch.tensor(-1.0)) * x
    
    # Test different solvers
    solvers = [
        ('Euler (h=0.01)', EulerSolver(step_size=0.01)),
        ('RK4 (h=0.1)', RK4Solver(step_size=0.1)),
        ('Dopri5 (rtol=1e-5)', Dopri5Solver(rtol=1e-5, atol=1e-6))
    ]
    
    print(f"Expected solution: {expected[0].numpy()}")
    print()
    
    for name, solver in solvers:
        func = ODEFunc(nn.Linear(2, 2, bias=False), time_dependent=False)
        with torch.no_grad():
            func.net.weight.copy_(-torch.eye(2))
        
        ode_layer = NeuralODE(func, solver=solver)
        ode_layer.reset_nfe()
        
        y = ode_layer(x, t)
        error = torch.norm(y - expected).item()
        nfe = ode_layer.get_nfe()
        
        print(f"{name}:")
        print(f"  Result: {y[0].detach().numpy()}")
        print(f"  Error: {error:.6f}")
        print(f"  NFE: {nfe['func_nfe']}")
        print()


if __name__ == '__main__':
    example_basic_usage()
    example_custom_time_span()
    example_in_sequential()
    example_trajectory_visualization()
    example_gradient_computation()
    example_different_solvers()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
