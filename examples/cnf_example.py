"""
Simple example demonstrating Continuous Normalizing Flow (CNF) usage.

This example shows how to:
1. Create a CNF model
2. Transform samples and compute log-probabilities
3. Generate samples from the learned distribution
"""

import torch
import torch.nn as nn
from neural_ode.layers.cnf import CNF


def main():
    print("=" * 60)
    print("Continuous Normalizing Flow (CNF) Example")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define the dynamics network
    # For CNF, the network should map from state_dim to state_dim
    state_dim = 2
    net = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, state_dim)
    )
    
    # Create CNF with Hutchinson trace estimator (faster for high dimensions)
    print("\n1. Creating CNF model...")
    cnf = CNF(
        net,
        trace_estimator='hutchinson',
        hutchinson_samples=1,
        rtol=1e-3,
        atol=1e-4
    )
    print(f"   CNF created with {sum(p.numel() for p in cnf.parameters())} parameters")
    
    # Define base distribution (standard Gaussian)
    base_dist = torch.distributions.Normal(
        torch.zeros(state_dim),
        torch.ones(state_dim)
    )
    
    # Generate some data samples
    print("\n2. Generating data samples...")
    num_samples = 100
    data = torch.randn(num_samples, state_dim) * 0.5 + 1.0  # Shifted Gaussian
    print(f"   Generated {num_samples} samples with shape {data.shape}")
    print(f"   Data mean: {data.mean(dim=0).tolist()}")
    print(f"   Data std: {data.std(dim=0).tolist()}")
    
    # Forward transformation: data -> latent
    print("\n3. Forward transformation (data -> latent)...")
    with torch.no_grad():
        latent, log_det = cnf(data, reverse=False)
    print(f"   Latent shape: {latent.shape}")
    print(f"   Log-determinant shape: {log_det.shape}")
    print(f"   Latent mean: {latent.mean(dim=0).tolist()}")
    print(f"   Latent std: {latent.std(dim=0).tolist()}")
    print(f"   Mean log-det: {log_det.mean().item():.4f}")
    
    # Compute log-probabilities
    print("\n4. Computing log-probabilities...")
    with torch.no_grad():
        log_probs = cnf.log_prob(data, base_dist)
    print(f"   Log-prob shape: {log_probs.shape}")
    print(f"   Mean log-prob: {log_probs.mean().item():.4f}")
    print(f"   Std log-prob: {log_probs.std().item():.4f}")
    
    # Reverse transformation: latent -> data (sampling)
    print("\n5. Reverse transformation (latent -> data)...")
    with torch.no_grad():
        samples = cnf.sample(50, base_dist)
    print(f"   Generated {samples.shape[0]} samples")
    print(f"   Sample mean: {samples.mean(dim=0).tolist()}")
    print(f"   Sample std: {samples.std(dim=0).tolist()}")
    
    # Check NFE (number of function evaluations)
    print("\n6. Function evaluation counts...")
    nfe = cnf.get_nfe()
    print(f"   Solver NFE: {nfe['solver_nfe']}")
    
    # Demonstrate gradient flow
    print("\n7. Testing gradient flow...")
    cnf.train()
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    
    # Single training step
    batch = data[:10]
    log_prob = cnf.log_prob(batch, base_dist)
    loss = -log_prob.mean()  # Negative log-likelihood
    
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed successfully!")
    
    # Check that parameters have gradients
    has_grads = all(p.grad is not None for p in cnf.parameters() if p.requires_grad)
    print(f"   All parameters have gradients: {has_grads}")
    
    print("\n" + "=" * 60)
    print("CNF example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
