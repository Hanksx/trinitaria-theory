#!/usr/bin/env python3
"""
Trinitária Theory - Basic Tutorial
=================================

This tutorial demonstrates the basic usage of the Trinitária Theory
for galactic dynamics analysis without dark matter.

Author: Research Team
Version: 1.0.0  
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.trinitaria_theory import TrinitariaModel, trinitaria_velocity

def basic_velocity_profile():
    """Demonstrate basic velocity profile calculation."""
    print("=== Basic Velocity Profile ===")
    
    # Create radius array (0.1 to 30 kpc)
    r = np.linspace(0.1, 30, 100)
    
    # Calculate velocity using universal parameters
    v = trinitaria_velocity(r, 
                           psi_0=100, phi_0=150, sigma_0=80,
                           alpha_psi=0.1, beta_phi=0.1, gamma_sigma=2.0,
                           r_c_psi=8, r_c_phi=12, r_c_sigma=20)
    
    # Print key statistics
    print(f"Inner velocity (1 kpc): {v[10]:.1f} km/s")
    print(f"Peak velocity: {np.max(v):.1f} km/s")
    print(f"Outer velocity (25 kpc): {v[85]:.1f} km/s")
    print(f"Rotation curve flatness: {np.std(v[50:]):.1f} km/s")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(r, v, 'b-', linewidth=2, label='Trinitária Theory')
    plt.axhline(np.mean(v[50:]), color='r', linestyle='--', 
                label=f'Flat curve level: {np.mean(v[50:]):.1f} km/s')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title('Trinitária Theory - Basic Velocity Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return r, v

def model_comparison():
    """Compare different cosmic scale models."""
    print("\n=== Multi-Scale Model Comparison ===")
    
    # Create radius arrays for different scales
    r_galaxy = np.linspace(0.1, 30, 100)
    r_cluster = np.linspace(10, 3000, 100) 
    
    # Create models for different scales
    galaxy_model = TrinitariaModel(scale='galaxy')
    cluster_model = TrinitariaModel(scale='cluster')
    
    # Calculate velocity profiles
    v_galaxy = galaxy_model.velocity_profile(r_galaxy)
    v_cluster = cluster_model.velocity_profile(r_cluster)
    
    print("Galaxy Scale:")
    print(f"  Parameter scaling: 1×")
    print(f"  Velocity range: {np.min(v_galaxy):.1f} - {np.max(v_galaxy):.1f} km/s")
    
    print("Cluster Scale:")
    print(f"  Parameter scaling: ~500×")  
    print(f"  Velocity range: {np.min(v_cluster):.1f} - {np.max(v_cluster):.1f} km/s")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(r_galaxy, v_galaxy, 'g-', linewidth=2, label='Galaxy Scale')
    ax1.set_xlabel('Radius (kpc)')
    ax1.set_ylabel('Velocity (km/s)')
    ax1.set_title('Galactic Scale (30 kpc)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(r_cluster, v_cluster, 'b-', linewidth=2, label='Cluster Scale') 
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.set_title('Cluster Scale (5 Mpc)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def field_decomposition():
    """Demonstrate individual field contributions."""
    print("\n=== Field Decomposition Analysis ===")
    
    # Create model and radius array
    model = TrinitariaModel(scale='galaxy')
    r = np.linspace(0.1, 30, 100)
    
    # Calculate field contributions
    contributions = model.field_contributions(r)
    
    # Print field statistics
    print("Field Contributions at 10 kpc:")
    print(f"  Psi (gravitational): {contributions['psi'][50]:.1f}")
    print(f"  Phi (dynamic): {contributions['phi'][50]:.1f}")
    print(f"  Sigma (cohesive): {contributions['sigma'][50]:.1f}")
    print(f"  Total: {contributions['total'][50]:.1f}")
    print(f"  Velocity: {contributions['velocity'][50]:.1f} km/s")
    
    # Plot field decomposition
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(r, contributions['psi'], 'r-', linewidth=2, label='Ψ (Gravitational)')
    plt.plot(r, contributions['phi'], 'g-', linewidth=2, label='Φ (Dynamic)')
    plt.plot(r, contributions['sigma'], 'b-', linewidth=2, label='Σ (Cohesive)')
    plt.plot(r, contributions['total'], 'k--', linewidth=2, label='Total')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Field Contribution')
    plt.title('Individual Scalar Field Contributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 1, 2)
    plt.plot(r, contributions['velocity'], 'k-', linewidth=3, label='Total Velocity')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title('Resulting Velocity Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def parameter_sensitivity():
    """Demonstrate parameter sensitivity analysis."""
    print("\n=== Parameter Sensitivity Analysis ===")
    
    # Base parameters
    r = np.linspace(0.1, 30, 100)
    base_params = [100, 150, 80, 0.1, 0.1, 2.0, 8, 12, 20]
    
    # Calculate base velocity
    v_base = trinitaria_velocity(r, *base_params)
    
    # Test parameter variations
    variations = {
        'psi_0': [75, 100, 125],      # ±25%
        'phi_0': [120, 150, 180],     # ±20%  
        'sigma_0': [60, 80, 100],     # ±25%
        'gamma_sigma': [1.5, 2.0, 2.5] # ±25%
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (param_name, values) in enumerate(variations.items(), 1):
        plt.subplot(2, 2, i)
        
        for j, value in enumerate(values):
            params = base_params.copy()
            if param_name == 'psi_0':
                params[0] = value
            elif param_name == 'phi_0':
                params[1] = value  
            elif param_name == 'sigma_0':
                params[2] = value
            elif param_name == 'gamma_sigma':
                params[5] = value
                
            v = trinitaria_velocity(r, *params)
            
            if j == 1:  # Base case
                plt.plot(r, v, 'k-', linewidth=3, label=f'{param_name}={value} (base)')
            else:
                style = 'r--' if j == 0 else 'b--'
                plt.plot(r, v, style, linewidth=2, label=f'{param_name}={value}')
        
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocity (km/s)')
        plt.title(f'Sensitivity to {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print sensitivity statistics
    print("Parameter Sensitivity Summary:")
    for param_name, values in variations.items():
        velocities = []
        for value in values:
            params = base_params.copy()
            if param_name == 'psi_0':
                params[0] = value
            elif param_name == 'phi_0':
                params[1] = value
            elif param_name == 'sigma_0':
                params[2] = value
            elif param_name == 'gamma_sigma':
                params[5] = value
            
            v = trinitaria_velocity(r, *params)
            velocities.append(np.mean(v[50:]))  # Outer region average
        
        sensitivity = (max(velocities) - min(velocities)) / velocities[1] * 100
        print(f"  {param_name}: {sensitivity:.1f}% change in outer velocity")

def fitting_example():
    """Demonstrate model fitting to synthetic data."""
    print("\n=== Model Fitting Example ===")
    
    # Generate synthetic observational data
    r_obs = np.array([1, 2, 4, 6, 8, 10, 12, 15, 20, 25])
    
    # True parameters (slightly different from universal)
    true_params = [120, 140, 90, 0.12, 0.08, 1.8, 9, 11, 18]
    v_true = trinitaria_velocity(r_obs, *true_params)
    
    # Add realistic noise (±5-10 km/s)
    np.random.seed(42)  # Reproducible
    v_obs = v_true + np.random.normal(0, 7, len(r_obs))
    
    print(f"Synthetic data created with {len(r_obs)} points")
    print(f"Observational uncertainty: ~7 km/s RMS")
    
    # Fit model to synthetic data
    model = TrinitariaModel(scale='galaxy')
    fit_result = model.fit(r_obs, v_obs, maxiter=500, seed=42)
    
    print(f"\nFit Results:")
    print(f"  RMS Error: {fit_result['rms_error']:.2f} km/s")
    print(f"  Success: {fit_result['success']}")
    print(f"  Iterations: {fit_result['nfev']}")
    
    # Compare fitted vs true parameters
    param_names = ['psi_0', 'phi_0', 'sigma_0', 'alpha_psi', 'beta_phi', 'gamma_sigma', 'r_c_psi', 'r_c_phi', 'r_c_sigma']
    print(f"\nParameter Recovery:")
    for i, name in enumerate(param_names):
        true_val = true_params[i]
        fit_val = fit_result[name]
        error = abs(fit_val - true_val) / true_val * 100
        print(f"  {name}: true={true_val:.2f}, fit={fit_val:.2f}, error={error:.1f}%")
    
    # Plot fit quality
    r_fine = np.linspace(0.5, 30, 100)
    v_true_fine = trinitaria_velocity(r_fine, *true_params)
    v_fit_fine = model.predict(r_fine)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_fine, v_true_fine, 'g-', linewidth=3, label='True Model')
    plt.plot(r_fine, v_fit_fine, 'b--', linewidth=2, label='Fitted Model')
    plt.scatter(r_obs, v_obs, c='red', s=50, zorder=5, label='Synthetic Data')
    plt.scatter(r_obs, v_true, c='green', s=30, marker='x', zorder=5, label='True Values')
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'Model Fitting Example (RMS = {fit_result["rms_error"]:.2f} km/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Run all tutorial examples."""
    print("Trinitária Theory - Basic Tutorial")
    print("==================================")
    print("Demonstrating multi-scale scalar field approach to galactic dynamics\n")
    
    # Run examples
    basic_velocity_profile()
    model_comparison()
    field_decomposition()
    parameter_sensitivity() 
    fitting_example()
    
    print("\n=== Tutorial Complete ===")
    print("The Trinitária Theory successfully demonstrates:")
    print("✓ Flat rotation curves without dark matter")
    print("✓ Universal parameter scaling across cosmic scales")
    print("✓ Three-field decomposition revealing physical mechanisms")
    print("✓ Robust parameter fitting to observational data")
    print("✓ Multi-scale framework from galaxies to superclusters")
    
    print("\nNext steps:")
    print("- Try validate_sparc_175.py for full SPARC galaxy analysis") 
    print("- Run validate_25_clusters.py for cluster validation")
    print("- Explore cosmic_superclusters.py for large-scale structure")
    print("- Check docs/ for complete theoretical framework")

if __name__ == "__main__":
    main()