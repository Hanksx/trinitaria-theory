#!/usr/bin/env python3
"""
Basic tests for Trinitária Theory implementation.

This test suite verifies core functionality and ensures the installation is working correctly.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.trinitaria_theory import TrinitariaModel, trinitaria_velocity

def test_basic_velocity_calculation():
    """Test basic velocity profile calculation."""
    print("Testing basic velocity calculation...")
    
    r = np.linspace(0.1, 30, 100)
    v = trinitaria_velocity(r, 100, 150, 80, 0.1, 0.1, 2.0, 8, 12, 20)
    
    assert len(v) == len(r), "Velocity array length mismatch"
    assert np.all(v > 0), "All velocities should be positive"
    assert np.std(v[50:]) < 50, "Should produce relatively flat rotation curve"
    
    print("✓ Basic velocity calculation passed")

def test_model_creation():
    """Test model creation for different scales.""" 
    print("Testing model creation...")
    
    galaxy_model = TrinitariaModel(scale='galaxy')
    cluster_model = TrinitariaModel(scale='cluster')
    cosmic_model = TrinitariaModel(scale='supercluster')
    
    assert galaxy_model.scale == 'galaxy'
    assert cluster_model.scale == 'cluster'
    assert cosmic_model.scale == 'supercluster'
    
    # Test parameter scaling
    assert cluster_model.universal_params['psi_0'] > galaxy_model.universal_params['psi_0']
    assert cosmic_model.universal_params['psi_0'] > cluster_model.universal_params['psi_0']
    
    print("✓ Model creation passed")

def test_field_decomposition():
    """Test individual field contribution calculation."""
    print("Testing field decomposition...")
    
    model = TrinitariaModel(scale='galaxy')
    r = np.linspace(1, 25, 50)
    
    contributions = model.field_contributions(r)
    
    required_keys = ['psi', 'phi', 'sigma', 'total', 'velocity']
    for key in required_keys:
        assert key in contributions, f"Missing key: {key}"
        assert len(contributions[key]) == len(r), f"Length mismatch for {key}"
        assert np.all(contributions[key] >= 0), f"Negative values in {key}"
    
    # Test field superposition
    total_manual = contributions['psi'] + contributions['phi'] + contributions['sigma']
    np.testing.assert_array_almost_equal(contributions['total'], total_manual, decimal=10)
    
    print("✓ Field decomposition passed")

def test_parameter_bounds():
    """Test parameter bounds for different scales."""
    print("Testing parameter bounds...")
    
    galaxy_model = TrinitariaModel(scale='galaxy')
    cluster_model = TrinitariaModel(scale='cluster')
    
    galaxy_bounds = galaxy_model.get_parameter_bounds()
    cluster_bounds = cluster_model.get_parameter_bounds()
    
    assert len(galaxy_bounds) == 9, "Galaxy model should have 9 parameters"
    assert len(cluster_bounds) == 9, "Cluster model should have 9 parameters"
    
    # Cluster bounds should generally be larger than galaxy bounds
    assert cluster_bounds[0][1] > galaxy_bounds[0][1], "Cluster psi_0 upper bound should be larger"
    
    print("✓ Parameter bounds passed")

def test_velocity_profile_method():
    """Test the velocity_profile method with custom parameters."""
    print("Testing velocity profile method...")
    
    model = TrinitariaModel(scale='galaxy')
    r = np.linspace(0.5, 20, 40)
    
    # Test with default universal parameters
    v_default = model.velocity_profile(r)
    assert len(v_default) == len(r)
    assert np.all(v_default > 0)
    
    # Test with custom parameters
    v_custom = model.velocity_profile(r, psi_0=120, phi_0=180, sigma_0=90)
    assert len(v_custom) == len(r)
    assert np.all(v_custom > 0)
    assert not np.array_equal(v_default, v_custom), "Custom parameters should change result"
    
    print("✓ Velocity profile method passed")

def test_universal_parameter_access():
    """Test access to universal parameters."""
    print("Testing universal parameter access...")
    
    model = TrinitariaModel(scale='galaxy')
    params = model.universal_params
    
    required_params = ['psi_0', 'phi_0', 'sigma_0', 'alpha_psi', 'beta_phi', 'gamma_sigma', 'r_c_psi', 'r_c_phi', 'r_c_sigma']
    
    for param in required_params:
        assert param in params, f"Missing universal parameter: {param}"
        assert isinstance(params[param], (int, float)), f"Parameter {param} should be numeric"
        assert params[param] > 0, f"Parameter {param} should be positive"
    
    print("✓ Universal parameter access passed")

def test_convenience_functions():
    """Test convenience functions for model creation."""
    print("Testing convenience functions...")
    
    from core.trinitaria_theory import create_galaxy_model, create_cluster_model, create_cosmic_model
    
    galaxy_model = create_galaxy_model()
    cluster_model = create_cluster_model()
    cosmic_model = create_cosmic_model()
    
    assert galaxy_model.scale == 'galaxy'
    assert cluster_model.scale == 'cluster'
    assert cosmic_model.scale == 'supercluster'
    
    print("✓ Convenience functions passed")

def run_all_tests():
    """Run all basic tests."""
    print("Trinitária Theory - Basic Test Suite")
    print("=" * 40)
    
    try:
        test_basic_velocity_calculation()
        test_model_creation()
        test_field_decomposition()
        test_parameter_bounds()
        test_velocity_profile_method()
        test_universal_parameter_access()
        test_convenience_functions()
        
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The Trinitária Theory implementation is working correctly.")
        print("\nNext steps:")
        print("- Run examples/basic_tutorial.py for interactive demonstration")
        print("- Try src/validation/ scripts for full analysis")
        print("- Check docs/ for complete documentation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nPlease check your installation and try again.")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()