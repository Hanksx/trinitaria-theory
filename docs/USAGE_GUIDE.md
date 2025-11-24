# Trinitária Theory - Usage Guide
## Complete Instructions for Multi-Scale Analysis

---

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Clone repository  
git clone https://github.com/research/trinitaria-theory.git
cd trinitaria-theory

# Create virtual environment
python -m venv trinitaria-env
source trinitaria-env/bin/activate  # Linux/Mac
# trinitaria-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, matplotlib; print('Dependencies OK')"
```

### 2. Basic Theory Implementation

```python
# Basic usage example
import numpy as np

def trinitaria_velocity(r, psi_0=100, phi_0=150, sigma_0=80,
                       alpha_psi=0.1, beta_phi=0.1, gamma_sigma=2.0,
                       r_c_psi=13.8, r_c_phi=19.1, r_c_sigma=5.3):
    """
    Calculate Trinitária velocity profile
    
    Args:
        r: Radius array (kpc)
        Universal parameters with default values
        
    Returns:
        v_total: Velocity profile (km/s)
    """
    # Three scalar fields
    psi = psi_0 * np.exp(-r/r_c_psi) * (r/r_c_psi)**alpha_psi
    phi = phi_0 * (r/r_c_phi)**beta_phi * np.exp(-r/r_c_phi)
    sigma = sigma_0 * np.exp(-r/r_c_sigma) * (1 + r/r_c_sigma)**gamma_sigma
    
    return np.sqrt(np.maximum(psi + phi + sigma, 0))

# Example usage
r = np.linspace(0.1, 30, 100)  # 0.1 to 30 kpc
v_profile = trinitaria_velocity(r)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(r, v_profile, 'b-', linewidth=2, label='Trinitária Theory')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Galaxy Rotation Curve - Trinitária Theory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 Scale-Specific Analysis

### Galactic Scale (30 kpc) - SPARC Dataset

#### 1. Single Galaxy Analysis

```bash
# Analyze individual SPARC galaxy
python validador_trinitaria_sparc_175.py

# Check specific galaxy results
python -c "
import json
with open('trinitaria_sparc_175_results.json', 'r') as f:
    results = json.load(f)
    
# Find best result
best_galaxy = min(results['individual_results'], 
                 key=lambda x: x['rms_optimized'])
print(f'Best galaxy: {best_galaxy[\"galaxy\"]} with {best_galaxy[\"rms_optimized\"]:.1f} km/s')
"
```

#### 2. Full SPARC Validation

```python
# Run complete 175 galaxy validation
import subprocess
result = subprocess.run(['python', 'validador_trinitaria_sparc_175.py'], 
                       capture_output=True, text=True)
print("SPARC validation completed")

# Load and analyze results
import json
with open('trinitaria_sparc_175_results.json', 'r') as f:
    results = json.load(f)

print(f"Mean RMS: {results['summary']['mean_rms_optimized']:.1f} km/s")
print(f"Success rate: {results['summary']['success_rate']:.0%}")
print(f"Perfect fits: {results['summary']['perfect_fits']}")
```

#### 3. Morphological Analysis

```python
# Analyze performance by galaxy type
results_by_type = {}
for galaxy in results['individual_results']:
    gal_type = galaxy.get('type', 'unknown')
    if gal_type not in results_by_type:
        results_by_type[gal_type] = []
    results_by_type[gal_type].append(galaxy['rms_optimized'])

for gal_type, rms_values in results_by_type.items():
    mean_rms = np.mean(rms_values)
    print(f"{gal_type}: {mean_rms:.1f} km/s ({len(rms_values)} galaxies)")
```

### Cluster Scale (5 Mpc) - Famous Clusters

#### 1. Extended Cluster Validation

```bash
# Run 25 cluster analysis
python trinitaria_25_clusters_validation.py

# Results will show:
# - Performance per cluster
# - Morphological dependencies  
# - Distance independence
# - Success rate statistics
```

#### 2. Individual Cluster Optimization

```python
# Optimize specific cluster
from trinitaria_clusters_optimized import OptimizedClusterTrinitaria

optimizer = OptimizedClusterTrinitaria()

# Test Virgo cluster
result = optimizer.validate_cluster('Virgo')
print(f"Virgo Cluster:")
print(f"  Predicted: {result['trinitaria_prediction']:.0f} km/s")
print(f"  Observed: {result['observed_dispersion']:.0f} km/s") 
print(f"  RMS: {result['rms_trinitaria']:.1f} km/s")
```

#### 3. Morphology-Specific Analysis

```bash
# Generate cluster morphology plots
python trinitaria_25_clusters_validation.py

# Check results by morphology:
# - Irregular clusters: ~92 km/s (excellent)
# - Regular clusters: ~218 km/s (competitive)
# - Cool-core: ~398 km/s (challenging)
# - Merger systems: ~720 km/s (difficult)
```

### Cosmic Scale (50+ Mpc) - Supercluster Implementation

#### 1. Supercluster Analysis

```bash
# Run cosmic structure analysis
python trinitaria_superclusters_simple.py

# Results include:
# - 5 major supercluster systems
# - Correlation function ξ(r) for 1-300 Mpc  
# - Performance vs distance
# - Framework validation
```

#### 2. Correlation Function Calculation

```python
# Calculate cosmic correlation function
from trinitaria_superclusters_simple import SuperclusterTrinitaria

cosmic = SuperclusterTrinitaria()
corr_data = cosmic.cosmic_correlation_function()

print(f"ξ(5 Mpc): {np.interp(5.0, corr_data['separations_Mpc'], corr_data['xi_trinitaria']):.3f}")
print(f"ξ(50 Mpc): {np.interp(50.0, corr_data['separations_Mpc'], corr_data['xi_trinitaria']):.3f}")
```

---

## 🔧 Parameter Optimization

### 1. Global Optimization Setup

```python
from scipy.optimize import differential_evolution

def optimize_trinitaria_parameters(observed_data):
    """
    Optimize Trinitária parameters for observational data
    
    Args:
        observed_data: Dict with 'radius' and 'velocity' arrays
        
    Returns:
        Optimized parameters and performance metrics
    """
    
    def objective(params):
        psi_0, phi_0, sigma_0, alpha_psi, beta_phi, gamma_sigma, r_c_psi, r_c_phi, r_c_sigma = params
        
        # Calculate model prediction
        v_model = trinitaria_velocity(
            observed_data['radius'], psi_0, phi_0, sigma_0,
            alpha_psi, beta_phi, gamma_sigma, r_c_psi, r_c_phi, r_c_sigma
        )
        
        # RMS error
        rms = np.sqrt(np.mean((v_model - observed_data['velocity'])**2))
        return rms
    
    # Parameter bounds (galactic scale)
    bounds = [
        (50, 150),      # psi_0
        (100, 200),     # phi_0  
        (40, 120),      # sigma_0
        (0.05, 0.5),    # alpha_psi
        (0.05, 0.5),    # beta_phi
        (1.0, 3.0),     # gamma_sigma
        (5.0, 25.0),    # r_c_psi
        (10.0, 30.0),   # r_c_phi
        (2.0, 15.0)     # r_c_sigma
    ]
    
    # Global optimization
    result = differential_evolution(objective, bounds, seed=42, maxiter=300)
    
    return {
        'optimal_params': result.x,
        'rms_error': result.fun,
        'success': result.success
    }
```

### 2. Multi-Scale Parameter Scaling

```python
def scale_parameters_for_clusters(galactic_params, mass_scale_factor=10):
    """
    Scale galactic parameters for cluster analysis
    
    Args:
        galactic_params: Optimized galactic parameters
        mass_scale_factor: Scaling factor for cluster mass
        
    Returns:
        Scaled parameters for cluster scale
    """
    
    cluster_params = galactic_params.copy()
    
    # Scale field amplitudes
    cluster_params['psi_0'] *= mass_scale_factor * 500   # 50k range
    cluster_params['phi_0'] *= mass_scale_factor * 533   # 80k range  
    cluster_params['sigma_0'] *= mass_scale_factor * 1500  # 120k range
    
    # Scale characteristic radii (kpc -> Mpc range)
    cluster_params['r_c_psi'] *= 100     # ~1.4 Mpc
    cluster_params['r_c_phi'] *= 100     # ~1.9 Mpc
    cluster_params['r_c_sigma'] *= 75    # ~0.4 Mpc
    
    # Adjust exponents for cluster dynamics
    cluster_params['alpha_psi'] *= 2     # 0.2 range
    cluster_params['beta_phi'] *= 1.5    # 0.15 range
    cluster_params['gamma_sigma'] *= 0.75  # 1.5 range
    
    return cluster_params
```

---

## 📈 Visualization and Analysis

### 1. Multi-Scale Performance Plot

```python
import matplotlib.pyplot as plt

def plot_multi_scale_performance():
    """Generate multi-scale performance visualization"""
    
    # Performance data
    scales = ['Galaxy\n(30 kpc)', 'Cluster\n(5 Mpc)', 'Cosmic\n(50+ Mpc)']
    rms_values = [57.6, 378.5, 1000]  # Example values
    success_rates = [100, 44, 20]
    colors = ['green', 'blue', 'red']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMS performance
    bars1 = ax1.bar(scales, rms_values, color=colors, alpha=0.7)
    ax1.set_ylabel('Mean RMS (km/s)')
    ax1.set_title('RMS Performance by Scale')
    ax1.set_yscale('log')
    
    # Add values on bars
    for bar, rms in zip(bars1, rms_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rms:.0f}', ha='center', va='bottom')
    
    # Success rates
    bars2 = ax2.bar(scales, success_rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by Scale')
    ax2.set_ylim(0, 100)
    
    for bar, rate in zip(bars2, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('multi_scale_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate plot
plot_multi_scale_performance()
```

### 2. Parameter Evolution Visualization

```python
def plot_parameter_scaling():
    """Visualize parameter scaling across scales"""
    
    scales = [1, 100, 1000, 10000, 50000]  # kpc
    psi_amplitudes = [100, 5000, 50000, 200000, 500000]
    phi_amplitudes = [150, 8000, 80000, 350000, 800000]
    sigma_amplitudes = [80, 12000, 120000, 500000, 1200000]
    
    plt.figure(figsize=(12, 8))
    plt.loglog(scales, psi_amplitudes, 'o-', linewidth=2, label='ψ₀ (Psi field)')
    plt.loglog(scales, phi_amplitudes, 's-', linewidth=2, label='φ₀ (Phi field)')  
    plt.loglog(scales, sigma_amplitudes, '^-', linewidth=2, label='σ₀ (Sigma field)')
    
    plt.xlabel('Physical Scale (kpc)')
    plt.ylabel('Field Amplitude')
    plt.title('Hierarchical Parameter Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark scale transitions
    plt.axvline(30, color='green', alpha=0.5, linestyle='--', label='Galaxy')
    plt.axvline(5000, color='blue', alpha=0.5, linestyle='--', label='Cluster')
    plt.axvline(50000, color='red', alpha=0.5, linestyle='--', label='Supercluster')
    
    plt.savefig('parameter_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_parameter_scaling()
```

---

## 🧪 Testing and Validation

### 1. Unit Tests

```python
# test_trinitaria_theory.py
import numpy as np
import pytest

def test_trinitaria_velocity_basic():
    """Test basic velocity calculation"""
    r = np.array([1.0, 5.0, 10.0, 20.0])
    v = trinitaria_velocity(r)
    
    # Check output shape
    assert v.shape == r.shape
    
    # Check positive values
    assert np.all(v >= 0)
    
    # Check monotonic behavior in inner region
    assert v[0] < v[1]  # Rising inner region

def test_parameter_bounds():
    """Test parameter boundary conditions"""
    r = np.linspace(0.1, 30, 100)
    
    # Test with extreme parameters
    v_extreme = trinitaria_velocity(r, psi_0=1000, phi_0=1000, sigma_0=1000)
    assert np.all(np.isfinite(v_extreme))
    
    # Test with minimal parameters  
    v_minimal = trinitaria_velocity(r, psi_0=1, phi_0=1, sigma_0=1)
    assert np.all(np.isfinite(v_minimal))

def test_scale_invariance():
    """Test that functional form works across scales"""
    r_galaxy = np.linspace(0.1, 30, 50)      # Galaxy scale (kpc)
    r_cluster = np.linspace(100, 3000, 50)   # Cluster scale (kpc)
    
    v_galaxy = trinitaria_velocity(r_galaxy)  # Galaxy parameters
    v_cluster = trinitaria_velocity(r_cluster, psi_0=50000, phi_0=80000, 
                                   sigma_0=120000, r_c_psi=1200, 
                                   r_c_phi=2000, r_c_sigma=400)  # Cluster parameters
    
    # Both should produce valid results
    assert np.all(np.isfinite(v_galaxy))
    assert np.all(np.isfinite(v_cluster))
    assert np.all(v_cluster > 0)

# Run tests
if __name__ == "__main__":
    test_trinitaria_velocity_basic()
    test_parameter_bounds() 
    test_scale_invariance()
    print("All tests passed!")
```

### 2. Performance Benchmarks

```python
import time
import numpy as np

def benchmark_trinitaria_performance():
    """Benchmark computational performance"""
    
    # Test different array sizes
    sizes = [100, 1000, 10000, 100000]
    times = []
    
    for size in sizes:
        r = np.linspace(0.1, 30, size)
        
        start_time = time.time()
        for _ in range(100):  # Average over 100 runs
            v = trinitaria_velocity(r)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        times.append(avg_time)
        
        print(f"Size {size:6d}: {avg_time*1000:.2f} ms per calculation")
    
    return sizes, times

# Run benchmark
sizes, times = benchmark_trinitaria_performance()
```

---

## 📋 Common Issues and Solutions

### 1. Installation Problems

```bash
# If matplotlib backend issues:
export MPLBACKEND=Agg  # Linux/Mac
# or
set MPLBACKEND=Agg     # Windows

# If dependency conflicts:
pip install --upgrade pip
pip install --no-deps -r requirements.txt

# If missing system libraries:
sudo apt-get install python3-dev libhdf5-dev  # Ubuntu
brew install hdf5                             # macOS
```

### 2. Memory Issues with Large Datasets

```python
# Process data in chunks for large datasets
def process_large_dataset(data, chunk_size=1000):
    """Process large dataset in chunks to avoid memory issues"""
    
    results = []
    n_chunks = len(data) // chunk_size + 1
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        
        chunk_data = data[start_idx:end_idx]
        chunk_results = process_chunk(chunk_data)
        results.extend(chunk_results)
        
        print(f"Processed chunk {i+1}/{n_chunks}")
    
    return results
```

### 3. Optimization Convergence Issues

```python
# If optimization doesn't converge, try:
# 1. Different random seeds
# 2. Looser tolerances  
# 3. More iterations
# 4. Different algorithms

from scipy.optimize import differential_evolution, dual_annealing

def robust_optimization(objective_func, bounds):
    """Try multiple optimization algorithms"""
    
    algorithms = [
        ('differential_evolution', differential_evolution),
        ('dual_annealing', dual_annealing)
    ]
    
    best_result = None
    best_score = np.inf
    
    for name, algorithm in algorithms:
        try:
            result = algorithm(objective_func, bounds, seed=42)
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                print(f"Best so far: {name} with score {best_score:.3f}")
        except Exception as e:
            print(f"Algorithm {name} failed: {e}")
    
    return best_result
```

---

## 🎯 Best Practices

### 1. Parameter Initialization

```python
# Always use physically motivated parameter ranges
GALACTIC_BOUNDS = {
    'psi_0': (50, 150),      # Field amplitudes
    'phi_0': (100, 200),
    'sigma_0': (40, 120),
    'alpha_psi': (0.05, 0.5),  # Exponents
    'beta_phi': (0.05, 0.5),
    'gamma_sigma': (1.0, 3.0),
    'r_c_psi': (5.0, 25.0),    # Characteristic radii (kpc)
    'r_c_phi': (10.0, 30.0),
    'r_c_sigma': (2.0, 15.0)
}
```

### 2. Error Handling

```python
def safe_trinitaria_calculation(r, params):
    """Safe calculation with error handling"""
    
    try:
        # Validate inputs
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0):
            raise ValueError("Radius must be positive")
        
        # Calculate velocity
        v = trinitaria_velocity(r, **params)
        
        # Check for invalid results
        if np.any(~np.isfinite(v)):
            raise ValueError("Invalid velocity values calculated")
            
        return v
        
    except Exception as e:
        print(f"Error in Trinitária calculation: {e}")
        return None
```

### 3. Performance Monitoring

```python
import functools
import time

def timing_decorator(func):
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.3f} seconds")
        return result
    return wrapper

@timing_decorator
def analyze_galaxy(galaxy_data):
    """Timed galaxy analysis"""
    # Analysis code here
    pass
```

---

## 📞 Support and Troubleshooting

### Getting Help

1. **Check documentation**: Review this guide and code comments
2. **Run tests**: Execute test scripts to verify installation  
3. **Check issues**: Browse GitHub issues for similar problems
4. **Contact support**: Create new issue with detailed error information

### Reporting Issues

When reporting problems, include:
- Python version and operating system
- Full error traceback
- Minimal code example reproducing the issue
- Environment information (`pip list`)

### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Ensure code passes all tests
5. Submit pull request with clear description

---

**Happy analyzing with Trinitária Theory! 🌌**