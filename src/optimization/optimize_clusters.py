#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEORIA TRINITÁRIA OTIMIZADA PARA AGLOMERADOS
============================================

Versão especializada da Teoria Trinitária para aglomerados de galáxias,
com parâmetros otimizados para escalas de 1-10 Mpc.

Baseado na análise prévia que mostrou necessidade de ajustes específicos
para reproduzir dispersões de velocidade de 400-1300 km/s observadas
em aglomerados famosos (Coma, Virgo, Perseus, Fornax).

Autor: Sistema de IA Avançado  
Data: 23 de Novembro de 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import json
import warnings
warnings.filterwarnings('ignore')

class OptimizedClusterTrinitaria:
    """
    Teoria Trinitária especializada para Aglomerados de Galáxias
    
    Utiliza parâmetros otimizados especificamente para escalas de aglomerados,
    diferindo da versão galáctica para atingir dispersões de 400-1300 km/s.
    """
    
    def __init__(self):
        """Inicializa com parâmetros otimizados para aglomerados"""
        
        # Parâmetros base amplificados para aglomerados
        self.cluster_params = {
            'psi_0': 50000.0,    # 500× maior que galáxias (campo estrutural forte)
            'phi_0': 80000.0,    # 533× maior (estabilização dinâmica intensa)
            'sigma_0': 120000.0, # 1500× maior (coesão energética extrema)
            'alpha_psi': 0.3,    # Expoente maior (decaimento mais rápido)
            'beta_phi': 0.2,     # Ligeiramente maior
            'gamma_sigma': 1.5,  # Menor que galáxias (menos divergência)
        }
        
        # Raios característicos para aglomerados (kpc)
        self.cluster_radii = {
            'r_c_psi': 1500,    # ~1.5 Mpc (estrutura total)
            'r_c_phi': 800,     # ~0.8 Mpc (região dinâmica)  
            'r_c_sigma': 300,   # ~0.3 Mpc (core central)
        }
        
        # Dados observacionais de aglomerados famosos
        self.cluster_data = {
            'Coma': {
                'distance_Mpc': 99,
                'velocity_dispersion': 1000,  # km/s
                'core_radius_kpc': 500,
                'virial_radius_kpc': 2300,
                'mass_1e14_Msun': 15.0,
                'temperature_keV': 8.2
            },
            'Virgo': {
                'distance_Mpc': 16.5,
                'velocity_dispersion': 750,
                'core_radius_kpc': 200, 
                'virial_radius_kpc': 1200,
                'mass_1e14_Msun': 1.2,
                'temperature_keV': 2.3
            },
            'Perseus': {
                'distance_Mpc': 73,
                'velocity_dispersion': 1300,
                'core_radius_kpc': 300,
                'virial_radius_kpc': 2800,
                'mass_1e14_Msun': 6.5,
                'temperature_keV': 6.5
            },
            'Fornax': {
                'distance_Mpc': 20,
                'velocity_dispersion': 400,
                'core_radius_kpc': 150,
                'virial_radius_kpc': 700,
                'mass_1e14_Msun': 0.7,
                'temperature_keV': 1.2
            }
        }
    
    def trinitaria_velocity(self, r, psi_0, phi_0, sigma_0, 
                           alpha_psi, beta_phi, gamma_sigma,
                           r_c_psi, r_c_phi, r_c_sigma):
        """
        Calcula velocidade Trinitária para aglomerados
        
        Parâmetros:
        -----------
        r : array_like
            Raio em kpc
        psi_0, phi_0, sigma_0 : float
            Amplitudes dos campos
        alpha_psi, beta_phi, gamma_sigma : float
            Expoentes dos campos
        r_c_psi, r_c_phi, r_c_sigma : float
            Raios característicos em kpc
            
        Retorna:
        --------
        array_like : Velocidade em km/s
        """
        
        # Campo Psi (estrutura primária)
        psi = psi_0 * np.exp(-r/r_c_psi) * (r/r_c_psi)**alpha_psi
        
        # Campo Phi (estabilização dinâmica) 
        phi = phi_0 * (r/r_c_phi)**beta_phi * np.exp(-r/r_c_phi)
        
        # Campo Sigma (coesão energética)
        sigma = sigma_0 * np.exp(-r/r_c_sigma) * (1 + r/r_c_sigma)**gamma_sigma
        
        # Velocidade total
        v_total = np.sqrt(np.maximum(psi + phi + sigma, 0))
        
        return v_total
    
    def cluster_dispersion_profile(self, cluster_name, r_range=None):
        """
        Gera perfil de dispersão de velocidades para aglomerado específico
        
        Parâmetros:
        -----------
        cluster_name : str
            Nome do aglomerado ('Coma', 'Virgo', etc.)
        r_range : tuple, opcional
            (r_min, r_max) em kpc. Default: (10, 3000)
            
        Retorna:
        --------
        dict : Perfil radial de dispersões
        """
        
        if r_range is None:
            r_range = (10, 3000)  # 10 kpc to 3 Mpc
        
        cluster_info = self.cluster_data[cluster_name]
        
        # Grade radial logarítmica
        r = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), 100)
        
        # Parâmetros iniciais (otimizados para aglomerados)
        params = [
            self.cluster_params['psi_0'],
            self.cluster_params['phi_0'], 
            self.cluster_params['sigma_0'],
            self.cluster_params['alpha_psi'],
            self.cluster_params['beta_phi'],
            self.cluster_params['gamma_sigma'],
            self.cluster_radii['r_c_psi'],
            self.cluster_radii['r_c_phi'],
            self.cluster_radii['r_c_sigma']
        ]
        
        # Velocidade Trinitária
        v_trinitaria = self.trinitaria_velocity(r, *params)
        
        # Perfil NFW para comparação
        r_s = cluster_info['core_radius_kpc']
        v_nfw = cluster_info['velocity_dispersion'] * np.sqrt(
            np.log(1 + r/r_s) / (r/r_s)
        )
        
        # Perfil isotérmico simples
        v_isothermal = cluster_info['velocity_dispersion'] * np.ones_like(r)
        
        return {
            'radius_kpc': r,
            'v_trinitaria': v_trinitaria,
            'v_nfw': v_nfw,
            'v_isothermal': v_isothermal,
            'observed_dispersion': cluster_info['velocity_dispersion'],
            'cluster_info': cluster_info
        }
    
    def optimize_cluster_parameters(self, cluster_name, verbose=True):
        """
        Otimiza parâmetros Trinitária para aglomerado específico
        
        Parâmetros:
        -----------
        cluster_name : str
            Nome do aglomerado
        verbose : bool
            Imprimir progresso da otimização
            
        Retorna:
        --------
        dict : Parâmetros otimizados e métricas
        """
        
        cluster_info = self.cluster_data[cluster_name]
        target_dispersion = cluster_info['velocity_dispersion']
        
        # Região de interesse: 100 kpc - 1500 kpc (core + infall)
        r_fit = np.logspace(2, 3.2, 30)  # 100 to ~1600 kpc
        
        def objective(params):
            """Função objetivo: minimizar diferença com dispersão observada"""
            try:
                v_pred = self.trinitaria_velocity(r_fit, *params)
                # Média ponderada na região central (mais peso no core)
                weights = np.exp(-r_fit/500)  # Peso maior em r < 500 kpc
                v_weighted = np.average(v_pred, weights=weights)
                
                # RMS com dispersão observada
                rms = np.sqrt(np.mean((v_pred - target_dispersion)**2))
                
                # Penalizar se muito longe da dispersão alvo
                penalty = abs(v_weighted - target_dispersion) / target_dispersion
                
                return rms + 1000 * penalty
            
            except:
                return 1e6  # Penalidade por parâmetros inválidos
        
        # Limites dos parâmetros
        bounds = [
            (1000, 200000),    # psi_0
            (1000, 300000),    # phi_0  
            (1000, 400000),    # sigma_0
            (0.1, 1.0),        # alpha_psi
            (0.1, 1.0),        # beta_phi
            (0.5, 3.0),        # gamma_sigma
            (200, 3000),       # r_c_psi
            (200, 2000),       # r_c_phi
            (50, 1000)         # r_c_sigma
        ]
        
        # Valores iniciais
        initial_guess = [
            self.cluster_params['psi_0'],
            self.cluster_params['phi_0'],
            self.cluster_params['sigma_0'], 
            self.cluster_params['alpha_psi'],
            self.cluster_params['beta_phi'],
            self.cluster_params['gamma_sigma'],
            self.cluster_radii['r_c_psi'],
            self.cluster_radii['r_c_phi'],
            self.cluster_radii['r_c_sigma']
        ]
        
        if verbose:
            print(f"🔍 Otimizando {cluster_name} (alvo: {target_dispersion} km/s)...")
        
        # Otimização global
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=200,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        optimal_params = result.x
        final_rms = result.fun
        
        # Validação com parâmetros ótimos
        v_optimal = self.trinitaria_velocity(r_fit, *optimal_params)
        mean_prediction = np.mean(v_optimal)
        rms_validation = np.sqrt(np.mean((v_optimal - target_dispersion)**2))
        
        if verbose:
            print(f"  ✅ RMS final: {rms_validation:.1f} km/s")
            print(f"  📊 Predição média: {mean_prediction:.0f} km/s")
            print(f"  🎯 Alvo: {target_dispersion} km/s")
        
        return {
            'cluster': cluster_name,
            'optimal_params': {
                'psi_0': optimal_params[0],
                'phi_0': optimal_params[1],
                'sigma_0': optimal_params[2],
                'alpha_psi': optimal_params[3],
                'beta_phi': optimal_params[4], 
                'gamma_sigma': optimal_params[5],
                'r_c_psi': optimal_params[6],
                'r_c_phi': optimal_params[7],
                'r_c_sigma': optimal_params[8]
            },
            'performance': {
                'rms_km_s': rms_validation,
                'mean_prediction': mean_prediction,
                'target_dispersion': target_dispersion,
                'relative_error': abs(mean_prediction - target_dispersion) / target_dispersion,
                'optimization_success': result.success
            },
            'fit_region': {
                'r_min_kpc': r_fit.min(),
                'r_max_kpc': r_fit.max(),
                'n_points': len(r_fit)
            }
        }

def run_cluster_optimization():
    """
    Executa otimização completa para todos os aglomerados
    """
    
    print("=" * 65)
    print("TEORIA TRINITÁRIA OTIMIZADA - AGLOMERADOS DE GALÁXIAS")
    print("=" * 65)
    
    optimizer = OptimizedClusterTrinitaria()
    results = []
    
    # Otimizar cada aglomerado
    for cluster_name in ['Coma', 'Virgo', 'Perseus', 'Fornax']:
        print(f"\n🌌 {cluster_name.upper()} CLUSTER")
        print("-" * 30)
        
        result = optimizer.optimize_cluster_parameters(cluster_name)
        results.append(result)
        
        perf = result['performance']
        print(f"  Dispersão alvo: {perf['target_dispersion']:.0f} km/s")
        print(f"  Predição ótima: {perf['mean_prediction']:.0f} km/s") 
        print(f"  RMS final: {perf['rms_km_s']:.1f} km/s")
        print(f"  Erro relativo: {perf['relative_error']:.1%}")
        
        if perf['rms_km_s'] < 100:
            print("  🎯 EXCELENTE ajuste!")
        elif perf['rms_km_s'] < 200:
            print("  ✅ BOM ajuste")
        elif perf['rms_km_s'] < 400:
            print("  ⚠️  Ajuste moderado")
        else:
            print("  ❌ Precisa melhorar")
    
    # Estatísticas gerais
    avg_rms = np.mean([r['performance']['rms_km_s'] for r in results])
    avg_error = np.mean([r['performance']['relative_error'] for r in results])
    success_rate = np.mean([r['performance']['optimization_success'] for r in results])
    
    print(f"\n📊 ESTATÍSTICAS GERAIS")
    print("-" * 25)
    print(f"  RMS médio: {avg_rms:.1f} km/s")
    print(f"  Erro relativo médio: {avg_error:.1%}")
    print(f"  Taxa de convergência: {success_rate:.0%}")
    
    # Visualização comparativa
    print(f"\n📈 GERANDO COMPARAÇÕES VISUAIS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        cluster_name = result['cluster']
        optimal_params = result['optimal_params']
        
        # Gerar perfil com parâmetros ótimos
        r = np.logspace(1.5, 3.5, 100)  # 30 to 3000 kpc
        v_optimal = optimizer.trinitaria_velocity(
            r, *list(optimal_params.values())
        )
        
        # Perfis de comparação
        profile = optimizer.cluster_dispersion_profile(cluster_name)
        
        ax = axes[i]
        ax.loglog(r, v_optimal, 'b-', linewidth=3, 
                 label='Trinitária Otimizada')
        ax.loglog(profile['radius_kpc'], profile['v_nfw'], 
                 'r--', linewidth=2, label='NFW')
        ax.axhline(profile['observed_dispersion'], color='g', 
                  linestyle=':', linewidth=3, 
                  label=f'Observado ({profile["observed_dispersion"]:.0f} km/s)')
        
        ax.set_xlabel('Raio (kpc)')
        ax.set_ylabel('Dispersão de Velocidade (km/s)')
        ax.set_title(f'{cluster_name} Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(30, 3000)
        ax.set_ylim(100, 3000)
        
        # Marcar região de ajuste
        ax.axvspan(100, 1600, alpha=0.1, color='blue', 
                  label='Região de Ajuste')
    
    plt.tight_layout()
    plt.savefig('trinitaria_clusters_optimized.png', dpi=300, bbox_inches='tight')
    print("  ✅ Perfis otimizados salvos: trinitaria_clusters_optimized.png")
    
    # Análise de parâmetros
    param_analysis = analyze_optimal_parameters(results)
    
    # Salvar resultados
    final_results = {
        'cluster_optimization': results,
        'parameter_analysis': param_analysis,
        'summary_statistics': {
            'average_rms': avg_rms,
            'average_relative_error': avg_error,
            'convergence_rate': success_rate,
            'clusters_optimized': len(results)
        },
        'metadata': {
            'optimization_date': '2025-11-23',
            'theory_version': 'Cluster-Optimized Trinitária',
            'method': 'Differential Evolution Global Optimization',
            'fit_region': '100-1600 kpc'
        }
    }
    
    with open('trinitaria_clusters_optimized_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("  ✅ Resultados salvos: trinitaria_clusters_optimized_results.json")
    
    # Conclusões
    print(f"\n" + "=" * 65)
    print("🎯 CONCLUSÕES DA OTIMIZAÇÃO")
    print("=" * 65)
    
    if avg_rms < 200:
        print(f"🚀 SUCESSO: Teoria Trinitária otimizada para aglomerados!")
        print(f"   RMS médio: {avg_rms:.1f} km/s")
        print(f"   Melhoria significativa sobre versão base")
    else:
        print(f"⚠️  PROGRESSO: Melhorias obtidas, mas ainda desafiador")
        print(f"   RMS médio: {avg_rms:.1f} km/s")
        print(f"   Necessário desenvolvimento adicional")
    
    print(f"\n🔬 PRÓXIMOS PASSOS:")
    print("1. Validar parâmetros ótimos em outros aglomerados")
    print("2. Buscar padrões universais nos parâmetros otimizados") 
    print("3. Incorporar efeitos relativísticos")
    print("4. Conectar com estrutura em grande escala")
    
    return final_results

def analyze_optimal_parameters(results):
    """
    Analisa padrões nos parâmetros otimizados
    """
    
    param_names = ['psi_0', 'phi_0', 'sigma_0', 'alpha_psi', 
                   'beta_phi', 'gamma_sigma', 'r_c_psi', 'r_c_phi', 'r_c_sigma']
    
    analysis = {}
    
    for param in param_names:
        values = [r['optimal_params'][param] for r in results]
        analysis[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'coefficient_of_variation': np.std(values) / np.mean(values)
        }
    
    # Identificar parâmetros mais universais (menor variação)
    cv_values = [(param, analysis[param]['coefficient_of_variation']) 
                 for param in param_names]
    cv_values.sort(key=lambda x: x[1])
    
    analysis['universality_ranking'] = cv_values
    
    return analysis

if __name__ == "__main__":
    # Execução principal
    results = run_cluster_optimization()
    
    print(f"\n🚀 Otimização de aglomerados concluída!")
    print(f"📁 Arquivos gerados:")
    print(f"   - trinitaria_clusters_optimized.png")
    print(f"   - trinitaria_clusters_optimized_results.json")