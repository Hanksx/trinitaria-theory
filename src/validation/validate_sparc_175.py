#!/usr/bin/env python3
"""
VALIDADOR TEORIA TRINITÁRIA - 175 GALÁXIAS SPARC OFICIAIS
==========================================================
Sistema de validação massiva da Teoria Trinitária usando o dataset
SPARC completo com 175 galáxias.

Baseado nos parâmetros universais ótimos descobertos:
- ψ₀=100, φ₀=150, σ₀=80
- α_ψ=0.1, β_φ=0.1, γ_σ=2.0
- r_c variáveis por galáxia
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import differential_evolution, dual_annealing
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TrinitariaResult:
    """Resultado da validação Trinitária para uma galáxia."""
    galaxy_name: str
    galaxy_type: str
    distance_mpc: float
    rms_original: float
    rms_optimized: float
    improvement_factor: float
    optimization_success: bool
    optimized_params: Dict
    execution_time: float
    n_data_points: int
    max_velocity: float

class TrinitariaValidator175:
    """Validador da Teoria Trinitária para 175 galáxias SPARC."""
    
    def __init__(self, sparc_data_file: str = "sparc_official_data/sparc_175_complete.json"):
        self.sparc_file = Path(sparc_data_file)
        self.galaxies_data = []
        self.results = []
        
        # Parâmetros universais ótimos descobertos
        self.universal_params = {
            'psi_0': 100.00,      # Máximo
            'phi_0': 150.00,      # Máximo
            'sigma_0': 80.00,     # Máximo
            'alpha_psi': 0.10,    # Mínimo
            'beta_phi': 0.10,     # Mínimo
            'gamma_sigma': 2.00,  # Máximo
            'r_c_psi': 13.78,     # Média otimizada
            'r_c_phi': 19.08,     # Média otimizada
            'r_c_sigma': 5.31     # Média otimizada
        }
        
        # Limites físicos para otimização
        self.param_bounds = {
            'psi_0': (5.0, 100.0),
            'phi_0': (5.0, 150.0),
            'sigma_0': (2.0, 80.0),
            'alpha_psi': (0.1, 2.0),
            'beta_phi': (0.1, 2.5),
            'gamma_sigma': (0.1, 2.0),
            'r_c_psi': (0.5, 15.0),
            'r_c_phi': (1.0, 25.0),
            'r_c_sigma': (0.5, 12.0)
        }
        
        print("🌟 Validador Teoria Trinitária - 175 Galáxias SPARC")
        print(f"📁 Arquivo de dados: {self.sparc_file}")
        
    def load_sparc_data(self) -> bool:
        """Carregar dados das 175 galáxias SPARC."""
        print("\n📡 Carregando dados SPARC...")
        
        if not self.sparc_file.exists():
            print(f"❌ Arquivo não encontrado: {self.sparc_file}")
            return False
        
        try:
            with open(self.sparc_file, 'r') as f:
                data = json.load(f)
            
            self.galaxies_data = data['galaxies']
            print(f"✅ {len(self.galaxies_data)} galáxias SPARC carregadas")
            
            # Estatísticas básicas
            types = [g['galaxy_type'] for g in self.galaxies_data]
            distances = [g['distance_mpc'] for g in self.galaxies_data]
            
            print(f"📊 Estatísticas:")
            print(f"   Distâncias: {min(distances):.1f} - {max(distances):.1f} Mpc")
            print(f"   Tipos: {len(set(types))} morfologias diferentes")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def trinitaria_velocity_profile(self, r: np.ndarray, params: Dict) -> np.ndarray:
        """Calcular velocidades da Teoria Trinitária."""
        
        # Campos trinitários
        psi = params['psi_0'] * np.exp(-r/params['r_c_psi']) * (r/params['r_c_psi'])**params['alpha_psi']
        phi = params['phi_0'] * (r/params['r_c_phi'])**params['beta_phi'] * np.exp(-r/params['r_c_phi'])
        sigma = params['sigma_0'] * np.exp(-r/params['r_c_sigma']) * (1 + r/params['r_c_sigma'])**params['gamma_sigma']
        
        # Velocidade total
        v_total = np.sqrt(psi + phi + sigma)
        
        return v_total
    
    def calculate_rms(self, v_obs: np.ndarray, v_theory: np.ndarray, v_err: np.ndarray) -> float:
        """Calcular RMS ponderado pelos erros."""
        weighted_residuals = (v_theory - v_obs) / v_err
        rms = np.sqrt(np.mean(weighted_residuals**2)) * np.mean(v_err)
        return rms
    
    def objective_function(self, param_array: np.ndarray, r_obs: np.ndarray, 
                          v_obs: np.ndarray, v_err: np.ndarray) -> float:
        """Função objetivo para otimização."""
        
        # Converter array em dicionário de parâmetros
        param_names = list(self.param_bounds.keys())
        params = {name: param_array[i] for i, name in enumerate(param_names)}
        
        try:
            # Calcular velocidades teóricas
            v_theory = self.trinitaria_velocity_profile(r_obs, params)
            
            # RMS ponderado
            rms = self.calculate_rms(v_obs, v_theory, v_err)
            
            return rms
            
        except Exception:
            return 1e6
    
    def validate_single_galaxy(self, galaxy_data: Dict, optimize: bool = True) -> TrinitariaResult:
        """Validar uma galáxia individual."""
        
        galaxy_name = galaxy_data['name']
        start_time = time.time()
        
        # Extrair dados observacionais
        r_obs = np.array(galaxy_data['r_kpc'])
        v_obs = np.array(galaxy_data['v_obs_kms'])
        v_err = np.array(galaxy_data['v_err_kms'])
        
        # Filtrar dados válidos
        valid_mask = (r_obs > 0) & (v_obs > 0) & (v_err > 0)
        r_obs = r_obs[valid_mask]
        v_obs = v_obs[valid_mask]
        v_err = v_err[valid_mask]
        
        if len(r_obs) < 3:
            # Dados insuficientes
            return TrinitariaResult(
                galaxy_name=galaxy_name,
                galaxy_type=galaxy_data['galaxy_type'],
                distance_mpc=galaxy_data['distance_mpc'],
                rms_original=999.0,
                rms_optimized=999.0,
                improvement_factor=1.0,
                optimization_success=False,
                optimized_params=self.universal_params.copy(),
                execution_time=time.time() - start_time,
                n_data_points=len(r_obs),
                max_velocity=0.0
            )
        
        # RMS com parâmetros universais
        v_universal = self.trinitaria_velocity_profile(r_obs, self.universal_params)
        rms_original = self.calculate_rms(v_obs, v_universal, v_err)
        
        final_params = self.universal_params.copy()
        rms_final = rms_original
        optimization_success = True
        
        # Otimização individual (se solicitado)
        if optimize:
            try:
                # Configurar bounds para otimização
                bounds = [self.param_bounds[name] for name in self.param_bounds.keys()]
                
                # Usar parâmetros universais como ponto inicial
                initial_params = [self.universal_params[name] for name in self.param_bounds.keys()]
                
                # Otimização global
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    args=(r_obs, v_obs, v_err),
                    seed=42,
                    maxiter=300,
                    popsize=15,
                    x0=initial_params,
                    atol=1e-6
                )
                
                if result.success:
                    # Converter resultado para dicionário
                    param_names = list(self.param_bounds.keys())
                    final_params = {name: result.x[i] for i, name in enumerate(param_names)}
                    rms_final = result.fun
                else:
                    optimization_success = False
                    
            except Exception as e:
                print(f"⚠️  Otimização falhou para {galaxy_name}: {e}")
                optimization_success = False
        
        # Calcular fator de melhoria
        improvement_factor = rms_original / rms_final if rms_final > 0 else 1.0
        
        return TrinitariaResult(
            galaxy_name=galaxy_name,
            galaxy_type=galaxy_data['galaxy_type'],
            distance_mpc=galaxy_data['distance_mpc'],
            rms_original=rms_original,
            rms_optimized=rms_final,
            improvement_factor=improvement_factor,
            optimization_success=optimization_success,
            optimized_params=final_params,
            execution_time=time.time() - start_time,
            n_data_points=len(r_obs),
            max_velocity=max(v_obs) if len(v_obs) > 0 else 0.0
        )
    
    def validate_all_galaxies(self, optimize: bool = True, save_results: bool = True) -> List[TrinitariaResult]:
        """Validar todas as 175 galáxias SPARC."""
        print(f"\n🚀 Iniciando validação de {len(self.galaxies_data)} galáxias SPARC...")
        print(f"🔧 Otimização individual: {'✅ ATIVADA' if optimize else '❌ DESATIVADA'}")
        
        results = []
        start_time = time.time()
        
        for i, galaxy_data in enumerate(self.galaxies_data):
            galaxy_name = galaxy_data['name']
            
            # Mostrar progresso
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                progress = (i + 1) / len(self.galaxies_data) * 100
                eta = elapsed / (i + 1) * len(self.galaxies_data) - elapsed
                print(f"🔄 Progresso: {i+1}/175 ({progress:.1f}%) - ETA: {eta/60:.1f}min")
            
            # Validar galáxia
            result = self.validate_single_galaxy(galaxy_data, optimize=optimize)
            results.append(result)
            
            # Log de resultados interessantes
            if result.rms_optimized < 30:
                print(f"🏆 {galaxy_name}: {result.rms_optimized:.1f} km/s (melhoria: {result.improvement_factor:.2f}x)")
        
        self.results = results
        total_time = time.time() - start_time
        
        print(f"\n✅ Validação completa em {total_time/60:.1f} minutos!")
        
        # Estatísticas finais
        self.print_summary_statistics()
        
        if save_results:
            self.save_results()
        
        return results
    
    def print_summary_statistics(self):
        """Imprimir estatísticas resumo."""
        if not self.results:
            return
        
        # Calcular estatísticas
        rms_original = [r.rms_original for r in self.results if r.rms_original < 1000]
        rms_optimized = [r.rms_optimized for r in self.results if r.rms_optimized < 1000]
        improvements = [r.improvement_factor for r in self.results if r.improvement_factor < 10]
        successes = [r for r in self.results if r.optimization_success]
        
        # Categorizar resultados
        perfect_fits = [r for r in self.results if r.rms_optimized < 30]
        good_fits = [r for r in self.results if 30 <= r.rms_optimized < 50]
        fair_fits = [r for r in self.results if 50 <= r.rms_optimized < 100]
        
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS FINAIS - TEORIA TRINITÁRIA vs SPARC")
        print("="*60)
        
        print(f"\n🎯 PERFORMANCE GERAL:")
        print(f"   Galáxias validadas: {len(self.results)}")
        print(f"   Otimizações bem-sucedidas: {len(successes)} ({len(successes)/len(self.results)*100:.1f}%)")
        
        if rms_original:
            print(f"\n📈 RMS ORIGINAL (Parâmetros Universais):")
            print(f"   Média: {np.mean(rms_original):.1f} ± {np.std(rms_original):.1f} km/s")
            print(f"   Mediana: {np.median(rms_original):.1f} km/s")
            print(f"   Faixa: {min(rms_original):.1f} - {max(rms_original):.1f} km/s")
        
        if rms_optimized:
            print(f"\n🏆 RMS OTIMIZADO (Individual):")
            print(f"   Média: {np.mean(rms_optimized):.1f} ± {np.std(rms_optimized):.1f} km/s")
            print(f"   Mediana: {np.median(rms_optimized):.1f} km/s")
            print(f"   Faixa: {min(rms_optimized):.1f} - {max(rms_optimized):.1f} km/s")
        
        if improvements:
            print(f"\n📊 FATOR DE MELHORIA:")
            print(f"   Média: {np.mean(improvements):.2f}x")
            print(f"   Mediana: {np.median(improvements):.2f}x")
            print(f"   Máxima: {max(improvements):.2f}x")
        
        print(f"\n🎯 CATEGORIAS DE RESULTADO:")
        print(f"   🏆 Ajustes perfeitos (RMS < 30): {len(perfect_fits)} ({len(perfect_fits)/len(self.results)*100:.1f}%)")
        print(f"   ✅ Ajustes bons (30-50): {len(good_fits)} ({len(good_fits)/len(self.results)*100:.1f}%)")
        print(f"   ⚖️  Ajustes razoáveis (50-100): {len(fair_fits)} ({len(fair_fits)/len(self.results)*100:.1f}%)")
        
        # Top 10 melhores resultados
        best_results = sorted(self.results, key=lambda x: x.rms_optimized)[:10]
        
        print(f"\n🏅 TOP 10 MELHORES RESULTADOS:")
        for i, result in enumerate(best_results):
            print(f"   {i+1:2d}. {result.galaxy_name:12s}: {result.rms_optimized:6.1f} km/s "
                  f"({result.improvement_factor:.2f}x melhoria)")
        
        # Análise por morfologia
        morphologies = {}
        for result in self.results:
            morph = result.galaxy_type[:2] if len(result.galaxy_type) >= 2 else result.galaxy_type
            if morph not in morphologies:
                morphologies[morph] = []
            morphologies[morph].append(result.rms_optimized)
        
        print(f"\n🌀 PERFORMANCE POR MORFOLOGIA:")
        for morph, rms_values in morphologies.items():
            if len(rms_values) >= 3:  # Só mostrar com amostra suficiente
                avg_rms = np.mean([r for r in rms_values if r < 1000])
                print(f"   {morph:4s}: {avg_rms:6.1f} km/s médio ({len(rms_values)} galáxias)")
    
    def save_results(self, output_file: str = "trinitaria_sparc_175_results.json"):
        """Salvar resultados em JSON."""
        if not self.results:
            print("❌ Nenhum resultado para salvar!")
            return False
        
        output_path = Path(output_file)
        
        # Converter resultados para dicionários serializáveis
        results_data = []
        for result in self.results:
            result_dict = {
                'galaxy_name': result.galaxy_name,
                'galaxy_type': result.galaxy_type,
                'distance_mpc': result.distance_mpc,
                'rms_original': result.rms_original,
                'rms_optimized': result.rms_optimized,
                'improvement_factor': result.improvement_factor,
                'optimization_success': result.optimization_success,
                'optimized_params': result.optimized_params,
                'execution_time': result.execution_time,
                'n_data_points': result.n_data_points,
                'max_velocity': result.max_velocity
            }
            results_data.append(result_dict)
        
        # Calcular estatísticas para metadados
        valid_results = [r for r in self.results if r.rms_optimized < 1000]
        rms_values = [r.rms_optimized for r in valid_results]
        improvements = [r.improvement_factor for r in valid_results if r.improvement_factor < 10]
        
        # Preparar dados completos
        output_data = {
            "metadata": {
                "source": "SPARC 175 Galaxies - Trinitária Theory Validation",
                "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_galaxies": len(self.results),
                "successful_optimizations": sum(1 for r in self.results if r.optimization_success),
                "universal_parameters_used": self.universal_params,
                "summary_statistics": {
                    "mean_rms_km_s": float(np.mean(rms_values)) if rms_values else 0.0,
                    "median_rms_km_s": float(np.median(rms_values)) if rms_values else 0.0,
                    "std_rms_km_s": float(np.std(rms_values)) if rms_values else 0.0,
                    "min_rms_km_s": float(min(rms_values)) if rms_values else 0.0,
                    "max_rms_km_s": float(max(rms_values)) if rms_values else 0.0,
                    "mean_improvement_factor": float(np.mean(improvements)) if improvements else 0.0,
                    "perfect_fits_count": sum(1 for r in valid_results if r.rms_optimized < 30),
                    "good_fits_count": sum(1 for r in valid_results if 30 <= r.rms_optimized < 50),
                    "success_rate_percent": len(valid_results) / len(self.results) * 100
                },
                "description": "Validation of Trinitária Theory on 175 SPARC galaxies with individual optimization"
            },
            "results": results_data
        }
        
        # Salvar
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Resultados salvos: {output_path}")
        print(f"📊 Tamanho: {output_path.stat().st_size / 1024:.1f} KB")
        
        return True
    
    def create_summary_plot(self, save_plot: bool = True):
        """Criar gráfico resumo dos resultados."""
        if not self.results:
            print("❌ Nenhum resultado para plotar!")
            return
        
        # Filtrar resultados válidos
        valid_results = [r for r in self.results if r.rms_optimized < 500]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Teoria Trinitária - Validação SPARC 175 Galáxias', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de RMS
        rms_values = [r.rms_optimized for r in valid_results]
        ax1.hist(rms_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(rms_values), color='red', linestyle='--', label=f'Média: {np.mean(rms_values):.1f} km/s')
        ax1.axvline(30, color='green', linestyle='--', label='Limiar Perfeito: 30 km/s')
        ax1.set_xlabel('RMS (km/s)')
        ax1.set_ylabel('Número de Galáxias')
        ax1.set_title('Distribuição de RMS Otimizado')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RMS Original vs Otimizado
        rms_orig = [r.rms_original for r in valid_results if r.rms_original < 500]
        rms_opt = [r.rms_optimized for r in valid_results if r.rms_original < 500]
        
        ax2.scatter(rms_orig, rms_opt, alpha=0.6, s=50)
        max_rms = max(max(rms_orig) if rms_orig else 0, max(rms_opt) if rms_opt else 0)
        ax2.plot([0, max_rms], [0, max_rms], 'r--', alpha=0.5, label='y=x (sem melhoria)')
        ax2.set_xlabel('RMS Original (km/s)')
        ax2.set_ylabel('RMS Otimizado (km/s)')
        ax2.set_title('Original vs Otimizado')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fator de melhoria
        improvements = [r.improvement_factor for r in valid_results if r.improvement_factor < 10]
        ax3.hist(improvements, bins=25, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(improvements), color='red', linestyle='--', 
                   label=f'Média: {np.mean(improvements):.2f}x')
        ax3.axvline(1.0, color='orange', linestyle='--', label='Sem melhoria: 1.0x')
        ax3.set_xlabel('Fator de Melhoria')
        ax3.set_ylabel('Número de Galáxias')
        ax3.set_title('Distribuição de Melhoria')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. RMS por morfologia
        morphologies = {}
        for result in valid_results:
            morph = result.galaxy_type[:2] if len(result.galaxy_type) >= 2 else result.galaxy_type
            if morph not in morphologies:
                morphologies[morph] = []
            morphologies[morph].append(result.rms_optimized)
        
        # Filtrar morfologias com pelo menos 3 exemplos
        morph_filtered = {k: v for k, v in morphologies.items() if len(v) >= 3}
        
        if morph_filtered:
            labels = list(morph_filtered.keys())
            means = [np.mean(morph_filtered[label]) for label in labels]
            stds = [np.std(morph_filtered[label]) for label in labels]
            
            bars = ax4.bar(labels, means, yerr=stds, capsize=5, alpha=0.7, 
                          color='purple', edgecolor='black')
            ax4.set_xlabel('Tipo Morfológico')
            ax4.set_ylabel('RMS Médio (km/s)')
            ax4.set_title('Performance por Morfologia')
            ax4.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.05,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = "trinitaria_sparc_175_summary.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Gráfico salvo: {plot_file}")
        
        plt.show()

def main():
    """Função principal."""
    print("🌟 VALIDAÇÃO TEORIA TRINITÁRIA - 175 GALÁXIAS SPARC")
    print("="*55)
    
    # Inicializar validador
    validator = TrinitariaValidator175()
    
    # Carregar dados SPARC
    if not validator.load_sparc_data():
        print("❌ Falha ao carregar dados SPARC")
        return False
    
    # Executar validação
    print("\n🤔 Executar otimização individual? (Demora ~30-60 min para 175 galáxias)")
    optimize = input("Digite 'y' para SIM, qualquer tecla para só parâmetros universais: ").lower().strip() == 'y'
    
    # Validar todas as galáxias
    results = validator.validate_all_galaxies(optimize=optimize, save_results=True)
    
    # Criar gráficos
    validator.create_summary_plot(save_plot=True)
    
    print("\n🎉 VALIDAÇÃO SPARC COMPLETA!")
    print("🏆 Teoria Trinitária testada em 175 galáxias oficiais!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Execução bem-sucedida!")
    else:
        print("❌ Execução falhou!")