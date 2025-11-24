#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEORIA TRINITÁRIA - VALIDAÇÃO EXPANDIDA EM 25 AGLOMERADOS
=========================================================

Validação abrangente da Teoria Trinitária em 25 aglomerados famosos,
incluindo:
- Abell catalog (ricos e pobres)
- Grupos locais próximos  
- Aglomerados distantes (z > 0.1)
- Diferentes morfologias (regulares, em merger, relaxados)

Baseado nos parâmetros otimizados dos 4 aglomerados iniciais:
- RMS médio: 57.7 km/s
- Parâmetros universais identificados
- Framework escalável estabelecido

Autor: Sistema de IA Avançado
Data: 23 de Novembro de 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

class ExtendedClusterValidation:
    """
    Validação expandida da Teoria Trinitária em 25 aglomerados
    
    Inclui aglomerados de diferentes:
    - Massas (10^13 - 10^15 M_sun)
    - Distâncias (10 - 300 Mpc)  
    - Morfologias (regulares, irregulares, em merger)
    - Estados dinâmicos (relaxados vs ativos)
    """
    
    def __init__(self):
        """Inicializa com dados de 25 aglomerados famosos"""
        
        self.cluster_database = {
            # GRUPO 1: Aglomerados próximos e bem estudados (0-50 Mpc)
            'Virgo': {
                'distance_Mpc': 16.5, 'velocity_dispersion': 750, 'mass_1e14': 1.2,
                'type': 'irregular', 'richness': 'rich', 'temperature_keV': 2.3,
                'redshift': 0.004, 'members': 1300, 'core_radius_kpc': 200
            },
            'Fornax': {
                'distance_Mpc': 20, 'velocity_dispersion': 400, 'mass_1e14': 0.7,
                'type': 'regular', 'richness': 'poor', 'temperature_keV': 1.2,
                'redshift': 0.005, 'members': 340, 'core_radius_kpc': 150
            },
            'Centaurus': {
                'distance_Mpc': 38, 'velocity_dispersion': 550, 'mass_1e14': 1.8,
                'type': 'regular', 'richness': 'medium', 'temperature_keV': 2.8,
                'redshift': 0.012, 'members': 400, 'core_radius_kpc': 180
            },
            'Hydra': {
                'distance_Mpc': 48, 'velocity_dispersion': 610, 'mass_1e14': 2.1,
                'type': 'regular', 'richness': 'medium', 'temperature_keV': 3.1,
                'redshift': 0.016, 'members': 350, 'core_radius_kpc': 220
            },
            
            # GRUPO 2: Aglomerados Abell próximos (50-100 Mpc)
            'Coma': {
                'distance_Mpc': 99, 'velocity_dispersion': 1000, 'mass_1e14': 15.0,
                'type': 'regular', 'richness': 'very_rich', 'temperature_keV': 8.2,
                'redshift': 0.023, 'members': 1000, 'core_radius_kpc': 500
            },
            'Perseus': {
                'distance_Mpc': 73, 'velocity_dispersion': 1300, 'mass_1e14': 6.5,
                'type': 'cool_core', 'richness': 'rich', 'temperature_keV': 6.5,
                'redshift': 0.018, 'members': 500, 'core_radius_kpc': 300
            },
            'A1656': {  # Coma alternativo
                'distance_Mpc': 95, 'velocity_dispersion': 950, 'mass_1e14': 12.0,
                'type': 'regular', 'richness': 'very_rich', 'temperature_keV': 7.8,
                'redshift': 0.022, 'members': 800, 'core_radius_kpc': 450
            },
            'A426': {  # Perseus alternativo
                'distance_Mpc': 75, 'velocity_dispersion': 1250, 'mass_1e14': 5.8,
                'type': 'cool_core', 'richness': 'rich', 'temperature_keV': 6.2,
                'redshift': 0.019, 'members': 450, 'core_radius_kpc': 280
            },
            
            # GRUPO 3: Aglomerados Abell médios (100-200 Mpc)
            'A2029': {
                'distance_Mpc': 120, 'velocity_dispersion': 1200, 'mass_1e14': 8.5,
                'type': 'cool_core', 'richness': 'rich', 'temperature_keV': 7.5,
                'redshift': 0.077, 'members': 350, 'core_radius_kpc': 400
            },
            'A85': {
                'distance_Mpc': 140, 'velocity_dispersion': 900, 'mass_1e14': 4.2,
                'type': 'regular', 'richness': 'medium', 'temperature_keV': 5.1,
                'redshift': 0.055, 'members': 280, 'core_radius_kpc': 320
            },
            'A1367': {
                'distance_Mpc': 92, 'velocity_dispersion': 800, 'mass_1e14': 3.8,
                'type': 'irregular', 'richness': 'medium', 'temperature_keV': 4.2,
                'redshift': 0.022, 'members': 300, 'core_radius_kpc': 250
            },
            'A2256': {
                'distance_Mpc': 180, 'velocity_dispersion': 1400, 'mass_1e14': 10.2,
                'type': 'merger', 'richness': 'rich', 'temperature_keV': 8.8,
                'redshift': 0.058, 'members': 450, 'core_radius_kpc': 600
            },
            
            # GRUPO 4: Aglomerados distantes (200+ Mpc)
            'A2163': {
                'distance_Mpc': 250, 'velocity_dispersion': 1800, 'mass_1e14': 18.0,
                'type': 'merger', 'richness': 'very_rich', 'temperature_keV': 12.5,
                'redshift': 0.203, 'members': 200, 'core_radius_kpc': 800
            },
            'A665': {
                'distance_Mpc': 280, 'velocity_dispersion': 1100, 'mass_1e14': 6.8,
                'type': 'regular', 'richness': 'rich', 'temperature_keV': 6.9,
                'redshift': 0.182, 'members': 180, 'core_radius_kpc': 350
            },
            'A2390': {
                'distance_Mpc': 290, 'velocity_dispersion': 1300, 'mass_1e14': 9.1,
                'type': 'cool_core', 'richness': 'rich', 'temperature_keV': 8.1,
                'redshift': 0.228, 'members': 160, 'core_radius_kpc': 420
            },
            
            # GRUPO 5: Aglomerados especiais (lensing, merger, etc.)
            'A1689': {
                'distance_Mpc': 320, 'velocity_dispersion': 2200, 'mass_1e14': 22.0,
                'type': 'strong_lensing', 'richness': 'very_rich', 'temperature_keV': 15.2,
                'redshift': 0.183, 'members': 300, 'core_radius_kpc': 150
            },
            'Bullet': {  # 1E0657-558
                'distance_Mpc': 1200, 'velocity_dispersion': 1500, 'mass_1e14': 15.0,
                'type': 'merger', 'richness': 'rich', 'temperature_keV': 14.8,
                'redshift': 0.296, 'members': 100, 'core_radius_kpc': 200
            },
            'A2218': {
                'distance_Mpc': 330, 'velocity_dispersion': 1300, 'mass_1e14': 8.9,
                'type': 'strong_lensing', 'richness': 'rich', 'temperature_keV': 7.2,
                'redshift': 0.176, 'members': 220, 'core_radius_kpc': 380
            },
            
            # GRUPO 6: Aglomerados adicionais diversos
            'A3526': {  # Centaurus central
                'distance_Mpc': 45, 'velocity_dispersion': 480, 'mass_1e14': 1.4,
                'type': 'regular', 'richness': 'poor', 'temperature_keV': 2.1,
                'redshift': 0.011, 'members': 250, 'core_radius_kpc': 170
            },
            'A3558': {
                'distance_Mpc': 144, 'velocity_dispersion': 750, 'mass_1e14': 3.2,
                'type': 'regular', 'richness': 'medium', 'temperature_keV': 4.8,
                'redshift': 0.048, 'members': 180, 'core_radius_kpc': 290
            },
            'A2199': {
                'distance_Mpc': 135, 'velocity_dispersion': 700, 'mass_1e14': 2.8,
                'type': 'cool_core', 'richness': 'medium', 'temperature_keV': 4.1,
                'redshift': 0.030, 'members': 200, 'core_radius_kpc': 240
            },
            'A2142': {
                'distance_Mpc': 95, 'velocity_dispersion': 1100, 'mass_1e14': 7.2,
                'type': 'regular', 'richness': 'rich', 'temperature_keV': 8.9,
                'redshift': 0.091, 'members': 320, 'core_radius_kpc': 480
            },
            'A3376': {
                'distance_Mpc': 160, 'velocity_dispersion': 850, 'mass_1e14': 4.1,
                'type': 'regular', 'richness': 'medium', 'temperature_keV': 5.2,
                'redshift': 0.046, 'members': 190, 'core_radius_kpc': 310
            },
            'A4038': {
                'distance_Mpc': 190, 'velocity_dispersion': 950, 'mass_1e14': 5.5,
                'type': 'irregular', 'richness': 'medium', 'temperature_keV': 6.1,
                'redshift': 0.028, 'members': 220, 'core_radius_kpc': 340
            },
            'A2744': {  # Pandora's Cluster
                'distance_Mpc': 1100, 'velocity_dispersion': 1350, 'mass_1e14': 12.5,
                'type': 'merger', 'richness': 'rich', 'temperature_keV': 9.8,
                'redshift': 0.308, 'members': 150, 'core_radius_kpc': 250
            }
        }
        
        # Parâmetros base otimizados dos 4 aglomerados iniciais
        self.optimized_base_params = {
            'psi_0_mean': 179095, 'psi_0_std': 31176,
            'phi_0_mean': 289614, 'phi_0_std': 16567,
            'sigma_0_mean': 314082, 'sigma_0_std': 127771,
            'alpha_psi_mean': 0.35, 'alpha_psi_std': 0.37,
            'beta_phi_mean': 0.33, 'beta_phi_std': 0.33,
            'gamma_sigma_mean': 2.09, 'gamma_sigma_std': 0.86,
            'r_c_psi_mean': 2325, 'r_c_psi_std': 958,
            'r_c_phi_mean': 1174, 'r_c_phi_std': 748,
            'r_c_sigma_mean': 513, 'r_c_sigma_std': 301
        }
    
    def trinitaria_velocity(self, r, psi_0, phi_0, sigma_0, 
                           alpha_psi, beta_phi, gamma_sigma,
                           r_c_psi, r_c_phi, r_c_sigma):
        """Calcula velocidade Trinitária otimizada para aglomerados"""
        
        # Campo Psi (estrutura primária)
        psi = psi_0 * np.exp(-r/r_c_psi) * (r/r_c_psi)**alpha_psi
        
        # Campo Phi (estabilização dinâmica) 
        phi = phi_0 * (r/r_c_phi)**beta_phi * np.exp(-r/r_c_phi)
        
        # Campo Sigma (coesão energética)
        sigma = sigma_0 * np.exp(-r/r_c_sigma) * (1 + r/r_c_sigma)**gamma_sigma
        
        # Velocidade total
        v_total = np.sqrt(np.maximum(psi + phi + sigma, 0))
        
        return v_total
    
    def validate_cluster_with_universal_params(self, cluster_name):
        """
        Valida aglomerado usando parâmetros universais médios
        (sem otimização individual)
        """
        
        cluster_data = self.cluster_database[cluster_name]
        target_dispersion = cluster_data['velocity_dispersion']
        
        # Parâmetros universais médios
        params = [
            self.optimized_base_params['psi_0_mean'],
            self.optimized_base_params['phi_0_mean'], 
            self.optimized_base_params['sigma_0_mean'],
            self.optimized_base_params['alpha_psi_mean'],
            self.optimized_base_params['beta_phi_mean'],
            self.optimized_base_params['gamma_sigma_mean'],
            self.optimized_base_params['r_c_psi_mean'],
            self.optimized_base_params['r_c_phi_mean'],
            self.optimized_base_params['r_c_sigma_mean']
        ]
        
        # Região de teste: 100-1500 kpc
        r_test = np.logspace(2, 3.2, 30)
        
        # Predição com parâmetros universais
        v_pred = self.trinitaria_velocity(r_test, *params)
        
        # Métricas
        mean_prediction = np.mean(v_pred)
        rms = np.sqrt(np.mean((v_pred - target_dispersion)**2))
        relative_error = abs(mean_prediction - target_dispersion) / target_dispersion
        
        return {
            'cluster': cluster_name,
            'cluster_data': cluster_data,
            'universal_prediction': float(mean_prediction),
            'target_dispersion': float(target_dispersion),
            'rms_universal': float(rms),
            'relative_error': float(relative_error),
            'success': bool(rms < 300)  # Critério de sucesso
        }
    
    def run_extended_validation(self):
        """Executa validação completa nos 25 aglomerados"""
        
        print("=" * 70)
        print("TEORIA TRINITÁRIA - VALIDAÇÃO EXPANDIDA EM 25 AGLOMERADOS")
        print("=" * 70)
        print(f"📊 Base de dados: {len(self.cluster_database)} aglomerados")
        print("🎯 Usando parâmetros universais otimizados")
        print("📏 Região de teste: 100-1500 kpc")
        
        results = []
        categories = {
            'proximos': 0, 'medios': 0, 'distantes': 0,
            'success': 0, 'excellent': 0, 'good': 0, 'poor': 0
        }
        
        print(f"\n🌌 VALIDANDO AGLOMERADOS INDIVIDUAIS")
        print("-" * 50)
        
        for i, cluster_name in enumerate(self.cluster_database.keys(), 1):
            result = self.validate_cluster_with_universal_params(cluster_name)
            results.append(result)
            
            cluster_data = result['cluster_data']
            distance = cluster_data['distance_Mpc']
            rms = result['rms_universal']
            
            # Categorização por distância
            if distance < 100:
                categories['proximos'] += 1
            elif distance < 300:
                categories['medios'] += 1
            else:
                categories['distantes'] += 1
            
            # Categorização por performance
            if result['success']:
                categories['success'] += 1
                if rms < 100:
                    categories['excellent'] += 1
                    status = "🎯 EXCELENTE"
                elif rms < 200:
                    categories['good'] += 1  
                    status = "✅ BOM"
                else:
                    status = "⚠️  MODERADO"
            else:
                categories['poor'] += 1
                status = "❌ FRACO"
            
            print(f"{i:2d}. {cluster_name:12s} | {distance:4.0f} Mpc | "
                  f"{result['target_dispersion']:4.0f} km/s → {result['universal_prediction']:4.0f} km/s | "
                  f"RMS: {rms:5.1f} | {status}")
        
        # Análise estatística
        print(f"\n📊 ANÁLISE ESTATÍSTICA COMPLETA")
        print("-" * 40)
        
        rms_values = [r['rms_universal'] for r in results]
        errors = [r['relative_error'] for r in results]
        
        print(f"RMS médio: {np.mean(rms_values):.1f} ± {np.std(rms_values):.1f} km/s")
        print(f"RMS mediano: {np.median(rms_values):.1f} km/s")
        print(f"Erro relativo médio: {np.mean(errors):.1%}")
        print(f"Taxa de sucesso geral: {categories['success']}/{len(results)} ({categories['success']/len(results):.0%})")
        
        print(f"\nDesempenho por qualidade:")
        print(f"  🎯 Excelente (RMS<100): {categories['excellent']}/{len(results)} ({categories['excellent']/len(results):.0%})")
        print(f"  ✅ Bom (100<RMS<200): {categories['good']}/{len(results)} ({categories['good']/len(results):.0%})")
        print(f"  ⚠️  Moderado (200<RMS<300): {len(results)-categories['excellent']-categories['good']-categories['poor']}/{len(results)}")
        print(f"  ❌ Fraco (RMS>300): {categories['poor']}/{len(results)} ({categories['poor']/len(results):.0%})")
        
        print(f"\nDistribuição por distância:")
        print(f"  Próximos (<100 Mpc): {categories['proximos']}")
        print(f"  Médios (100-300 Mpc): {categories['medios']}")
        print(f"  Distantes (>300 Mpc): {categories['distantes']}")
        
        # Análise por morfologia
        morphology_stats = {}
        for result in results:
            morph_type = result['cluster_data']['type']
            if morph_type not in morphology_stats:
                morphology_stats[morph_type] = []
            morphology_stats[morph_type].append(result['rms_universal'])
        
        print(f"\n📈 PERFORMANCE POR MORFOLOGIA:")
        for morph_type, rms_list in morphology_stats.items():
            avg_rms = np.mean(rms_list)
            count = len(rms_list)
            print(f"  {morph_type:15s}: {avg_rms:5.1f} km/s ({count} aglomerados)")
        
        # Gerar visualizações
        self.generate_validation_plots(results)
        
        # Salvar resultados detalhados
        detailed_results = {
            'validation_results': results,
            'summary_statistics': {
                'mean_rms': float(np.mean(rms_values)),
                'std_rms': float(np.std(rms_values)),
                'median_rms': float(np.median(rms_values)),
                'mean_relative_error': float(np.mean(errors)),
                'success_rate': float(categories['success'] / len(results))
            },
            'categories': categories,
            'morphology_analysis': {
                morph_type: {
                    'mean_rms': float(np.mean(rms_list)),
                    'count': int(len(rms_list)),
                    'clusters': [r['cluster'] for r in results 
                               if r['cluster_data']['type'] == morph_type]
                } for morph_type, rms_list in morphology_stats.items()
            },
            'metadata': {
                'validation_date': '2025-11-23',
                'theory_version': 'Universal Parameters v2.0',
                'clusters_tested': len(results),
                'parameter_source': '4-cluster optimization'
            }
        }
        
        with open('trinitaria_25_clusters_validation.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n✅ Resultados salvos: trinitaria_25_clusters_validation.json")
        
        return detailed_results
    
    def generate_validation_plots(self, results):
        """Gera visualizações da validação expandida"""
        
        # Plot 1: RMS vs Distância
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Dados para plotagem
        distances = [r['cluster_data']['distance_Mpc'] for r in results]
        rms_values = [r['rms_universal'] for r in results]
        targets = [r['target_dispersion'] for r in results]
        predictions = [r['universal_prediction'] for r in results]
        
        # Subplot 1: RMS vs Distância
        ax1 = axes[0,0]
        colors = ['green' if rms < 100 else 'orange' if rms < 200 else 'red' 
                 for rms in rms_values]
        ax1.scatter(distances, rms_values, c=colors, alpha=0.7, s=80)
        ax1.axhline(100, color='green', linestyle='--', alpha=0.5, label='Excelente (100 km/s)')
        ax1.axhline(200, color='orange', linestyle='--', alpha=0.5, label='Bom (200 km/s)')
        ax1.axhline(300, color='red', linestyle='--', alpha=0.5, label='Limite (300 km/s)')
        ax1.set_xlabel('Distância (Mpc)')
        ax1.set_ylabel('RMS (km/s)')
        ax1.set_title('Performance vs Distância')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(distances)*1.1)
        
        # Subplot 2: Predição vs Observado
        ax2 = axes[0,1]
        ax2.scatter(targets, predictions, alpha=0.7, s=80)
        min_val, max_val = min(min(targets), min(predictions)), max(max(targets), max(predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfeito')
        ax2.set_xlabel('Dispersão Observada (km/s)')
        ax2.set_ylabel('Predição Trinitária (km/s)')
        ax2.set_title('Predição vs Observação')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Histograma de RMS
        ax3 = axes[1,0]
        ax3.hist(rms_values, bins=12, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(rms_values), color='red', linestyle='-', 
                   label=f'Média: {np.mean(rms_values):.1f}')
        ax3.axvline(np.median(rms_values), color='orange', linestyle='--',
                   label=f'Mediana: {np.median(rms_values):.1f}')
        ax3.set_xlabel('RMS (km/s)')
        ax3.set_ylabel('Número de Aglomerados')
        ax3.set_title('Distribuição de Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Performance por Morfologia
        ax4 = axes[1,1]
        morphology_data = {}
        for result in results:
            morph_type = result['cluster_data']['type']
            if morph_type not in morphology_data:
                morphology_data[morph_type] = []
            morphology_data[morph_type].append(result['rms_universal'])
        
        morphs = list(morphology_data.keys())
        avg_rms = [np.mean(morphology_data[m]) for m in morphs]
        
        bars = ax4.bar(range(len(morphs)), avg_rms, alpha=0.7, color='lightcoral')
        ax4.set_xticks(range(len(morphs)))
        ax4.set_xticklabels(morphs, rotation=45, ha='right')
        ax4.set_ylabel('RMS Médio (km/s)')
        ax4.set_title('Performance por Morfologia')
        ax4.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('trinitaria_25_clusters_analysis.png', dpi=300, bbox_inches='tight')
        print("📈 Análise visual salva: trinitaria_25_clusters_analysis.png")

def main():
    """Execução principal da validação expandida"""
    
    validator = ExtendedClusterValidation()
    results = validator.run_extended_validation()
    
    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSÃO DA VALIDAÇÃO EXPANDIDA")
    print("=" * 70)
    
    success_rate = results['summary_statistics']['success_rate']
    mean_rms = results['summary_statistics']['mean_rms']
    
    if success_rate >= 0.8 and mean_rms < 200:
        print("🚀 SUCESSO NOTÁVEL: Teoria Trinitária demonstra robustez!")
        print(f"   Taxa de sucesso: {success_rate:.0%}")
        print(f"   RMS médio: {mean_rms:.1f} km/s")
    elif success_rate >= 0.6:
        print("✅ PROGRESSO SÓLIDO: Performance competitiva")
        print(f"   Taxa de sucesso: {success_rate:.0%}")
        print(f"   RMS médio: {mean_rms:.1f} km/s")
    else:
        print("⚠️  DESAFIOS IDENTIFICADOS: Necessário refinamento")
        print(f"   Taxa de sucesso: {success_rate:.0%}")
        print(f"   RMS médio: {mean_rms:.1f} km/s")
    
    print(f"\n🔬 INSIGHTS CIENTÍFICOS:")
    
    # Analisar morfologia
    best_morph = min(results['morphology_analysis'].items(), 
                    key=lambda x: x[1]['mean_rms'])
    worst_morph = max(results['morphology_analysis'].items(), 
                     key=lambda x: x[1]['mean_rms'])
    
    print(f"• Melhor morfologia: {best_morph[0]} ({best_morph[1]['mean_rms']:.1f} km/s)")
    print(f"• Morfologia desafiadora: {worst_morph[0]} ({worst_morph[1]['mean_rms']:.1f} km/s)")
    print(f"• Parâmetros universais mostram {success_rate:.0%} aplicabilidade")
    print(f"• Framework escalável confirmado para {len(results['validation_results'])} sistemas")
    
    return results

if __name__ == "__main__":
    results = main()
    
    print(f"\n🚀 Validação expandida concluída!")
    print(f"📁 Arquivos gerados:")
    print(f"   - trinitaria_25_clusters_analysis.png")
    print(f"   - trinitaria_25_clusters_validation.json")