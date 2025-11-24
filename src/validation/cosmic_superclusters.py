#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEORIA TRINITÁRIA - SUPERAGLOMERADOS (VERSÃO SIMPLIFICADA)
========================================================

Implementação funcional da Teoria Trinitária para escalas 10-200 Mpc

Autor: Sistema de IA Avançado
Data: 23 de Novembro de 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import json

class SuperclusterTrinitaria:
    """Teoria Trinitária para Superaglomerados"""
    
    def __init__(self):
        # Superaglomerados conhecidos
        self.superclusters = {
            'Local_Supercluster': {
                'extent_Mpc': 110, 'mass_1e15': 1.2, 'velocity_km_s': 627, 'type': 'sheet'
            },
            'Perseus_Pisces': {
                'extent_Mpc': 180, 'mass_1e15': 8.5, 'velocity_km_s': 400, 'type': 'filament'  
            },
            'Coma_Supercluster': {
                'extent_Mpc': 160, 'mass_1e15': 15.2, 'velocity_km_s': 200, 'type': 'node'
            },
            'Shapley_Supercluster': {
                'extent_Mpc': 400, 'mass_1e15': 45.0, 'velocity_km_s': 850, 'type': 'massive'
            },
            'Hercules_Supercluster': {
                'extent_Mpc': 220, 'mass_1e15': 12.8, 'velocity_km_s': 300, 'type': 'elongated'
            }
        }
    
    def supercluster_velocity(self, r_Mpc, extent_Mpc, mass_1e15):
        """Calcula velocidade em superaglomerado usando escalas ampliadas"""
        
        # Parâmetros escalados para superaglomerados
        psi_0 = mass_1e15 * 50000  # Proporcional à massa
        phi_0 = mass_1e15 * 80000
        sigma_0 = mass_1e15 * 120000
        
        # Raios característicos baseados na extensão
        r_c_psi = extent_Mpc * 0.3   # 30% da extensão
        r_c_phi = extent_Mpc * 0.5   # 50% da extensão  
        r_c_sigma = extent_Mpc * 0.15 # 15% da extensão
        
        # Expoentes otimizados para superaglomerados
        alpha_psi = 0.25
        beta_phi = 0.2
        gamma_sigma = 1.2
        
        # Campos Trinitários
        psi = psi_0 * np.exp(-r_Mpc/r_c_psi) * (r_Mpc/r_c_psi)**alpha_psi
        phi = phi_0 * (r_Mpc/r_c_phi)**beta_phi * np.exp(-r_Mpc/r_c_phi)
        sigma = sigma_0 * np.exp(-r_Mpc/r_c_sigma) * (1 + r_Mpc/r_c_sigma)**gamma_sigma
        
        # Velocidade total
        v_total = np.sqrt(np.maximum(psi + phi + sigma, 0))
        
        return v_total
    
    def validate_superclusters(self):
        """Valida teoria em superaglomerados"""
        
        print("=" * 65)
        print("TEORIA TRINITÁRIA - SUPERAGLOMERADOS")  
        print("=" * 65)
        
        results = {}
        
        for name, data in self.superclusters.items():
            extent = data['extent_Mpc']
            mass = data['mass_1e15'] 
            v_obs = data['velocity_km_s']
            
            # Velocidade central predita
            r_center = extent * 0.1  # 10% da extensão (região central)
            v_pred = self.supercluster_velocity(r_center, extent, mass)
            
            error = abs(v_pred - v_obs) / v_obs
            
            print(f"\n🌌 {name.replace('_', ' ')}")
            print(f"   Extensão: {extent} Mpc")
            print(f"   Massa: {mass:.1f} × 10¹⁵ M☉")
            print(f"   V. observada: {v_obs} km/s")
            print(f"   V. predita: {v_pred:.0f} km/s")
            print(f"   Erro: {error:.1%}")
            
            if error < 0.3:
                status = "✅ BOM"
            elif error < 0.5:
                status = "⚠️  MODERADO"
            else:
                status = "❌ DESAFIADOR"
                
            print(f"   Status: {status}")
            
            results[name] = {
                'predicted': float(v_pred),
                'observed': float(v_obs),
                'error': float(error),
                'extent_Mpc': extent,
                'mass_1e15': mass
            }
        
        # Estatísticas
        errors = [r['error'] for r in results.values()]
        mean_error = np.mean(errors)
        success_rate = sum(e < 0.5 for e in errors) / len(errors)
        
        print(f"\n📊 ESTATÍSTICAS")
        print(f"   Erro médio: {mean_error:.1%}")
        print(f"   Taxa de sucesso: {success_rate:.0%}")
        
        return results
    
    def cosmic_correlation_function(self):
        """Calcula função de correlação cósmica"""
        
        print(f"\n🌐 FUNÇÃO DE CORRELAÇÃO CÓSMICA")
        print("-" * 35)
        
        # Separações de 1 a 300 Mpc
        r_Mpc = np.logspace(0, 2.5, 50)
        
        # Parâmetros cósmicos médios
        extent_cosmic = 200  # Mpc
        mass_cosmic = 20     # 10^15 M_sun
        
        # Velocidades em função da separação
        v_profile = self.supercluster_velocity(r_Mpc, extent_cosmic, mass_cosmic)
        
        # Função de correlação aproximada
        v_0 = 1000  # km/s normalização
        xi_trinitaria = (v_profile / v_0)**1.8 - 1
        
        # Modelo clássico para comparação
        r_0 = 5.0   # Mpc
        xi_classic = (r_Mpc / r_0)**(-1.8)
        
        print(f"ξ(5 Mpc) Trinitária: {np.interp(5.0, r_Mpc, xi_trinitaria):.3f}")
        print(f"ξ(50 Mpc) Trinitária: {np.interp(50.0, r_Mpc, xi_trinitaria):.3f}")
        
        return {
            'separations_Mpc': r_Mpc,
            'xi_trinitaria': xi_trinitaria,
            'xi_classic': xi_classic
        }

def run_cosmic_analysis():
    """Execução principal"""
    
    print("=" * 70)
    print("TEORIA TRINITÁRIA - ANÁLISE CÓSMICA COMPLETA")
    print("=" * 70)
    
    cosmic = SuperclusterTrinitaria()
    
    # 1. Validação de superaglomerados
    sc_results = cosmic.validate_superclusters()
    
    # 2. Função de correlação
    corr_data = cosmic.cosmic_correlation_function()
    
    # 3. Visualização
    print(f"\n📈 GERANDO VISUALIZAÇÃO...")
    
    plt.figure(figsize=(12, 8))
    
    plt.loglog(corr_data['separations_Mpc'], np.abs(corr_data['xi_trinitaria']), 
              'b-', linewidth=3, label='Trinitária (superaglomerados)')
    plt.loglog(corr_data['separations_Mpc'], corr_data['xi_classic'],
              'r--', linewidth=2, label='Modelo clássico')
    
    plt.xlabel('Separação (Mpc)')
    plt.ylabel('|ξ(r)|') 
    plt.title('Função de Correlação Cósmica - Teoria Trinitária')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Marcar escalas
    plt.axvline(5, color='orange', alpha=0.5, label='Aglomerados')
    plt.axvline(50, color='purple', alpha=0.5, label='Superaglomerados')
    
    plt.savefig('trinitaria_cosmic_simple.png', dpi=300, bbox_inches='tight')
    print("   ✅ Visualização: trinitaria_cosmic_simple.png")
    
    # 4. Salvar resultados
    results = {
        'supercluster_validation': sc_results,
        'correlation_function': {
            'separations_Mpc': corr_data['separations_Mpc'].tolist(),
            'xi_trinitaria': corr_data['xi_trinitaria'].tolist(),
            'xi_classic': corr_data['xi_classic'].tolist()
        },
        'summary': {
            'mean_error': float(np.mean([r['error'] for r in sc_results.values()])),
            'success_rate': float(sum(r['error'] < 0.5 for r in sc_results.values()) / len(sc_results)),
            'superclusters_tested': len(sc_results)
        },
        'metadata': {
            'date': '2025-11-23',
            'version': 'Simplified Cosmic Trinitária',
            'scales': '10-300 Mpc'
        }
    }
    
    with open('trinitaria_cosmic_simple.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("   ✅ Resultados: trinitaria_cosmic_simple.json")
    
    # 5. Conclusões
    success_rate = results['summary']['success_rate'] 
    mean_error = results['summary']['mean_error']
    
    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSÕES")
    print("=" * 70)
    
    if success_rate >= 0.6:
        print("🚀 SUCESSO: Framework cósmico promissor!")
    else:
        print("⚠️  DESENVOLVIMENTO: Necessário refinamento")
        
    print(f"   Taxa de sucesso: {success_rate:.0%}")
    print(f"   Erro médio: {mean_error:.1%}")
    print(f"   Escalas: 10-300 Mpc implementadas")
    
    return results

if __name__ == "__main__":
    results = run_cosmic_analysis()
    print(f"\n🚀 Análise cósmica concluída!")
    print(f"📁 Arquivos: trinitaria_cosmic_simple.png, trinitaria_cosmic_simple.json")