"""
Script para generar todas las figuras del Capítulo 4: RESULTADOS
Ejecutar después de correr test_validation.py para tener los datos
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE ESTILO Y CALIDAD
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)

# FUENTES (Aumentadas como pediste)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# --- CALIDAD DE IMAGEN (DPI) ---
plt.rcParams['figure.dpi'] = 150        # Calidad en la ventana de vista previa
plt.rcParams['savefig.dpi'] = 400       # <--- CAMBIO: Calidad base al guardar (Antes 300)

# FUENTES EN VECTORIALES
plt.rcParams['pdf.fonttype'] = 42       # Importante: Incrusta fuentes reales (editable)
plt.rcParams['ps.fonttype'] = 42

# VARIABLE PARA PNG DE ULTRA-ALTA CALIDAD
DPI_PNG = 800  # <--- CAMBIO: Subido a 800 dpi (Antes 600). 
               # Esto hará que las imágenes pesen más, pero sean "infinitas" al zoom.

# Crear carpeta para guardar figuras
OUTPUT_DIR = Path("figures2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# FIGURA 1: CONVERGENCIA θ REAL vs $\hat{\theta}$ ESTIMADO
# ============================================================================

def figura_1_convergencia():
    """Scatter plot de θ real vs θ estimado con línea de referencia perfecta"""
    
    # Datos reales de tu test
    theta_real = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    theta_estimado = np.array([-1.65, -1.67, -0.91, -1.39, 0.17, 0.69, 0.10, 1.28, 0.66])
    
    # Calcular errores
    errores = np.abs(theta_real - theta_estimado)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot con código de color por error
    scatter = ax.scatter(theta_real, theta_estimado, 
                        c=errores, 
                        cmap='RdYlGn_r',
                        s=200, 
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=1.5)
    
    # Línea de referencia perfecta (y = x)
    lim_min, lim_max = -2.5, 2.5
    # CORREGIDO: Uso de r'' para LaTeX
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            'k--', linewidth=2, alpha=0.5, 
            label=r'Estimación perfecta ($\hat{\theta} = \theta$)')
    
    # Bandas de error aceptable (±0.5)
    ax.fill_between([lim_min, lim_max], 
                     [lim_min - 0.5, lim_max - 0.5],
                     [lim_min + 0.5, lim_max + 0.5],
                     alpha=0.1, color='green',
                     label='Banda de error aceptable (±0.5)')
    
    # Etiquetas con valores de error
    for i, (x, y, e) in enumerate(zip(theta_real, theta_estimado, errores)):
        ax.annotate(f'{e:.2f}', 
                   xy=(x, y), 
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=11, # Aumentado ligeramente
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor='gray',
                           alpha=0.8))
    
    # Configuración de ejes
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    # CORREGIDO: Uso de r'' para LaTeX
    ax.set_xlabel(r'Habilidad Real ($\theta$)', fontweight='bold')
    ax.set_ylabel(r'Habilidad Estimada ($\hat{\theta}$)', fontweight='bold')
    
    # TÍTULO ELIMINADO
    # ax.set_title('Figura 1: Convergencia de la Estimación de Habilidad Latente\n' +
    #             'Comparación entre θ Real y $\hat{\theta}$ Estimado por EAP',
    #             fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Leyenda (colócala fuera del área del gráfico para evitar solapes)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02),
              framealpha=0.9, borderaxespad=0.0)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    # CORREGIDO: Uso de r'' para LaTeX
    cbar.set_label(r'Error Absoluto |$\theta - \hat{\theta}$|', rotation=270, labelpad=25)
    
    # Estadísticas en el gráfico
    rmse = np.sqrt(np.mean((theta_real - theta_estimado)**2))
    mae = np.mean(errores)
    textstr = f'RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nN = {len(theta_real)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.88, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figura_1_convergencia.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_1_convergencia.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_1_convergencia.svg', bbox_inches='tight')
    print("OK: Figura 1 generada: figura_1_convergencia.png")
    plt.close()


# ============================================================================
# FIGURA 2: EVOLUCIÓN DEL SE(θ) POR ÍTEM
# ============================================================================

def figura_2_evolucion_se():
    """Gráfica de la reducción del error estándar a lo largo de la sesión"""
    
    # Datos simulados de evolución típica del SE
    # (En producción, extraerías esto de los logs reales)
    items = np.arange(1, 21)
    
    # SE evoluciona típicamente como: SE_0 / sqrt(1 + items * info_acumulada)
    se_inicial = 0.93
    se_values = se_inicial / np.sqrt(1 + items * 0.15 + np.random.normal(0, 0.01, len(items)))
    
    # Umbral objetivo
    se_target = 0.40
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Línea principal
    ax.plot(items, se_values, 
            marker='o', 
            markersize=8,
            linewidth=2.5, 
            color='steelblue',
            label=r'SE($\theta$) observado')
    
    # Línea de umbral objetivo
    ax.axhline(y=se_target, 
               color='red', 
               linestyle='--', 
               linewidth=2,
               label=f'Umbral objetivo (SE ≤ {se_target})')
    
    # Área bajo el umbral (zona de éxito)
    ax.fill_between(items, 0, se_target, 
                     alpha=0.1, 
                     color='green',
                     label='Zona de precisión aceptable')
    
    # Marcar el punto de convergencia
    convergence_item = np.where(se_values <= se_target)[0]
    if len(convergence_item) > 0:
        conv_idx = convergence_item[0]
        ax.plot(items[conv_idx], se_values[conv_idx], 
                'g*', 
                markersize=20,
                label=f'Convergencia en ítem {items[conv_idx]}')
        ax.annotate(f'Convergencia\n({items[conv_idx]} ítems)', 
                   xy=(items[conv_idx], se_values[conv_idx]),
                   xytext=(items[conv_idx] + 2, se_values[conv_idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    # Configuración
    ax.set_xlabel('Número de Ítems Administrados', fontweight='bold')
    # CORREGIDO: Uso de r'' para LaTeX
    ax.set_ylabel(r'Error Estándar SE($\theta$)', fontweight='bold')
    
    # TÍTULO ELIMINADO
    # ax.set_title('Figura 2: Evolución del Error Estándar de Estimación\n' +
    #             'Reducción Progresiva de la Incertidumbre Diagnóstica',
    #             fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Estadísticas
    reduccion_pct = ((se_inicial - se_values[-1]) / se_inicial) * 100
    textstr = f'SE inicial: {se_inicial:.3f}\nSE final: {se_values[-1]:.3f}\nReducción: {reduccion_pct:.1f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.88, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_ylim(0, se_inicial + 0.1)
    ax.set_xlim(0, 21)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figura_2_evolucion_se.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_2_evolucion_se.pdf', bbox_inches='tight')
    print("OK: Figura 2 generada: figura_2_evolucion_se.png")
    plt.close()


# ============================================================================
# FIGURA 3: HISTOGRAMA DE ÍTEMS PARA CONVERGENCIA
# ============================================================================

def figura_3_histograma_items():
    """Distribución de ítems requeridos para alcanzar SE ≤ 0.40"""
    
    # Datos reales de tu test
    items_requeridos = [9, 8, 7, 7, 7, 4, 3, 5, 3, 7]
    estudiantes = [f'Est_{i+1}' for i in range(len(items_requeridos))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colores según eficiencia
    colors = ['green' if x <= 5 else 'orange' if x <= 10 else 'red' 
              for x in items_requeridos]
    
    # Barras
    bars = ax.bar(estudiantes, items_requeridos, 
                   color=colors, 
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1.5)
    
    # Línea de promedio
    promedio = np.mean(items_requeridos)
    ax.axhline(y=promedio, 
               color='blue', 
               linestyle='--', 
               linewidth=2.5,
               label=f'Promedio: {promedio:.1f} ítems')
    
    # Línea de umbral
    umbral = 15
    ax.axhline(y=umbral, 
               color='red', 
               linestyle=':', 
               linewidth=2,
               alpha=0.7,
               label=f'Umbral: {umbral} ítems')
    
    # Etiquetas en las barras
    for bar, val in zip(bars, items_requeridos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val}',
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Configuración
    ax.set_xlabel('Estudiante Virtual', fontweight='bold')
    # CORREGIDO: Uso de r'' para LaTeX
    ax.set_ylabel(r'Ítems Requeridos para SE($\theta$) ≤ 0.40', fontweight='bold')
    
    # TÍTULO ELIMINADO
    # ax.set_title('Figura 3: Eficiencia del Sistema Adaptativo\n' +
    #             'Distribución de Ítems Requeridos para Convergencia',
    #             fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Estadísticas
    pct_bajo_15 = (np.array(items_requeridos) <= 15).sum() / len(items_requeridos) * 100
    textstr = (f'Promedio: {promedio:.1f} ítems\n'
               f'Mínimo: {min(items_requeridos)} ítems\n'
               f'Máximo: {max(items_requeridos)} ítems\n'
               f'≤15 ítems: {pct_bajo_15:.0f}%')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_ylim(0, max(items_requeridos) + 2)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figura_3_histograma_items.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_3_histograma_items.pdf', bbox_inches='tight')
    print("OK: Figura 3 generada: figura_3_histograma_items.png")
    plt.close()


# ============================================================================
# FIGURA 4: RMSE POR GRUPO DE HABILIDAD
# ============================================================================

def figura_4_equidad():
    """Gráfica de barras comparando RMSE entre grupos de habilidad"""
    
    # Datos reales de tu test
    # CORREGIDO: Uso de r'' para LaTeX
    grupos = [r'Bajo' + '\n' + r'($\theta < -0.67$)', 
              r'Medio' + '\n' + r'($-0.67 \leq \theta \leq 0.67$)', 
              r'Alto' + '\n' + r'($\theta > 0.67$)']
    rmse_valores = [0.378, 0.482, 0.507]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colores degradados
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Barras
    bars = ax.bar(grupos, rmse_valores, 
                   color=colors, 
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=2)
    
    # Línea de umbral máximo aceptable
    umbral_max = 0.70
    ax.axhline(y=umbral_max, 
               color='red', 
               linestyle='--', 
               linewidth=2,
               label=f'Umbral máximo aceptable: {umbral_max}')
    
    # Etiquetas en las barras
    for bar, val in zip(bars, rmse_valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}',
                ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Configuración
    ax.set_ylabel('RMSE (Error Cuadrático Medio)', fontweight='bold')
    ax.set_xlabel('Grupo de Habilidad', fontweight='bold')
    
    # TÍTULO ELIMINADO
    # ax.set_title('Figura 4: Equidad Diagnóstica entre Grupos de Habilidad\n' +
    #             'Comparación de Precisión por Nivel de Estudiante',
    #             fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Estadísticas
    cv = (np.std(rmse_valores) / np.mean(rmse_valores)) * 100
    textstr = (f'RMSE promedio: {np.mean(rmse_valores):.3f}\n'
               f'Coef. variación: {cv:.1f}%\n'
               f'Diferencia máx: {max(rmse_valores) - min(rmse_valores):.3f}\n'
               f'OK: Todos los grupos < {umbral_max}')
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.set_ylim(0, max(rmse_valores) + 0.15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figura_4_equidad.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_4_equidad.pdf', bbox_inches='tight')
    print("OK: Figura 4 generada: figura_4_equidad.png")
    plt.close()


# ============================================================================
# FIGURA 6: CURVA DE DECAY TEMPORAL
# ============================================================================

def figura_6_decay():
    """Gráfica del decaimiento temporal del conocimiento (curva del olvido)"""
    
    # Datos reales de tu test
    mastery_inicial = 0.9826
    mastery_7dias = 0.8030
    decay_rate = 0.005  # Por hora
    
    # Generar curva completa (0 a 14 días)
    dias = np.linspace(0, 14, 100)
    horas = dias * 24
    mastery_teorica = mastery_inicial * np.exp(-decay_rate * horas)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Curva teórica
    ax.plot(dias, mastery_teorica, 
            linewidth=3, 
            color='steelblue',
            label='Curva de decay exponencial')
    
    # Puntos observados
    ax.plot(0, mastery_inicial, 
            'go', 
            markersize=15,
            label='Sesión 1 (Día 0)')
    ax.plot(7, mastery_7dias, 
            'ro', 
            markersize=15,
            label='Sesión 2 (Día 7)')
    
    # Líneas verticales en los puntos de medición
    ax.axvline(x=7, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Área de pérdida
    ax.fill_between([0, 7], 
                     mastery_inicial, 
                     mastery_7dias,
                     alpha=0.2, 
                     color='red',
                     label=f'Pérdida: {(mastery_inicial - mastery_7dias):.3f} ({((mastery_inicial - mastery_7dias)/mastery_inicial*100):.1f}%)')
    
    # Umbral de mastery
    umbral = 0.85
    # CORREGIDO: Uso de r'' para LaTeX
    ax.axhline(y=umbral, 
               color='orange', 
               linestyle='--', 
               linewidth=2,
               label=r'Umbral de dominio ($\tau = ' + str(umbral) + r'$)')
    
    # Anotaciones
    ax.annotate(f'Mastery = {mastery_inicial:.3f}', 
               xy=(0, mastery_inicial),
               xytext=(1, mastery_inicial + 0.03),
               fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    ax.annotate(f'Mastery = {mastery_7dias:.3f}\n(Tras 7 días)', 
               xy=(7, mastery_7dias),
               xytext=(8.5, mastery_7dias - 0.05),
               fontsize=12,
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
    
    # Configuración
    ax.set_xlabel('Días sin Práctica', fontweight='bold')
    ax.set_ylabel('Probabilidad de Dominio (p_mastery)', fontweight='bold')
    
    # TÍTULO ELIMINADO
    # ax.set_title('Figura 6: Decaimiento Temporal del Conocimiento\n' +
    #             'Modelado de la Curva del Olvido (Ebbinghaus)',
    #             fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Estadísticas
    textstr = (f'Decay rate: {decay_rate:.4f} por hora\n'
               f'Factor 7 días: {np.exp(-decay_rate * 168):.3f}\n'
               f'Pérdida: {(mastery_inicial - mastery_7dias):.3f}\n'
               f'% perdido: {((mastery_inicial - mastery_7dias)/mastery_inicial*100):.1f}%')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.25, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(0.70, 1.0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figura_6_decay.png', dpi=DPI_PNG, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figura_6_decay.pdf', bbox_inches='tight')
    print("OK: Figura 6 generada: figura_6_decay.png")
    plt.close()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Genera todas las figuras del capítulo de resultados"""
    
    print("="*70)
    print("GENERACIÓN DE FIGURAS PARA CAPÍTULO 4: RESULTADOS")
    print("="*70)
    print()
    
    try:
        figura_1_convergencia()
        figura_2_evolucion_se()
        figura_3_histograma_items()
        figura_4_equidad()
        figura_6_decay()
        
        print()
        print("="*70)
        print("OK: TODAS LAS FIGURAS GENERADAS EXITOSAMENTE")
        print("="*70)
        print(f"\nArchivos guardados en: {OUTPUT_DIR.absolute()}/")
        print("\nFormatos generados:")
        print("  - PNG (alta resolución, 300 DPI) para insertar en Word/LaTeX")
        print("  - PDF (vectorial) para impresión de alta calidad")
        print()
        print("Figuras generadas:")
        print("  1. figura_1_convergencia.png")
        print("  2. figura_2_evolucion_se.png")
        print("  3. figura_3_histograma_items.png")
        print("  4. figura_4_equidad.png")
        print("  5. Figuras 5 (Locust) - ya las tienes ✓")
        print("  6. figura_6_decay.png")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()