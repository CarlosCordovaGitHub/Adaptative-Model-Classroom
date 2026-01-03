"""
Test de Larga Duración (Simulación de Olvido y Retención).
Versión Corregida: Lectura correcta de formatos de API vs Estado Interno.
"""

import json
import os
import shutil
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from app.engine import build_engine
from simulator import StudentSimulator, StudentProfile, simulate_full_session

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

SIMULATED_DAYS_PASSED = 7   # Días que pasan entre sesión 1 y 2
TARGET_SKILL = "k_regla_potencia"

def clean_runtime():
    if os.path.exists("runtime"):
        shutil.rmtree("runtime")
    print("[SETUP] Memoria limpia.")

def time_travel_hack(student_id: str, days_back: int, engine) -> None:
    """
    Hackea el archivo de estado para simular que la última sesión fue hace N días.
    """
    state_path = engine.get_state_path(student_id)
    
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    print(f"\n[TIME TRAVEL] Hackeando reloj: Retrocediendo {days_back} días...")
    
    mastery = state.get("mastery", {})
    modified_count = 0
    
    # Restamos días a los timestamps
    fake_past_date = (datetime.now() - timedelta(days=days_back)).astimezone().isoformat()
    
    for skill, data in mastery.items():
        if data.get("last_update_ts"):
            data["last_update_ts"] = fake_past_date
            modified_count += 1
            
    # También modificar timestamps del historial para consistencia
    for h in state.get("history", []):
        h["ts"] = fake_past_date
            
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        
    print(f"[TIME TRAVEL] Éxito. {modified_count} skills envejecidos.")


# ==============================================================================
# TEST PRINCIPAL
# ==============================================================================

def run_longterm_test():
    print("="*80)
    print("TEST DE LARGA DURACIÓN: CURVA DEL OLVIDO")
    print("="*80)
    
    clean_runtime()
    engine = build_engine()
    
    # 1. Crear un estudiante promedio
    profile = StudentProfile(
        student_id="student_longterm_01",
        true_theta=0.5, # Capacidad media
        true_mastery={TARGET_SKILL: 0.2}, # Empieza sin saber
        learning_rate=0.3
    )
    
    simulator = StudentSimulator(profile, seed=123)
    
    # --------------------------------------------------------------------------
    # FASE 1: APRENDIZAJE (DÍA 0)
    # --------------------------------------------------------------------------
    print(f"\n>>> FASE 1: Sesión de Aprendizaje (Día 0)")
    print("Objetivo: Alcanzar dominio (> 0.85)")
    
    results_1 = simulate_full_session(simulator, engine, max_items=25)
    
    # Lectura del Estado Interno (Formato complejo)
    final_mastery_1 = results_1["final_state"]["mastery"].get(TARGET_SKILL, {}).get("p_mastery", 0.0)
    print(f"Mastery Final (Sesión 1): {final_mastery_1:.4f}")
    
    if final_mastery_1 < 0.8:
        print("⚠ Forzando mastery alto para demostración...")
        state = engine.get_student_state(profile.student_id)
        state["mastery"][TARGET_SKILL] = {"p_mastery": 0.95, "last_update_ts": engine._now_local_iso()}
        engine._save_student_state(profile.student_id, state)
        final_mastery_1 = 0.95

    # --------------------------------------------------------------------------
    # FASE 2: EL PASO DEL TIEMPO
    # --------------------------------------------------------------------------
    time_travel_hack(profile.student_id, SIMULATED_DAYS_PASSED, engine)
    
    # --------------------------------------------------------------------------
    # FASE 3: EL RETORNO (DÍA 7)
    # --------------------------------------------------------------------------
    print(f"\n>>> FASE 2: El Retorno (Día {SIMULATED_DAYS_PASSED})")
    print("Objetivo: Verificar que el mastery ha decaído.")
    
    # Ejecutamos 1 solo ítem para que el motor procese el decay
    results_2 = simulate_full_session(simulator, engine, max_items=1)
    
    # CORRECCIÓN AQUÍ: 
    # La API devuelve mastery plano {"skill": 0.5}, no el objeto complejo.
    # Así que leemos el valor directamente.
    first_item_mastery = results_2["items_administered"][0]["mastery"].get(TARGET_SKILL, 0.0)
    
    print(f"\nANÁLISIS DE RESULTADOS:")
    print(f"Mastery al cerrar Sesión 1:     {final_mastery_1:.4f}")
    print(f"Mastery al abrir Sesión 2:      {first_item_mastery:.4f} (Tras aplicar decay)")
    
    drop = final_mastery_1 - first_item_mastery
    print(f"Caída de conocimiento (Decay):  {drop:.4f}")
    
    if drop > 0.01:
        print("\n[ÉXITO] ✓ El sistema simuló el olvido correctamente.")
        print(f"          El estudiante perdió {drop*100:.1f}% de dominio en {SIMULATED_DAYS_PASSED} días.")
    else:
        print("\n[FALLO] ✗ El mastery se mantuvo estático.")

if __name__ == "__main__":
    run_longterm_test()