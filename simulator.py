"""
Simulador de estudiantes para validación del motor adaptativo.

Permite generar respuestas sintéticas basadas en modelos IRT y BKT
para probar el comportamiento del engine sin estudiantes reales.
"""

from __future__ import annotations

import random
import math
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import uuid


@dataclass
class StudentProfile:
    """
    Perfil de estudiante simulado con parámetros conocidos.
    """
    student_id: str
    true_theta: float  # Habilidad real IRT
    true_mastery: Dict[str, float]  # Probabilidad real de dominio por skill
    response_consistency: float = 0.9  # Qué tan consistente es (0-1)
    learning_rate: float = 0.15  # Tasa de aprendizaje durante sesión
    fatigue_factor: float = 0.02  # Incremento en tiempo de respuesta por fatiga
    
    def __post_init__(self):
        # Validaciones
        assert -4.0 <= self.true_theta <= 4.0, "theta debe estar en [-4, 4]"
        for skill, mastery in self.true_mastery.items():
            assert 0.0 <= mastery <= 1.0, f"Mastery de {skill} debe estar en [0, 1]"
        assert 0.0 <= self.response_consistency <= 1.0
        assert 0.0 <= self.learning_rate <= 1.0
        assert self.fatigue_factor >= 0.0


class StudentSimulator:
    """
    Simula comportamiento de estudiantes para validación del engine.
    """
    
    def __init__(self, profile: StudentProfile, seed: int = None):
        self.profile = profile
        self.items_answered = 0
        self.session_history: List[Dict[str, Any]] = []
        
        if seed is not None:
            random.seed(seed)
    
    def generate_response(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera una respuesta simulada para un ítem.
        
        Returns:
            {
                "is_correct": bool,
                "response_time_ms": int,
                "hint_used": bool
            }
        """
        # Determinar si respuesta es correcta
        is_correct = self._determine_correctness(item)
        
        # Generar tiempo de respuesta
        response_time_ms = self._generate_response_time(item, is_correct)
        
        # Determinar si usa hint (más probable si tiene dificultades)
        hint_used = self._determine_hint_usage(item, is_correct)
        
        # Actualizar historia
        self.items_answered += 1
        self.session_history.append({
            "item_id": item["item_id"],
            "is_correct": is_correct,
            "response_time_ms": response_time_ms,
            "hint_used": hint_used
        })
        
        # Aprendizaje incremental (el estudiante mejora ligeramente con cada ítem)
        self._apply_learning_gain(item)
        
        return {
            "is_correct": is_correct,
            "response_time_ms": response_time_ms,
            "hint_used": hint_used
        }
    
    def _determine_correctness(self, item: Dict[str, Any]) -> bool:
        """
        Determina si la respuesta es correcta basado en IRT o BKT.
        """
        # Si el ítem tiene parámetros IRT, usar modelo IRT
        if item.get("irt"):
            p_correct_irt = self._irt_3pl_prob(self.profile.true_theta, item)
            
            # Aplicar consistency: reduce la probabilidad de respuestas aleatorias
            if random.random() > self.profile.response_consistency:
                # Respuesta inconsistente (ruido)
                return random.random() < 0.5
            
            return random.random() < p_correct_irt
        
        # Si no hay IRT, usar BKT
        skill_ids = item.get("skill_ids", [])
        if skill_ids:
            skill_id = skill_ids[0]  # Usar primer skill
            mastery = self.profile.true_mastery.get(skill_id, 0.5)
            
            # Modelo simplificado: P(correct) = mastery * (1-slip) + (1-mastery) * guess
            slip = 0.1
            guess = 0.25
            p_correct_bkt = mastery * (1 - slip) + (1 - mastery) * guess
            
            if random.random() > self.profile.response_consistency:
                return random.random() < 0.5
            
            return random.random() < p_correct_bkt
        
        # Fallback: 50% random
        return random.random() < 0.5
    
    def _irt_3pl_prob(self, theta: float, item: Dict[str, Any]) -> float:
        """Probabilidad de respuesta correcta según modelo 3PL."""
        irt = item.get("irt", {})
        a = float(irt.get("a", 1.0))
        b = float(irt.get("b", 0.0))
        c = float(irt.get("c", 0.0))
        
        z = a * (theta - b)
        try:
            logistic = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            logistic = 1.0 if z > 0 else 0.0
        
        return c + (1.0 - c) * logistic
    
    def _generate_response_time(self, item: Dict[str, Any], is_correct: bool) -> int:
        """
        Genera tiempo de respuesta realista.
        
        Factores:
        - Dificultad del ítem
        - Fatiga acumulada
        - Correctitud (incorrectas tienden a ser más rápidas o más lentas)
        """
        # Tiempo base según dificultad
        difficulty = item.get("difficulty_label", "media")
        base_times = {
            "facil": 30000,    # 30 segundos
            "media": 60000,    # 60 segundos
            "dificil": 90000   # 90 segundos
        }
        base_time = base_times.get(difficulty, 60000)
        
        # Ajuste por fatiga
        fatigue_multiplier = 1.0 + (self.items_answered * self.profile.fatigue_factor)
        
        # Ajuste por correctitud
        if is_correct:
            # Respuestas correctas: distribución normal alrededor del tiempo base
            time = base_time * fatigue_multiplier * random.gauss(1.0, 0.2)
        else:
            # Respuestas incorrectas: más variabilidad
            # Puede ser muy rápido (adivinanza) o muy lento (confusión)
            if random.random() < 0.3:
                # Adivinanza rápida
                time = base_time * fatigue_multiplier * random.uniform(0.3, 0.6)
            else:
                # Intentó pero falló
                time = base_time * fatigue_multiplier * random.gauss(1.2, 0.3)
        
        # Límites razonables
        time = max(5000, min(300000, time))  # Entre 5 seg y 5 min
        
        return int(time)
    
    def _determine_hint_usage(self, item: Dict[str, Any], is_correct: bool) -> bool:
        """
        Determina si el estudiante usa hint.
        
        Más probable en ítems difíciles y si la respuesta fue incorrecta.
        """
        difficulty = item.get("difficulty_label", "media")
        
        # Probabilidad base de usar hint
        hint_probs = {
            "facil": 0.05,
            "media": 0.15,
            "dificil": 0.30
        }
        p_hint = hint_probs.get(difficulty, 0.15)
        
        # Aumenta si respondió incorrectamente (podría haber ayudado)
        if not is_correct:
            p_hint *= 1.5
        
        # Estudiantes con baja mastery usan más hints
        skill_ids = item.get("skill_ids", [])
        if skill_ids:
            skill_id = skill_ids[0]
            mastery = self.profile.true_mastery.get(skill_id, 0.5)
            if mastery < 0.5:
                p_hint *= 1.3
        
        return random.random() < min(p_hint, 0.5)
    
    def _apply_learning_gain(self, item: Dict[str, Any]) -> None:
        """
        Aplica ganancia de aprendizaje incremental después de responder.
        
        El estudiante mejora ligeramente en los skills del ítem.
        """
        skill_ids = item.get("skill_ids", [])
        
        for skill_id in skill_ids:
            if skill_id in self.profile.true_mastery:
                current_mastery = self.profile.true_mastery[skill_id]
                
                # Ganancia proporcional a learning_rate y espacio para mejorar
                gain = self.profile.learning_rate * (1.0 - current_mastery) * 0.1
                
                new_mastery = min(1.0, current_mastery + gain)
                self.profile.true_mastery[skill_id] = new_mastery
        
        # El theta también puede mejorar ligeramente (efecto de práctica)
        theta_gain = self.profile.learning_rate * 0.02
        self.profile.true_theta = min(4.0, self.profile.true_theta + theta_gain)


def create_student_profiles(n_students: int = 10) -> List[StudentProfile]:
    """
    Crea perfiles de estudiantes diversos para simulación.
    
    Args:
        n_students: Número de perfiles a generar
    
    Returns:
        Lista de StudentProfile con variedad de habilidades
    """
    profiles = []
    
    # Distribuir theta uniformemente
    theta_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for i in range(n_students):
        student_id = f"sim_student_{i+1:03d}"
        
        # Theta del pool o aleatorio
        if i < len(theta_values):
            true_theta = theta_values[i]
        else:
            true_theta = random.uniform(-2.0, 2.0)
        
        # Mastery correlacionado con theta pero con variación
        # Estudiantes con alto theta tienden a tener alto mastery
        base_mastery = (true_theta + 2.0) / 4.0  # Normalizar [-2, 2] a [0, 1]
        
        true_mastery = {
            "k_regla_potencia": max(0.1, min(0.9, base_mastery + random.gauss(0, 0.15))),
            "k_regla_cadena": max(0.05, min(0.85, base_mastery * 0.7 + random.gauss(0, 0.15)))
        }
        
        # Variación en consistencia y learning rate
        response_consistency = random.uniform(0.8, 0.95)
        learning_rate = random.uniform(0.10, 0.20)
        fatigue_factor = random.uniform(0.01, 0.03)
        
        profile = StudentProfile(
            student_id=student_id,
            true_theta=true_theta,
            true_mastery=true_mastery,
            response_consistency=response_consistency,
            learning_rate=learning_rate,
            fatigue_factor=fatigue_factor
        )
        
        profiles.append(profile)
    
    return profiles


def simulate_full_session(
    simulator: StudentSimulator,
    engine,
    topic_id: str = "t_derivadas",
    skill_ids: List[str] = None,
    max_items: int = 20
) -> Dict[str, Any]:
    """
    Simula una sesión completa de evaluación adaptativa.
    
    Args:
        simulator: Simulador de estudiante
        engine: Engine del motor adaptativo
        topic_id: ID del tema
        skill_ids: Lista de skills a evaluar
        max_items: Máximo de ítems a administrar
    
    Returns:
        Diccionario con resultados de la sesión simulada
    """
    if skill_ids is None:
        skill_ids = ["k_regla_potencia", "k_regla_cadena"]
    
    session_id = f"sim_session_{uuid.uuid4().hex[:8]}"
    student_id = simulator.profile.student_id
    
    # Obtener primer ítem (sin evento previo, usar ítem medio)
    # En producción, el Componente A manejaría esto
    first_item = None
    for item in engine.item_bank.get("items", []):
        if item["topic_id"] == topic_id and item["difficulty_label"] == "media":
            first_item = item
            break
    
    if not first_item:
        first_item = engine.item_bank["items"][0]
    
    results = {
        "student_id": student_id,
        "session_id": session_id,
        "true_theta": simulator.profile.true_theta,
        "true_mastery": dict(simulator.profile.true_mastery),
        "items_administered": [],
        "final_state": None,
        "convergence_history": []
    }
    
    current_item = first_item
    
    for i in range(max_items):
        # Estudiante responde
        response = simulator.generate_response(current_item)
        
        # Construir payload A->B
        payload = {
            "event_type": "activity_result",
            "event_version": "1.0",
            "student": {
                "student_id": student_id,
                "session_id": session_id
            },
            "context": {
                "topic_id": topic_id,
                "skill_ids": skill_ids,
                "language": "es",
                "mode": "practice"
            },
            "activity": {
                "item_id": current_item["item_id"],
                "item_format": current_item.get("format", "mcq"),
                "difficulty_label": current_item.get("difficulty_label", "media")
            },
            "response": response,
            "telemetry": {}
        }
        
        # Enviar a engine
        recommendation = engine.process_activity_result(payload)
        
        # Registrar resultados
        results["items_administered"].append({
            "item_id": current_item["item_id"],
            "difficulty": current_item.get("difficulty_label"),
            "is_correct": response["is_correct"],
            "response_time_ms": response["response_time_ms"],
            "theta_hat": recommendation["state"]["theta_hat"],
            "se_theta": recommendation["state"]["se_theta"],
            "mastery": dict(recommendation["state"]["mastery"])
        })
        
        results["convergence_history"].append({
            "item_number": i + 1,
            "theta_hat": recommendation["state"]["theta_hat"],
            "se_theta": recommendation["state"]["se_theta"],
            "theta_error": abs(recommendation["state"]["theta_hat"] - simulator.profile.true_theta)
        })
        
        # Verificar stopping
        if not recommendation["recommendation"].get("present_item", True):
            results["stopped_by"] = recommendation["recommendation"].get("stop_reason", "unknown")
            break
        
        # Obtener siguiente ítem
        next_item_id = recommendation["recommendation"]["item_id"]
        current_item = engine.items_by_id.get(next_item_id)
        
        if not current_item:
            break
    
    # Estado final
    results["final_state"] = engine.get_student_state(student_id)
    
    return results


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from app.engine import build_engine
    
    # Crear engine
    engine = build_engine()
    
    # Crear perfiles de estudiantes
    profiles = create_student_profiles(n_students=5)
    
    print("=" * 80)
    print("SIMULACIÓN DE ESTUDIANTES")
    print("=" * 80)
    
    for profile in profiles:
        print(f"\nEstudiante: {profile.student_id}")
        print(f"  θ real: {profile.true_theta:.2f}")
        print(f"  Mastery real: {profile.true_mastery}")
        
        simulator = StudentSimulator(profile, seed=42)
        
        results = simulate_full_session(
            simulator=simulator,
            engine=engine,
            max_items=15
        )
        
        final_theta = results["final_state"]["theta"]["theta_hat"]
        final_se = results["final_state"]["theta"]["se_theta"]
        theta_error = abs(final_theta - profile.true_theta)
        
        print(f"  θ estimado: {final_theta:.2f} (error: {theta_error:.2f})")
        print(f"  SE final: {final_se:.2f}")
        print(f"  Ítems administrados: {len(results['items_administered'])}")
        print(f"  Razón de parada: {results.get('stopped_by', 'max_items')}")