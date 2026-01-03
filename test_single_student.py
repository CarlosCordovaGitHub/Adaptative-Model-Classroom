# tests/test_single_student.py
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def simulate_student_session(student_id: str, true_theta: float, n_items: int = 20):
    """
    Simula un estudiante con habilidad conocida.
    """
    session_id = f"sess_{student_id}"
    current_item = "itm_001"  # Ítem inicial
    
    print(f"\n{'='*60}")
    print(f"Simulando estudiante: {student_id}")
    print(f"Theta real: {true_theta:.2f}")
    print(f"{'='*60}\n")
    
    responses = []
    
    for i in range(n_items):
        # Obtén el ítem actual del banco
        item = get_item_from_bank(current_item)
        
        # Calcula probabilidad de respuesta correcta usando IRT
        p_correct = irt_3pl(true_theta, item['irt'])
        is_correct = random.random() < p_correct
        
        # Envía respuesta al engine
        payload = {
            "event_type": "activity_result",
            "event_version": "1.0",
            "student": {
                "student_id": student_id,
                "session_id": session_id
            },
            "context": {
                "topic_id": "t_derivadas",
                "skill_ids": item["skill_ids"],
                "language": "es",
                "mode": "practice"
            },
            "activity": {
                "activity_id": f"act_{i:03d}",
                "item_id": current_item,
                "item_format": "mcq",
                "difficulty_label": item["difficulty_label"],
                "seed": i * 100
            },
            "response": {
                "is_correct": is_correct,
                "selected_option": "A" if is_correct else "B",
                "attempts_count": 1,
                "hint_used": False,
                "hints_count": 0,
                "response_time_ms": random.randint(10000, 30000)
            },
            "telemetry": {
                "started_at": datetime.now().isoformat(),
                "ended_at": (datetime.now() + timedelta(seconds=15)).isoformat(),
                "client_device": "simulator",
                "client_version": "v1"
            }
        }
        
        # POST al engine
        resp = requests.post(f"{BASE_URL}/b/events", json=payload)
        
        if resp.status_code != 200:
            print(f"❌ Error en ítem {i}: {resp.text}")
            break
        
        rec = resp.json()
        
        # Extrae estado actual
        theta_hat = rec["state"]["theta_hat"]
        se_theta = rec["state"]["se_theta"]
        next_item = rec["recommendation"]["item_id"]
        
        responses.append({
            "item_num": i + 1,
            "item_id": current_item,
            "difficulty": item["irt"]["b"],
            "correct": is_correct,
            "theta_hat": theta_hat,
            "se_theta": se_theta,
            "error": abs(theta_hat - true_theta)
        })
        
        print(f"Ítem {i+1:2d} | {current_item} (b={item['irt']['b']:+.2f}) | "
              f"{'✓' if is_correct else '✗'} | "
              f"θ̂={theta_hat:+.2f} (SE={se_theta:.2f}) | "
              f"Error={abs(theta_hat - true_theta):.2f}")
        
        # Siguiente ítem
        current_item = next_item
        
        # Stopping rule simple
        if se_theta < 0.4 and i >= 10:
            print(f"\n✓ Convergencia alcanzada en {i+1} ítems")
            break
    
    return responses


def get_item_from_bank(item_id: str):
    """Carga ítem del banco JSON."""
    with open("resources/data/item_bank.json", "r") as f:
        bank = json.load(f)
    
    for item in bank["items"]:
        if item["item_id"] == item_id:
            return item
    
    raise ValueError(f"Item {item_id} not found")


def irt_3pl(theta: float, irt: dict) -> float:
    """Calcula P(correct|theta) usando modelo 3PL."""
    import math
    a = irt["a"]
    b = irt["b"]
    c = irt["c"]
    z = a * (theta - b)
    return c + (1 - c) / (1 + math.exp(-z))


if __name__ == "__main__":
    import random
    random.seed(42)
    
    # Prueba con un estudiante de habilidad media
    responses = simulate_student_session(
        student_id="sim_001",
        true_theta=0.5,  # Habilidad real conocida
        n_items=20
    )
    
    # Análisis final
    final_theta = responses[-1]["theta_hat"]
    final_error = responses[-1]["error"]
    
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL:")
    print(f"  Theta real:      {0.5:.2f}")
    print(f"  Theta estimado:  {final_theta:.2f}")
    print(f"  Error absoluto:  {final_error:.2f}")
    print(f"  SE final:        {responses[-1]['se_theta']:.2f}")
    print(f"  Ítems usados:    {len(responses)}")
    print(f"{'='*60}\n")
    
    # Guarda resultados
    with open("test_results_single.json", "w") as f:
        json.dump(responses, f, indent=2)
    
    print("✓ Resultados guardados en test_results_single.json")