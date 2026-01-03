import json
import random
import math
from pathlib import Path

def generate_dummy_bank():
    print("Generando banco de ítems sintético masivo...")
    
    base_items = []
    # Definimos los skills explícitamente para asegurar cobertura balanceada
    skills = ["k_regla_potencia", "k_regla_cadena"]
    difficulties = ["facil", "media", "dificil"]
    
    # Generar 200 ítems (100 por skill para asegurar profundidad)
    count = 0
    for skill in skills:
        for i in range(100):
            count += 1
            # Distribución de dificultad (b) uniforme amplia [-3, +3]
            b_val = random.uniform(-3.0, 3.0)
            
            if b_val < -0.6:
                diff_label = "facil"
            elif b_val < 0.6:
                diff_label = "media"
            else:
                diff_label = "dificil"
            
            # Discriminación (a) log-normal
            # a alto (>1.5) discrimina muy bien, a bajo (<0.8) es más indulgente
            a_val = math.exp(random.normalvariate(0.3, 0.4))
            a_val = max(0.5, min(2.5, a_val)) # Clamp razonable
            
            # Adivinanza (c)
            c_val = random.uniform(0.0, 0.25)
            
            item = {
                "item_id": f"itm_gen_{count:03d}",
                "topic_id": "t_derivadas",
                "skill_ids": [skill],
                "difficulty_label": diff_label,
                "format": "mcq",
                "irt": {
                    "model": "3PL",
                    "a": round(a_val, 3),
                    "b": round(b_val, 3),
                    "c": round(c_val, 3)
                },
                "content": {
                    "prompt_ref": f"derivadas/gen/{count}",
                    "stem_template": f"Pregunta {count} sobre {skill} (b={b_val:.2f})",
                    "choices_ref": "placeholder"
                },
                "constraints": {
                    "exposure_group": f"gen_{skill}_{diff_label}",
                    "time_limit_ms": 60000
                },
                "analytics": {
                    "exposure_count": 0
                }
            }
            base_items.append(item)

    final_json = {
        "bank_id": "bank_derivadas_synthetic_v2",
        "version": "2.0",
        "items": base_items
    }
    
    # Ruta de guardado
    output_path = Path("resources/data/item_bank.json")
    if not output_path.parent.exists():
        output_path = Path("item_bank.json")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Generados {len(base_items)} ítems balanceados en {output_path.absolute()}")

if __name__ == "__main__":
    generate_dummy_bank()