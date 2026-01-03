"""
Script de Prueba de Carga (Stress Test) usando Locust.
Simula múltiples estudiantes interactuando concurrentemente con el Componente B.
"""

import random
import uuid
from locust import HttpUser, task, between

# Datos base para generar tráfico realista
TOPICS = ["t_derivadas"]
SKILLS = ["k_regla_potencia", "k_regla_cadena"]
ITEMS = ["itm_001", "itm_002"] # Se usarán los del banco

class StudentUser(HttpUser):
    # Tiempo de espera entre tareas (simula el tiempo de pensar del estudiante)
    # Entre 1 y 3 segundos para ser agresivos en la prueba
    wait_time = between(1, 3)

    def on_start(self):
        """Se ejecuta cuando un usuario 'nace' en la prueba."""
        self.student_id = f"stress_student_{uuid.uuid4().hex[:6]}"
        self.session_id = f"sess_{uuid.uuid4().hex[:6]}"
        self.items_answered = 0

    @task(1)
    def health_check(self):
        """Tarea ligera: Verificar que el servidor está vivo."""
        self.client.get("/health", name="/health")

    @task(3)
    def submit_activity(self):
        """Tarea pesada: Enviar una respuesta y procesar algoritmos (IRT+BKT)."""
        
        # Generamos una respuesta sintética
        payload = {
            "event_type": "activity_result",
            "event_version": "1.0",
            "student": {
                "student_id": self.student_id,
                "session_id": self.session_id
            },
            "context": {
                "topic_id": random.choice(TOPICS),
                "skill_ids": [random.choice(SKILLS)],
                "mode": "practice",
                "language": "es"
            },
            "activity": {
                "item_id": random.choice(ITEMS),
                "item_format": "mcq",
                "difficulty_label": "media"
            },
            "response": {
                "is_correct": random.choice([True, False]),
                "response_time_ms": random.randint(5000, 30000),
                "hint_used": random.choice([True, False]),
                "attempts_count": 1
            },
            "telemetry": {}
        }
        
        # Enviamos la petición POST
        with self.client.post("/b/events", json=payload, catch_response=True, name="/b/events") as response:
            if response.status_code == 200:
                self.items_answered += 1
                response.success()
            else:
                response.failure(f"Status {response.status_code}: {response.text}")

    @task(2)
    def check_metrics(self):
        """Tarea media: Consultar el dashboard de métricas."""
        # Solo consultamos si ya hemos respondido algo para que exista la sesión
        if self.items_answered > 0:
            url = f"/metrics/{self.student_id}/{self.session_id}"
            self.client.get(url, name="/metrics/{id}/{sess}")