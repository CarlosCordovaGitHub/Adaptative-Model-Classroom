from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List

from app.engine import (
    build_engine, 
    EngineError, 
    ResourceNotFound, 
    ContractValidationError, 
    BKTParamsMissing
)

app = FastAPI(
    title="Componente B - Motor Adaptativo IRT+BKT",
    version="1.1.0",
    description="Motor de evaluación adaptativa con IRT (3PL) + BKT y métricas avanzadas"
)
engine = build_engine()


class Health(BaseModel):
    status: str
    engine_version: str
    engine_name: str


@app.get("/")
def root() -> Dict[str, Any]:
    """Endpoint raíz con información del servicio."""
    return {
        "service": "Componente B - Motor Adaptativo",
        "version": "1.1.0",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "process_activity": "/b/events",
            "get_metrics": "/metrics/{student_id}/{session_id}",
            "debug": {
                "state": "/debug/state/{student_id}",
                "logs": "/debug/logs/{student_id}/{session_id}",
                "log_file": "/debug/logs/{student_id}/{session_id}/{filename}",
                "replay": "/debug/replay/{student_id}/{session_id}"
            }
        },
        "features": [
            "IRT 3PL with EAP estimation",
            "Bayesian Knowledge Tracing",
            "Hybrid mode selection",
            "Advanced metrics (precision, efficiency, progress, quality)",
            "Temporal decay in BKT",
            "Deterministic session replay",
            "Audit logging"
        ]
    }


@app.get("/health", response_model=Health)
def health() -> Health:
    """Health check endpoint."""
    return Health(
        status="ok",
        engine_version=engine.version,
        engine_name=engine.name
    )


@app.post("/b/events")
def b_events(payload: Dict[str, Any]) -> JSONResponse:
    """
    Procesa resultado de actividad A->B y genera recomendación B->A.
    
    Request body: activity_result según schema a_to_b_activity_result.schema.json
    Response: recommendation según schema b_to_a_recommendation.schema.json
    
    La respuesta incluye métricas avanzadas de desempeño.
    """
    try:
        rec = engine.process_activity_result(payload)
        return JSONResponse(content=rec)
    except ContractValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (ResourceNotFound, BKTParamsMissing) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EngineError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.get("/metrics/{student_id}/{session_id}")
def get_session_metrics(student_id: str, session_id: str) -> JSONResponse:
    """
    Retorna métricas detalladas de desempeño de una sesión activa.
    
    Métricas incluidas:
    - Precisión: theta_hat, se_theta, información acumulada, ítems restantes
    - Eficiencia: ítems administrados, tiempo, tiempo promedio por ítem, ETA
    - Progreso/Dominio: mastery por skill, skills dominados, velocidad de progreso
    - Calidad Predictiva: Brier score, log-likelihood, accuracy reciente, consistencia
    """
    try:
        state = engine.get_student_state(student_id)
        
        # Validar que la sesión coincida
        current_session = state.get("session", {}).get("session_id")
        if current_session != session_id:
            raise HTTPException(
                status_code=404, 
                detail=f"Session '{session_id}' not found. Current session: '{current_session}'"
            )
        
        metrics = engine.compute_session_metrics(state)
        
        return JSONResponse(content={
            "student_id": student_id,
            "session_id": session_id,
            "metrics": metrics,
            "timestamp": engine._now_local_iso()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {e}")


@app.get("/debug/state/{student_id}")
def debug_state(student_id: str) -> JSONResponse:
    """
    Retorna el estado completo del estudiante.
    
    Incluye:
    - Sesión actual
    - Historial de respuestas
    - Estado IRT (theta_hat, se_theta)
    - Estado BKT (mastery por skill)
    """
    try:
        state = engine.get_student_state(student_id)
        return JSONResponse(content=state)
    except ResourceNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.get("/debug/logs/{student_id}/{session_id}", response_model=List[str])
def debug_list_logs(student_id: str, session_id: str) -> List[str]:
    """
    Lista todos los archivos de log de auditoría para una sesión.
    
    Returns: Lista de nombres de archivo (timestamps).
    """
    try:
        return engine.list_audit_logs(student_id, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.get("/debug/logs/{student_id}/{session_id}/{filename}")
def debug_read_log(student_id: str, session_id: str, filename: str) -> JSONResponse:
    """
    Lee un archivo específico de log de auditoría.
    
    Returns: Registro completo con evento, estado y recomendación.
    """
    try:
        rec = engine.read_audit_log(student_id, session_id, filename)
        return JSONResponse(content=rec)
    except ResourceNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.post("/debug/replay/{student_id}/{session_id}")
def debug_replay(student_id: str, session_id: str) -> JSONResponse:
    """
    Replay determinista de una sesión desde logs de auditoría.
    
    Reconstruye el estado final aplicando todos los eventos guardados
    en orden, sin escribir estado ni logs nuevos.
    
    Útil para:
    - Validación de determinismo
    - Debugging de políticas
    - Análisis retrospectivo
    """
    try:
        result = engine.replay_session(student_id, session_id)
        return JSONResponse(content=result)
    except ContractValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ResourceNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


@app.get("/debug/state/{student_id}/path")
def debug_state_path(student_id: str) -> Dict[str, str]:
    """
    Retorna la ruta del archivo de estado persistido.
    
    Útil para debugging y gestión de archivos.
    """
    try:
        path = engine.get_state_path(student_id)
        return {"student_id": student_id, "state_file_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)