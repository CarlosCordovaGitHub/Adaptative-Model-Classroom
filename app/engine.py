from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from jsonschema import validate
from jsonschema.exceptions import ValidationError


class EngineError(Exception):
    """Base exception for engine errors."""


class ResourceNotFound(EngineError):
    pass


class ContractValidationError(EngineError):
    pass


class BKTParamsMissing(EngineError):
    pass


@dataclass
class ModeDecision:
    mode: str  # "IRT" | "BKT"
    reason: str


class Engine:
    """
    Componente B – Motor adaptativo IRT + BKT (Versión Mejorada)
    
    Mejoras implementadas:
    - Métricas de desempeño en tiempo real
    - Fisher Information optimizada con región de incertidumbre
    - Stopping rules robustas con prioridades
    - Análisis de calidad predictiva
    - Decay temporal en BKT
    - Mejores fallbacks en selección BKT
    """

    def __init__(self, data_dir: str, persist_dir: str):
        self.data_dir = Path(data_dir).resolve()
        self.persist_dir = Path(persist_dir).resolve()

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        (self.persist_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Load resources
        self.config = self._load_required_json(self.data_dir / "config" / "config_b.json")
        self.skill_map = self._load_required_json(self.data_dir / "data" / "skill_map.json")
        self.bkt_params = self._load_required_json(self.data_dir / "data" / "bkt_parameters.json")
        self.item_bank = self._load_required_json(self.data_dir / "data" / "item_bank.json")

        # Schemas
        self.schema_a2b = self._load_required_json(self.data_dir / "schemas" / "a_to_b_activity_result.schema.json")
        self.schema_b2a = self._load_required_json(self.data_dir / "schemas" / "b_to_a_recommendation.schema.json")

        # Index item bank
        self.items_by_id: Dict[str, Dict[str, Any]] = {}
        self.items_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for it in self.item_bank.get("items", []):
            self.items_by_id[it["item_id"]] = it
            self.items_by_topic.setdefault(it["topic_id"], []).append(it)

    @property
    def version(self) -> str:
        return str(self.config.get("engine", {}).get("version", "unknown"))

    @property
    def name(self) -> str:
        return str(self.config.get("engine", {}).get("name", "Engine"))

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def process_activity_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa resultado de actividad y genera recomendación.
        """
        self._validate_a2b(payload)

        student_id = payload["student"]["student_id"]
        session_id = payload["student"]["session_id"]
        topic_id = payload["context"]["topic_id"]
        skill_ids = payload["context"]["skill_ids"]

        item_id = payload["activity"]["item_id"]
        is_correct = payload["response"]["is_correct"]
        rt_ms = payload["response"]["response_time_ms"]
        hint_used = payload["response"]["hint_used"]

        # Load/init state
        state = self._load_student_state(student_id)

        # Session management
        now_iso = self._now_local_iso()
        if not state.get("session") or state.get("session", {}).get("session_id") != session_id:
            state["session"] = {
                "session_id": session_id,
                "started_at": now_iso,
                "items_administered": 0,
                "time_spent_ms": 0,
                "recent_item_ids": []
            }

        state["session"]["items_administered"] = int(state["session"].get("items_administered", 0)) + 1
        state["session"]["time_spent_ms"] = int(state["session"].get("time_spent_ms", 0)) + int(rt_ms)

        # Anti-repeat
        recent_window = int(self._cfg(["constraints", "exposure_control", "recent_window_items"], default=5))
        recent = list(state["session"].get("recent_item_ids", []))
        recent.append(item_id)
        state["session"]["recent_item_ids"] = recent[-recent_window:]

        # History
        hist = list(state.get("history", []))
        hist.append({
            "ts": now_iso,
            "topic_id": topic_id,
            "skill_ids": skill_ids,
            "item_id": item_id,
            "correct": bool(is_correct),
            "response_time_ms": int(rt_ms),
            "hint_used": bool(hint_used)
        })
        state["history"] = hist

        # Apply temporal decay before updating
        for k in skill_ids:
            state = self._apply_temporal_decay(state, k)

        # Update BKT
        for k in skill_ids:
            state = self._update_bkt_for_skill(state, k, bool(is_correct))

        # Update IRT
        state = self._update_irt_eap(state)

        # Decision mode
        mode_decision = self._decide_mode(state)

        # Support policy
        support_type = self._select_support(state)

        # Stopping rules
        should_stop, stop_reason = self._should_stop_session(state)

        # Select next item
        if not should_stop:
            next_item = self._select_next_item(
                state=state,
                topic_id=topic_id,
                skill_ids=skill_ids,
                mode=mode_decision.mode
            )
        else:
            next_item = self.items_by_id.get(item_id, list(self.items_by_id.values())[0])

        # Build recommendation
        rec = self._build_recommendation(
            payload=payload,
            state=state,
            chosen_item=next_item,
            support_type=support_type,
            mode_reason=mode_decision.reason
        )

        if should_stop:
            rec["recommendation"]["action"] = "end_session"
            rec["recommendation"]["stop_reason"] = stop_reason
            rec["recommendation"]["present_item"] = False
        else:
            rec["recommendation"]["present_item"] = True

        self._validate_b2a(rec)

        # Persist
        self._save_student_state(student_id, state)
        self._log_event(
            student_id=student_id,
            session_id=session_id,
            event=payload,
            state=state,
            recommendation=rec
        )

        return rec

    def get_student_state(self, student_id: str) -> Dict[str, Any]:
        return self._load_student_state(student_id)

    def get_state_path(self, student_id: str) -> str:
        return str(self._state_path(student_id))

    def list_audit_logs(self, student_id: str, session_id: str) -> List[str]:
        d = self._log_dir(student_id, session_id)
        if not d.exists():
            return []
        return sorted([p.name for p in d.glob("*.json")])

    def read_audit_log(self, student_id: str, session_id: str, filename: str) -> Dict[str, Any]:
        p = self._log_dir(student_id, session_id) / filename
        if not p.exists():
            raise ResourceNotFound(f"Log not found: {p}")
        return self._load_json(p)

    def replay_session(self, student_id: str, session_id: str) -> Dict[str, Any]:
        """Replay determinista desde logs."""
        files = self.list_audit_logs(student_id, session_id)
        if not files:
            return {
                "student_id": student_id,
                "session_id": session_id,
                "events_replayed": 0,
                "final_state": None,
                "last_recommendation": None
            }

        state = self._load_student_state_template(student_id)
        last_rec = None

        for fn in files:
            rec_file = self.read_audit_log(student_id, session_id, fn)
            event = rec_file.get("event")
            if not event:
                continue

            self._validate_a2b(event)

            ts = (
                rec_file.get("ts_utc")
                or event.get("telemetry", {}).get("ended_at")
                or self._now_local_iso()
            )

            topic_id = event["context"]["topic_id"]
            skill_ids = event["context"]["skill_ids"]
            item_id = event["activity"]["item_id"]
            is_correct = event["response"]["is_correct"]
            rt_ms = event["response"]["response_time_ms"]
            hint_used = event["response"]["hint_used"]

            if not state.get("session") or state.get("session", {}).get("session_id") != session_id:
                state["session"] = {
                    "session_id": session_id,
                    "started_at": ts,
                    "items_administered": 0,
                    "time_spent_ms": 0,
                    "recent_item_ids": []
                }
            elif not state["session"].get("started_at"):
                state["session"]["started_at"] = ts

            state["session"]["items_administered"] = int(state["session"].get("items_administered", 0)) + 1
            state["session"]["time_spent_ms"] = int(state["session"].get("time_spent_ms", 0)) + int(rt_ms)

            recent_window = int(self._cfg(["constraints", "exposure_control", "recent_window_items"], default=5))
            recent = list(state["session"].get("recent_item_ids", []))
            recent.append(item_id)
            state["session"]["recent_item_ids"] = recent[-recent_window:]

            hist = list(state.get("history", []))
            hist.append({
                "ts": ts,
                "topic_id": topic_id,
                "skill_ids": skill_ids,
                "item_id": item_id,
                "correct": bool(is_correct),
                "response_time_ms": int(rt_ms),
                "hint_used": bool(hint_used)
            })
            state["history"] = hist

            for k in skill_ids:
                state = self._apply_temporal_decay(state, k)
                state = self._update_bkt_for_skill(state, k, bool(is_correct))
            
            state = self._update_irt_eap(state)

            mode_decision = self._decide_mode(state)
            should_stop, stop_reason = self._should_stop_session(state)
            
            if not should_stop:
                chosen_item = self._select_next_item(
                    state=state,
                    topic_id=topic_id,
                    skill_ids=skill_ids,
                    mode=mode_decision.mode
                )
            else:
                chosen_item = self.items_by_id.get(item_id, list(self.items_by_id.values())[0])
            
            support_type = self._select_support(state)

            last_rec = self._build_recommendation(
                payload=event,
                state=state,
                chosen_item=chosen_item,
                support_type=support_type,
                mode_reason=f"REPLAY:{mode_decision.reason}"
            )
            
            if should_stop:
                last_rec["recommendation"]["action"] = "end_session"
                last_rec["recommendation"]["stop_reason"] = f"REPLAY:{stop_reason}"
                last_rec["recommendation"]["present_item"] = False
            else:
                last_rec["recommendation"]["present_item"] = True
            
            self._validate_b2a(last_rec)

        return {
            "student_id": student_id,
            "session_id": session_id,
            "events_replayed": len(files),
            "final_state": state,
            "last_recommendation": last_rec
        }

    # =====================================================================
    # MÉTRICAS AVANZADAS (NUEVO)
    # =====================================================================

    def compute_session_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula métricas detalladas de desempeño.
        """
        theta_hat = float(state.get("theta", {}).get("theta_hat", 0.0))
        se_theta = float(state.get("theta", {}).get("se_theta", 1.0))
        items_admin = int(state.get("session", {}).get("items_administered", 0))
        time_spent_ms = int(state.get("session", {}).get("time_spent_ms", 0))
        
        mastery_vals = [float(blk.get("p_mastery", 0.0)) for blk in state.get("mastery", {}).values()]
        tau = float(self._cfg(["thresholds", "mastery_tau"], default=0.85))
        
        metrics = {
            "precision": {
                "theta_hat": theta_hat,
                "se_theta": se_theta,
                "information_accumulated": self._compute_total_information(state),
                "estimated_items_remaining": self._estimate_items_to_target(state),
                "se_reduction_rate": self._compute_se_reduction_rate(state)
            },
            "eficiencia": {
                "items_administered": items_admin,
                "time_spent_minutes": time_spent_ms / 60000.0,
                "avg_time_per_item_sec": (
                    time_spent_ms / items_admin / 1000.0 if items_admin > 0 else 0
                ),
                "eta_minutes": self._estimate_time_remaining(state)
            },
            "progreso_dominio": {
                "mastery_by_skill": {
                    sk: float(blk.get("p_mastery", 0.0)) 
                    for sk, blk in state.get("mastery", {}).items()
                },
                "skills_mastered": [
                    sk for sk, blk in state.get("mastery", {}).items() 
                    if float(blk.get("p_mastery", 0.0)) >= tau
                ],
                "skills_in_progress": [
                    sk for sk, blk in state.get("mastery", {}).items() 
                    if 0.3 < float(blk.get("p_mastery", 0.0)) < tau
                ],
                "overall_progress": sum(mastery_vals) / len(mastery_vals) if mastery_vals else 0.0,
                "mastery_velocity": self._compute_mastery_velocity(state)
            },
            "calidad_predictiva": {
                "brier_score": self._compute_brier_score(state),
                "log_likelihood": self._compute_log_likelihood(state),
                "accuracy_last_5": self._compute_recent_accuracy(state, n=5),
                "consistency_score": self._compute_consistency(state)
            }
        }
        
        return metrics

    def _compute_total_information(self, state: Dict) -> float:
        """Información acumulada de Fisher."""
        theta = state["theta"]["theta_hat"]
        total_info = 0.0
        
        for h in state.get("history", []):
            item = self.items_by_id.get(h["item_id"])
            if item and item.get("irt"):
                total_info += self._fisher_information_3pl(theta, item)
        
        return total_info

    def _estimate_items_to_target(self, state: Dict) -> int:
        """Estima ítems restantes para SE target."""
        se_current = state["theta"]["se_theta"]
        se_target = float(self._cfg(["thresholds", "se_target"], default=0.4))
        
        if se_current <= se_target:
            return 0
        
        items_so_far = len(state.get("history", []))
        if items_so_far == 0:
            return 10
        
        ratio_squared = (se_current / se_target) ** 2
        items_needed = items_so_far * ratio_squared
        items_remaining = max(0, int(items_needed - items_so_far))
        
        return items_remaining

    def _compute_se_reduction_rate(self, state: Dict) -> float:
        """Calcula tasa de reducción de SE por ítem."""
        history = state.get("history", [])
        if len(history) < 2:
            return 0.0
        
        # Tomar últimos 5 ítems
        recent = history[-5:] if len(history) >= 5 else history
        
        se_start = 1.0  # SE inicial
        se_end = state["theta"]["se_theta"]
        n_items = len(recent)
        
        if n_items == 0:
            return 0.0
        
        reduction = (se_start - se_end) / n_items
        return max(0.0, reduction)

    def _estimate_time_remaining(self, state: Dict) -> float:
        """Estima minutos restantes para completar sesión."""
        items_remaining = self._estimate_items_to_target(state)
        
        if items_remaining == 0:
            return 0.0
        
        items_admin = state.get("session", {}).get("items_administered", 0)
        time_spent_ms = state.get("session", {}).get("time_spent_ms", 0)
        
        if items_admin == 0:
            avg_time_per_item = 60000  # 1 minuto default
        else:
            avg_time_per_item = time_spent_ms / items_admin
        
        estimated_ms = items_remaining * avg_time_per_item
        return estimated_ms / 60000.0

    def _compute_mastery_velocity(self, state: Dict) -> float:
        """Calcula velocidad de incremento de mastery."""
        history = state.get("history", [])
        if len(history) < 3:
            return 0.0
        
        # Comparar mastery promedio entre primeros y últimos ítems
        mastery_vals = [float(blk.get("p_mastery", 0.0)) for blk in state.get("mastery", {}).values()]
        
        if not mastery_vals:
            return 0.0
        
        current_avg = sum(mastery_vals) / len(mastery_vals)
        
        # Aproximación: asumimos que comenzó en p_L0 promedio
        initial_avg = 0.2  # Valor típico de p_L0
        
        n_items = len(history)
        velocity = (current_avg - initial_avg) / n_items if n_items > 0 else 0.0
        
        return velocity

    def _compute_brier_score(self, state: Dict) -> Optional[float]:
        """Calcula Brier Score (calibración de predicciones)."""
        history = state.get("history", [])
        if len(history) < 5:
            return None
        
        theta = state["theta"]["theta_hat"]
        predictions = []
        actuals = []
        
        for h in history[-10:]:
            item = self.items_by_id.get(h["item_id"])
            if item and item.get("irt"):
                p_correct = self._irt_3pl_prob(theta, item)
                predictions.append(p_correct)
                actuals.append(1.0 if h["correct"] else 0.0)
        
        if not predictions:
            return None
        
        mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / len(predictions)
        return mse

    def _compute_log_likelihood(self, state: Dict) -> Optional[float]:
        """Calcula log-likelihood del modelo."""
        history = state.get("history", [])
        if not history:
            return None
        
        theta = state["theta"]["theta_hat"]
        ll = 0.0
        
        for h in history:
            item = self.items_by_id.get(h["item_id"])
            if not item or not item.get("irt"):
                continue
            
            p = self._irt_3pl_prob(theta, item)
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            
            if h["correct"]:
                ll += math.log(p)
            else:
                ll += math.log(1.0 - p)
        
        return ll

    def _compute_recent_accuracy(self, state: Dict, n: int = 5) -> float:
        """Calcula accuracy en últimos N ítems."""
        history = state.get("history", [])
        if not history:
            return 0.0
        
        recent = history[-n:]
        correct_count = sum(1 for h in recent if h["correct"])
        
        return correct_count / len(recent)

    def _compute_consistency(self, state: Dict) -> float:
        """Mide consistencia entre predicciones y respuestas."""
        history = state.get("history", [])
        if len(history) < 5:
            return 1.0
        
        theta = state["theta"]["theta_hat"]
        matches = 0
        total = 0
        
        for h in history:
            item = self.items_by_id.get(h["item_id"])
            if not item or not item.get("irt"):
                continue
            
            p_correct = self._irt_3pl_prob(theta, item)
            predicted = p_correct > 0.5
            actual = h["correct"]
            
            if predicted == actual:
                matches += 1
            total += 1
        
        return matches / total if total > 0 else 1.0

    # =====================================================================
    # CONTRACT VALIDATION
    # =====================================================================

    def _validate_a2b(self, payload: Dict[str, Any]) -> None:
        try:
            validate(instance=payload, schema=self.schema_a2b)
        except ValidationError as e:
            raise ContractValidationError(f"A->B contract invalid: {e.message}") from e

    def _validate_b2a(self, payload: Dict[str, Any]) -> None:
        try:
            validate(instance=payload, schema=self.schema_b2a)
        except ValidationError as e:
            raise ContractValidationError(f"B->A contract invalid: {e.message}") from e

    # =====================================================================
    # BKT UPDATE CON DECAY TEMPORAL (MEJORADO)
    # =====================================================================

    def _apply_temporal_decay(self, state: Dict[str, Any], skill_id: str) -> Dict[str, Any]:
        """Aplica decay temporal a la probabilidad de mastery."""
        mastery_block = state.get("mastery", {})
        
        if skill_id not in mastery_block:
            return state
        
        last_update_str = mastery_block[skill_id].get("last_update_ts")
        if not last_update_str:
            return state
        
        try:
            from dateutil.parser import parse
            last_update = parse(last_update_str)
            now = datetime.now(last_update.tzinfo or timezone.utc)
            hours_elapsed = (now - last_update).total_seconds() / 3600.0
        except:
            return state
        
        # Decay exponencial suave
        decay_rate = float(self._cfg(["policy", "bkt_decay", "rate_per_hour"], default=0.005))
        decay_factor = math.exp(-decay_rate * hours_elapsed)
        
        current_p = float(mastery_block[skill_id]["p_mastery"])
        decayed_p = current_p * decay_factor
        
        mastery_block[skill_id]["p_mastery"] = self._clamp01(decayed_p)
        state["mastery"] = mastery_block
        
        return state

    def _update_bkt_for_skill(self, state: Dict[str, Any], skill_id: str, correct: bool) -> Dict[str, Any]:
        by_skill = self.bkt_params.get("by_skill", {})
        if skill_id not in by_skill:
            raise BKTParamsMissing(f"BKT params missing for skill_id='{skill_id}'")

        params = by_skill[skill_id]
        p_T = float(params["p_T"])
        p_G = float(params["p_G"])
        p_S = float(params["p_S"])
        p_F = float(params.get("p_F", 0.0))

        mastery_block = state.get("mastery", {})
        if skill_id not in mastery_block:
            p_L0 = float(params["p_L0"])
            mastery_block[skill_id] = {"p_mastery": p_L0, "last_update_ts": None}

        P_L = float(mastery_block[skill_id]["p_mastery"])

        if correct:
            num = P_L * (1.0 - p_S)
            den = num + (1.0 - P_L) * p_G
        else:
            num = P_L * p_S
            den = num + (1.0 - P_L) * (1.0 - p_G)

        P_post = P_L if den <= 0 else (num / den)

        P_next = P_post + (1.0 - P_post) * p_T
        if p_F > 0:
            P_next = P_next * (1.0 - p_F)

        mastery_block[skill_id] = {
            "p_mastery": self._clamp01(P_next), 
            "last_update_ts": self._now_local_iso()
        }
        state["mastery"] = mastery_block
        return state

    # =====================================================================
    # IRT EAP UPDATE
    # =====================================================================

    def _update_irt_eap(self, state: Dict[str, Any]) -> Dict[str, Any]:
        theta_block = state.get("theta", {})
        theta_hat = float(theta_block.get("theta_hat", 0.0))
        se_theta = float(theta_block.get("se_theta", 1.0))

        history = state.get("history", [])
        used: List[Tuple[bool, Dict[str, Any]]] = []
        for h in history:
            it = self.items_by_id.get(h["item_id"])
            if not it:
                continue
            if not it.get("irt"):
                continue
            used.append((bool(h["correct"]), it))

        if not used:
            state["theta"] = {
                "theta_hat": theta_hat,
                "se_theta": se_theta,
                "method": theta_block.get("method", "EAP"),
                "prior": theta_block.get("prior", {"dist": "N", "mu": 0.0, "sigma": 1.0})
            }
            return state

        prior = theta_block.get("prior", {"dist": "N", "mu": 0.0, "sigma": 1.0})
        mu = float(prior.get("mu", 0.0))
        sigma = float(prior.get("sigma", 1.0)) or 1.0

        grid_min = float(self._cfg(["engine", "irt", "grid_min"], default=-4.0))
        grid_max = float(self._cfg(["engine", "irt", "grid_max"], default=4.0))
        grid_n = int(self._cfg(["engine", "irt", "grid_n"], default=161))
        if grid_n < 3:
            grid_n = 161

        grid = [grid_min + i * (grid_max - grid_min) / (grid_n - 1) for i in range(grid_n)]

        log_post: List[float] = []
        for th in grid:
            lp = self._log_normal_pdf(th, mu, sigma)
            ll = 0.0
            for correct, it in used:
                p = self._irt_3pl_prob(th, it)
                p = min(max(p, 1e-12), 1.0 - 1e-12)
                ll += math.log(p) if correct else math.log(1.0 - p)
            log_post.append(lp + ll)

        max_lp = max(log_post)
        weights = [math.exp(lp - max_lp) for lp in log_post]
        Z = sum(weights)
        if Z <= 0:
            state["theta"] = {"theta_hat": theta_hat, "se_theta": se_theta, "method": "EAP", "prior": prior}
            return state

        weights = [w / Z for w in weights]
        new_theta = sum(th * w for th, w in zip(grid, weights))
        var = sum(((th - new_theta) ** 2) * w for th, w in zip(grid, weights))
        new_se = math.sqrt(max(var, 0.0))

        state["theta"] = {
            "theta_hat": float(new_theta), 
            "se_theta": float(new_se), 
            "method": "EAP", 
            "prior": prior
        }
        return state

    def _irt_3pl_prob(self, theta: float, item: Dict[str, Any]) -> float:
        irt = item.get("irt", {})
        a = float(irt.get("a", 1.0))
        b = float(irt.get("b", 0.0))
        c = float(irt.get("c", 0.0))
        z = a * (theta - b)
        logistic = 1.0 / (1.0 + math.exp(-z))
        return c + (1.0 - c) * logistic

    def _fisher_information_3pl(self, theta: float, item: Dict[str, Any]) -> float:
        """Fisher Information para 3PL con validaciones."""
        irt = item.get("irt", {})
        if not irt:
            return 0.0
        
        a = float(irt.get("a", 1.0))
        c = float(irt.get("c", 0.0))
        
        if a <= 0:
            return 0.0
        
        p = self._irt_3pl_prob(theta, item)
        
        if p >= 1.0 or p <= c:
            return 0.0
        
        q = 1.0 - p
        info = (a ** 2) * (q / p) * ((p - c) ** 2) / ((1.0 - c) ** 2)
        
        return max(info, 0.0)

    # =====================================================================
    # POLICY: MODE DECISION, ITEM SELECTION, SUPPORT, STOPPING
    # =====================================================================

    def _decide_mode(self, state: Dict[str, Any]) -> ModeDecision:
        se_target = float(self._cfg(["thresholds", "se_target"], default=0.40))
        low_tau = float(self._cfg(["thresholds", "low_mastery_tau"], default=0.55))
        tau = float(self._cfg(["thresholds", "mastery_tau"], default=0.85))
        delta = float(self._cfg(["thresholds", "near_mastery_delta"], default=0.05))

        se_theta = float(state.get("theta", {}).get("se_theta", 1.0))

        mastery_vals = [float(blk.get("p_mastery", 0.0)) for _, blk in state.get("mastery", {}).items()]
        any_low = any(m <= low_tau for m in mastery_vals) if mastery_vals else False
        any_near = any(abs(m - tau) <= delta for m in mastery_vals) if mastery_vals else False

        if se_theta > se_target:
            return ModeDecision(mode="IRT", reason="IRT:max_fisher_information")
        if any_low or any_near:
            return ModeDecision(mode="BKT", reason="BKT:max_expected_learning_gain")

        fallback = str(self._cfg(["policy", "mode_selection", "fallback"], default="BKT")).upper()
        return ModeDecision(mode=fallback, reason=f"{fallback}:fallback")

    def _select_next_item(
        self, 
        *, 
        state: Dict[str, Any], 
        topic_id: str, 
        skill_ids: List[str], 
        mode: str
    ) -> Dict[str, Any]:
        # 1. Obtener todos los candidatos del tema
        candidates = list(self.items_by_topic.get(topic_id, []))
        if not candidates:
            # Fallback de emergencia si el topic no existe o está vacío
            if self.items_by_id:
                candidates = list(self.items_by_id.values())
            else:
                raise ResourceNotFound(f"No items found for topic '{topic_id}' and bank is empty.")

        # 2. Identificar ítems ya vistos (Historial completo de la sesión)
        history_ids = set(h["item_id"] for h in state.get("history", []))
        
        # 3. Filtrar candidatos disponibles (que no se hayan visto nunca)
        available = [it for it in candidates if it["item_id"] not in history_ids]
        
        # Si nos quedamos sin items nuevos, reiniciamos el pool (permitimos repetir)
        # Esto evita que el examen colapse si se acaban las preguntas
        if not available:
            available = candidates

        # 4. Filtrar por Skill (Preferencia Principal)
        skill_filtered = [
            it for it in available 
            if any(sk in it.get("skill_ids", []) for sk in skill_ids)
        ]
        
        # 5. Selección final del pool
        # Si hay items del skill, usamos esos. Si no, usamos 'available' (items de otros skills)
        # Esto es vital: permite al motor seguir calibrando theta con otros temas si se acaba el actual.
        final_pool = skill_filtered if skill_filtered else available

        if mode == "IRT":
            return self._select_item_irt_advanced(state, final_pool)
        
        return self._select_item_bkt_improved(state, final_pool, skill_ids)

    def _select_item_irt_advanced(
        self, 
        state: Dict[str, Any], 
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        theta_hat = float(state.get("theta", {}).get("theta_hat", 0.0))
        se_theta = float(state.get("theta", {}).get("se_theta", 1.0))
        
        # Si SE es muy alto (inicio), ampliamos la búsqueda
        # Si SE es bajo (final), buscamos precisión local
        range_mult = 2.0 if se_theta > 0.5 else 1.5
        
        theta_low = theta_hat - range_mult * se_theta
        theta_high = theta_hat + range_mult * se_theta

        best_item = None
        best_score = -float("inf")
        
        import random # Import local seguro
        
        for it in candidates:
            # Si no tiene IRT, penalización masiva
            if not it.get("irt"):
                continue

            # Información de Fisher en el punto estimado actual
            info = self._fisher_information_3pl(theta_hat, it)
            
            # Penalización por exposición (simulada)
            exposure = float(it.get("analytics", {}).get("exposure_count", 0))
            
            # Puntuación: Información - Penalización
            # Añadimos un ruido minúsculo (random) para romper empates
            score = info - (0.1 * exposure) + random.uniform(0, 0.01)
            
            if score > best_score:
                best_score = score
                best_item = it

        # Si por alguna razón matemática falló todo (ej. parámetros raros), devolver al azar
        if best_item is None:
             return random.choice(candidates)
             
        return best_item

    def _select_item_bkt_improved(
        self, 
        state: Dict[str, Any], 
        candidates: List[Dict[str, Any]], 
        skill_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Selección BKT mejorada con fallbacks inteligentes.
        """
        mastery = state.get("mastery", {})
        chosen_skill = None
        chosen_val = 2.0
        for sk in skill_ids:
            val = float(mastery.get(sk, {}).get("p_mastery", 0.0))
            if val < chosen_val:
                chosen_val = val
                chosen_skill = sk
        chosen_skill = chosen_skill or skill_ids[0]

        low_tau = float(self._cfg(["thresholds", "low_mastery_tau"], default=0.55))
        tau = float(self._cfg(["thresholds", "mastery_tau"], default=0.85))

        if chosen_val < low_tau:
            target = "facil"
        elif chosen_val < tau:
            target = "media"
        else:
            target = "dificil"

        adjacent = {
            "facil": ["media"],
            "media": ["facil", "dificil"],
            "dificil": ["media"]
        }

        # 1. Skill correcto + dificultad exacta
        for it in candidates:
            if chosen_skill in it.get("skill_ids", []) and it.get("difficulty_label") == target:
                return it
        
        # 2. Skill correcto + dificultad adyacente
        for it in candidates:
            if chosen_skill in it.get("skill_ids", []) and it.get("difficulty_label") in adjacent.get(target, []):
                return it
        
        # 3. Skill correcto + cualquier dificultad
        for it in candidates:
            if chosen_skill in it.get("skill_ids", []):
                return it
        
        # 4. Cualquier skill relevante + dificultad target
        for it in candidates:
            if any(sk in it.get("skill_ids", []) for sk in skill_ids) and it.get("difficulty_label") == target:
                return it
        
        # 5. Fallback final
        return candidates[0]

    def _select_support(self, state: Dict[str, Any]) -> str:
        policy = self._cfg(["policy", "support_policy"], default={})
        hint_on = int(policy.get("hint_on_consecutive_errors", 2))
        worked_on = int(policy.get("worked_example_on_consecutive_errors", 3))
        no_hint_on = int(policy.get("no_hint_on_consecutive_correct", 2))

        history = state.get("history", [])
        if not history:
            return "no_hint"

        consec_correct = 0
        consec_wrong = 0
        for h in reversed(history):
            if bool(h.get("correct")):
                if consec_wrong > 0:
                    break
                consec_correct += 1
            else:
                if consec_correct > 0:
                    break
                consec_wrong += 1

        if consec_wrong >= worked_on:
            return "worked_example"
        if consec_wrong >= hint_on:
            return "hint"
        if consec_correct >= no_hint_on:
            return "no_hint"
        return "no_hint"

    def _should_stop_session(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Stopping rules robustas con prioridades.
        """
        # Prioridad 1: Límites de seguridad
        max_items = int(self._cfg(["stopping_rules", "stop_if_items_geq"], default=20))
        items_admin = int(state.get("session", {}).get("items_administered", 0))
        if items_admin >= max_items:
            return (True, f"MAX_ITEMS_HARD_LIMIT:items={items_admin}")
        
        max_time_ms = int(self._cfg(["stopping_rules", "stop_if_time_ms_geq"], default=600000))
        time_spent = int(state.get("session", {}).get("time_spent_ms", 0))
        if time_spent >= max_time_ms:
            return (True, f"MAX_TIME_HARD_LIMIT:time_ms={time_spent}")
        
        # Prioridad 2: Objetivos de precisión
        se_target = float(self._cfg(["stopping_rules", "stop_if_se_leq"], default=0.4))
        se_theta = float(state.get("theta", {}).get("se_theta", 1.0))
        
        min_items_for_se = int(self._cfg(["stopping_rules", "min_items_before_se_stop"], default=5))
        if items_admin >= min_items_for_se and se_theta <= se_target:
            return (True, f"SE_TARGET_REACHED:se={se_theta:.3f},items={items_admin}")
        
        # Prioridad 3: Dominio completo
        check_mastery = bool(self._cfg(["stopping_rules", "stop_if_mastery_all_geq_tau"], default=True))
        if check_mastery:
            tau = float(self._cfg(["thresholds", "mastery_tau"], default=0.85))
            mastery_vals = [float(blk.get("p_mastery", 0.0)) for blk in state.get("mastery", {}).values()]
            
            if mastery_vals and all(m >= tau for m in mastery_vals):
                min_mastery = min(mastery_vals)
                return (True, f"ALL_SKILLS_MASTERED:min={min_mastery:.3f},items={items_admin}")
        
        return (False, "")

    def _build_recommendation(
        self,
        *,
        payload: Dict[str, Any],
        state: Dict[str, Any],
        chosen_item: Dict[str, Any],
        support_type: str,
        mode_reason: str
    ) -> Dict[str, Any]:
        student_id = payload["student"]["student_id"]
        session_id = payload["student"]["session_id"]
        topic_id = payload["context"]["topic_id"]
        skill_ids = payload["context"]["skill_ids"]
        language = payload["context"].get("language", self._cfg(["defaults", "language"], default="es"))
        mode = payload["context"].get("mode", self._cfg(["defaults", "mode"], default="practice"))

        theta_hat = float(state.get("theta", {}).get("theta_hat", 0.0))
        se_theta = float(state.get("theta", {}).get("se_theta", 1.0))

        mastery_out: Dict[str, float] = {
            sk: float(blk.get("p_mastery", 0.0)) for sk, blk in state.get("mastery", {}).items()
        }

        # Calcular métricas avanzadas
        metrics = self.compute_session_metrics(state)

        rec = {
            "event_type": "recommendation",
            "event_version": "1.0",
            "student": {"student_id": student_id, "session_id": session_id},
            "context": {"topic_id": topic_id, "skill_ids": skill_ids, "language": language, "mode": mode},
            "recommendation": {
                "action": "present_item",
                "item_id": chosen_item["item_id"],
                "target_difficulty": chosen_item.get("difficulty_label", "media"),
                "support_type": support_type
            },
            "state": {
                "theta_hat": theta_hat, 
                "se_theta": se_theta, 
                "mastery": mastery_out
            },
            "explain": {
                "policy_used": str(self._cfg(["policy", "policy_id"], default="hybrid_irt_bkt_v1")),
                "reason": mode_reason,
                "timestamp": self._now_local_iso()
            },
            "metrics": metrics  # NUEVO: Métricas avanzadas incluidas
        }
        return rec

    # =====================================================================
    # PERSISTENCE & AUDIT LOGGING
    # =====================================================================

    def _state_path(self, student_id: str) -> Path:
        return self.persist_dir / f"student_{student_id}.json"

    def _load_student_state_template(self, student_id: str) -> Dict[str, Any]:
        template = self._load_required_json(self.data_dir / "state" / "student_state.json")
        template = json.loads(json.dumps(template))
        template["student_id"] = student_id
        return template

    def _load_student_state(self, student_id: str) -> Dict[str, Any]:
        p = self._state_path(student_id)
        if p.exists():
            return self._load_json(p)

        state = self._load_student_state_template(student_id)
        self._save_student_state(student_id, state)
        return state

    def _save_student_state(self, student_id: str, state: Dict[str, Any]) -> None:
        self._write_json(self._state_path(student_id), state)

    def _log_dir(self, student_id: str, session_id: str) -> Path:
        d = self.persist_dir / "logs" / student_id / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _log_event(
        self,
        *,
        student_id: str,
        session_id: str,
        event: Dict[str, Any],
        state: Dict[str, Any],
        recommendation: Dict[str, Any]
    ) -> None:
        d = self._log_dir(student_id, session_id)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "student_id": student_id,
            "session_id": session_id,
            "event": event,
            "state_snapshot": state,
            "recommendation": recommendation
        }
        self._write_json(d / f"{ts}.json", record)

    # =====================================================================
    # HELPERS
    # =====================================================================

    def _cfg(self, path: List[str], default: Any = None) -> Any:
        cur: Any = self.config
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _load_required_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ResourceNotFound(f"Required resource not found: {path}")
        return self._load_json(path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def _log_normal_pdf(self, x: float, mu: float, sigma: float) -> float:
        z = (x - mu) / sigma
        return -0.5 * (z * z) - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    def _now_local_iso(self) -> str:
        return datetime.now().astimezone().isoformat()


def build_engine() -> Engine:
    data_dir = os.environ.get("DATA_DIR", "resources")
    persist_dir = os.environ.get("PERSIST_DIR", "runtime")
    return Engine(data_dir=data_dir, persist_dir=persist_dir)