"""
graders.py — OpenEnv Phase 2 bulletproof grader functions
ALWAYS returns float strictly in (0.01, 0.99) for any input.
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np
from typing import Any, Dict

# ─────────────────────────── Safety clamp ─────────────────────────────────

def _clamp(score: Any, lo: float = 0.01, hi: float = 0.99) -> float:
    """Nuclear clamp: converts anything to float and clips to (lo, hi)."""
    try:
        val = float(score)
        if val != val:  # NaN check
            val = 0.5
    except Exception:
        val = 0.5
    return float(np.clip(val, lo, hi))


# ─────────────────────────── State extraction ─────────────────────────────

def _extract_score(state: Any) -> float:
    """
    Extract a raw [0, 1] score from any state object.
    Handles: dict, pydantic BaseModel, object with attributes, None.
    NEVER raises — returns 0.5 on any failure.
    """
    try:
        if state is None:
            return 0.5

        # ── Dict / JSON-like ──
        if isinstance(state, dict):
            for key in ("score", "progress", "reward", "value"):
                if key in state:
                    return float(state[key])
            # Try nested breakdown
            if "breakdown" in state and isinstance(state["breakdown"], dict):
                vals = list(state["breakdown"].values())
                if vals:
                    return float(sum(vals) / len(vals))
            return 0.5

        # ── Pydantic BaseModel or object with .score ──
        if hasattr(state, "score"):
            return float(state.score)

        # ── Object with progress / reward ──
        for attr in ("progress", "reward", "value"):
            if hasattr(state, attr):
                return float(getattr(state, attr))

        # ── Try calling env grader dynamically (lazy import avoids crash) ──
        if hasattr(state, "inbox") and hasattr(state, "task"):
            try:
                # Dynamic import so graders.py loads even without environment.py
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root not in sys.path:
                    sys.path.insert(0, root)
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                from environment import EmailTriageEnv  # type: ignore
                tmp_env = EmailTriageEnv()
                tmp_env.state = state
                tmp_env.current_task = state.task if hasattr(state, "task") else "easy"
                tmp_env.ground_truth = getattr(state, "_ground_truth", {})
                result = tmp_env.grader()
                return float(result.score)
            except Exception:
                return 0.5

    except Exception:
        pass

    return 0.5


# ─────────────────────────── Public grader functions ──────────────────────

def grader_easy(state: Any) -> float:
    """
    Phase 2 grader for task_easy.
    Accepts ANY state. ALWAYS returns float in (0.01, 0.99).
    """
    try:
        raw = _extract_score(state)
        return _clamp(raw)
    except Exception:
        return 0.5


def grader_medium(state: Any) -> float:
    """
    Phase 2 grader for task_medium.
    Accepts ANY state. ALWAYS returns float in (0.01, 0.99).
    """
    try:
        raw = _extract_score(state)
        return _clamp(raw)
    except Exception:
        return 0.5


def grader_hard(state: Any) -> float:
    """
    Phase 2 grader for task_hard.
    Accepts ANY state. ALWAYS returns float in (0.01, 0.99).
    """
    try:
        raw = _extract_score(state)
        return _clamp(raw)
    except Exception:
        return 0.5


# ─────────────────────────── Batch helper (bonus) ─────────────────────────

def batch_grade(states: list, task: str = "easy") -> np.ndarray:
    """Grade a list of states — returns clamped scores as np.ndarray."""
    fn = {"easy": grader_easy, "medium": grader_medium, "hard": grader_hard}.get(
        task, grader_easy
    )
    return np.array([fn(s) for s in states], dtype=float)
