"""
validator_fix.py — Phase 2 simulation test suite.

Replicates exactly what openenv validate does:
  1. Imports each grader function by dotted path from openenv.yaml
  2. Calls it with various state types (dict, None, empty, Pydantic)
  3. Asserts 0.0 < score < 1.0  (strictly, not on boundary)

Run:
    python validator_fix.py

Expected: ALL TESTS PASS
"""
from __future__ import annotations

import sys
import os
import importlib
import traceback

import numpy as np

# ── Add repo root to path so imports resolve the same way the validator does ──
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ──────────────────────────── Helpers ────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"

results: list[bool] = []


def check(name: str, score, lo: float = 0.0, hi: float = 1.0):
    """Assert score is strictly within (lo, hi)."""
    try:
        val = float(score)
        ok = lo < val < hi
    except Exception as exc:
        print(f"  {FAIL}  [{name}]  could not cast to float: {exc}")
        results.append(False)
        return
    sym = PASS if ok else FAIL
    print(f"  {sym}  [{name}]  score={val:.6f}  range=({lo},{hi})")
    results.append(ok)


# ──────────────────────────── Load grader module ─────────────────────────────

def load_grader(dotted_path: str):
    """
    Mimics importlib-based dynamic loading that openenv validate uses.
    dotted_path e.g. 'graders.grader_easy'
    """
    module_name, fn_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


# ──────────────────────────── Build test states ──────────────────────────────

def make_states():
    states = []

    # 1. None
    states.append(("None", None))

    # 2. Empty dict
    states.append(("empty_dict", {}))

    # 3. Dict with score=0 (boundary — clamp must move it up)
    states.append(("dict_score_0", {"score": 0.0}))

    # 4. Dict with score=1 (boundary — clamp must move it down)
    states.append(("dict_score_1", {"score": 1.0}))

    # 5. Dict with valid mid score
    states.append(("dict_score_0.5", {"score": 0.5}))

    # 6. Dict with progress key
    states.append(("dict_progress_0.8", {"progress": 0.8}))

    # 7. Dict with negative score
    states.append(("dict_score_neg", {"score": -99.0}))

    # 8. Dict with huge score
    states.append(("dict_score_big", {"score": 999.0}))

    # 9. Dict with NaN
    states.append(("dict_score_nan", {"score": float("nan")}))

    # 10. Dict with inf
    states.append(("dict_score_inf", {"score": float("inf")}))

    # 11. String state (garbage)
    states.append(("string_state", "garbage"))

    # 12. Integer state
    states.append(("int_state", 42))

    # 13. List state
    states.append(("list_state", [0.3, 0.7]))

    # 14. Pydantic-like object (mock BaseModel)
    try:
        from pydantic import BaseModel

        class MockState(BaseModel):
            score: float = 0.6
            task: str = "easy"

        states.append(("pydantic_score_0.6", MockState()))
    except ImportError:
        pass

    # 15. Real environment state (if environment.py is importable)
    try:
        from environment import EmailTriageEnv  # type: ignore
        env = EmailTriageEnv()
        env.reset(task="easy")
        real_state = env.state
        states.append(("real_env_state_easy", real_state))
    except Exception:
        pass

    return states


# ──────────────────────────── Main test runner ────────────────────────────────

def run_tests():
    print("=" * 60)
    print("  OpenEnv Phase 2 Validator Replica — grader score tests")
    print("=" * 60)

    # Grader paths from openenv.yaml
    grader_paths = [
        ("task_easy",   "graders.grader_easy"),
        ("task_medium", "graders.grader_medium"),
        ("task_hard",   "graders.grader_hard"),
    ]

    # ── Step 1: verify grader modules load without error ──────────────────
    print("\n[Step 1] Loading grader functions via importlib...")
    fns = {}
    for task_name, dotted_path in grader_paths:
        try:
            fn = load_grader(dotted_path)
            fns[task_name] = fn
            print(f"  {PASS}  Loaded '{dotted_path}'  → {fn}")
        except Exception as exc:
            print(f"  {FAIL}  Could not load '{dotted_path}': {exc}")
            traceback.print_exc()
            results.append(False)

    if not fns:
        print("\nFATAL: No grader functions loaded. Cannot continue.")
        return

    # ── Step 2: run each grader against every test state ──────────────────
    states = make_states()
    print(f"\n[Step 2] Testing {len(fns)} graders × {len(states)} states = "
          f"{len(fns) * len(states)} checks\n")

    for task_name, fn in fns.items():
        print(f"  --- {task_name} ({fn.__name__}) ---")
        for state_name, state_val in states:
            try:
                score = fn(state_val)
            except Exception as exc:
                print(f"  {FAIL}  [{state_name}]  raised exception: {exc}")
                results.append(False)
                continue
            check(f"{state_name}", score)
        print()

    # ── Step 3: batch / vectorised test ────────────────────────────────────
    print("[Step 3] Batch score array check (np.clip applied)...")
    try:
        from graders import batch_grade  # type: ignore
        edge_states = [{"score": 0.0}, {"score": 1.0}, {"score": 0.5}, None, {}]
        scores = batch_grade(edge_states, task="easy")
        for i, s in enumerate(scores):
            check(f"batch[{i}]", s)
    except Exception as exc:
        print(f"  (batch_grade not tested: {exc})")

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    failed = total - passed
    if failed == 0:
        print(f"  \033[92mALL {total} TESTS PASSED — Ready for Phase 3 ✓\033[0m")
    else:
        print(f"  \033[91m{failed}/{total} TESTS FAILED\033[0m")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
