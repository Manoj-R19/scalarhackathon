#!/usr/bin/env python3
"""
inference.py — Email Triage OpenEnv baseline inference script

MANDATORY stdout format (key=value, NOT JSON):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Each task score in [END] is strictly in (0.01, 0.99) — never 0.0 or 1.0.
"""
import os
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------
# Mandatory Environment Variables
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "google/gemma-2-9b-it")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Environment server URL — set ENV_URL to your HF Space URL when deploying
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK         = "email_triage"
MAX_STEPS         = 30
SUCCESS_THRESHOLD = 0.10   # score >= this → success=true
TASKS             = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert email triage agent.
Process inbox emails one at a time with a single JSON action.

Actions available:
1. Label priority : {"type": "label",   "email_id": "<id>", "value": "<spam|low|med|high|escalate>"}
2. Delete spam    : {"type": "delete",  "email_id": "<id>"}
3. Draft reply    : {"type": "draft",   "email_id": "<id>", "value": "<reply text 10-80 words>"}
4. Escalate P0    : {"type": "escalate","email_id": "<id>"}
5. Archive        : {"type": "archive", "email_id": "<id>"}

Rules:
- Respond with the JSON object ONLY — no markdown, no explanation.
- Pick the FIRST unprocessed email (labeled=false) each turn.
- Spam → delete. Critical/P0 → escalate. Others → label then draft if legit."""


# ---------------------------------------------------------
# Score safety clamp — NEVER returns 0.0 or 1.0
# ---------------------------------------------------------
def _clamp(val) -> float:
    """Clamp to strictly (0.01, 0.99). Safe against NaN/inf/non-numeric."""
    try:
        v = float(val)
        if v != v:          # NaN
            v = 0.50
    except Exception:
        v = 0.50
    return round(max(0.01, min(0.99, v)), 4)


# ---------------------------------------------------------
# Mandatory log helpers (key=value, one line each)
# ---------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    action_safe = action.replace("\n", " ").replace("\r", "")
    error_val   = str(error).replace("\n", " ") if error else "null"
    done_val    = str(done).lower()
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------
def get_client() -> OpenAI:
    key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-token")
    return OpenAI(api_key=key, base_url=API_BASE_URL)


# ---------------------------------------------------------
# Task runner
# ---------------------------------------------------------
def run_task(task_name: str, client: OpenAI) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score       = 0.05   # safe non-zero default
    success     = False

    try:
        # ── Reset ──────────────────────────────────────────────
        r = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observation"]

        step_num = 0
        while not obs.get("done", False) and step_num < MAX_STEPS:
            step_num += 1
            action_str = "{}"
            error_msg  = None

            # ── LLM call ───────────────────────────────────────
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": (
                            f"Step {step_num}. Current inbox state:\n"
                            f"{json.dumps(obs, indent=2)}\n\nNext action?"
                        )},
                    ],
                    temperature=0,
                    max_tokens=200,
                )
                raw = (completion.choices[0].message.content or "{}").strip()
                # Strip markdown fences if present
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1]).strip()
                action_str = raw or "{}"
            except Exception as e:
                error_msg  = f"llm_error:{e}"
                action_str = "{}"

            # ── Step ───────────────────────────────────────────
            reward = 0.05
            done   = False
            try:
                step_r = requests.post(
                    f"{ENV_URL}/step", json={"action": action_str}, timeout=30
                )
                if step_r.ok:
                    res    = step_r.json()
                    reward = _clamp(res.get("reward", 0.05))
                    done   = bool(res.get("done", False))
                    obs    = res["observation"]
                else:
                    error_msg = f"step_http_{step_r.status_code}"
            except Exception as e:
                error_msg = f"step_error:{e}"

            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=action_str, reward=reward,
                     done=done, error=error_msg)

            if done:
                break

            time.sleep(0.3)

        # ── Final grading ───────────────────────────────────────
        try:
            grade_r = requests.post(f"{ENV_URL}/grader", json={}, timeout=30)
            if grade_r.ok:
                score = _clamp(grade_r.json().get("score", 0.05))
            else:
                # Fallback: mean of step rewards
                score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.05
        except Exception:
            score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.05

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        # Guarantee [END] is always emitted even on hard failure
        error_step_reward = 0.05
        if not rewards:
            rewards = [error_step_reward]
        if steps_taken == 0:
            steps_taken = 1
        log_step(step=steps_taken, action="{}", reward=error_step_reward,
                 done=True, error=str(e))
        score   = 0.05
        success = False

    # Safety guard — ensure rewards list is never empty
    if not rewards:
        rewards = [0.05]

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main() -> None:
    client = get_client()
    for task in TASKS:
        run_task(task, client)


if __name__ == "__main__":
    main()
