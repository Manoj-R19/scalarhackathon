"""
train_frontier_v5.py — EmailTriage Sovereign Agent GRPO Training
=================================================================
Stack : Unsloth + GRPO v2 + Qwen2.5-7B-Instruct
Method: Group Relative Policy Optimisation (GRPO) — DeepSeek-R1 style
Env   : EmailTriageEnv v5.0.0 (RLVE — Verifiable Environments)
Reward: Multi-headed (Outcome + Logic + Format + Crisis) — RLVR

Reference papers:
  - DeepSeekMath (2024): GRPO algorithm
  - Process Reward Models (Lightman et al., 2023)
  - Unsloth GRPO: memory-efficient RL fine-tuning
  - RLVE/RLVR framework: OpenEnv Hackathon 2025
"""

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

# ── Guard imports for non-training environments (e.g., demo mode) ─────────────
try:
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

from environment import (
    EmailTriageEnv,
    SovereignAgent,
    BaselineAgent,
    run_episode,
    benchmark,
    EMAIL_CORPUS,
    CRISIS_CORPUS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str            = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length: int        = 2048
    load_in_4bit: bool         = True
    dtype: Optional[str]       = None   # auto-detect

    # LoRA
    lora_r: int                = 16
    lora_alpha: int            = 16
    lora_dropout: float        = 0.0
    target_modules: tuple      = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # GRPO
    learning_rate: float       = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int       = 6       # G in GRPO — group size
    max_new_tokens: int        = 512
    temperature: float         = 0.9
    num_train_epochs: int      = 3
    max_steps: int             = 500
    save_steps: int            = 100
    logging_steps: int         = 10
    warmup_ratio: float        = 0.05
    beta: float                = 0.04   # KL penalty coefficient

    # Data
    n_train_episodes: int      = 2000
    n_val_episodes: int        = 100
    crisis_probability: float  = 0.6    # 60% of training episodes have crisis injection

    # Output
    output_dir: str            = "./outputs/sovereign_v5"
    hub_model_id: str          = "ManojR19/EmailTriage-Sovereign-v5"


CFG = TrainConfig()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Enterprise Email Triage Sovereign Agent.

Your task is to manage a corporate inbox, calendar, and crisis response workflow.

CRITICAL RULES:
1. Always wrap your reasoning in <thought>...</thought> tags BEFORE choosing a tool.
2. Your thought must explain WHY you are using this specific tool.
3. If a CRISIS is detected (breach, outage, PII leak), STOP all other tasks and escalate immediately.
4. Never schedule a meeting without first checking the calendar.
5. Never send a reply without first reading the email.

OUTPUT FORMAT — respond ONLY with valid JSON:
{
  "thought": "<your causal reasoning here — minimum 15 words>",
  "tool": "<tool_name>",
  "args": {<tool-specific arguments>}
}

Available tools: read_email, check_calendar, schedule_meeting, send_reply,
                 escalate_crisis, archive_email, flag_priority, search_inbox, mark_done
"""

def build_user_prompt(obs: dict) -> str:
    """Format the environment observation into a structured user message."""
    inbox_str = "\n".join(
        f"  - [{e['priority']}] {e['id']}: {e['subject']} (from: {e['from']})"
        for e in obs.get("inbox", [])
    )
    calendar_str = "\n".join(
        f"  {slot}: {status}"
        for slot, status in obs.get("calendar", {}).items()
    )
    crisis_str = ""
    if obs.get("crisis"):
        c = obs["crisis"]
        crisis_str = (
            f"\n\n⚠️  CRISIS ALERT [{c['type'].upper()}]:\n"
            f"  ID: {c['id']}\n"
            f"  Subject: {c['subject']}\n"
            f"  Priority: {c['priority']}"
        )

    completed = obs.get("completed", [])
    completed_str = ", ".join(completed) if completed else "none"

    return f"""=== ENTERPRISE INBOX ===
{inbox_str}

=== CALENDAR (Today) ===
{calendar_str}
{crisis_str}

=== COMPLETED TASKS ===
{completed_str}

=== STEP {obs.get('step', 0)} / 20 | TOTAL REWARD: {obs.get('total_reward', 0.0)} ===

What is your next action? Respond with valid JSON only."""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_dataset(
    n_episodes: int = CFG.n_train_episodes,
    seed: int = 42,
):
    """
    Generate a dataset of (prompt, completion, reward) triples for GRPO.
    The sovereign agent's optimal actions become the "gold" completions.
    GRPO will then fine-tune the LLM to match / exceed this quality.
    """
    rng = random.Random(seed)
    records = []

    sovereign = SovereignAgent()
    baseline  = BaselineAgent()

    for i in range(n_episodes):
        use_crisis = rng.random() < CFG.crisis_probability
        env = EmailTriageEnv(enable_crisis=use_crisis, seed=seed + i)
        obs = env.reset()
        sovereign.reset()
        done = False

        while not done:
            action = sovereign.act(obs)
            prompt = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": build_user_prompt(obs)},
            ]
            completion = json.dumps(action, ensure_ascii=False)

            obs, reward, done, info = env.step(action)

            records.append({
                "prompt":      prompt,
                "completion":  completion,
                "reward":      reward,
                "logic":       info["logic_score"],
                "outcome":     info["outcome_score"],
                "crisis_step": info["crisis_active"],
                "step":        info["step"],
            })

    print(f"Generated {len(records)} training steps from {n_episodes} episodes.")
    return Dataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MULTI-HEADED REWARD FUNCTIONS (for GRPO verifiable rewards)
# ─────────────────────────────────────────────────────────────────────────────

def reward_format(completions: list[str], **kwargs) -> list[float]:
    """Is the output valid JSON with required keys?"""
    scores = []
    for c in completions:
        try:
            parsed = json.loads(c)
            has_thought = "thought" in parsed and len(parsed["thought"].split()) >= 5
            has_tool    = "tool" in parsed and isinstance(parsed["tool"], str)
            has_args    = "args" in parsed and isinstance(parsed["args"], dict)
            score = 0.0
            if has_tool:    score += 0.4
            if has_thought: score += 0.4
            if has_args:    score += 0.2
            scores.append(score)
        except (json.JSONDecodeError, Exception):
            scores.append(0.0)
    return scores


def reward_logic(completions: list[str], **kwargs) -> list[float]:
    """Does the thought block align with the chosen tool?"""
    TOOL_KEYWORDS = {
        "read_email":       ["email", "read", "open", "check"],
        "check_calendar":   ["calendar", "schedule", "availability", "slot"],
        "schedule_meeting": ["meeting", "book", "schedule", "slot"],
        "send_reply":       ["reply", "respond", "answer"],
        "escalate_crisis":  ["crisis", "breach", "outage", "escalate", "incident"],
        "archive_email":    ["archive", "dismiss", "low priority"],
        "flag_priority":    ["flag", "priority", "mark"],
        "search_inbox":     ["search", "find", "filter"],
        "mark_done":        ["done", "complete", "finished"],
    }
    scores = []
    for c in completions:
        try:
            parsed = json.loads(c)
            tool = parsed.get("tool", "")
            thought = parsed.get("thought", "").lower()
            keywords = TOOL_KEYWORDS.get(tool, [])
            hit = any(kw in thought for kw in keywords)
            long_enough = len(thought.split()) >= 15
            score = (0.6 if hit else 0.0) + (0.4 if long_enough else 0.0)
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return scores


def reward_crisis_awareness(completions: list[str], crisis_step=None, **kwargs) -> list[float]:
    """Penalise if crisis is active but agent doesn't escalate."""
    scores = []
    crisis_flags = crisis_step or [False] * len(completions)
    for c, is_crisis in zip(completions, crisis_flags):
        if not is_crisis:
            scores.append(0.5)   # neutral — no crisis in this step
            continue
        try:
            parsed = json.loads(c)
            tool = parsed.get("tool", "")
            thought = parsed.get("thought", "").lower()
            crisis_keywords = ["crisis", "breach", "outage", "escalate", "incident", "critical"]
            if tool == "escalate_crisis":
                scores.append(1.0)
            elif any(kw in thought for kw in crisis_keywords):
                scores.append(0.6)  # at least aware of crisis
            else:
                scores.append(0.0)  # completely ignored crisis
        except Exception:
            scores.append(0.0)
    return scores


def reward_causal(completions: list[str], **kwargs) -> list[float]:
    """
    Heuristic causal reward: check if thought references prerequisite context
    for tools that have causal dependencies.
    """
    CAUSAL_HINTS = {
        "schedule_meeting": ["calendar", "checked", "available", "free slot"],
        "send_reply":       ["read", "opened", "reviewed", "email"],
        "escalate_crisis":  ["detected", "active", "read", "critical"],
    }
    scores = []
    for c in completions:
        try:
            parsed = json.loads(c)
            tool   = parsed.get("tool", "")
            thought = parsed.get("thought", "").lower()
            hints  = CAUSAL_HINTS.get(tool, [])
            if not hints:
                scores.append(0.8)  # no causal requirement for this tool
            else:
                scores.append(1.0 if any(h in thought for h in hints) else 0.2)
        except Exception:
            scores.append(0.0)
    return scores


# Combined reward for GRPO trainer
def combined_reward(completions: list[str], **kwargs) -> list[float]:
    W = {"format": 0.15, "logic": 0.30, "crisis": 0.20, "causal": 0.35}
    fmt    = reward_format(completions, **kwargs)
    logic  = reward_logic(completions, **kwargs)
    crisis = reward_crisis_awareness(completions, **kwargs)
    causal = reward_causal(completions, **kwargs)
    return [
        W["format"] * f + W["logic"] * l + W["crisis"] * cr + W["causal"] * ca
        for f, l, cr, ca in zip(fmt, logic, crisis, causal)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def train():
    if not TRAINING_AVAILABLE:
        raise RuntimeError(
            "Training dependencies not available. Install: unsloth, trl, transformers, datasets."
        )

    print("=" * 70)
    print(f"EmailTriage Sovereign Agent v5.0.0 — GRPO Training")
    print(f"Model: {CFG.model_name}")
    print("=" * 70)

    # ── 5a. Load model ────────────────────────────────────────────────────────
    print("\n[1/5] Loading Qwen2.5-7B with Unsloth 4-bit quantisation...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = CFG.model_name,
        max_seq_length = CFG.max_seq_length,
        dtype        = CFG.dtype,
        load_in_4bit = CFG.load_in_4bit,
    )

    # ── 5b. Apply LoRA ────────────────────────────────────────────────────────
    print("[2/5] Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                 = CFG.lora_r,
        target_modules    = list(CFG.target_modules),
        lora_alpha        = CFG.lora_alpha,
        lora_dropout      = CFG.lora_dropout,
        bias              = "none",
        use_gradient_checkpointing = "unsloth",
        random_state      = 42,
    )

    # ── 5c. Generate dataset ──────────────────────────────────────────────────
    print(f"[3/5] Generating {CFG.n_train_episodes} training episodes...")
    train_ds = generate_training_dataset(CFG.n_train_episodes)
    val_ds   = generate_training_dataset(CFG.n_val_episodes, seed=9999)

    # ── 5d. Configure GRPO ────────────────────────────────────────────────────
    print("[4/5] Configuring GRPO v2 trainer...")
    grpo_config = GRPOConfig(
        learning_rate                  = CFG.learning_rate,
        per_device_train_batch_size    = CFG.per_device_train_batch_size,
        gradient_accumulation_steps    = CFG.gradient_accumulation_steps,
        num_generations                = CFG.num_generations,
        max_new_tokens                 = CFG.max_new_tokens,
        temperature                    = CFG.temperature,
        num_train_epochs               = CFG.num_train_epochs,
        max_steps                      = CFG.max_steps,
        save_steps                     = CFG.save_steps,
        logging_steps                  = CFG.logging_steps,
        warmup_ratio                   = CFG.warmup_ratio,
        beta                           = CFG.beta,
        output_dir                     = CFG.output_dir,
        report_to                      = "none",
        remove_unused_columns          = False,
        optim                          = "adamw_8bit",
        fp16                           = not torch.cuda.is_bf16_supported(),
        bf16                           = torch.cuda.is_bf16_supported(),
        seed                           = 42,
    )

    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        reward_funcs  = [combined_reward],
        args          = grpo_config,
    )

    # ── 5e. Train ─────────────────────────────────────────────────────────────
    print("[5/5] Starting GRPO training...\n")
    trainer.train()

    # ── 5f. Save & push ───────────────────────────────────────────────────────
    print(f"\nSaving model to {CFG.output_dir}...")
    model.save_pretrained(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)

    if CFG.hub_model_id:
        print(f"Pushing to HuggingFace Hub: {CFG.hub_model_id}...")
        model.push_to_hub(CFG.hub_model_id)
        tokenizer.push_to_hub(CFG.hub_model_id)

    print("\n✅ Training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUICK BENCHMARK (no training required)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(n_episodes: int = 30):
    print(f"\nBenchmarking {n_episodes} episodes per agent...\n")
    results = benchmark(n_episodes=n_episodes)

    print("=" * 55)
    print(f"{'METRIC':<30} {'BASELINE':>10} {'SOVEREIGN':>10}")
    print("-" * 55)
    for key in ["success_rate", "avg_reward", "avg_logic",
                "crisis_resolve_rate", "avg_causal_violations"]:
        b = results["baseline"].get(key, "—")
        s = results["sovereign"].get(key, "—")
        print(f"  {key:<28} {str(b):>10} {str(s):>10}")
    print("=" * 55)
    return results


if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train()
    else:
        run_benchmark(n_episodes=20)
