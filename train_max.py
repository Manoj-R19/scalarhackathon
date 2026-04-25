"""
train_max.py — Hyper-Optimized RL Training (15min Target)
Specifically for Theme 3.1 Expert Dynamics.
"""
import torch
import time
import json
import numpy as np
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from environment import EmailTriageEnv
from datasets import Dataset

# 1. Hardware Detection & Model Load
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct",
    dtype=dtype,
    load_in_4bit=True if DEVICE == "cuda" else False,
    max_seq_length=1024
)

# Light LoRA for speed
model = FastLanguageModel.get_peft_model(model, r=32)

# 2. Ultra-Fast Synthetic Data Generation (10k episodes)
def gen_fast_dataset(n_ep=200):
    """
    Parallel-ready generator for synthetic trajectories.
    Note: In real competition, use baseline rollouts.
    """
    envs = [EmailTriageEnv() for _ in range(8)]
    data = []
    for i in range(n_ep):
        env = envs[i % 8]
        obs = env.reset("expert")
        prompt = f"System State: {obs.model_dump_json()}\nAction:"
        # Mocking logical sequence: check -> schedule -> reply
        trajectory = [
            {"prompt": prompt, "action": '{"tool": "check_calendar", "params": {}}'},
            {"prompt": "Updated obs...", "action": '{"tool": "schedule_meeting", "params": {"time": 16.0}}'},
            {"prompt": "Final obs...", "action": '{"tool": "reply_email", "params": {"email_id": "e1"}}'}
        ]
        data.append({
            "prompt": [t["prompt"] for t in trajectory],
            "completion": [t["action"] for t in trajectory]
        })
    return Dataset.from_list(data)

def max_reward_fn(completions, **kwargs):
    """Vectorized-compatible reward function."""
    env = EmailTriageEnv()
    rewards = []
    for comp in completions:
        try:
            env.reset("expert")
            # Rapid step execution
            res = env.step(comp)
            # Combine stepwise reward with final grader check
            final = env.grader().score
            rewards.append(res.reward * 0.4 + final * 0.6)
        except:
            rewards.append(0.01)
    return rewards

# 3. High-Throughput Training Config
config = GRPOConfig(
    output_dir="./enterprise-max-rl",
    num_generations=16, # Max throughput
    beta=0.01,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    warmup_steps=10,
    logging_steps=10,
    save_steps=50,
    report_to="tensorboard",
    max_steps=100
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=gen_fast_dataset(50),
    reward_funcs=[max_reward_fn]
)

if __name__ == "__main__":
    start = time.time()
    print("🚀 LAUNCHING HYPER-PERF TRAINING...")
    trainer.train()
    trainer.save_model("enterprise_max_final")
    print(f"🏁 Trained in {time.time()-start:.0f}s")
