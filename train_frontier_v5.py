"""
train_frontier_v5.py — Sovereign Agent Training (Qwen2.5-7B + Thought-Verify)
Phase: Frontier SOTA
"""
import time, torch, json, numpy as np
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_from_disk
from environment import EmailTriageEnv

def sovereign_reward_fn(completions, **kwargs):
    """
    Sovereign Multi-Verifier Logic.
    Rewards Format + Causal Action + Reasoning Consistency.
    """
    env = EmailTriageEnv()
    rewards = []
    
    for comp in completions:
        try:
            # 1. Parse Rationality
            parsed = json.loads(comp)
            thought = parsed.get("thought", "")
            tool = parsed.get("tool", "")
            
            env.reset("expert")
            res = env.step(comp)
            
            # Weighted Signal Combination
            r_outcome = res.reward # Base env reward
            r_format = 0.2 if len(thought) > 10 else -0.1 # Penalty for skip reasoning
            r_causal = 0.3 if res.reward > 0.5 and tool == "schedule_meeting" else 0.0
            
            total = 0.5 * r_outcome + 0.3 * r_causal + 0.2 * r_format
            rewards.append(max(0.01, min(0.99, total)))
        except:
            rewards.append(0.01)
            
    return rewards

# --- Hyper-SOTA Load ---
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048, 
    dtype=torch.bfloat16,
    load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=64)

# Load high-capacity 50k dataset
dataset = load_from_disk("enterprise_dataset")

config = GRPOConfig(
    output_dir="sovereign-frontier-v5",
    beta=0.03, # Increased stability
    num_generations=16,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=7e-6,
    num_train_epochs=1,
    logging_steps=10,
    report_to="tensorboard"
)

trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer, args=config,
    train_dataset=dataset,
    reward_funcs=[sovereign_reward_fn]
)

if __name__ == "__main__":
    print("🔥 LAUNCHING SOVEREIGN TRAINING [v5.0.0]...")
    trainer.train()
    trainer.save_model("enterprise_sovereign_final")
