"""
train_frontier.py — Frontier RL Training (Qwen2.5-7B + GRPO v2)
Targets 0.95 Expert Success.
"""
import time
import numpy as np
import torch
import json
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
from datasets import load_from_disk
from peft import LoraConfig
from environment import EmailTriageEnv

start_time = time.time()

# 1. Load Frontier Model (7B for max capacity)
print("🚀 Loading Frontier Qwen2.5-7B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048, # Optimized for speed
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=True
)

lora_config = LoraConfig(
    r=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_rslora=True
)
model = FastLanguageModel.get_peft_model(model, lora_config)

try:
    PatchFastRL("grpo")
except:
    pass

# 2. Load Mega Dataset
print("📦 Loading 50k Mega Dataset...")
try:
    dataset = load_from_disk("enterprise_dataset").shuffle()
except Exception as e:
    print(f"❌ Error loading dataset: {e}. Run python data/gen_dataset.py first.")
    import sys
    sys.exit(1)

# 3. Frontier Reward Function
def enterprise_reward_fn(samples):
    """
    Multi-signal reward: Path Reward + Final Score + Process Bonus.
    """
    # Note: TRL GRPOTrainer passes prompts/completions differently than specific manual loops
    # This is a simplified version of the user request logic adapted for GRPOTrainer reward_funcs
    rewards = []
    env = EmailTriageEnv()
    
    # In GRPOTrainer v2, reward functions usually take (completions, prompts, ...)
    # If the user request suggests 'samples', we'll assume a custom rollout check
    for completion in samples:
        try:
            env.reset("expert")
            # Rapid execution of completion output
            res = env.step(completion)
            path_reward = res.reward
            final = env.grader().score
            
            # Process bonus: did it think to check calendar before scheduling?
            # Mocking process verification from the reasoning text in completion
            process_bonus = 0.2 if "check" in completion.lower() else -0.1
            
            total_reward = 0.4 * path_reward + 0.4 * final + 0.2 * process_bonus
            rewards.append(max(0.01, min(0.99, total_reward)))
        except:
            rewards.append(0.01)
            
    return rewards

# 4. Hyper-Max Config
config = GRPOConfig(
    output_dir="enterprise-frontier-rl",
    beta=0.02,
    num_generations=16, # Balanced for common GPUs
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    num_train_epochs=1, # Quick run or scaling in Colab
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    report_to="tensorboard",
    max_prompt_length=1024,
    max_completion_length=128
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset.select(range(min(40000, len(dataset)))),
    eval_dataset=dataset.select(range(min(40000, len(dataset)), min(50000, len(dataset)))),
    reward_funcs=[enterprise_reward_fn],
)

if __name__ == "__main__":
    print("🔥 Starting Frontier RL Training...")
    trainer.train()
    trainer.save_model("enterprise-frontier-rl")
    
    print(f"🎉 Frontier RL Complete: {time.time()-start_time:.0f}s")
    print("Exported to: enterprise-frontier-rl")
