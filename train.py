"""
train.py — Training pipeline using Hugging Face TRL GRPOTrainer and Unsloth.
This script demonstrates how to train the OpenEnv via GRPO without a heavy critic model.

For Google Colab or local GPU setup:
pip install unsloth trl peft transformers accelerate openenv
"""
import os
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
try:
    from unsloth import is_bfloat16_supported
except ImportError:
    def is_bfloat16_supported(): return False

from trl import GRPOTrainer, GRPOConfig
from environment import EmailTriageEnv

# 1. Configuration for Fast Iteration
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16

def format_env_prompt(obs_dict):
    """Simple converter mapping OpenEnv obs to a list of dicts for generation."""
    return [
        {"role": "system", "content": "You are an Enterprise AI Agent. Output ONLY tool commands in JSON."},
        {"role": "user", "content": f"Current State: {obs_dict}. Process tools."}
    ]

def main():
    print("🚀 Initializing Unsloth FastLanguageModel...")
    # Load Model with Unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        load_in_4bit=True,  # 4bit quantization to save memory
        device_map="auto"
    )

    # Configure LoRA Adapters
    print("⚡ Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    PatchDPOTrainer() # Unsloth performance patch

    print("🌍 Binding OpenEnv for TRL...")
    def custom_environment_factory():
        # Inject the OpenEnv email-triage environment correctly
        env = EmailTriageEnv()
        return env

    # Configure GRPO (Generalized Reward Policy Optimization)
    training_args = GRPOConfig(
        output_dir="./grpo_checkpoints",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        max_steps=200,          # Quick hackathon demo run
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        logging_steps=10,
        beta=0.1,               # KL divergence penalty
        report_to="none"        # Disable wandb to prevent Colab hangs
    )

    print("🤖 Starting GRPOTrainer...")
    # Note: TRL 0.10+ GRPOTrainer allows passing an environment factory directly
    # and programmatically computing rewards without an external critic model.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        env_factory=custom_environment_factory,  # Passes env object directly
        train_dataset=None, # Uses env for internal rollout generation
    )

    # Kick off training
    trainer.train()

    print("✅ Training Complete. Saving LoRA adapter...")
    model.save_pretrained("email-triage-grpo-qwen2.5")
    tokenizer.save_pretrained("email-triage-grpo-qwen2.5")
    
    print("🎉 Done! Ready to push to Hugging Face or run inference.")

if __name__ == "__main__":
    main()
