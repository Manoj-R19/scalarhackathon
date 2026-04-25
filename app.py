import json
import time
import random
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from environment import (
    EmailTriageEnv,
    SovereignAgent,
    BaselineAgent,
    run_episode,
    benchmark,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SIMULATION LOGIC (V5 Logic, V4 Simplicity)
# ─────────────────────────────────────────────────────────────────────────────

def gradio_simulate(agent_type, enable_crisis, seed):
    env = EmailTriageEnv(enable_crisis=enable_crisis, seed=int(seed))
    agent = SovereignAgent() if "Sovereign" in agent_type else BaselineAgent()
    
    metrics = run_episode(agent, env, verbose=False)
    steps_data = metrics.pop("steps_data", [])

    # Simple formatted log
    log = f"=== {agent_type.upper()} EPISODE (Seed {seed}) ===\n\n"
    for s in steps_data:
        causal = "✓" if s['causal_ok'] else "✗ BLOCK"
        crisis = "🚨" if s['crisis_active'] else ""
        log += f"Step {s['step']:02d} | {s['tool']:<20} | R={s['reward']:+.2f} | Causal={causal} {crisis}\n"
        log += f"  Thought: {s['thought'][:80]}...\n\n"
    
    # Simple dictionary for the standard Gradio Label/JSON components
    summary_metrics = {
        "Result": "✅ SUCCESS" if metrics.get("success") else "❌ FAILED",
        "Total Reward": round(metrics.get("total_reward", 0), 3),
        "Causal Violations": metrics.get("causal_violations", 0),
        "Logic Score": round(metrics.get("avg_logic", 0), 3),
        "Tasks Done": metrics.get("tasks_completed", 0),
        "Crisis Status": "RESOLVED" if metrics.get("crisis_resolved") else ("MISSED" if metrics.get("crisis_active") else "N/A")
    }
    
    return log, summary_metrics


def gradio_benchmark(n_episodes):
    results = benchmark(n_episodes=int(n_episodes))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLEAN GRADIO UI (No custom CSS)
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="EmailTriage Sovereign Agent") as demo:
    
    gr.Markdown("# 🛡️ EmailTriage Sovereign Agent v5.0.0")
    gr.Markdown("Enterprise-grade RL agent with verifiable reasoning and causal grounding.")

    with gr.Tabs():
        with gr.Tab("🎬 Live Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    agent_selector = gr.Radio(
                        choices=["🛡️ Sovereign Agent", "🔴 Baseline Agent"],
                        value="🛡️ Sovereign Agent",
                        label="Select Agent"
                    )
                    crisis_toggle = gr.Checkbox(value=True, label="Enable Crisis injection")
                    seed_input = gr.Number(value=42, label="Environment Seed")
                    run_btn = gr.Button("Run Simulation", variant="primary")
                
                with gr.Column(scale=1):
                    # Using standard Gradio Label for metrics (clean and reliable)
                    metrics_output = gr.Label(label="Episode Summary")

            log_output = gr.Textbox(label="Action Log", lines=20)

            run_btn.click(
                fn=gradio_simulate,
                inputs=[agent_selector, crisis_toggle, seed_input],
                outputs=[log_output, metrics_output]
            )

        with gr.Tab("📊 Benchmark"):
            n_ep = gr.Slider(5, 50, value=20, step=5, label="Episodes per Agent")
            bench_btn = gr.Button("Run Benchmark")
            bench_out = gr.JSON(label="Comparative Results")
            bench_btn.click(fn=gradio_benchmark, inputs=[n_ep], outputs=[bench_out])

        with gr.Tab("🏗️ Architecture"):
            gr.Markdown("""
            ### Sovereign Agent Framework
            - **Environment**: RLVE (Verifiable Environment) with Causal Gates.
            - **Reward**: RLVR (Verifiable Reward) - Multi-headed logic.
            - **Optimization**: GRPO v2 (Group Relative Policy Optimisation).
            - **Base Model**: Qwen2.5-7B-Instruct.
            """)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  API SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/meta")
async def get_meta():
    return JSONResponse(content={
        "name": "EmailTriage Sovereign Agent",
        "version": "5.0.0",
        "compliance": "OpenEnv v0.3.0"
    })

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
