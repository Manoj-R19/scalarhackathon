import json
import time
import random
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd

from environment import (
    EmailTriageEnv,
    SovereignAgent,
    BaselineAgent,
    run_episode,
    benchmark,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SIMULATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def gradio_simulate(agent_type, enable_crisis, seed):
    try:
        env = EmailTriageEnv(enable_crisis=enable_crisis, seed=int(seed))
        agent = SovereignAgent() if "Sovereign" in agent_type else BaselineAgent()
        
        metrics = run_episode(agent, env, verbose=False)
        steps_data = metrics.pop("steps_data", [])

        # 📋 PREPARE LOG
        log_header = f"🚀 {agent_type.upper()} SESSION STARTED\n"
        log_header += f"Environment Seed: {int(seed)} | Crisis: {'Active' if enable_crisis else 'Inactive'}\n"
        log_header += "─" * 60 + "\n\n"
        
        log_steps = []
        for s in steps_data:
            causal = "✓" if s['causal_ok'] else "✗ BLOCK"
            crisis = "🚨" if s['crisis_active'] else ""
            line = f"Step {s['step']:02d} | {s['tool']:<18} | R={s['reward']:+.2f} | {causal} {crisis}\n"
            line += f"  Thought: {s['thought'][:90]}...\n"
            log_steps.append(line)
        
        full_log = log_header + "\n".join(log_steps)
        
        # 📊 PREPARE METRICS TABLE
        df_metrics = pd.DataFrame([
            {"Metric": "Episode Result", "Value": "✅ SUCCESS" if metrics.get("success") else "❌ FAILED"},
            {"Metric": "Reward Accumulation", "Value": f"{metrics.get('total_reward', 0):.3f}"},
            {"Metric": "Logic Consistency", "Value": f"{metrics.get('avg_logic', 0)*100:.1f}%"},
            {"Metric": "Outcome Accuracy", "Value": f"{metrics.get('avg_outcome', 0)*100:.1f}%"},
            {"Metric": "Causal Violations", "Value": str(metrics.get("causal_violations", 0))},
            {"Metric": "Crisis Mitigation", "Value": "RESOLVED" if metrics.get("crisis_resolved") else ("MISSED" if metrics.get("crisis_active") else "NONE")},
        ])
        
        return full_log, df_metrics
        
    except Exception as e:
        return f"Error during simulation: {str(e)}", pd.DataFrame([{"Metric": "Error", "Value": str(e)}])


def gradio_benchmark(n_episodes):
    try:
        results = benchmark(n_episodes=int(n_episodes))
        b, s = results["baseline"], results["sovereign"]
        
        comp_df = pd.DataFrame([
            {"Metric": "Success Rate", "Baseline": f"{b['success_rate']*100:.1f}%", "Sovereign": f"{s['success_rate']*100:.1f}%"},
            {"Metric": "Avg Reward", "Baseline": f"{b['avg_reward']:.3f}", "Sovereign": f"{s['avg_reward']:.3f}"},
            {"Metric": "Logic Score", "Baseline": f"{b['avg_logic']:.3f}", "Sovereign": f"{s['avg_logic']:.3f}"},
            {"Metric": "Crisis Resolve Rate", "Baseline": f"{b['crisis_resolve_rate']*100:.1f}%", "Sovereign": f"{s['crisis_resolve_rate']*100:.1f}%"},
            {"Metric": "Causal Violations", "Baseline": f"{b['avg_causal_violations']:.1f}", "Sovereign": f"{s['avg_causal_violations']:.1f}"},
        ])
        return comp_df
    except Exception as e:
        return pd.DataFrame([{"Metric": "Error", "Value": str(e)}])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  "DEEP AZURE" GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

# Clean professional styles
CUSTOM_CSS = """
.container { max-width: 1100px; margin: auto; padding-top: 2rem; }
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-size: 2.2rem; font-weight: 800; color: #1e3a8a; }
.header p { font-size: 1rem; color: #64748b; }
.log-box textarea { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; line-height: 1.5 !important; }
"""

with gr.Blocks(
    title="EmailTriage Sovereign Agent",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css=CUSTOM_CSS
) as demo:
    
    with gr.Div(elem_classes=["header"]):
        gr.Markdown("# 🛡️ EmailTriage Sovereign Agent")
        gr.Markdown("v5.0.0 · Enterprise AI Reasoning Engine · Verified by OpenEnv")

    with gr.Tabs():
        with gr.Tab("💠 Live Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    agent_selector = gr.Dropdown(
                        choices=["🛡️ Sovereign (Research Agent)", "🔴 Baseline (Standard LLM)"],
                        value="🛡️ Sovereign (Research Agent)",
                        label="Agent Type"
                    )
                    crisis_toggle = gr.Checkbox(value=True, label="Enable High-Entropy Crisis Injection")
                    seed_input = gr.Slider(0, 999, value=42, step=1, label="Environment Seed")
                    run_btn = gr.Button("▶ Run Episode", variant="primary")
                
                with gr.Column(scale=1):
                    metrics_output = gr.Dataframe(
                        headers=["Metric", "Value"],
                        datatype=["str", "str"],
                        label="Episode Intelligence Report",
                        interactive=False
                    )

            log_output = gr.Textbox(
                label="Step-by-Step Reasoned Trace",
                lines=18,
                elem_classes=["log-box"]
            )

            run_btn.click(
                fn=gradio_simulate,
                inputs=[agent_selector, crisis_toggle, seed_input],
                outputs=[log_output, metrics_output]
            )

        with gr.Tab("📈 Performance Benchmarking"):
            gr.Markdown("Stochastic evaluation across multiple environment seeds.")
            n_ep = gr.Slider(5, 100, value=20, step=5, label="Evaluation Episodes")
            bench_btn = gr.Button("🏁 Run Benchmark Comparison", variant="primary")
            bench_table = gr.Dataframe(label="Comparative Metrics")
            
            bench_btn.click(fn=gradio_benchmark, inputs=[n_ep], outputs=[bench_table])

        with gr.Tab("🔬 Architecture & Metadata"):
            gr.Markdown("""
            ### Sovereign Agent v5.0.0
            
            **Framework Implementation:**
            - **RLVE**: Verifiable Environments with strict causal gates (Check-before-Act).
            - **RLVR**: Verifiable Rewards with multi-headed process signals.
            - **Algorithm**: GRPO v2 (Group Relative Policy Optimisation).
            
            **Hardware & Model:**
            - **Model**: Qwen2.5-7B (Unsloth 4-bit LoRA).
            - **Deployment**: Dockerized FastAPI + Uvicorn.
            """)
            
            meta_btn = gr.Button("Get Discovery Metadata", size="sm")
            meta_out = gr.JSON(label="OpenEnv Endpoint")
            meta_btn.click(fn=lambda: {"name": "EmailTriage", "version": "5.0.0", "engine": "GRPO v2"}, outputs=[meta_out])

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FASTAPI BACKEND
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/meta")
async def get_meta():
    return JSONResponse(content={"status": "ready", "agent": "sovereign_v5", "compliance": "v0.3.0"})

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
