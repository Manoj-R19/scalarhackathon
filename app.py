import json
import time
import random
import gradio as gr
import plotly.graph_objects as go
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

        # Clean Replay Log
        log = f"SESSION LOG: {agent_type}\n{'='*30}\n\n"
        for s in steps_data:
            causal = "✓" if s['causal_ok'] else "✗"
            crisis = "🚨" if s['crisis_active'] else "  "
            log += f"Step {s['step']:02d} | {crisis} | {s['tool']:<18} | R={s['reward']:+.2f} | Causal={causal}\n"
            log += f"  > {s['thought'][:90]}...\n\n"
        
        # Professional Metric Dataframe
        df = pd.DataFrame([
            {"Metric": "Success Status", "Value": "✅ SUCCESS" if metrics.get("success") else "❌ FAILED"},
            {"Metric": "Reward Accumulation", "Value": f"{metrics.get('total_reward', 0):.4f}"},
            {"Metric": "Logic Accuracy", "Value": f"{metrics.get('avg_logic', 0)*100:.1f}%"},
            {"Metric": "Causal Consistency", "Value": f"{metrics.get('causal_violations', 0)} Violations"},
            {"Metric": "Crisis Mitigation", "Value": "SOLVED" if metrics.get("crisis_resolved") else ("FAILED" if metrics.get("crisis_active") else "N/A")}
        ])

        # Obsidian Chart
        steps = [s["step"] for s in steps_data]
        rewards = [s["reward"] for s in steps_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=rewards, name="Total Reward", line=dict(color='#3b82f6', width=3)))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            height=250,
            title="Sovereign Reward Trace"
        )
        
        return log, df, fig
        
    except Exception as e:
        return f"CRITICAL ERROR: {str(e)}", pd.DataFrame([{"Metric": "Error", "Value": "Fail"}]), None

def gradio_benchmark(episodes):
    try:
        n = int(episodes)
        res = benchmark(n_episodes=n)
        b, s = res["baseline"], res["sovereign"]
        
        df = pd.DataFrame([
            {"Metric": "Success Rate", "Baseline": f"{b['success_rate']*100:.1f}%", "Sovereign": f"{s['success_rate']*100:.1f}%"},
            {"Metric": "Logic Alignment", "Baseline": f"{b['avg_logic']:.3f}", "Sovereign": f"{s['avg_logic']:.3f}"},
            {"Metric": "Crisis Resolve %", "Baseline": f"{b['crisis_resolve_rate']*100:.1f}%", "Sovereign": f"{s['crisis_resolve_rate']*100:.1f}%"},
            {"Metric": "Causal Integrity", "Baseline": f"{b['avg_causal_violations']:.2f}", "Sovereign": f"{s['avg_causal_violations']:.2f}"}
        ])
        return df
    except Exception as e:
        return pd.DataFrame([{"Metric": "Error", "Value": str(e)}])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  "OBSIDIAN MINIMALIST" UI (My Preferred Design)
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { background-color: #000000 !important; }
.header-text { margin-bottom: 2rem; border-left: 4px solid #3b82f6; padding-left: 1rem; }
.mono-log textarea { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=CSS) as demo:
    
    with gr.Column(elem_classes=["header-text"]):
        gr.Markdown("# EmailTriage Sovereign Agent")
        gr.Markdown("Enterprise Reinforcement Learning Framework · v5.0.0")

    with gr.Tabs():
        with gr.Tab("Live Sim"):
            with gr.Row():
                with gr.Column(scale=1):
                    agent = gr.Dropdown(["🛡️ Sovereign Agent", "🔴 Baseline Agent"], value="🛡️ Sovereign Agent", label="Active Agent")
                    crisis = gr.Checkbox(True, label="Crisis Active")
                    seed = gr.Slider(0, 999, 42, step=1, label="Seed")
                    run = gr.Button("RUN EPISODE", variant="primary")
                
                with gr.Column(scale=1):
                    metrics = gr.Dataframe(label="Intelligence Metrics", interactive=False)
                    chart = gr.Plot(label="Performance Visualisation")

            log = gr.Textbox(label="Reasoned Action Log", lines=20, elem_classes=["mono-log"])
            
            run.click(gradio_simulate, [agent, crisis, seed], [log, metrics, chart])

        with gr.Tab("Benchmark"):
            n_ep = gr.Slider(5, 50, 10, step=5, label="Evaluation Count")
            b_btn = gr.Button("START BENCHMARK", variant="primary")
            b_table = gr.Dataframe(label="Comparative Performance")
            
            b_btn.click(gradio_benchmark, [n_ep], [b_table])

        with gr.Tab("Architecture"):
            gr.Markdown("""
            ### System Design
            - **Environment**: OpenEnv v0.3.0 compliant.
            - **Logic**: Causal grounding via RLVE logic gates.
            - **Reward**: Multi-headed RLVR lattice.
            - **Base Model**: Qwen2.5-7B (Unsloth optimized).
            """)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SERVER MOUNT
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/meta")
async def get_meta():
    return JSONResponse(content={"name": "Sovereign Agent", "version": "5.0.0"})

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
