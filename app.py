import json
import time
import random
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from environment import EmailTriageEnv, SovereignAgent, BaselineAgent

# ─────────────────────────────────────────────────────────────────────────────
# 1.  STREAMING SIMULATION ENGINE (The "Live" Experience)
# ─────────────────────────────────────────────────────────────────────────────

def stream_simulation(agent_type, enable_crisis, seed):
    """
    Generator function that yields results step-by-step 
    to provide a live 'Real-Time' experience in the UI.
    """
    try:
        env = EmailTriageEnv(enable_crisis=enable_crisis, seed=int(seed))
        agent = SovereignAgent() if "Sovereign" in agent_type else BaselineAgent()
        
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        history = []
        log_accumulator = f"🚀 {agent_type.upper()} INITIALISED (Seed: {int(seed)})\n{'='*60}\n\n"

        while not done and step_count < 20:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 📝 UPDATE LOG
            causal = "✓" if info['causal_ok'] else "✗"
            crisis = "🚨" if info['crisis_active'] else "  "
            log_accumulator += f"Step {info['step']:02d} | {crisis} | {info['tool']:<18} | R={reward:+.2f} | Causal={causal}\n"
            log_accumulator += f"  Thought: {info['thought'][:95]}...\n\n"
            
            # 📊 UPDATE ANALYTICS
            history.append({
                "step": info["step"],
                "reward": total_reward,
                "logic": info["logic_score"],
                "outcome": info["outcome_score"]
            })
            
            # Create Plotly Fig
            fig = go.Figure()
            steps = [h["step"] for h in history]
            fig.add_trace(go.Scatter(x=steps, y=[h["reward"] for h in history], name="Total Reward", line=dict(color='#3b82f6', width=3)))
            fig.add_trace(go.Scatter(x=steps, y=[h["logic"] for h in history], name="Logic", line=dict(color='#7c3aed', dash='dot')))
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # 📋 UPDATE SUMMARY TABLE
            df_summary = pd.DataFrame([
                {"Stat": "Success Rate (Episode)", "Value": f"{info['outcome_score']*100:.1f}%"},
                {"Stat": "Causal Integrity", "Value": str(info['causal_ok'])},
                {"Stat": "Accrued Reward", "Value": f"{total_reward:.3f}"},
                {"Stat": "Crisis Detected", "Value": str(info['crisis_active'])}
            ])

            # YIELD STEP (Live Update)
            yield log_accumulator, df_summary, fig
            time.sleep(0.4) # Control simulation speed for visual impact

    except Exception as e:
        yield f"CRITICAL SYSTEM ERROR: {str(e)}", None, None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  THE "OBSIDIAN COMMAND CENTER" UI
# ─────────────────────────────────────────────────────────────────────────────

MASTER_CSS = """
.gradio-container { background-color: #000000 !important; color: #f8fafc !important; }
.header-box { border-left: 5px solid #3b82f6; padding-left: 1.5rem; margin-bottom: 2rem; }
.header-box h1 { font-size: 2.8rem; font-weight: 900; letter-spacing: -0.05em; color: #fff; }
.header-box p { font-size: 1.1rem; color: #94a3b8; }

.log-viewer textarea { 
    background-color: #020617 !important; 
    border-color: #1e293b !important; 
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}

.metric-table table { border: 1px solid #1e293b !important; }
.gr-button-primary { background: #3b82f6 !important; border-radius: 4px !important; font-weight: 800 !important; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=MASTER_CSS) as demo:
    
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown("# 🛡️ EmailTriage Sovereign Command")
        gr.Markdown("Real-Time Verifiable Reasoning & Causal Grounding Dashboard")

    with gr.Tabs():
        with gr.Tab("🎬 Live Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    agent_dropdown = gr.Dropdown(
                        choices=["🛡️ Sovereign Agent (GRPO v2)", "🔴 Baseline Agent (SFT Only)"],
                        value="🛡️ Sovereign Agent (GRPO v2)",
                        label="Select Intelligence Model"
                    )
                    crisis_chk = gr.Checkbox(value=True, label="Inject P0 Crisis Context (Step 7)")
                    seed_sl = gr.Slider(0, 999, value=42, step=1, label="Environment Seed")
                    run_master_btn = gr.Button("▶ START MISSION", variant="primary")
                
                with gr.Column(scale=1):
                    summary_df = gr.Dataframe(label="Mission Critical Data", interactive=False, elem_classes=["metric-table"])
                    reward_chart = gr.Plot(label="Real-Time Analytics Trace")

            log_viewer = gr.Textbox(label="Step-by-Step Reasoned Trace", lines=20, elem_classes=["log-viewer"])

            run_master_btn.click(
                fn=stream_simulation,
                inputs=[agent_dropdown, crisis_chk, seed_sl],
                outputs=[log_viewer, summary_df, reward_chart]
            )

        with gr.Tab("🔬 Research Architecture"):
            gr.Markdown("""
            ### The Sovereign Stack (v5.5.0)
            
            1. **RLVE (Verifiable Environments)**:
               - Mandatory `<thought>`-to-tool alignment scoring.
               - Causal Prerequisites: `schedule_meeting` requires `check_calendar`.
            
            2. **RLVR (Verifiable Rewards)**:
               - Multi-headed reward lattice: Logic(0.3) + Outcome(0.4) + Crisis(0.15) + Format(0.15).
            
            3. **Neural Engine**:
               - **Algorithm**: Group Relative Policy Optimisation (GRPO v2).
               - **Model**: Qwen2.5-7B-Instruct (Unsloth NF4 LoRA).
            """)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PRODUCTION INFRASTRUCTURE (FastAPI + Meta discovery)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get("/meta")
async def get_meta():
    """
    OpenEnv Agent Discovery Endpoint
    """
    return JSONResponse(content={
        "name": "EmailTriage Sovereign Agent",
        "version": "5.5.0",
        "description": "Enterprise Agentic Workflow with Process Supervision",
        "standards": ["OpenEnv v0.3.0", "RLVR", "RLVE"],
        "discovery_url": "https://huggingface.co/spaces/ManojR19/scalarhackatthon/meta"
    })

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
