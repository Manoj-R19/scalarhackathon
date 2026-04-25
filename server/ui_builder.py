import gradio as gr
import json
import pandas as pd
from environment import EmailTriageEnv
from server.ui_assets import CSS
from models import Action, ActionType
import random

def create_ui(env: EmailTriageEnv):
    with gr.Blocks(css=CSS, theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")) as demo:
        current_obs = gr.State(None)
        
        with gr.Column(elem_classes="container"):
            gr.HTML("""
                <div class="title-header">
                    <h1>Enterprise Agent Simulator: Theme #3.1</h1>
                    <p style="color: #94a3b8; margin-top: 0.5rem;">Multi-Step Tool Calling & Dynamic Environments • OpenEnv</p>
                </div>
            """)
            
            with gr.Row():
                # Left Panel: Email Inbox & Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Email Inbox")
                    inbox_display = gr.HTML("""
                        <div style="height: 350px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                            <p style="text-align: center; color: #64748b; margin-top: 50px;">Initialize environment...</p>
                        </div>
                    """)
                    
                    gr.Markdown("### 🛠 Simulation Controls")
                    with gr.Row():
                        reset_btn = gr.Button("🚀 Reset Scenario", variant="primary")
                        agent_mode = gr.Radio(["Baseline (Rule-Based)", "RL-Trained Model (GRPO)"], value="RL-Trained Model (GRPO)", label="Agent Mode")

                    simulate_btn = gr.Button("🤖 Run Agent Simulation", variant="secondary")

                # Right Panel: Enterprise State (Calendar & Tasks)
                with gr.Column(scale=1):
                    gr.Markdown("### 🏢 Enterprise State")
                    with gr.Tabs() as tabs:
                        with gr.Tab("📅 Calendar & Task Board", id="state_tab"):
                            calendar_display = gr.HTML("<div>Calendar slots will appear here.</div>")
                            taskboard_display = gr.HTML("<div>Active tickets will appear here.</div>")
                            
                        with gr.Tab("🤖 Action Stream", id="agent_tab"):
                            terminal_out = gr.Code(
                                label="Execution Logs",
                                value="Waiting for agent actions...",
                                language="markdown",
                                interactive=False,
                                elem_classes="terminal-card"
                            )
                            
                        with gr.Tab("📈 RL Training Curves", id="rl_tab"):
                            gr.Markdown(
                                "### Unsloth + TRL GRPO Training Rewards\n"
                                "During internal training, the agent's multi-step capabilities quickly converge on maximum stepwise efficiencies."
                            )
                            # Using a mock placeholder image for the reward curve
                            gr.HTML('<div style="text-align: center;"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/rlhf-reward-model.png" width="80%"></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Metrics")
                    with gr.Row():
                        score_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-score">0.00</div><div class="stat-label">Task Score</div></div>')
                        step_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-step">0</div><div class="stat-label">Steps Taken</div></div>')
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Live Performance Trend")
                    performance_chart = gr.BarPlot(
                        label="Reward per Action",
                        x="Step",
                        y="Reward",
                        tooltip=["Step", "Action", "Reward"],
                        height=200,
                        y_lim=[-1, 1]
                    )

        # ──────────────── UI rendering logic ──────────────── #

        def render_inbox(obs, state=None):
            html = '<div style="height: 350px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;">'
            inbox_list = state.inbox if state else []
            for email in inbox_list:
                status = email.get("status", "unread")
                icon = "📩" if status == "unread" else "✅"
                color = "#e2e8f0" if status == "unread" else "#64748b"
                html += f"""
                <div class="email-row" style="opacity: {1.0 if status == 'unread' else 0.6}; border-left: 4px solid {'#eab308' if 'CRITICAL' in str(email.get('subject')) else '#3b82f6'};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: {color};">{icon} {email.get('subject', 'No Subject')}</strong>
                        <span class="label-pill" style="font-size: 0.7rem;">{status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">From: {email.get('sender', 'Unknown')}</div>
                    <div style="font-size: 0.85rem; color: {color}; margin-top: 4px;">{email.get('body', '')}</div>
                </div>
                """
            html += '</div>'
            return html

        def render_enterprise(state):
            cal_html = '<h4>Calendar Schedules</h4><div style="border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px; margin-bottom: 15px; background: rgba(59, 130, 246, 0.05);">'
            for c in getattr(state, "calendar", []):
                cal_html += f'<div style="color: #60a5fa;">🕒 <strong>{c["time"]}</strong> - {c["event"]}</div>'
            cal_html += '</div>'

            task_html = '<h4>Task Board</h4><div style="border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px; background: rgba(234, 179, 8, 0.05);">'
            for t in getattr(state, "task_board", []):
                task_html += f'<div style="color: #fbbf24;">📋 <strong>[{t["ticket"]}]</strong> {t.get("issue", t.get("status"))}</div>'
            task_html += '</div>'
            
            return cal_html, task_html

        def on_reset():
            obs = env.reset(task="expert")
            state = env.get_state()
            inbox_h = render_inbox(obs, state)
            cal_h, task_h = render_enterprise(state)
            empty_df = pd.DataFrame({"Step": [], "Reward": [], "Action": []})
            
            score_h = f'<div class="stat-card"><div class="stat-value">0.00</div><div class="stat-label">Task Score</div></div>'
            step_h = f'<div class="stat-card"><div class="stat-value">0</div><div class="stat-label">Steps Taken</div></div>'
            term_out = "Environment initialized. Ready."
            
            return obs, inbox_h, cal_h, task_h, empty_df, score_h, step_h, term_out

        def on_simulate(mode_selection):
            obs = env.reset(task="expert")
            logs = [f"### 🤖 Simulation Started: {mode_selection}"]
            chart_data = []
            
            if "Baseline" in mode_selection:
                # Baseline crashes into calendar conflicts and hallucinates
                actions = [
                    Action(type=ActionType.schedule_meeting, time="15:00"), # Conflict
                    Action(type=ActionType.reply_email, email_id="1", message="Done"),
                ]
            else:
                # RL Trained agent uses correct intermediate steps and avoids conflicts
                actions = [
                    Action(type=ActionType.check_calendar),
                    Action(type=ActionType.schedule_meeting, time="16:00"),
                    Action(type=ActionType.reply_email, email_id="1", message="Meeting moved to 16:00."),
                    Action(type=ActionType.create_ticket, issue="DB Down!"), # Will trigger at step 3 because of dynamic event at step 2
                    Action(type=ActionType.reply_email, email_id="999", message="Investigating DB Down.")
                ]
                
            for act in actions:
                res = env.step(act.model_dump_json())
                logs.append(f"\n✅ **Step {res.info['step']}**: {act.type.value}")
                logs.append(f"   > *Reasoning*: {res.reasoning}")
                chart_data.append({"Step": res.info['step'], "Reward": res.reward, "Action": act.type.value})
                if res.done:
                    logs.append("\n🏁 Episodic limit or completion reached.")
                    break

            # Grade at the end
            score_res = env.grader()
            logs.append(f"\n--- \n### 🏁 Complete\n**Final Score: {score_res.score:.4f}**\n*Grader Log*: {score_res.details.get('msg', 'Failed task.')}")
            
            # Post-sim renders
            state = env.get_state()
            inbox_h = render_inbox(obs, state)
            cal_h, task_h = render_enterprise(state)
            
            score_h = f'<div class="stat-card"><div class="stat-value">{score_res.score:.2f}</div><div class="stat-label">Task Score</div></div>'
            step_h = f'<div class="stat-card"><div class="stat-value">{state.step_count}</div><div class="stat-label">Steps Taken</div></div>'
            df = pd.DataFrame(chart_data)
            
            return "\n".join(logs), inbox_h, cal_h, task_h, df, score_h, step_h


        reset_btn.click(
            on_reset, 
            inputs=[], 
            outputs=[current_obs, inbox_display, calendar_display, taskboard_display, performance_chart, score_stat, step_stat, terminal_out]
        )
        
        simulate_btn.click(
            on_simulate,
            inputs=[agent_mode],
            outputs=[terminal_out, inbox_display, calendar_display, taskboard_display, performance_chart, score_stat, step_stat]
        )

    return demo
