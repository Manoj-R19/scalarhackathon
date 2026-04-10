
import gradio as gr
import json
from environment import EmailTriageEnv
from server.ui_assets import CSS
from models import Action, ActionType

def create_ui(env: EmailTriageEnv):
    with gr.Blocks(css=CSS, theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")) as demo:
        # State
        current_obs = gr.State(None)
        
        with gr.Column(elem_classes="container"):
            # Header
            gr.HTML("""
                <div class="title-header">
                    <h1>EmailTriage : Next Generation Benchmark</h1>
                    <p style="color: #94a3b8; margin-top: 0.5rem;">Interactive Agent Control Center • OpenEnv Scalar Hackathon 2025</p>
                </div>
            """)
            
            with gr.Row():
                # Left Panel: Tasks & Controls
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠 Environment Control")
                    task_dropdown = gr.Dropdown(
                        choices=["easy", "medium", "hard", "expert"],
                        value="easy",
                        label="Difficulty Level"
                    )
                    reset_btn = gr.Button("🚀 Initialize / Reset", variant="primary")
                    
                    gr.Markdown("### 📊 Live Performance")
                    with gr.Row():
                        labeled_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-labeled">0</div><div class="stat-label">Labeled</div></div>')
                        drafts_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-drafts">0</div><div class="stat-label">Drafts</div></div>')
                    
                    with gr.Row():
                        deleted_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-deleted">0</div><div class="stat-label">Deleted</div></div>')
                        score_stat = gr.HTML('<div class="stat-card"><div class="stat-value" id="stat-score">0.00</div><div class="stat-label">Task Score</div></div>')

                # Right Panel: Inbox & Agent
                with gr.Column(scale=2):
                    with gr.Tabs() as tabs:
                        with gr.Tab("📥 Virtual Inbox", id="inbox_tab"):
                            inbox_display = gr.HTML("""
                                <div style="height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                                    <p style="text-align: center; color: #64748b; margin-top: 50px;">Reset environment to load emails...</p>
                                </div>
                            """)
                            
                        with gr.Tab("🤖 Agent Simulation", id="agent_tab"):
                            terminal_out = gr.Code(
                                label="Action Stream",
                                value="Waiting for agent actions...",
                                language="markdown",
                                interactive=False,
                                elem_classes="terminal-card"
                            )
                            simulate_btn = gr.Button("🤖 Run Mock Agent (Automated Triage)", variant="secondary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔍 Environment Inspect")
                    state_json = gr.JSON(label="Full Environment State")

        # ──────────────── Logic ────────────────

        def on_reset(task):
            obs = env.reset(task=task)
            
            # Update Statistics
            stats = obs.stats
            labeled_html = f'<div class="stat-card"><div class="stat-value">{stats.labeled}</div><div class="stat-label">Labeled</div></div>'
            drafts_html = f'<div class="stat-card"><div class="stat-value">{stats.drafts}</div><div class="stat-label">Drafts</div></div>'
            deleted_html = f'<div class="stat-card"><div class="stat-value">{stats.deleted}</div><div class="stat-label">Deleted</div></div>'
            
            # Update Inbox HTML
            inbox_html = '<div style="height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;">'
            for email in obs.current_emails:
                label_html = ""
                if email.labeled:
                    label_class = f"label-{email.label}"
                    label_html = f'<span class="label-pill {label_class}">{email.label}</span>'
                
                inbox_html += f"""
                <div class="email-row">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #e2e8f0;">{email.subject}</strong>
                        {label_html}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">From: {email.sender}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{email.body}</div>
                </div>
                """
            inbox_html += '</div>'
            
            return obs, labeled_html, drafts_html, deleted_html, inbox_html, env.get_state().model_dump()

        def on_simulate(task):
            # Run a basic heuristic agent
            env.reset(task=task)
            logs = [f"### Starting Automated Triage for task: {task}"]
            
            obs = env._get_obs()
            emails = obs.current_emails
            
            for email in emails:
                body = email.body.lower()
                sender = email.sender.lower()
                
                # Simple logic for simulation
                action = None
                if "nigerian prince" in body or "macbook" in body or "win" in body or ".biz" in sender or ".ru" in sender:
                    action = Action(type=ActionType.delete, email_id=email.id)
                elif "urgent" in body or "crash" in body or "broken" in body or "legal" in body or "subpoena" in body:
                    action = Action(type=ActionType.escalate, email_id=email.id)
                elif "billing" in body or "payment" in body or "invoice" in body:
                    action = Action(type=ActionType.label, email_id=email.id, value="high")
                else:
                    action = Action(type=ActionType.label, email_id=email.id, value="low")
                
                res = env.step(action.model_dump_json())
                logs.append(f"✅ Step {res.info['step']}: {action.type.value} on `{email.id}` | Reward: {res.reward:.2f}")

            # Grade at the end
            score_res = env.grader()
            logs.append(f"\n### Simulation Complete\n**Final Score: {score_res.score:.4f}**")
            
            # Wrap up
            _, l, d, del_h, inbox_h, state = on_reset(task)
            score_html = f'<div class="stat-card"><div class="stat-value">{score_res.score:.2f}</div><div class="stat-label">Final Score</div></div>'
            
            return "\n".join(logs), l, d, del_h, score_html, inbox_h, state

        reset_btn.click(
            on_reset, 
            inputs=[task_dropdown], 
            outputs=[current_obs, labeled_stat, drafts_stat, deleted_stat, inbox_display, state_json]
        )
        
        simulate_btn.click(
            on_simulate,
            inputs=[task_dropdown],
            outputs=[terminal_out, labeled_stat, drafts_stat, deleted_stat, score_stat, inbox_display, state_json]
        )

    return demo
