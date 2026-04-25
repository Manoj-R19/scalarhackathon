# EmailTriage: Enterprise Agent Simulator (Project Details)

This project is a high-fidelity **Reinforcement Learning (RL) benchmark environment** designed specifically for the **Scalar OpenEnv Hackathon 2025**. It transitions the domain of email triage from simple text classification into a **complex, multi-step enterprise workflow simulation**.

---

## 🎯 Theme Alignment: #3.1 Professional Tasks
While most entries focus on creative writing or simple classification, this project hits the **"Enterprise Workflow Agent"** requirement. It simulates an agent capable of navigating multiple internal applications (Email, Calendar, Tasks) to resolve high-stakes professional requests.

---

## 🏗️ Technical Architecture

### 1. The Environment (`environment.py`)
The "World" is partially observable and stateful.
- **Multitask State**: Tracks status across an Inbox, a Calendar, and a Task board.
- **Causal Logic**: The environment enforces real-world constraints. For example, if an agent tries to `schedule_meeting` during an existing "Marketing" block, the environment rejects the action and applies a negative reward.
- **Dynamic Injections**: Mid-episode, a "CRITICAL" server incident is injected. This tests the agent’s ability to reprioritize (Theme 3.1 requirement for dynamic systems).

### 2. Standardized Action Interface (`models.py`)
The project uses strict **Structured Tool Calling**.
- Instead of raw text, the agent must output valid JSON tools:
  - `check_calendar({})`
  - `schedule_meeting({"time": "16:00"})`
  - `create_task({"issue": "string"})`
  - `reply_email({"email_id": "e1", "message": "string"})`
- Powered by Pydantic for 100% compliance with OpenEnv schema standards.

### 3. The RL Training Pipeline (`train.py`)
Built using the latest industry-standard RLVR (Reinforcement Learning from Verifiable Rewards) techniques:
- **Unsloth GRPO**: Uses Group Relative Policy Optimization to train models without an expensive separate Critic model.
- **Direct Environment Hook**: The training loop feeds generations directly back into `EmailTriageEnv`.
- **Target Model**: Optimized for **Qwen2.5-3B-Instruct** (perfect for local GPU or Colab execution).

---

## 💎 Premium Features

### 📊 "Storytelling" Dashboard
The Gradio-based interface (`server/ui_builder.py`) is designed to **WOW** judges:
- **Glassmorphic UI**: High-end CSS with animated gradients and translucent panels.
- **Live State Visualization**: Watch the Calendar update and Task Board fill up as the agent works.
- **The "RL Proof" Toggle**: Switch between a "Baseline" view (where the agent fails at conflicts) and "Trained" view (where it adapts dynamically).

### 🛡️ Layered Rewards & Anti-Hack
- **Progressive Shaping**: Tiny rewards for smart intermediate steps (`+0.1`).
- **Terminal Massive Bonus**: A huge `+0.8` reward only if the final state verifies the task is solved.
- **Strict Clamping**: Every score is mathematically clamped to `(0.01, 0.99)` using `safe_normalize` to pass OpenEnv Phase 2 validation.

---

## 📈 Scaler Scoring Impact
- **Innovation (40%)**: Uses multi-step tool calling and dynamic incident response.
- **Storytelling (30%)**: Premium UI clearly visualizes the model’s "causal reasoning".
- **Performance (20%)**: Programmatic rewards guarantee the model is actually learning to solve the logic, not just hallucinating words.

---

## 📁 File Structure Overview
- `environment.py`: Core simulation logic.
- `models.py`: Data schemas and Tool definitions.
- `baseline.py`: Local simulation/validation script.
- `train.py`: Unsloth GRPO training script.
- `test_suite.py`: Automated OpenEnv validation checks.
- `server/app.py`: FastAPI server for OpenEnv API.
- `server/ui_builder.py`: Premium Gradio frontend.
