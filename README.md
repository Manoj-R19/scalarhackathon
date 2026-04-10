---
title: Email Triage NextGen 🚀
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.12"
app_file: app.py
pinned: false
---

# EmailTriage NextGen 🚀

> **State-of-the-Art Email Support Triage Benchmark.**  
> Built for the OpenEnv Scalar Hackathon. Version 2.0 brings a premium visual dashboard, expert-level scenarios, and ultra-robust reward shaping.

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.dev)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

---

## 🎯 Why Email Triage?

Every support team processes hundreds of emails daily — spam, urgent bug reports, feature requests, billing issues. Training AI agents on this task develops:

- **Classification** (spam vs. legit, priority levels)
- **Judgment** (when to escalate a P0 production outage vs. a feature request)
- **Communication** (drafting concise, relevant replies)
- **Hierarchy** (label before draft, escalate before reply)

No existing OpenEnv benchmark covers this domain. This fills a real gap.

---

## 🗂️ Tasks

| Task | Emails | Description | Key Skill |
|------|--------|-------------|-----------|
| **easy** | 5 | Binary classify: spam or legit (low/med/high) | Pattern recognition |
| **medium** | 10 | Prioritize mixed inbox + delete spam | Urgency judgment |
| **hard** | 15 | Full triage: classify + draft replies + escalate P0s | Compositionality |
| **expert** | 5 | Extreme cases: phishing, legal, infrastructure cost spikes | Critical Judgment |

### Grader Weights

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| Label Accuracy | 0.6 | 0.5 | 0.3 |
| Spam Recall | 0.1 | 0.2 | 0.1 |
| Reply Relevance | 0.2 | 0.2 | 0.2 |
| Escalation Recall | — | — | 0.3 |
| Inbox Cleared | 0.1 | 0.1 | 0.1 |

## 💎 NextGen Features (v2.0)

-   **Premium Dashboard**: Real-time visual control center at `/ui` (built with Gradio + Custom CSS).
-   **Expert Difficulty**: High-stakes scenarios including data subpoenas and phishing detection.
-   **Agent Simulation**: Built-in mock agent to visualize environment transitions and reward signals.
-   **Bulletproof Graders**: Strict (0, 1) score clamping for seamless OpenEnv integration.

---

## 📊 Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Rule-Based (keyword) | 0.75 | 0.55 | 0.40 |
| GPT-4o-mini | 0.82 | 0.62 | 0.48 |
| Human (simulated) | 0.95 | 0.91 | 0.87 |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
docker build -t email-triage .
docker run -p 7860:7860 email-triage
```

### Option 2: Local Dev

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Option 3: HuggingFace Space

```bash
huggingface-cli login
huggingface-cli repo create email-triage --type space --sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/email-triage
git push hf main
```

---

## 🔌 API Reference

Base URL: `http://localhost:7860`  
Interactive docs: `http://localhost:7860/docs`

### `GET /tasks`
List all tasks with descriptions and grader weights.

```bash
curl http://localhost:7860/tasks
```

### `POST /reset`
Reset environment with a specific task.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

**Response:**
```json
{
  "observation": {
    "current_emails": [
      {"id": "e1", "subject": "WIN $1000 NOW!!!", "body": "...", "sender": "...", "labeled": false}
    ],
    "stats": {"total": 5, "unread": 5, "labeled": 0, "deleted": 0, "escalated": 0, "drafts": 0},
    "history": [],
    "done": false,
    "step": 0
  }
}
```

### `POST /step`
Take one action in the environment.

```bash
# Label an email
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "{\"type\": \"label\", \"email_id\": \"e1\", \"value\": \"spam\"}"}'

# Delete spam
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "{\"type\": \"delete\", \"email_id\": \"e1\"}"}'

# Draft a reply
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "{\"type\": \"draft\", \"email_id\": \"e2\", \"value\": \"We are investigating your issue immediately.\"}"}'

# Escalate a P0 (hard task)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "{\"type\": \"escalate\", \"email_id\": \"h1\"}"}'
```

**Response:**
```json
{
  "observation": {...},
  "reward": 0.3,
  "done": false,
  "info": {"step": 1, "task": "easy", "action_applied": "delete"}
}
```

### `POST /grader`
Score the current episode (0.0 – 1.0).

```bash
curl -X POST http://localhost:7860/grader -H "Content-Type: application/json" -d '{}'
```

**Response:**
```json
{
  "score": 0.74,
  "breakdown": {
    "label_accuracy": 0.8,
    "spam_recall": 1.0,
    "reply_relevance": 0.65,
    "escalation_recall": 0.0,
    "inbox_cleared": 1.0
  },
  "details": {"task": "easy", "steps_used": "5", ...}
}
```

### `GET /state`
Full internal state for debugging.

### `GET /health`
Health check — returns `{"status": "ok"}`.

---

## 🤖 Action Space

```json
{
  "type": "label | draft | delete | escalate | archive",
  "email_id": "string",
  "value": "string (label value or reply text, optional)"
}
```

### Action Details

| Action | `value` | Effect | Reward |
|--------|---------|--------|--------|
| `label` | `spam\|low\|med\|high\|escalate` | Tag email | +0.2 correct, -0.1 wrong |
| `delete` | — | Remove from inbox | +0.3 if spam, **-0.4** if legit |
| `draft` | reply text | Write reply | +0.05 to +0.35 (keyword match) |
| `escalate` | — | Mark as P0 | **+0.4** if P0, -0.2 if not |
| `archive` | — | Archive email | +0.05/+0.1, -0.1 for high |

---

## 🧠 Escalation Mechanic (Hard Task)

The `escalate` action is unique to production-ready support workflows. Emails requiring escalation are P0 incidents:

- 🔴 Production database corruption
- 🔴 Security vulnerability (SQL injection, etc.)  
- 🔴 All-user authentication failures

Escalating correctly gives the highest reward (+0.4). Missing an escalation significantly hurts the hard task score (escalation recall weight: 0.3).

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected: **All 40+ tests pass**.

---

## 🏃 Running the Baseline

```bash
# Rule-based (no API key needed)
python baseline.py --mode rule

# LLM-based (requires OpenAI API key)
OPENAI_API_KEY=sk-... python baseline.py --mode llm

# Both modes
OPENAI_API_KEY=sk-... python baseline.py --mode both
```

---

## 📁 Project Structure

```
email-triage-env/
├── openenv.yaml          # OpenEnv manifest (tasks, schemas, metadata)
├── models.py             # Pydantic schemas (Email, Action, Observation, State)
├── environment.py        # Core env logic (step, reset, grader, reward shaping)
├── baseline.py           # Rule-based + GPT-4o-mini baseline agents
├── requirements.txt      # Pinned dependencies
├── pyproject.toml        # Package metadata
├── Dockerfile            # HF Spaces compatible (port 7860)
├── .gitignore
├── server/
│   ├── __init__.py
│   └── app.py            # FastAPI server (all endpoints)
├── data/
│   └── inboxes.json      # Task emails + ground truth (easy/medium/hard)
├── tests/
│   ├── __init__.py
│   └── test_env.py       # 40+ TDD tests
└── README.md
```

---

## 🌟 Key Design Decisions

1. **No external LLM in grader** — All scoring is deterministic (keyword match, exact label comparison). Reproducible, fair, fast.
2. **Shaped rewards** — Dense reward signal every step (progress bonus + action quality). Agents learn incrementally.
3. **Escalation mechanic** — Novel action not found in any existing benchmark. Models judgment under pressure.
4. **Seeded randomness** — `random.seed(42)` in reset guarantees reproducible episodes.
5. **Hard penalties for irreversible mistakes** — Deleting a legit email (−0.4) teaches caution.

---

## 📄 License

MIT License — free to use, modify, and distribute.
