"""
baseline.py — Rule-based + LLM baseline agent for EmailTriage OpenEnv
Demonstrates reproducible scores across all three tasks.

Usage (rule-based, no API key):
    python baseline.py --mode rule

Usage (LLM-based, requires OPENAI_API_KEY):
    OPENAI_API_KEY=sk-... python baseline.py --mode llm

Usage (both):
    python baseline.py --mode both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import requests

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TASKS = ["easy", "medium", "hard"]


# ─────────────────── Rule-Based Agent ─────────────────────────

SPAM_KEYWORDS = [
    "win", "congratulations", "claim", "prize", "click here", "free", "limited",
    "offer", "guaranteed", "make money", "cheap", "buy now", "earn $", "nigerian",
    "prince", "million", "usd", "viagra", "cialis", "seo", "backlinks", "crypto",
    "10x return", "risk free"
]

ESCALATE_KEYWORDS = [
    "corrupted", "all data lost", "p0", "production down", "sql injection",
    "security vulnerability", "sso broken", "all users locked", "revenue impact",
    "legal team", "gdpr", "dpa", "immediately escalate"
]

HIGH_KEYWORDS = [
    "crash", "urgent", "asap", "broken", "not working", "bug", "error",
    "failed", "critical", "2fa", "login loop", "payment failed", "api broken",
    "503", "can't login", "export", "data loss"
]

MED_KEYWORDS = [
    "billing", "invoice", "refund", "overdue", "storage", "upgrade",
    "rate limit", "sla", "partnership", "gdpr", "compliance"
]


def rule_classify(email: dict) -> tuple[str, str]:
    """Returns (action_type, label_or_none)."""
    subject = (email.get("subject", "") + " " + email.get("body", "")).lower()

    # Check escalate first (hard task P0s)
    if any(kw in subject for kw in ESCALATE_KEYWORDS):
        return "escalate", "escalate"

    # Check spam
    if any(kw in subject for kw in SPAM_KEYWORDS):
        return "delete", "spam"

    # High priority
    if any(kw in subject for kw in HIGH_KEYWORDS):
        return "label", "high"

    # Medium priority
    if any(kw in subject for kw in MED_KEYWORDS):
        return "label", "med"

    # Default: low
    return "label", "low"


def rule_draft(email: dict, label: str) -> Optional[str]:
    """Generate a template-based reply."""
    if label in ("spam", "escalate"):
        return None

    subject = email.get("subject", "").lower()
    sender = email.get("sender", "User")

    if label == "high":
        return (
            f"Thank you for reaching out. We have received your report and our team is "
            f"investigating immediately. We will update you within 1 hour."
        )
    elif label == "med":
        return (
            f"Thank you for contacting support. We have received your request and our "
            f"billing/support team will follow up within 24 hours."
        )
    else:
        return (
            f"Thank you for your feedback! We have noted your suggestion and will "
            f"consider it for a future update."
        )


def run_rule_agent(task: str) -> dict:
    """Run the deterministic rule-based agent on a task."""
    # Reset environment
    r = requests.post(f"{BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    obs = r.json()["observation"]

    total_reward = 0.0
    steps = 0

    while not obs["done"]:
        emails = [e for e in obs["current_emails"] if not e.get("labeled", False)]

        if not emails:
            break  # nothing left to process

        email = emails[0]  # Process one at a time
        action_type, label = rule_classify(email)

        # Build action JSON
        if action_type == "delete":
            action = {"type": "delete", "email_id": email["id"]}
        elif action_type == "escalate":
            action = {"type": "escalate", "email_id": email["id"]}
        else:
            action = {"type": "label", "email_id": email["id"], "value": label}

        step_r = requests.post(
            f"{BASE_URL}/step",
            json={"action": json.dumps(action)},
            timeout=30
        )
        step_r.raise_for_status()
        result = step_r.json()
        obs = result["observation"]
        total_reward += result["reward"]
        steps += 1

        # After labeling legit emails in hard task, also draft a reply
        if task == "hard" and action_type == "label" and label not in ("spam",):
            draft = rule_draft(email, label)
            if draft:
                draft_action = {"type": "draft", "email_id": email["id"], "value": draft}
                dr = requests.post(
                    f"{BASE_URL}/step",
                    json={"action": json.dumps(draft_action)},
                    timeout=30
                )
                if dr.ok:
                    result = dr.json()
                    obs = result["observation"]
                    total_reward += result["reward"]
                    steps += 1

        if steps >= 100:  # Safety limit
            break

    # Get final grade
    grade_r = requests.post(f"{BASE_URL}/grader", json={}, timeout=30)
    grade_r.raise_for_status()
    grade = grade_r.json()

    return {
        "task": task,
        "score": grade["score"],
        "breakdown": grade["breakdown"],
        "total_steps": steps,
        "total_reward": round(total_reward, 3),
    }


# ─────────────────── LLM Agent ────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent for a software company's support inbox.

Your job is to process each unread email by performing ONE action at a time.

Available actions (respond with ONLY valid JSON, no markdown):
1. Label an email: {"type": "label", "email_id": "<id>", "value": "<spam|low|med|high>"}
2. Delete spam:    {"type": "delete", "email_id": "<id>"}
3. Draft a reply:  {"type": "draft", "email_id": "<id>", "value": "<concise 1-2 sentence reply>"}
4. Escalate P0:   {"type": "escalate", "email_id": "<id>"}

Priority rules:
- ESCALATE: Production outages, data loss, security vulnerabilities, all-user lockouts
- HIGH: Crashes, login failures, payment failures, broken APIs
- MED: Billing questions, refunds, feature upgrades, compliance
- LOW: Feature requests, documentation, marketing, partnership
- SPAM: Promotional scams, phishing, get-rich-quick schemes

Process the MOST URGENT unread email first. Delete spam immediately.
"""


def run_llm_agent(task: str) -> dict:
    """Run GPT-4o-mini as the triage agent."""
    try:
        import openai
    except ImportError:
        return {"error": "openai not installed. pip install openai"}

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set"}

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    r = requests.post(f"{BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    obs = r.json()["observation"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    steps = 0

    while not obs["done"] and steps < 60:
        unread = [e for e in obs["current_emails"] if not e.get("labeled")]
        if not unread:
            break

        user_msg = (
            f"Current inbox state:\n{json.dumps(obs, indent=2)}\n\n"
            f"Unread emails to process: {len(unread)}\n"
            f"Pick ONE action for the highest priority unread email. Return JSON only."
        )
        messages.append({"role": "user", "content": user_msg})

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=150,
            )
            action_str = resp.choices[0].message.content.strip()
            # Strip markdown code blocks if present
            if action_str.startswith("```"):
                action_str = action_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            messages.append({"role": "assistant", "content": action_str})

        except Exception as e:
            print(f"  LLM error: {e}")
            break

        step_r = requests.post(
            f"{BASE_URL}/step",
            json={"action": action_str},
            timeout=30
        )
        if not step_r.ok:
            print(f"  Step error: {step_r.text}")
            break

        result = step_r.json()
        obs = result["observation"]
        total_reward += result["reward"]
        steps += 1

        time.sleep(0.2)  # Rate limit courtesy

    grade_r = requests.post(f"{BASE_URL}/grader", json={}, timeout=30)
    grade = grade_r.json()

    return {
        "task": task,
        "score": grade["score"],
        "breakdown": grade["breakdown"],
        "total_steps": steps,
        "total_reward": round(total_reward, 3),
    }


# ─────────────────── Main ─────────────────────────────────────

def print_results(label: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"  {label} Results")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Score':<8} {'Steps':<8} {'Reward'}")
    print(f"{'-'*45}")
    for r in results:
        if "error" in r:
            print(f"{r.get('task','?'):<10} ERROR: {r['error']}")
        else:
            print(f"{r['task']:<10} {r['score']:<8.3f} {r['total_steps']:<8} {r['total_reward']:.3f}")
            for k, v in r.get("breakdown", {}).items():
                print(f"  └─ {k}: {v:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="EmailTriage OpenEnv Baseline")
    parser.add_argument("--mode", choices=["rule", "llm", "both"], default="rule")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"✅ Server at {BASE_URL} is healthy")
    except Exception as e:
        print(f"❌ Server not reachable at {BASE_URL}: {e}")
        print("   Start it with: uvicorn server.app:app --port 7860")
        sys.exit(1)

    if args.mode in ("rule", "both"):
        print("\n🤖 Running Rule-Based Agent...")
        rule_results = []
        for task in args.tasks:
            print(f"  ▶ Task: {task}")
            result = run_rule_agent(task)
            rule_results.append(result)
            print(f"    Score: {result.get('score', 'ERR')}")
        print_results("Rule-Based Agent", rule_results)

    if args.mode in ("llm", "both"):
        print("\n🧠 Running LLM Agent (GPT-4o-mini)...")
        llm_results = []
        for task in args.tasks:
            print(f"  ▶ Task: {task}")
            result = run_llm_agent(task)
            llm_results.append(result)
            print(f"    Score: {result.get('score', 'ERR')}")
        print_results("LLM Agent (GPT-4o-mini)", llm_results)


if __name__ == "__main__":
    main()
