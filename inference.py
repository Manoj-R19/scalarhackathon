import os
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------
# Configuration & Mandatory Environment Variables
# ---------------------------------------------------------
# Configured for Hugging Face Inference API (OpenAI-compatible)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-9b-it")
HF_TOKEN = os.getenv("HF_TOKEN")

# Environment Server URL (Current project's FastAPI app)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

def get_client():
    # If HF_TOKEN is not provided, use a placeholder 'no-token' for local/build validation
    key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-token")
    return OpenAI(
        api_key=key,
        base_url=API_BASE_URL
    )

client = get_client()

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert email triage agent. 
Process unread emails by performing ONE action at a time.
Respond ONLY with valid JSON.

Available actions:
1. Label an email: {"type": "label", "email_id": "<id>", "value": "<spam|low|med|high|escalate>"}
2. Delete spam:    {"type": "delete", "email_id": "<id>"}
3. Draft a reply:  {"type": "draft", "email_id": "<id>", "value": "<concise reply text>"}
4. Escalate P0:   {"type": "escalate", "email_id": "<id>"}

Respond with the JSON object only."""

def run_task(task_name: str):
    """Run a single triage task with structured logging."""
    print(f"[START] {task_name}")
    
    try:
        # Reset environment
        r = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=15)
        r.raise_for_status()
        obs = r.json()["observation"]
        
        step_count = 0
        while not obs["done"] and step_count < 50:
            # Call LLM using OpenAI client as mandated
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current Inbox Observation:\n{json.dumps(obs, indent=2)}\n\nNext Action?"}
            ]
            
            completion = client.chat.completions.create(
                model=MODEL_NAME if MODEL_NAME else "gpt-4o-mini",
                messages=messages,
                temperature=0,
            )
            
            action_str = completion.choices[0].message.content.strip()
            
            # Clean JSON markdown if necessary
            if action_str.startswith("```"):
                action_str = action_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # Perform Step
            step_r = requests.post(
                f"{ENV_URL}/step", 
                json={"action": action_str}, 
                timeout=15
            )
            
            if not step_r.ok:
                # Log the error in step format for traceability
                error_log = {
                    "step": step_count,
                    "error": step_r.text,
                    "action": action_str
                }
                print(f"[STEP] {json.dumps(error_log)}")
                break
                
            res = step_r.json()
            
            # MANDATORY: Structured [STEP] stdout logs
            # Format: [STEP] {json_payload}
            step_log = {
                "step": step_count,
                "action": json.loads(action_str) if action_str else {},
                "reward": res.get("reward"),
                "done": res.get("done"),
                "observation_summary": {
                    "unread": res["observation"]["stats"]["unread"],
                    "labeled": res["observation"]["stats"]["labeled"]
                }
            }
            print(f"[STEP] {json.dumps(step_log)}")
            
            obs = res["observation"]
            step_count += 1
            
            if obs["done"]:
                break
                
            time.sleep(0.2)

        # Final Grading
        grade_r = requests.post(f"{ENV_URL}/grader", json={}, timeout=15)
        grade = grade_r.json()
        
        # MANDATORY: Structured [END] stdout log
        # Including score in the [END] block is often required by deep validators
        end_log = {
            "task": task_name,
            "score": grade.get("score", 0.01),
            "status": "success"
        }
        print(f"[END] {json.dumps(end_log)}")

    except Exception as e:
        # Final [END] tag must exist even on failure
        # Use a small non-zero score to avoid "out of range" errors
        error_end = {
            "task": task_name,
            "score": 0.01,
            "status": "error",
            "message": str(e)
        }
        print(f"[END] {json.dumps(error_end)}")        # print(f"Runtime Exception: {e}", file=sys.stderr)

def main():
    for task in TASKS:
        run_task(task)

if __name__ == "__main__":
    main()
