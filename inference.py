import os
import sys
import json
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------
API_BASE_URL = os.getenv( "API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv( "MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv( "HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv( "LOCAL_IMAGE_NAME")

# Environment Server URL (Current project's FastAPI app)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Initialize OpenAI Client (OpenAI-compatible endpoint)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "no-key"), 
    base_url=API_BASE_URL
)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert email triage agent. 
Process unread emails by performing ONE action at a time.
Respond ONLY with valid JSON.

Available actions:
1. Label: {"type": "label", "email_id": "<id>", "value": "<spam|low|med|high>"}
2. Delete: {"type": "delete", "email_id": "<id>"}
3. Draft: {"type": "draft", "email_id": "<id>", "value": "<reply text>"}
4. Escalate: {"type": "escalate", "email_id": "<id>"}
"""

def run_task(task_name: str):
    """Run a single triage task with structured logging."""
    print(f"[START] {task_name}")
    
    try:
        # Reset environment
        r = requests.post(f"{ENV_URL}/reset", json={"task": task_name}, timeout=10)
        r.raise_for_status()
        obs = r.json()["observation"]
        
        step_count = 0
        while not obs["done"] and step_count < 50:
            unread = [e for e in obs["current_emails"] if not e.get("labeled")]
            if not unread:
                break

            # Call LLM
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Inbox:\n{json.dumps(obs, indent=2)}\nAction?"}
            ]
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
            )
            
            action_str = completion.choices[0].message.content.strip()
            # Clean JSON if LLM included block identifiers
            if action_str.startswith("```"):
                action_str = action_str.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            print(f"[STEP] {action_str}")
            
            # Perform Step
            step_r = requests.post(
                f"{ENV_URL}/step", 
                json={"action": action_str}, 
                timeout=10
            )
            
            if not step_r.ok:
                print(f"[STEP] ERROR: {step_r.text}")
                break
                
            res = step_r.json()
            obs = res["observation"]
            step_count += 1
            
            time.sleep(0.5) # Courtesy delay

        # Final Grading
        grade_r = requests.post(f"{ENV_URL}/grader", json={}, timeout=10)
        grade = grade_r.json()
        print(f"[END] {task_name} Score: {grade['score']}")

    except Exception as e:
        print(f"[END] {task_name} FAILED: {str(e)}")

def main():
    # Attempt to run despite possible missing keys (for validation checks)
    for task in TASKS:
        run_task(task)

if __name__ == "__main__":
    main()
