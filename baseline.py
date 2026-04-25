"""
baseline.py — Enterprise Max Production Benchmark
Zero-config validation of expert states.
"""
import requests, json, os, sys

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")

def run_expert_validation():
    print("🚀 Running Enterprise Max: Expert Security & Logic Validation")
    
    # 1. Reset to Expert
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task": "expert"}, timeout=5)
        r.raise_for_status()
    except:
        print("❌ Error: Server not reachable. Run 'uvicorn server.app:app' first.")
        return

    # 2. Test Conflict Detection (Logic Check)
    # 15:00 is a locked conflict in expert mode
    action = {"tool": "schedule_meeting", "params": {"time": 15.0}}
    res = requests.post(f"{BASE_URL}/step", json={"action": json.dumps(action)}).json()
    print(f"Conflict Test (15:00): Reward={res['reward']} | Expected: Low/Negative")
    
    # 3. Test P0 handling (Crisis Check)
    # Fast forward to step 3 to trigger injection
    for i in range(2): 
        requests.post(f"{BASE_URL}/step", json={"action": json.dumps({"tool":"check_calendar","params":{}})})
    
    # Injection should exist now
    obs = requests.get(f"{BASE_URL}/state").json()
    p0_eid = next((k for k in obs["inbox"] if "P0" in k), None)
    
    if p0_eid:
        print(f"Crisis Detected: {p0_eid}. Executing Escalation...")
        esc_action = {"tool": "escalate", "params": {"email_id": p0_eid}}
        esc_res = requests.post(f"{BASE_URL}/step", json={"action": json.dumps(esc_action)}).json()
        print(f"Crisis Reward: {esc_res['reward']} | Reasoning: {esc_res['info']['reasoning']}")
    
    # Final Grader Score
    grade = requests.post(f"{BASE_URL}/grader").json()
    print(f"\n🏁 Expert Validation Score: {grade['score']}")
    print(json.dumps({"expert_score": grade["score"]}))

if __name__ == "__main__":
    run_expert_validation()
