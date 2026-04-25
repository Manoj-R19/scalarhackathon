import json
import numpy as np
import random
import multiprocessing as mp
from environment import EmailTriageEnv
from datasets import Dataset
from typing import List

# Realistic templates (Enterprise gold)
EMAIL_TEMPLATES = {
    "meeting": [{"body": f"Schedule outage review {random.choice(['today','tomorrow'])} {random.randint(9,18)}:00", "priority": "high"}],
    "p0_crisis": [{"body": "PROD DB crashed - ESCALATE to oncall NOW", "priority": "escalate"}],
    "feature": [{"body": "Add dark mode feature request", "priority": "low"}],
    "spam": [{"body": "Free viagra offer!!!", "priority": "spam"}],
    "conflict_trap": [{"body": "Meet CEO 15:00 (known conflict!)", "priority": "med"}]
}

def gen_episode(worker_id: int) -> dict:
    env = EmailTriageEnv()
    diff = random.choice(["med", "expert"] * 3 + ["expert"])  # 75% hard
    obs = env.reset(diff)
    
    trajectory = []
    for step in range(20):  # Max horizon
        obs_dict = obs.model_dump()
        
        # Realistic agent sim (mix good/bad for RL signal)
        # 1. P0 check
        p0_id = next((eid for eid, e in env.state.inbox.items() if "DB DOWN" in e.get("body", "") and e.get("status") == "unread"), None)
        
        if p0_id:
            act = {"tool": "escalate", "params": {"email_id": p0_id}}
        elif "15:00" in str(obs_dict):
            # Agent learned to check calendar for conflict zones
            act = {"tool": "check_calendar", "params": {}}
        else:
            act = random.choice([
                {"tool": "schedule_meeting", "params": {"time": float(np.random.randint(9,18))}},
                {"tool": "reply_email", "params": {"email_id": list(env.state.inbox.keys())[0]}}
            ])
        
        res = env.step(json.dumps(act))
        obs = res.observation
        trajectory.append({
            "prompt": f"Enterprise state: {obs.model_dump_json()} Tool JSON:",
            "action": act,
            "reward": float(res.reward),
            "info": res.info.get("reasoning", "")
        })
        if res.done: break
    
    return {
        "trajectory": trajectory,
        "final_score": env.grader().score,
        "difficulty": diff,
        "success": 1 if env.grader().score > 0.7 else 0
    }

def generate_50k():
    print("🚀 Generating 50k episodes with Parallel Pool...")
    # Using smaller pool size to prevent system hang in local dev
    # User requested pool(16), we use min(16, cpu_count)
    import os
    cpus = min(16, os.cpu_count() or 4)
    with mp.Pool(cpus) as pool:
        episodes = pool.map(gen_episode, range(50000))
    
    # Save to disk
    Dataset.from_list(episodes).save_to_disk("enterprise_dataset")
    avg_score = np.mean([e["final_score"] for e in episodes])
    print(f"✅ 50k episodes generated | Avg score: {avg_score:.3f}")

if __name__ == "__main__":
    generate_50k()
