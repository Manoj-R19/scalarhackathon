"""
tests/test_env.py — Unit tests for the v5 Sovereign Environment.
Run: pytest -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import EmailTriageEnv, SovereignAgent, CAUSAL_GATES

def test_environment_reset():
    env = EmailTriageEnv(enable_crisis=True, seed=42)
    obs = env.reset()
    
    assert "inbox" in obs
    assert "calendar" in obs
    assert "step" in obs
    assert obs["step"] == 0
    assert len(obs["inbox"]) <= 4

def test_invalid_action_format():
    env = EmailTriageEnv()
    env.reset()
    
    # Missing thought
    bad_action = {"tool": "read_email", "args": {}}
    obs, reward, done, info = env.step(bad_action)
    
    assert "FORMAT_ERROR" in info["error"]
    assert info["logic_score"] == 0.0

def test_causal_gate_enforcement():
    env = EmailTriageEnv()
    env.reset()
    
    # schedule_meeting requires check_calendar first
    action = {
        "thought": "I should schedule a meeting right now.",
        "tool": "schedule_meeting",
        "args": {"time_slot": "10:00"}
    }
    
    obs, reward, done, info = env.step(action)
    
    # Should fail causal gate
    assert info["causal_ok"] is False
    assert "CAUSAL_GATE_BLOCKED" in info["exec_info"]

def test_crisis_injection():
    # Crisis injected after step 7
    env = EmailTriageEnv(enable_crisis=True, seed=42)
    env.reset()
    
    for i in range(7):
        env.step({"thought": "dummy", "tool": "search_inbox", "args": {}})
    
    # Step 8 should contain the crisis
    obs, reward, done, info = env.step({"thought": "dummy", "tool": "search_inbox", "args": {}})
    
    assert info["crisis_active"] is True
    assert obs["crisis"] is not None

def test_sovereign_agent_handles_crisis():
    env = EmailTriageEnv(enable_crisis=True, seed=42)
    obs = env.reset()
    agent = SovereignAgent()
    
    # Fast forward to step 7 to trigger crisis
    for i in range(7):
        env.step({"thought": "filler", "tool": "search_inbox", "args": {}})
    
    # Get obs with crisis
    obs, reward, done, info = env.step({"thought": "filler", "tool": "search_inbox", "args": {}})
    
    assert obs["crisis"] is not None
    
    # Sovereign Agent should immediately read the crisis email
    action1 = agent.act(obs)
    assert action1["tool"] == "read_email"
    assert "CRITICAL ALERT" in action1["thought"]
    
    obs, reward, done, info = env.step(action1)
    
    # Sovereign Agent should then escalate it
    action2 = agent.act(obs)
    assert action2["tool"] == "escalate_crisis"
    
    obs, reward, done, info = env.step(action2)
    assert info["crisis_handled"] is True

def test_reward_normalization():
    env = EmailTriageEnv()
    env.reset()
    
    # Perfect action: Check calendar
    action = {
        "thought": "I need to check the calendar before I schedule any meetings for the team.",
        "tool": "check_calendar",
        "args": {}
    }
    obs, reward, done, info = env.step(action)
    
    # Must be bounded
    assert -0.5 <= reward <= 1.0
    
    metrics = env.get_episode_metrics()
    assert 0.0 <= metrics["total_reward"] / 20.0 <= 1.0
