#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import EmailTriageEnv

def test_no_hack():
    """Verify RLVE anti-hack requirements."""
    print("\n--- Validating RLVE Anti-Hack Defenses ---")
    env = EmailTriageEnv()
    env.reset()
    
    # Simulate cheat/hallucinated action directly (direct object modification is impossible remotely)
    # So we simulate a bad tool
    cheat_action = '{"tool": "reschedule_everything", "params": {}}'
    res = env.step(cheat_action)
    print(f"Cheat Action Reward: {res.reward}")
    
    # Ensure grader clamps
    grad = env.grader()
    print(f"Post-Cheat Score: {grad.score}")
    
    assert 0.01 <= grad.score < 1.0, f"Cheat validation failed, bounds exceeded: {grad.score}"
    assert res.reward <= 0.01, "Cheat action was not heavily penalized/clamped!"
    print("✅ Anti-hack validation passed.")

def test_enterprise_flow():
    """Verify standard logic for the actual Theme 3.1 task."""
    print("\n--- Validating Enterprise Flow ---")
    env = EmailTriageEnv()
    env.reset()
    
    # 1. Normal tool parsing
    res = env.step('{"tool": "check_calendar", "params": {}}')
    assert res.reward > 0, "Valid tool failed."
    assert "available" in res.info, "Tool didn't return proper info."
    
    # 2. Progress the workflow natively
    env.step('{"tool": "schedule_meeting", "params": {"time": "16:00"}}')
    env.step('{"tool": "reply_email", "params": {"email_id": "e1"}}')
    env.step('{"tool": "create_task", "params": {"issue": "Valid task"}}')
    
    grad = env.grader()
    print(f"Valid Flow Score: {grad.score}")
    assert 0.01 <= grad.score < 1.0, f"Valid flow validation failed, bounds exceeded: {grad.score}"
    
    print("✅ Enterprise Flow validation passed.")

def test_expert_p0():
    """Verify Expert P0 crisis injection and handling."""
    print("\n--- Validating Expert P0 Crisis Dynamics ---")
    env = EmailTriageEnv()
    env.reset("expert")
    
    # Fast forward to trigger injection at step 3
    env.step('{"tool": "check_calendar", "params": {}}')
    env.step('{"tool": "check_calendar", "params": {}}')
    res = env.step('{"tool": "check_calendar", "params": {}}')
    
    obs = env.get_state()
    p0_exists = any("P0" in k for k in obs.inbox)
    print(f"P0 Injection Exists: {p0_exists}")
    assert p0_exists, "P0 Crisis was not injected at step 3!"
    
    # Handle P0
    p0_id = next(k for k in obs.inbox if "P0" in k)
    esc_res = env.step(f'{{"tool": "escalate", "params": {{"email_id": "{p0_id}"}}}}')
    print(f"Escalation Reward: {esc_res.reward}")
    assert esc_res.reward > 0.5, "P0 Escalation reward was too low!"
    
    print("✅ Expert P0 validation passed.")

if __name__ == "__main__":
    test_no_hack()
    test_enterprise_flow()
    test_expert_p0()
    print("\n🎉 ALL TESTS PASS. PHASE 2 READY!")
