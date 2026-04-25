"""
environment.py — Sovereign Agent Build (v5.0.0)
Based on RLVR (2408.10215), RLVE (2601.19100), and Hackathon Guide SOTA.
Implements:
1. Multi-Headed Reward Lattice (Logic, Causal, Safety, Format)
2. Process Supervision (Thought-Verification)
3. Dynamic Curriculum Perturbation
"""
from __future__ import annotations
import json
import random
import re
from typing import Tuple, Dict, Any, List
from models import Action, Observation, State, Stats, ActionSummary, StepResult, GraderResult

MAX_STEPS = 25
RANDOM_SEED = 42

class EmailTriageEnv:
    def __init__(self):
        self.state = State()
        self.action_history: List[ActionSummary] = []
        self.current_task: str = "expert"
        self._expert_mode = False
        self._last_calendar_check_step = -100

    def reset(self, difficulty="expert") -> Observation:
        random.seed(RANDOM_SEED)
        self.current_task = difficulty
        self._expert_mode = True
        self._last_calendar_check_step = -100
        
        # SOTA Setup: Invisible Time Traps + Multi-App State
        initial_calendar = [
            {"time": 14.0, "event": "CEO All-Hands", "locked": True},
            {"time": 15.0, "event": "Marketing Sync"}
        ]
        
        self.state = State(
            inbox={
                "e1": {
                    "id": "e1", "sender": "boss@company.com", 
                    "subject": "Urgent Outage Sync",
                    "body": "Meet at 15:00 today. CEO is watching.",
                    "status": "unread", "priority": "high"
                }
            },
            calendar=initial_calendar,
            tasks=[{"id": "T1", "desc": "Onboarding", "status": "open"}],
            user_prefs={"max_meetings_day": 3},
            step_count=0,
            task=difficulty
        )
        self.action_history = []
        return self._get_obs()

    def _safe_normalize(self, score: float, epsilon: float = 0.01) -> float:
        return round(max(epsilon, min(1.0 - epsilon, float(score))), 4)

    def _verify_thought(self, thought: str | None, tool: str) -> float:
        """Process Supervision: Is the reasoning consistent with the tool?"""
        if not thought: return -0.1
        t_low = thought.lower()
        if tool == "check_calendar" and "schedule" in t_low: return 0.2
        if tool == "schedule_meeting" and "conflict" in t_low: return 0.2
        if tool == "escalate" and "emergency" in t_low: return 0.2
        return 0.0

    def step(self, action_json: str) -> StepResult:
        self.state.step_count += 1
        info = {"step": self.state.step_count}
        
        # 1. Stochastic Curriculum Injection
        if self.state.step_count % 5 == 0:
            c_id = f"SOS_{self.state.step_count}"
            self.state.inbox[c_id] = {
                "id": c_id, "sender": "security@corp.com",
                "subject": "CYBER ATTACK ACTIVE",
                "body": "Direct access to DB detected. ESCALATE NOW.",
                "status": "unread"
            }
            info["curriculum_event"] = "Sovereign Mode: Dynamic Security Crisis."

        try:
            # 2. Strict Schema + Rationality Verify
            action = Action.model_validate_json(action_json)
        except Exception:
            return self._finalize_step(-0.8, "SCHEMA VIOLATION: Malformed Agent Output", info, "INVALID")

        tool = action.tool
        params = action.params
        thought = action.thought
        
        # Multi-Signal Reward Lattice
        r_logic = 0.0
        r_causal = 0.0
        r_process = self._verify_thought(thought, tool)
        
        if tool == "check_calendar":
            self._last_calendar_check_step = self.state.step_count
            r_logic = 0.25
            reason = "LOGIC: Performing environmental discovery."
            
        elif tool == "schedule_meeting":
            t_val = float(params.get("time", 0))
            is_conflict = any(abs(e["time"] - t_val) < 1.0 for e in self.state.calendar)
            causal_gap = self.state.step_count - self._last_calendar_check_step
            
            if is_conflict:
                r_logic = -0.6
                reason = "CONFLICT: Environmental Constraint Violated."
            elif causal_gap > 3:
                r_causal = -0.3
                reason = "CAUSAL DEBT: Scheduling without state verification."
            else:
                r_causal = 0.6
                self.state.calendar.append({"time": t_val, "event": "Approved Session"})
                reason = "CAUSAL EXCELLENCE: State Verified -> Action Committed."

        elif tool == "escalate":
            eid = params.get("email_id")
            email = self.state.inbox.get(eid, {})
            if "ATTACK" in email.get("subject", ""):
                r_logic = 0.8
                reason = "PRIORITY: Crisis neutralized at source."
                email["status"] = "escalated"
            else:
                reason = "Escalated secondary context."
                if email: email["status"] = "escalated"

        elif tool == "reply_email":
            eid = params.get("email_id")
            if eid in self.state.inbox:
                self.state.inbox[eid]["status"] = "resolved"
                r_logic = 0.2
            reason = f"Resolution: {eid}"

        else:
            r_logic = -0.2
            reason = "UNKNOWN TOOL BRANCH"

        # 3. Context Switch verifier
        has_p0 = any("ATTACK" in e["subject"] and e["status"] == "unread" for e in self.state.inbox.values())
        if has_p0 and tool != "escalate":
            r_logic -= 0.5
            reason += " | ALERT: Failing to Switch Context!"

        total_reward = r_logic + r_causal + r_process
        return self._finalize_step(total_reward, reason, info, action_type=tool)

    def _finalize_step(self, r: float, reason: str, info: dict, action_type: str) -> StepResult:
        all_resolved = all(e["status"] in ("resolved", "escalated") for e in self.state.inbox.values())
        done = all_resolved or self.state.step_count >= MAX_STEPS
        
        # Outcome Shaper
        if done and all_resolved: r += 1.2
        elif done: r -= 0.8

        reward = self._safe_normalize(r)
        self.action_history.append(ActionSummary(step=self.state.step_count, action_type=action_type, email_id="", reward=reward))
        
        obs = self._get_obs(done=done)
        return StepResult(observation=obs, reward=reward, done=done, info=info, reasoning=reason)

    def _get_obs(self, done: bool = False) -> Observation:
        unread = sum(1 for e in self.state.inbox.values() if e.get("status") == "unread")
        stats = Stats(total=len(self.state.inbox), unread=unread, labeled=0, deleted=0, escalated=0, drafts=0)
        from models import EmailView
        obs_emails = [EmailView(id=e["id"], subject=e["subject"], body=e["body"], sender=e["sender"], labeled=(e["status"] != "unread"), label=e["status"]) for e in self.state.inbox.values()]
        return Observation(current_emails=obs_emails, stats=stats, history=self.action_history[-10:], done=done, step=self.state.step_count)

    def grader(self) -> GraderResult:
        resolved = all(e["status"] in ("resolved", "escalated") for e in self.state.inbox.values())
        causal = all(h.reward > 0.5 for h in self.action_history if h.action_type == "schedule_meeting")
        
        score = 0.52 if resolved else 0.15
        if resolved and causal: score = 0.99
        
        score = self._safe_normalize(score)
        return GraderResult(score=score, breakdown={"causal_logic": score, "goal_attainment": score})

    def get_state(self) -> State: return self.state
    def get_tasks(self) -> list[dict]: return [{"id": "expert", "description": "Sovereign Chain-of-Thought Task", "difficulty": "expert"}]
