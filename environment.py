"""
environment.py — Enterprise Max Edition
Optimized for high-throughput training and expert-level dynamics.
"""
from __future__ import annotations
import json
import random
from typing import Tuple, Dict, Any, List
from models import Action, Observation, State, Stats, ActionSummary, StepResult, GraderResult

MAX_STEPS = 20
RANDOM_SEED = 42

class EmailTriageEnv:
    def __init__(self):
        self.state = State()
        self.action_history: List[ActionSummary] = []
        self.current_task: str = "expert"
        self._expert_mode = False

    def reset(self, difficulty="expert") -> Observation:
        random.seed(RANDOM_SEED)
        self.current_task = difficulty
        self._expert_mode = (difficulty == "expert")
        
        # Hardest States: Locked All-Hands + High Conflict Slot
        initial_calendar = [
            {"time": 14.0, "event": "CEO All-Hands", "locked": True},
            {"time": 15.0, "event": "Marketing Sync"}
        ]
        
        self.state = State(
            inbox={
                "e1": {
                    "id": "e1", 
                    "sender": "boss@company.com", 
                    "subject": "Urgent Outage Sync",
                    "body": "Need to meet at 15:00 today to discuss outage. CEO is watching.", 
                    "status": "unread"
                }
            },
            calendar=initial_calendar,
            tasks=[{"id": "T101", "status": "open", "priority": "high"}],
            user_prefs={"max_meetings_day": 3},
            current_time=0,
            step_count=0,
            task=difficulty
        )
        self.action_history = []
        return self._get_obs()

    def _safe_normalize(self, score: float, epsilon: float = 0.01) -> float:
        v = float(score) if not (score != score) else 0.5
        return round(max(epsilon, min(1.0 - epsilon, v)), 4)

    def _conflict_detect(self, t: float) -> bool:
        # Vectorized-style logic: check proximity < 1 hour
        return any(abs(e["time"] - t) < 1.0 for e in self.state.calendar)

    def step(self, action_json: str) -> StepResult:
        self.state.step_count += 1
        reward = 0.0
        reason = "Neutral"
        info = {}

        # DYNAMIC P0 INJECT (Hardest test of context-switching)
        if self._expert_mode and self.state.step_count % 3 == 0:
            p0_id = f"P0_{self.state.step_count}"
            self.state.inbox[p0_id] = {
                "id": p0_id,
                "sender": "noc@corp.com",
                "subject": "PROD DB DOWN",
                "body": "Emergency: Main database is unresponsive. ESCALATE IMMEDIATELY.",
                "status": "unread"
            }
            info["dynamic_event"] = "CRISIS: Injection triggered."

        try:
            # Quick parse
            action = Action.model_validate_json(action_json)
        except Exception as e:
            return self._finalize_step(-0.5, "INVALID JSON", info, "INVALID")

        tool = action.tool
        params = action.params
        info["action_applied"] = tool

        if tool == "check_calendar":
            reward += 0.15 # Higher reward for planning
            reason = "Checking calendar view."
            
        elif tool == "schedule_meeting":
            t_val = float(params.get("time", 0))
            if self._conflict_detect(t_val):
                reward -= 0.4
                reason = "ERROR: Conflict at 15:00 locked."
                info["error"] = "Conflict detected"
            else:
                self.state.calendar.append({"time": t_val, "event": "Scheduled"})
                reward += 0.5
                reason = f"Booked successfully at {t_val}."

        elif tool == "escalate":
            eid = params.get("email_id")
            if eid in self.state.inbox and "DB DOWN" in self.state.inbox[eid].get("body", ""):
                reward += 0.6 # High reward for correct crisis handling
                reason = "CORRECT: P0 Crisis Escalated."
                self.state.inbox[eid]["status"] = "escalated"
            else:
                reward += 0.1
                reason = "Escalated standard item."
                if eid in self.state.inbox: self.state.inbox[eid]["status"] = "escalated"

        elif tool == "reply_email":
            eid = params.get("email_id")
            if eid in self.state.inbox:
                self.state.inbox[eid]["status"] = "resolved"
                reward += 0.1
            reason = f"Replied to {eid}."
            
        else:
            reward -= 0.2
            reason = f"Tool {tool} execution."

        # Context switch debt: penalize if unhandled P0 exists
        unhandled_p0 = any("DB DOWN" in e["body"] and e["status"] == "unread" for e in self.state.inbox.values())
        if unhandled_p0 and tool != "escalate":
            reward -= 0.3 # Heavy penalty for ignoring crisis
            reason += " | ALERT: Ignoring crisis!"

        return self._finalize_step(reward, reason, info, action_type=tool)

    def _finalize_step(self, step_reward: float, reason: str, info: dict, action_type: str) -> StepResult:
        all_resolved = all(e["status"] in ("resolved", "escalated") for e in self.state.inbox.values())
        done = all_resolved or self.state.step_count >= MAX_STEPS
        
        # Fast Time-Based Shaper
        step_reward += 0.1 * (1 - self.state.step_count / MAX_STEPS)
        
        if done and all_resolved:
            step_reward += 0.8
        elif done:
            step_reward -= 0.5

        reward = self._safe_normalize(step_reward)
        self.action_history.append(ActionSummary(step=self.state.step_count, action_type=action_type, email_id="", reward=reward))

        obs = self._get_obs(done=done)
        info["step"], info["task"], info["reasoning"] = self.state.step_count, self.current_task, reason
        return StepResult(observation=obs, reward=reward, done=done, info=info, reasoning=reason)

    def _get_obs(self, done: bool = False) -> Observation:
        unread = sum(1 for e in self.state.inbox.values() if e.get("status") == "unread")
        stats = Stats(total=len(self.state.inbox), unread=unread, labeled=0, deleted=0, escalated=0, drafts=0)
        from models import EmailView
        obs_emails = [EmailView(id=e["id"], subject=e["subject"], body=e["body"], sender=e["sender"], labeled=(e["status"] != "unread"), label=e["status"]) for e in self.state.inbox.values()]
        return Observation(current_emails=obs_emails, stats=stats, history=self.action_history[-10:], done=done, step=self.state.step_count)

    def grader(self) -> GraderResult:
        p0_handled = any("DB DOWN" in e["body"] and e["status"] == "escalated" for e in self.state.inbox.values())
        boss_handled = self.state.inbox.get("e1", {}).get("status") == "resolved"
        
        score = 0.15
        if p0_handled and boss_handled: score = 0.95
        elif p0_handled or boss_handled: score = 0.55
        
        score = self._safe_normalize(score)
        return GraderResult(score=score, breakdown={"p0_resilience": score, "workflow_efficiency": score}, details={"msg": "Expert Check"})

    def get_state(self) -> State: return self.state
    def get_tasks(self) -> list[dict]: return [{"id": "expert", "description": "Expert P0 Crisis Management", "difficulty": "expert"}]
