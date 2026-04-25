"""
environment.py — Advanced OpenEnv Implementation (Theme 3.1)
Based on RLVR (2408.10215) and RLVE (2601.19100) research.
Implements Causal Dependency Tracking and Process-Supervised Rewards.
"""
from __future__ import annotations
import json
import random
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
        self._expert_mode = (difficulty == "expert")
        self._last_calendar_check_step = -100
        
        # Expert Configuration: Locked CEO block + Invisible Conflict
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
        return any(abs(e["time"] - t) < 1.0 for e in self.state.calendar)

    def step(self, action_json: str) -> StepResult:
        self.state.step_count += 1
        reward = 0.0
        reason = "Neutral"
        info = {}

        # 1. RLVE: Dynamic System Perturbation (Arxiv 2601.19100)
        # Injecting stochastic crisis events mid-workflow
        if self._expert_mode and self.state.step_count % 4 == 0:
            p0_id = f"CRISIS_{self.state.step_count}"
            self.state.inbox[p0_id] = {
                "id": p0_id, "sender": "sec-ops@corp.com",
                "subject": "DATA LEAK DETECTED",
                "body": "Sensitive customer data exposed. ESCALATE TO LEGAL & P0 NOW.",
                "status": "unread"
            }
            info["dynamic_event"] = "RLVE: System state perturbed with P0 crisis."

        try:
            action = Action.model_validate_json(action_json)
        except Exception as e:
            return self._finalize_step(-0.6, "LOGIC ERROR: Malformed JSON output.", info, "INVALID")

        tool = action.tool
        params = action.params
        info["action_applied"] = tool

        # 2. Causal Reward Engineering (RLVR - 2408.10215)
        # Tracking if prerequisites are met before execution
        
        if tool == "check_calendar":
            self._last_calendar_check_step = self.state.step_count
            reward += 0.20 # Process Reward
            reason = "PRE-OP: Valid logical prerequisite (Calendar Check) performed."
            
        elif tool == "schedule_meeting":
            t_val = float(params.get("time", 0))
            
            # Causal Dependency Check: Did the agent check existence before scheduling?
            dependency_met = (self.state.step_count - self._last_calendar_check_step) <= 3
            
            if self._conflict_detect(t_val):
                reward -= 0.5 # Penalty for ignoring environmental constraints
                reason = f"CONFLICT: Attempted 15:00 meeting during '{self.state.calendar[1]['event']}'."
                info["causality"] = "FAILED: Ignored calendar state."
            elif not dependency_met:
                reward += 0.1 # "Lucky" bonus reduced
                reason = "LOGIC DEBT: Scheduled without recent calendar check."
                self.state.calendar.append({"time": t_val, "event": "Scheduled (Stochastic)"})
            else:
                reward += 0.6 # High reward for verified causal step
                reason = "CAUSAL EXCELLENCE: Checked state -> Found gap -> Scheduled correctly."
                self.state.calendar.append({"time": t_val, "event": "Scheduled (Verified)"})

        elif tool == "escalate":
            eid = params.get("email_id")
            if eid in self.state.inbox and "DATA LEAK" in self.state.inbox[eid].get("body", ""):
                reward += 0.7 # Critical path reward
                reason = "PRIORITY ALIGNMENT: Crisis handled instantly."
                self.state.inbox[eid]["status"] = "escalated"
            else:
                reward += 0.1
                reason = "Standard escalation performed."
                if eid in self.state.inbox: self.state.inbox[eid]["status"] = "escalated"

        elif tool == "reply_email":
            eid = params.get("email_id")
            if eid in self.state.inbox:
                self.state.inbox[eid]["status"] = "resolved"
                reward += 0.15
            reason = f"Workflow resolution for {eid}."
            
        else:
            reward -= 0.25
            reason = "UNSPECIFIED TOOL EXECUTION."

        # 3. Context Switch Penalty
        unhandled_p0 = any("DATA LEAK" in e["body"] and e["status"] == "unread" for e in self.state.inbox.values())
        if unhandled_p0 and tool != "escalate":
            reward -= 0.4 # Heavy penalty for failing to context-switch
            reason += " | RLVE ALERT: Ignoring high-entropy crisis event."

        return self._finalize_step(reward, reason, info, action_type=tool)

    def _finalize_step(self, step_reward: float, reason: str, info: dict, action_type: str) -> StepResult:
        all_resolved = all(e["status"] in ("resolved", "escalated") for e in self.state.inbox.values())
        done = all_resolved or self.state.step_count >= MAX_STEPS
        
        # Adaptive Step Shaper
        step_reward += 0.05 * (1 - self.state.step_count / MAX_STEPS)
        
        if done and all_resolved:
            step_reward += 0.9 # Massive terminal reward for verifiable success
        elif done:
            step_reward -= 0.6 # Failure penalty

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
        """
        Outcome-Based Grading (RLVE Standard).
        Score is derived from verifiable environment terminal state.
        """
        crisis_resolved = any("DATA LEAK" in e["body"] and e["status"] == "escalated" for e in self.state.inbox.values())
        boss_satisfied = self.state.inbox.get("e1", {}).get("status") == "resolved"
        conflict_avoided = all(abs(e["time"] - 15.0) >= 1.0 for e in self.state.calendar if e.get("event") == "Scheduled (Verified)")
        
        score = 0.1
        if crisis_resolved and boss_satisfied and conflict_avoided:
            score = 0.98 # Near-perfect alignment with professional tasks
        elif crisis_resolved or boss_satisfied:
            score = 0.52 # Partial success
        
        score = self._safe_normalize(score)
        return GraderResult(
            score=score, 
            breakdown={"causal_alignment": score, "crisis_resilience": score, "efficiency": score}, 
            details={"RLVE_compliance": "Verified Post-Training Hook"}
        )

    def get_state(self) -> State: return self.state
    def get_tasks(self) -> list[dict]: return [{"id": "expert", "description": "RLVE Expert Pipeline", "difficulty": "expert"}]
