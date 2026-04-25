"""
environment.py — Core EmailTriage OpenEnv logic
Updated for Theme #3.1: Multi-step Enterprise Agent Simulator
"""
from __future__ import annotations

import json
import random
import time
from typing import Tuple, Dict, Any

from models import (
    Action, ActionType, Observation, State, Stats, ActionSummary, StepResult, GraderResult
)

MAX_STEPS = 5
RANDOM_SEED = 42

class EmailTriageEnv:
    def __init__(self):
        self.state = State()
        self.action_history: list[ActionSummary] = []
        self.current_task: str = "expert"

    def reset(self, task: str = "expert") -> Observation:
        random.seed(RANDOM_SEED)
        self.current_task = task
        
        self.state = State(
            inbox=[
                {
                    "id": "1", 
                    "sender": "boss@company.com", 
                    "subject": "Urgent Sync",
                    "body": "Need to meet at 15:00 today. Clear your calendar if needed.", 
                    "status": "unread"
                }
            ],
            calendar=[{"time": "15:00", "event": "Marketing Sync"}],
            task_board=[{"ticket": "T-101", "status": "open"}],
            step_count=0,
            task=task
        )
        self.action_history = []
        return self._get_obs()

    def _safe_normalize(self, score: float, epsilon: float = 0.01) -> float:
        try:
            v = float(score)
            if v != v: v = 0.5
        except Exception:
            v = 0.5
        clamped = max(epsilon, min(1.0 - epsilon, v))
        return round(clamped, 4)

    def step(self, action_json: str) -> StepResult:
        self.state.step_count += 1
        reward = 0.0
        reason = ""
        info = {}

        # 5. Introduce Dynamic Events
        if self.state.step_count == 2:
            self.state.inbox.append({
                "id": "999",
                "sender": "sysadmin@corp.com",
                "subject": "CRITICAL",
                "body": "Prod DB is down. Create a ticket immediately.",
                "status": "unread"
            })
            info["dynamic_event"] = "Injected an urgent email mid-workflow."

        try:
            action = Action.model_validate_json(action_json)
        except Exception as e:
            reward = -0.2
            reason = f"Failed to parse action JSON: {e}"
            info["error"] = "Invalid action JSON."
            return self._finalize_step(reward, reason, info, action_type="INVALID")

        at = action.type
        info["action_applied"] = at

        # 2. Transition to Tool-Calling Actions
        if at == ActionType.check_calendar:
            reward = 0.1
            reason = "Checked calendar successfully. See observation for availability."
            
        elif at == ActionType.schedule_meeting:
            time_val = action.time
            if not time_val:
                reward = -0.2
                reason = "Error: time argument missing."
            else:
                conflict = any(ev["time"] == time_val for ev in self.state.calendar)
                if conflict:
                    reward = -0.2
                    reason = "Error: Time unavailable due to conflict."
                    info["error"] = "Conflict detected."
                else:
                    self.state.calendar.append({"time": time_val, "event": "Scheduled Meeting"})
                    reward = 0.1
                    reason = f"Successfully scheduled meeting at {time_val}."
                    
        elif at == ActionType.reply_email:
            eid = action.email_id
            msg = action.message
            if not eid or not msg:
                reward = -0.2
                reason = "Error: missing id or message."
            else:
                # Find email and mark resolved
                found = False
                for email in self.state.inbox:
                    if email["id"] == eid:
                        email["status"] = "resolved"
                        found = True
                if not found:
                    reward = -0.2
                    reason = f"Error: Email {eid} not found."
                else:
                    reward = 0.1
                    reason = f"Replied to email {eid}."
                    
        elif at == ActionType.create_ticket:
            issue = action.issue
            if not issue:
                reward = -0.2
                reason = "Error: missing issue description."
            else:
                self.state.task_board.append({"ticket": f"T-10{len(self.state.task_board)+1}", "status": "open", "issue": issue})
                reward = 0.1
                reason = f"Ticket created for: {issue}."
                
        else:
            reward = -0.2
            reason = "Hallucinated or unknown tool call."

        return self._finalize_step(reward, reason, info, action_type=at)

    def _finalize_step(self, step_reward: float, reason: str, info: dict, action_type: str) -> StepResult:
        # Check task completion
        done, final_reason = self._is_task_complete()
        if self.state.step_count >= MAX_STEPS:
            done = True
            
        # Final Reward logic
        if done and final_reason:
            step_reward = max(step_reward, 0) + 0.8  # Enormous reward at completion
            reason = f"{reason} | {final_reason}"
        elif done and not final_reason:
            reason = f"{reason} | Episode ended, but task was incomplete."

        clamped_reward = self._safe_normalize(step_reward)

        self.action_history.append(ActionSummary(
            step=self.state.step_count,
            action_type=action_type,
            email_id="",
            reward=clamped_reward,
        ))

        obs = self._get_obs(done=done)
        info["step"] = self.state.step_count
        info["task"] = self.current_task

        return StepResult(
            observation=obs,
            reward=clamped_reward,
            done=done,
            info=info,
            reasoning=reason
        )

    def _is_task_complete(self) -> Tuple[bool, str]:
        # Boss email must be resolved, and urgent "Prod DB" ticket must be created (if injected)
        boss_resolved = any(e["id"] == "1" and e["status"] == "resolved" for e in self.state.inbox)
        critical_injected = any(e["id"] == "999" for e in self.state.inbox)
        critical_resolved = any(e["id"] == "999" and e["status"] == "resolved" for e in self.state.inbox)
        ticket_created = any("Prod DB" in str(t.get("issue","")) for t in self.state.task_board)

        # Has to have moved the conflicting meeting, or scheduled a new one at available time.
        # This is a bit flexible. Basically if they replied and created a ticket, they win.
        if boss_resolved:
            if critical_injected and not (critical_resolved or ticket_created):
                return False, ""
            return True, "Task Completed Successfully! (resolved boss email + urgent events)"
        return False, ""

    def _get_obs(self, done: bool = False) -> Observation:
        unread_count = sum(1 for e in self.state.inbox if e.get("status") == "unread")
        
        stats = Stats(
            total=len(self.state.inbox),
            unread=unread_count,
            labeled=0,
            deleted=0,
            escalated=0,
            drafts=0,
        )

        obs_emails = []
        for e in self.state.inbox:
            # We mock the EmailView for the observation
            from models import EmailView
            obs_emails.append(EmailView(
                id=e["id"],
                subject=e.get("subject", ""),
                body=e.get("body", ""),
                sender=e.get("sender", ""),
                labeled=(e.get("status") != "unread"),
                label=e.get("status")
            ))

        obs = Observation(
            current_emails=obs_emails,
            stats=stats,
            history=self.action_history[-10:],
            done=done,
            step=self.state.step_count,
        )
        
        # We also need to expose the calendar and task board to the agent observation 
        # (Since OpenEnv Observation is strict, we might inject it via current_emails or assume the agent can query it directly).
        # We will let check_calendar action handle calendar visibility by returning it in the step result reasoning, or extend Observation dynamically if needed.
        
        return obs

    def grader(self) -> GraderResult:
        done, msg = self._is_task_complete()
        score = 0.95 if done else 0.15
        
        breakdown = {
            "calendar_management": score,
            "urgent_interruption_handled": score if any("ticket" in t.get("issue","").lower() or "db down" in t.get("issue","").lower() for t in self.state.task_board) else 0.05,
            "boss_email_resolved": 0.95 if any(e["id"] == "1" and e["status"] == "resolved" for e in self.state.inbox) else 0.05,
            "inbox_cleared": score,
            "efficiency": 0.5
        }

        # Clamp all
        score = self._safe_normalize(score)
        for k, v in breakdown.items():
            breakdown[k] = self._safe_normalize(v)

        return GraderResult(
            score=score,
            breakdown=breakdown,
            details={"msg": msg}
        )

    def get_state(self) -> State:
        return self.state

    def get_tasks(self) -> list[dict]:
        return [{"id": "expert", "description": "Enterprise Agent Simulator", "difficulty": "expert"}]
