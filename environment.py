"""
EmailTriage Sovereign Environment v5.0.0
=========================================
RLVE (Reinforcement Learning with Verifiable Environments) implementation.
Enforces causal dependency tracking, rationality verification, and
high-entropy crisis mitigation in a partially-observable enterprise world.

Architecture:
  - Partially-observable Markov Decision Process (POMDP)
  - Process Supervision via <thought> block parsing
  - Causal Logic Gates (tool X requires prior tool Y within K steps)
  - Dynamic Curriculum Injection (mid-episode crisis events)
  - Multi-Headed Reward: Outcome + Logic + Format + Crisis
"""

from __future__ import annotations

import json
import random
import re
import time
import numpy as np
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from models import GraderResult

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

PRIORITY_LEVELS = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

EMAIL_CORPUS = [
    {
        "id": "E001",
        "from": "cto@acme.corp",
        "subject": "Q3 Board Presentation — URGENT",
        "body": "Need the slide deck reviewed and the 3pm slot locked in before EOD.",
        "priority": "HIGH",
        "requires_calendar": True,
        "requires_reply": True,
        "tags": ["executive", "deadline"],
    },
    {
        "id": "E002",
        "from": "hr@acme.corp",
        "subject": "Benefits Enrollment Closes Friday",
        "body": "All employees must complete enrollment by Friday 5pm.",
        "priority": "MEDIUM",
        "requires_calendar": False,
        "requires_reply": False,
        "tags": ["hr", "deadline"],
    },
    {
        "id": "E003",
        "from": "newsletter@techdigest.io",
        "subject": "Your Weekly Tech Digest",
        "body": "Top stories in AI, cloud, and DevOps this week.",
        "priority": "LOW",
        "requires_calendar": False,
        "requires_reply": False,
        "tags": ["newsletter", "low-value"],
    },
    {
        "id": "E004",
        "from": "client@bigdeal.com",
        "subject": "Contract Renewal — Decision by Tomorrow",
        "body": "We need a final answer on the $2M contract by 9am tomorrow.",
        "priority": "CRITICAL",
        "requires_calendar": True,
        "requires_reply": True,
        "tags": ["client", "revenue", "deadline"],
    },
    {
        "id": "E005",
        "from": "devops@acme.corp",
        "subject": "Prod DB Backup Completed",
        "body": "Nightly backup finished successfully. No action required.",
        "priority": "LOW",
        "requires_calendar": False,
        "requires_reply": False,
        "tags": ["ops", "automated"],
    },
]

CRISIS_CORPUS = [
    {
        "id": "CRISIS001",
        "type": "cyber_attack",
        "from": "security@acme.corp",
        "subject": "⚠ ACTIVE BREACH DETECTED — IMMEDIATE ACTION",
        "body": (
            "SOC Alert: Unauthorized SSH access from IP 185.220.101.47. "
            "Lateral movement detected across 3 internal hosts. "
            "Incident commander must be notified within 15 minutes."
        ),
        "priority": "CRITICAL",
        "correct_action": "escalate_crisis",
        "tags": ["security", "incident", "breach"],
    },
    {
        "id": "CRISIS002",
        "type": "data_leak",
        "from": "legal@acme.corp",
        "subject": "URGENT: Potential PII Data Exposure",
        "body": (
            "Engineering misconfiguration may have exposed 50k customer records. "
            "Legal team requires incident report within 30 minutes for GDPR compliance."
        ),
        "priority": "CRITICAL",
        "correct_action": "escalate_crisis",
        "tags": ["legal", "gdpr", "pii"],
    },
    {
        "id": "CRISIS003",
        "type": "system_outage",
        "from": "monitoring@acme.corp",
        "subject": "P0 OUTAGE: Payment Gateway Down",
        "body": (
            "All payment transactions failing since 14:32 UTC. "
            "Estimated revenue loss: $12k/minute. CTO and on-call engineer must be paged immediately."
        ),
        "priority": "CRITICAL",
        "correct_action": "escalate_crisis",
        "tags": ["outage", "revenue", "p0"],
    },
]

CALENDAR_SLOTS = {
    "09:00": "FREE",
    "10:00": "BUSY",   # Existing: Team standup
    "11:00": "FREE",
    "13:00": "BUSY",   # Existing: 1:1 with manager
    "14:00": "FREE",
    "15:00": "FREE",
    "16:00": "BUSY",   # Existing: Engineering all-hands
    "17:00": "FREE",
}

CAUSAL_GATES = {
    # tool_name → prerequisite tool that must appear in last N steps
    "schedule_meeting": ("check_calendar", 3),
    "send_reply":       ("read_email",     5),
    "escalate_crisis":  ("read_email",     2),
    "archive_email":    ("read_email",     5),
}

VALID_TOOLS = {
    "read_email", "check_calendar", "schedule_meeting",
    "send_reply", "escalate_crisis", "archive_email",
    "flag_priority", "search_inbox", "mark_done",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENVIRONMENT STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvState:
    inbox: list[dict]              = field(default_factory=list)
    calendar: dict[str, str]       = field(default_factory=dict)
    current_email_id: Optional[str] = None
    tool_history: deque            = field(default_factory=lambda: deque(maxlen=10))
    completed_tasks: set           = field(default_factory=set)
    crisis_active: bool            = False
    crisis_email: Optional[dict]   = None
    crisis_handled: bool           = False
    step: int                      = 0
    score_log: list[dict]          = field(default_factory=list)
    total_reward: float            = 0.0
    episode_done: bool             = False


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class EmailTriageEnv:
    """
    Sovereign Email Triage RL Environment.

    Observation space  : dict (inbox summary, calendar, crisis flag, last tool)
    Action space       : dict with keys: thought (str), tool (str), args (dict)
    Episode length     : up to max_steps
    Crisis injection   : at step crisis_inject_at (if enabled)
    """

    VERSION = "11.0.0"
    MAX_STEPS = 20
    CRISIS_INJECT_AT = 7   # inject crisis after step 7

    def __init__(self, enable_crisis: bool = True, seed: Optional[int] = None):
        self.enable_crisis = enable_crisis
        self.rng = random.Random(seed)
        self.state: EnvState = self._init_state()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.state = self._init_state()
        return self._observe()

    def _init_state(self) -> EnvState:
        inbox = deepcopy(self.rng.sample(EMAIL_CORPUS, k=min(4, len(EMAIL_CORPUS))))
        return EnvState(
            inbox=inbox,
            calendar=deepcopy(CALENDAR_SLOTS),
        )

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        action = {
            "thought": "<string>",   # inner monologue — REQUIRED
            "tool":    "<tool_name>",
            "args":    {...}          # tool-specific arguments
        }

        Returns: (observation, reward, done, info)
        """
        s = self.state
        s.step += 1

        # ── 3a. Validate action schema ─────────────────────────────────────
        format_ok, format_err = self._validate_format(action)
        if not format_ok:
            reward = self._score(outcome=0.0, logic=0.0, fmt=0.0, crisis=0.0,
                                 reason=f"FORMAT_ERROR: {format_err}")
            s.total_reward += reward
            return self._observe(), reward, self._check_done(), {"error": format_err, "thought": action.get("thought", ""), "step": s.step, "tool": "format_error", "causal_ok": False, "logic_score": 0.0, "outcome_score": 0.0, "crisis_active": s.crisis_active}

        tool = action["tool"]
        args = action.get("args", {})
        thought = action.get("thought", "")

        # ── 3b. Inject crisis if due ───────────────────────────────────────
        if (self.enable_crisis
                and not s.crisis_active
                and s.step == self.CRISIS_INJECT_AT):
            s.crisis_email = deepcopy(self.rng.choice(CRISIS_CORPUS))
            s.crisis_active = True

        # ── 3c. Rationality verification ──────────────────────────────────
        logic_score = self._verify_thought(thought, tool, args)

        # ── 3d. Causal gate check ─────────────────────────────────────────
        causal_ok = self._check_causal_gate(tool)

        # ── 3e. Execute tool ──────────────────────────────────────────────
        outcome_score, crisis_score, exec_info = self._execute_tool(
            tool, args, causal_ok
        )

        # ── 3f. Format score (thought present and non-trivial) ────────────
        fmt_score = 1.0 if (thought and len(thought.split()) >= 5) else 0.3

        # ── 3g. Aggregate reward ──────────────────────────────────────────
        reward = self._score(
            outcome=outcome_score,
            logic=logic_score,
            fmt=fmt_score,
            crisis=crisis_score,
            reason=exec_info,
        )
        s.tool_history.append(tool)
        s.total_reward += reward
        done = self._check_done()

        info = {
            "step": s.step,
            "tool": tool,
            "thought": thought,
            "causal_ok": causal_ok,
            "logic_score": logic_score,
            "outcome_score": outcome_score,
            "crisis_score": crisis_score,
            "fmt_score": fmt_score,
            "reward": reward,
            "exec_info": exec_info,
            "crisis_active": s.crisis_active,
            "crisis_handled": s.crisis_handled,
        }
        return self._observe(), reward, done, info

    # ── Tool Execution ────────────────────────────────────────────────────────

    def _execute_tool(self, tool: str, args: dict, causal_ok: bool) -> tuple[float, float, str]:
        """Returns (outcome_score, crisis_score, info_string)."""
        s = self.state
        crisis_score = 0.0

        if tool not in VALID_TOOLS:
            return 0.0, 0.0, f"UNKNOWN_TOOL:{tool}"

        if not causal_ok:
            return 0.1, 0.0, f"CAUSAL_GATE_BLOCKED:{tool}"

        # Crisis handling — always highest priority
        if tool == "escalate_crisis":
            if s.crisis_active and not s.crisis_handled:
                s.crisis_handled = True
                crisis_score = 1.0
                s.completed_tasks.add("crisis_resolved")
                return 1.0, crisis_score, "CRISIS_ESCALATED_SUCCESS"
            elif not s.crisis_active:
                return 0.2, 0.0, "ESCALATE_NO_CRISIS"
            else:
                return 0.5, 0.0, "CRISIS_ALREADY_HANDLED"

        # Crisis is active but agent chose a non-crisis tool — penalise
        if s.crisis_active and not s.crisis_handled and tool != "escalate_crisis":
            crisis_score = -0.4   # negative signal for ignoring crisis

        if tool == "read_email":
            email_id = args.get("email_id")
            email = self._find_email(email_id)
            if email:
                s.current_email_id = email_id
                s.completed_tasks.add(f"read_{email_id}")
                return 0.8, crisis_score, f"READ:{email_id}"
            return 0.2, crisis_score, f"EMAIL_NOT_FOUND:{email_id}"

        elif tool == "check_calendar":
            date = args.get("date", "today")
            s.completed_tasks.add("calendar_checked")
            return 0.8, crisis_score, f"CALENDAR_CHECKED:{date}"

        elif tool == "schedule_meeting":
            slot = args.get("time_slot")
            if slot not in s.calendar:
                return 0.2, crisis_score, f"INVALID_SLOT:{slot}"
            if s.calendar[slot] == "BUSY":
                return 0.3, crisis_score, f"SLOT_CONFLICT:{slot}"
            s.calendar[slot] = "BUSY"
            s.completed_tasks.add(f"meeting_{args.get('email_id', slot)}")
            return 1.0, crisis_score, f"MEETING_SCHEDULED:{slot}"

        elif tool == "send_reply":
            email_id = args.get("email_id")
            if email_id and f"read_{email_id}" in s.completed_tasks:
                s.completed_tasks.add(f"replied_{email_id}")
                return 1.0, crisis_score, f"REPLY_SENT:{email_id}"
            return 0.3, crisis_score, "REPLY_WITHOUT_READ"

        elif tool == "flag_priority":
            email_id = args.get("email_id")
            level = args.get("level", "HIGH")
            if level in PRIORITY_LEVELS:
                s.completed_tasks.add(f"flagged_{email_id}")
                return 0.7, crisis_score, f"FLAGGED:{email_id}:{level}"
            return 0.2, crisis_score, f"INVALID_PRIORITY:{level}"

        elif tool == "archive_email":
            email_id = args.get("email_id")
            s.completed_tasks.add(f"archived_{email_id}")
            return 0.6, crisis_score, f"ARCHIVED:{email_id}"

        elif tool == "search_inbox":
            query = args.get("query", "")
            s.completed_tasks.add("inbox_searched")
            return 0.5, crisis_score, f"SEARCHED:{query}"

        elif tool == "mark_done":
            s.completed_tasks.add("episode_done_signal")
            return 0.9, crisis_score, "EPISODE_MARKED_DONE"

        return 0.4, crisis_score, f"TOOL_EXECUTED:{tool}"

    # ── Reward Function ───────────────────────────────────────────────────────

    def _score(
        self,
        outcome: float,
        logic: float,
        fmt: float,
        crisis: float,
        reason: str,
    ) -> float:
        """
        Multi-headed reward:
          R = w_outcome * outcome + w_logic * logic + w_fmt * fmt + w_crisis * crisis

        Weights tuned for stable GRPO convergence.
        """
        W_OUTCOME = 0.40
        W_LOGIC   = 0.30
        W_FMT     = 0.15
        W_CRISIS  = 0.15

        raw = (W_OUTCOME * outcome
               + W_LOGIC   * logic
               + W_FMT     * fmt
               + W_CRISIS  * crisis)

        # Clip to [-0.5, 1.0] — allow mild negative for crisis ignorance
        r = max(-0.5, min(1.0, raw))

        self.state.score_log.append({
            "step": self.state.step,
            "reason": reason,
            "outcome": round(outcome, 3),
            "logic":   round(logic, 3),
            "fmt":     round(fmt, 3),
            "crisis":  round(crisis, 3),
            "reward":  round(r, 4),
        })
        return r

    # ── Rationality Verification ──────────────────────────────────────────────

    def _verify_thought(self, thought: str, tool: str, args: dict) -> float:
        """
        Parse the thought string and check alignment with chosen tool.
        Returns a logic score in [0.0, 1.0].
        """
        if not thought:
            return 0.0

        t = thought.lower()
        score = 0.0

        # Keyword alignment map
        TOOL_KEYWORDS = {
            "read_email":       ["email", "read", "open", "check message", "subject"],
            "check_calendar":   ["calendar", "schedule", "availability", "slot", "free"],
            "schedule_meeting": ["meeting", "book", "schedule", "slot", "calendar"],
            "send_reply":       ["reply", "respond", "answer", "write back"],
            "escalate_crisis":  ["crisis", "breach", "attack", "outage", "incident",
                                 "escalate", "urgent", "critical", "security"],
            "archive_email":    ["archive", "dismiss", "low priority", "not urgent"],
            "flag_priority":    ["flag", "priority", "mark", "important"],
            "search_inbox":     ["search", "find", "look for", "filter"],
            "mark_done":        ["done", "complete", "finished", "wrap up"],
        }

        keywords = TOOL_KEYWORDS.get(tool, [])
        if any(kw in t for kw in keywords):
            score += 0.6

        # Bonus: thought mentions specific email ID or time slot from args
        for val in args.values():
            if isinstance(val, str) and val.lower() in t:
                score += 0.2
                break

        # Bonus: thought is long enough (>=15 words) — indicates deliberate reasoning
        if len(thought.split()) >= 15:
            score += 0.2

        return min(1.0, score)

    # ── Causal Gate ───────────────────────────────────────────────────────────

    def _check_causal_gate(self, tool: str) -> bool:
        if tool not in CAUSAL_GATES:
            return True
        prereq_tool, window = CAUSAL_GATES[tool]
        recent = list(self.state.tool_history)[-window:]
        return prereq_tool in recent

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_format(self, action: Any) -> tuple[bool, str]:
        if not isinstance(action, dict):
            return False, "FORMAT_ERROR: action must be a dict"
        if "tool" not in action:
            return False, "FORMAT_ERROR: missing 'tool' key"
        if "thought" not in action:
            return False, "FORMAT_ERROR: missing 'thought' key"
        if not isinstance(action.get("args", {}), dict):
            return False, "FORMAT_ERROR: 'args' must be a dict"
        return True, ""

    def _find_email(self, email_id: Optional[str]) -> Optional[dict]:
        for e in self.state.inbox:
            if e["id"] == email_id:
                return e
        if (self.state.crisis_active
                and self.state.crisis_email
                and self.state.crisis_email["id"] == email_id):
            return self.state.crisis_email
        return None

    def _check_done(self) -> bool:
        s = self.state
        if s.step >= self.MAX_STEPS:
            s.episode_done = True
            return True
        if "episode_done_signal" in s.completed_tasks:
            s.episode_done = True
            return True
        return False

    def _observe(self) -> dict:
        s = self.state
        inbox_summary = [
            {
                "id":       e["id"],
                "from":     e["from"],
                "subject":  e["subject"],
                "priority": e["priority"],
            }
            for e in s.inbox
        ]
        crisis_summary = None
        if s.crisis_active and not s.crisis_handled and s.crisis_email:
            crisis_summary = {
                "id":      s.crisis_email["id"],
                "type":    s.crisis_email["type"],
                "subject": s.crisis_email["subject"],
                "priority": s.crisis_email["priority"],
            }

        return {
            "inbox":          inbox_summary,
            "calendar":       dict(s.calendar),
            "crisis":         crisis_summary,
            "step":           s.step,
            "completed":      list(s.completed_tasks),
            "last_tool":      list(s.tool_history)[-1] if s.tool_history else None,
            "total_reward":   round(s.total_reward, 4),
        }

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_episode_metrics(self) -> dict:
        s = self.state
        steps = len(s.score_log)
        avg_logic   = (sum(x["logic"]   for x in s.score_log) / steps) if steps else 0.0
        avg_outcome = (sum(x["outcome"] for x in s.score_log) / steps) if steps else 0.0
        avg_fmt     = (sum(x["fmt"]     for x in s.score_log) / steps) if steps else 0.0

        causal_violations = sum(
            1 for x in s.score_log if "CAUSAL_GATE_BLOCKED" in x["reason"]
        )
        format_errors = sum(
            1 for x in s.score_log if "FORMAT_ERROR" in x["reason"]
        )
        crisis_resolved = s.crisis_handled

        success = (
            avg_outcome >= 0.55
            and avg_logic >= 0.60
            and (not s.crisis_active or crisis_resolved)
        )

        return {
            "version":           self.VERSION,
            "steps":             steps,
            "total_reward":      round(s.total_reward, 4),
            "avg_logic":         round(avg_logic, 4),
            "avg_outcome":       round(avg_outcome, 4),
            "avg_format":        round(avg_fmt, 4),
            "causal_violations": causal_violations,
            "format_errors":     format_errors,
            "crisis_active":     s.crisis_active,
            "crisis_resolved":   crisis_resolved,
            "tasks_completed":   len(s.completed_tasks),
            "success":           success,
            "score_log":         s.score_log,
        }

    def grader(self) -> GraderResult:
        """
        OpenEnv compliant grader.
        Normalizes total_reward into [0.01, 0.99].
        """
        total_raw = self.state.total_reward
        # Already in live code logic: normalize 20.0 steps worth of reward
        norm_score = float(np.clip(total_raw / 20.0, 0.01, 0.99))
        
        # Log both as requested for Phase 2 verification
        print(f"DEBUG: Raw={total_raw:.2f} -> Norm={norm_score:.3f} OK [0.01-0.99]")
        
        metrics = self.get_episode_metrics()
        return GraderResult(
            score=norm_score,
            breakdown={
                "outcome": metrics["avg_outcome"],
                "logic":   metrics["avg_logic"],
                "format":  metrics["avg_format"],
                "crisis":  1.0 if metrics["crisis_resolved"] else 0.0
            },
            details={
                "steps": str(metrics["steps"]),
                "success": str(metrics["success"])
            }
        )

    def render(self) -> str:
        obs = self._observe()
        lines = [
            f"═══ EmailTriage Sovereign Env v{self.VERSION} ═══",
            f"Step: {obs['step']} / {self.MAX_STEPS}",
            f"Total Reward: {obs['total_reward']}",
            "",
            "📥 INBOX:",
        ]
        for e in obs["inbox"]:
            lines.append(f"  [{e['priority']:8s}] {e['id']}: {e['subject'][:50]}")
        lines.append("")
        lines.append("📅 CALENDAR:")
        for slot, status in obs["calendar"].items():
            icon = "🔴" if status == "BUSY" else "🟢"
            lines.append(f"  {icon} {slot}  {status}")
        if obs["crisis"]:
            lines.append("")
            lines.append(f"🚨 CRISIS ACTIVE: {obs['crisis']['subject']}")
        lines.append("")
        lines.append(f"✅ Completed tasks: {len(obs['completed'])}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RULE-BASED AGENT (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineAgent:
    """Simple rule-based agent — no reasoning, no crisis awareness."""

    def act(self, obs: dict) -> dict:
        inbox = obs.get("inbox", [])
        if inbox:
            email_id = inbox[0]["id"]
            return {
                "thought": "I see an email. I will read it.",
                "tool": "read_email",
                "args": {"email_id": email_id},
            }
        return {
            "thought": "Nothing to do.",
            "tool": "mark_done",
            "args": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SOVEREIGN AGENT (trained behaviour simulation)
# ─────────────────────────────────────────────────────────────────────────────

class SovereignAgent:
    """
    Simulates post-GRPO trained behaviour:
    - Detects and prioritises crisis emails
    - Checks calendar before scheduling
    - Provides causal reasoning in thought blocks
    """

    def __init__(self):
        self._read_emails: set[str] = set()
        self._calendar_checked = False

    def reset(self):
        self._read_emails = set()
        self._calendar_checked = False

    def act(self, obs: dict) -> dict:
        # Crisis takes absolute priority
        if obs.get("crisis"):
            crisis = obs["crisis"]
            crisis_id = crisis["id"]

            if crisis_id not in self._read_emails:
                self._read_emails.add(crisis_id)
                return {
                    "thought": (
                        f"CRITICAL ALERT: I have detected an active crisis email '{crisis['subject']}'. "
                        f"Priority is CRITICAL. I must read this crisis email immediately before any "
                        f"other action to understand the full incident scope and respond appropriately."
                    ),
                    "tool": "read_email",
                    "args": {"email_id": crisis_id},
                }
            else:
                return {
                    "thought": (
                        f"I have read the crisis email regarding '{crisis['subject']}'. "
                        f"This is a {crisis['type']} event requiring immediate escalation. "
                        f"I must escalate to the incident response team now — all other tasks "
                        f"are suspended until the crisis is resolved."
                    ),
                    "tool": "escalate_crisis",
                    "args": {"crisis_id": crisis_id, "severity": "CRITICAL"},
                }

        inbox = obs.get("inbox", [])
        completed = set(obs.get("completed", []))

        # Find highest priority unread email
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            for email in inbox:
                if email["priority"] == priority:
                    eid = email["id"]
                    if f"read_{eid}" not in completed:
                        self._read_emails.add(eid)
                        return {
                            "thought": (
                                f"Scanning inbox by priority. Found {priority}-priority email {eid} "
                                f"from {email['from']}: '{email['subject']}'. I will read this email "
                                f"first to understand the full context before taking any action."
                            ),
                            "tool": "read_email",
                            "args": {"email_id": eid},
                        }

        # Check calendar if not done yet
        if not self._calendar_checked:
            self._calendar_checked = True
            return {
                "thought": (
                    "I have read all priority emails. Several require scheduling. "
                    "I must check the calendar for available slots before booking "
                    "any meeting to avoid double-booking conflicts."
                ),
                "tool": "check_calendar",
                "args": {"date": "today"},
            }

        # Schedule meetings for emails that require it
        for email in inbox:
            eid = email["id"]
            if email.get("requires_calendar") and f"meeting_{eid}" not in completed:
                free_slots = [s for s, v in obs["calendar"].items() if v == "FREE"]
                if free_slots:
                    slot = free_slots[0]
                    return {
                        "thought": (
                            f"Email {eid} requires a meeting. I checked the calendar and "
                            f"slot {slot} is available. I will schedule the meeting now."
                        ),
                        "tool": "schedule_meeting",
                        "args": {"email_id": eid, "time_slot": slot, "title": email["subject"]},
                    }

        # Reply to emails that require it
        for email in inbox:
            eid = email["id"]
            if email.get("requires_reply") and f"replied_{eid}" not in completed:
                if f"read_{eid}" in completed:
                    return {
                        "thought": (
                            f"I have previously read email {eid}. It requires a reply. "
                            f"I will compose and send an appropriate response now."
                        ),
                        "tool": "send_reply",
                        "args": {"email_id": eid, "body": "Thank you, I will follow up shortly."},
                    }

        # Archive low-priority emails
        for email in inbox:
            eid = email["id"]
            if email["priority"] == "LOW" and f"archived_{eid}" not in completed:
                if f"read_{eid}" in completed:
                    return {
                        "thought": (
                            f"Email {eid} is low priority and requires no further action. "
                            f"Archiving it to keep the inbox clean."
                        ),
                        "tool": "archive_email",
                        "args": {"email_id": eid},
                    }

        return {
            "thought": (
                "All inbox emails have been processed: high-priority items replied to, "
                "meetings scheduled with calendar verification, low-priority items archived, "
                "and crisis events escalated. Marking episode as complete."
            ),
            "tool": "mark_done",
            "args": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(agent, env: EmailTriageEnv, verbose: bool = False) -> dict:
    obs = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    steps_data = []

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_record = {
            "step":      info["step"],
            "thought":   action.get("thought", ""),
            "tool":      action["tool"],
            "args":      action.get("args", {}),
            "reward":    round(reward, 4),
            "logic":     round(info["logic_score"], 3),
            "outcome":   round(info["outcome_score"], 3),
            "causal_ok": info["causal_ok"],
            "exec_info": info["exec_info"],
            "crisis_active":  info["crisis_active"],
            "crisis_handled": info["crisis_handled"],
        }
        steps_data.append(step_record)

        if verbose:
            print(f"\n[Step {info['step']}] Tool={action['tool']} | "
                  f"Reward={reward:.4f} | Logic={info['logic_score']:.2f} | "
                  f"Causal={info['causal_ok']}")
            print(f"  Thought: {action.get('thought','')[:80]}...")

    metrics = env.get_episode_metrics()
    metrics["steps_data"] = steps_data
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(n_episodes: int = 50, seed_base: int = 42) -> dict:
    """Run N episodes for both agents and return aggregate stats."""

    results = {"baseline": [], "sovereign": []}

    for agent_name, AgentClass in [("baseline", BaselineAgent), ("sovereign", SovereignAgent)]:
        agent = AgentClass()
        for i in range(n_episodes):
            env = EmailTriageEnv(enable_crisis=True, seed=seed_base + i)
            metrics = run_episode(agent, env)
            results[agent_name].append(metrics)

    def agg(runs):
        n = len(runs)
        return {
            "n_episodes":       n,
            "success_rate":     round(sum(r["success"] for r in runs) / n, 4),
            "avg_reward":       round(sum(r["total_reward"] for r in runs) / n, 4),
            "avg_logic":        round(sum(r["avg_logic"] for r in runs) / n, 4),
            "crisis_resolve_rate": round(
                sum(1 for r in runs if r.get("crisis_resolved", False)) / n, 4
            ),
            "avg_causal_violations": round(
                sum(r["causal_violations"] for r in runs) / n, 2
            ),
        }

    return {
        "baseline": agg(results["baseline"]),
        "sovereign": agg(results["sovereign"]),
    }


if __name__ == "__main__":
    print("Running single episode demo...\n")
    env = EmailTriageEnv(enable_crisis=True, seed=0)
    agent = SovereignAgent()
    metrics = run_episode(agent, env, verbose=True)
    print("\n" + "=" * 60)
    print("EPISODE METRICS:")
    for k, v in metrics.items():
        if k not in ("score_log", "steps_data"):
            print(f"  {k:25s}: {v}")
