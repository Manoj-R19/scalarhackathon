"""
models.py — Pydantic schemas for EmailTriage OpenEnv
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union, Any
from enum import Enum


# ─────────────────────────── Enums ────────────────────────────

class Label(str, Enum):
    spam    = "spam"
    low     = "low"
    med     = "med"
    high    = "high"
    escalate = "escalate"


class ActionType(str, Enum):
    check_calendar = "check_calendar"
    schedule_meeting = "schedule_meeting"
    create_task = "create_task"
    reply_email = "reply_email"
    escalate = "escalate"


# ─────────────────────────── Core Data ────────────────────────

class Email(BaseModel):
    """Visible email fields (priority/ground_truth hidden from agent)."""
    id: str
    subject: str
    body: str
    sender: str

    model_config = {"frozen": True}


class EmailView(BaseModel):
    """What the agent actually sees per email."""
    id: str
    subject: str
    body: str
    sender: str
    labeled: bool = False
    label: Optional[str] = None


# ─────────────────────────── Action ───────────────────────────

class Action(BaseModel):
    tool: Literal["check_calendar", "schedule_meeting", "create_task", "reply_email", "escalate"]
    params: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────── Observation ──────────────────────

class Stats(BaseModel):
    total: int
    unread: int
    labeled: int
    deleted: int
    escalated: int
    drafts: int


class ActionSummary(BaseModel):
    step: int
    action_type: str
    email_id: str
    reward: float


class Observation(BaseModel):
    """Full observation returned to agent after each step."""
    current_emails: List[EmailView]
    stats: Stats
    history: List[ActionSummary] = []
    done: bool = False
    step: int = 0


# ─────────────────────────── State ────────────────────────────

class State(BaseModel):
    """Full internal environment state for Enterprise."""
    inbox: Dict[str, Any] = {}
    calendar: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    user_prefs: Dict[str, Any] = {"max_meetings_day": 3}
    current_time: int = 0
    step_count: int = 0
    task: str = "enterprise_workflow"


# ─────────────────────────── Step Result ──────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict
    reasoning: Optional[str] = Field(default=None, description="Explanation for the given reward")


# ─────────────────────────── Grader ───────────────────────────

class GraderResult(BaseModel):
    score: float = Field(gt=0, lt=1)
    breakdown: Dict[str, float]
    details: Dict[str, str] = {}
