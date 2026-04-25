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
    check_calendar = "CHECK_CALENDAR"
    schedule_meeting = "SCHEDULE_MEETING"
    reply_email = "REPLY_EMAIL"
    create_ticket = "CREATE_TICKET"


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
    """
    Tool-Calling Actions:
      - CHECK_CALENDAR()
      - SCHEDULE_MEETING(time)
      - REPLY_EMAIL(id, message)
      - CREATE_TICKET(issue)
    """
    type: ActionType
    email_id: Optional[str] = Field(default=None)
    time: Optional[str] = Field(default=None)
    message: Optional[str] = Field(default=None)
    issue: Optional[str] = Field(default=None)

    model_config = {"use_enum_values": True}


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
    inbox: List[Dict[str, Any]] = []
    calendar: List[Dict[str, Any]] = []
    task_board: List[Dict[str, Any]] = []
    step_count: int = 0
    task: str = "easy"


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
