"""
models.py — Pydantic schemas for EmailTriage OpenEnv
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union
from enum import Enum


# ─────────────────────────── Enums ────────────────────────────

class Label(str, Enum):
    spam    = "spam"
    low     = "low"
    med     = "med"
    high    = "high"
    escalate = "escalate"


class ActionType(str, Enum):
    label   = "label"
    draft   = "draft"
    delete  = "delete"
    escalate = "escalate"
    archive = "archive"


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
    Agent action. One of:
      - label(email_id, value: spam|low|med|high|escalate)
      - draft(email_id, value: <reply text>)
      - delete(email_id)
      - escalate(email_id)
      - archive(email_id)
    """
    type: ActionType
    email_id: str
    value: Optional[str] = Field(default=None, description="Label value or draft reply text")

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
    """Full internal environment state (not exposed to agent directly)."""
    inbox: Dict[str, Email] = {}
    labels: Dict[str, str] = {}          # email_id -> label string
    replies: Dict[str, str] = {}         # email_id -> draft text
    deleted: List[str] = []
    archived: List[str] = []
    task: str = "easy"
    step_count: int = 0


# ─────────────────────────── Step Result ──────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict


# ─────────────────────────── Grader ───────────────────────────

class GraderResult(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    breakdown: Dict[str, float]
    details: Dict[str, str] = {}
