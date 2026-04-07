"""
environment.py — Core EmailTriage OpenEnv logic
step(), reset(), grader(), get_obs()
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Tuple, Dict, Any

from models import (
    Action, ActionType, Email, EmailView, Label,
    Observation, State, Stats, ActionSummary, StepResult, GraderResult
)

# ─────────────────────────── Constants ────────────────────────

DATA_PATH   = Path(__file__).parent / "data" / "inboxes.json"
MAX_STEPS   = 50
RANDOM_SEED = 42

TASK_CONFIG = {
    "easy":   {"label_w": 0.6, "reply_w": 0.2, "empty_w": 0.1, "spam_w": 0.1, "escalate_w": 0.0},
    "medium": {"label_w": 0.5, "reply_w": 0.2, "empty_w": 0.1, "spam_w": 0.2, "escalate_w": 0.0},
    "hard":   {"label_w": 0.3, "reply_w": 0.2, "empty_w": 0.1, "spam_w": 0.1, "escalate_w": 0.3},
}


class EmailTriageEnv:
    """
    OpenEnv-compatible Email Triage environment.

    Lifecycle:
        env = EmailTriageEnv()
        obs = env.reset(task="easy")
        result = env.step(action_json)
        score = env.grader()
    """

    def __init__(self):
        self._raw_data: Dict = {}
        self.state: State = State()
        self.ground_truth: Dict[str, Dict] = {}
        self.action_history: list[ActionSummary] = []
        self.current_task: str = "easy"
        self._load_data()

    # ──────────────── Data Loading ────────────────

    def _load_data(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

    # ──────────────── Reset ───────────────────────

    def reset(self, task: str = "easy") -> Observation:
        if task not in self._raw_data:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(self._raw_data.keys())}")

        random.seed(RANDOM_SEED)
        self.current_task = task
        task_data = self._raw_data[task]

        inbox: Dict[str, Email] = {}
        for e in task_data["emails"]:
            inbox[e["id"]] = Email(
                id=e["id"],
                subject=e["subject"],
                body=e["body"],
                sender=e["sender"],
            )

        self.ground_truth = task_data["ground_truth"]
        self.state = State(inbox=inbox, task=task)
        self.action_history = []

        return self._get_obs()

    def _scale_reward(self, raw_reward: float) -> float:
        """Scales raw reward in [-0.5, 0.5] range to strict [0.01, 0.99]."""
        # Clamp raw_reward first to be safe
        clamped_raw = max(-0.5, min(0.5, raw_reward))
        return round(0.01 + (clamped_raw + 0.5) * 0.98, 4)

    # ──────────────── Step ────────────────────────

    def step(self, action_json: str) -> StepResult:
        # Parse action
        try:
            action = Action.model_validate_json(action_json)
        except Exception as e:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._scale_reward(-0.5),
                done=False,
                info={"error": f"Invalid action JSON: {e}"}
            )

        email_id = action.email_id

        # Validate email_id exists
        if email_id not in self.state.inbox:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._scale_reward(-0.2),
                done=False,
                info={"error": f"Email '{email_id}' not found in inbox"}
            )

        # Prevent re-acting on deleted/archived
        if email_id in self.state.deleted or email_id in self.state.archived:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._scale_reward(-0.1),
                done=False,
                info={"warning": f"Email '{email_id}' already processed"}
            )

        raw_reward = self._compute_reward(action)
        reward = self._scale_reward(raw_reward)

        # Apply action to state
        self._apply_action(action)

        self.state.step_count += 1

        # Record in history
        self.action_history.append(ActionSummary(
            step=self.state.step_count,
            action_type=action.type,
            email_id=email_id,
            reward=reward,
        ))

        done = self._is_done()
        obs = self._get_obs(done=done)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self.state.step_count,
                "task": self.current_task,
                "action_applied": action.type,
            }
        )

    # ──────────────── Internal Helpers ────────────

    def _apply_action(self, action: Action):
        eid = action.email_id

        if action.type == ActionType.label:
            try:
                self.state.labels[eid] = Label(action.value).value
            except ValueError:
                pass  # invalid label handled in reward

        elif action.type == ActionType.delete:
            self.state.deleted.append(eid)
            # Also record it as spam label
            self.state.labels[eid] = Label.spam.value

        elif action.type == ActionType.escalate:
            self.state.labels[eid] = Label.escalate.value

        elif action.type == ActionType.draft:
            self.state.replies[eid] = action.value or ""

        elif action.type == ActionType.archive:
            self.state.archived.append(eid)

    def _compute_reward(self, action: Action) -> float:
        eid = action.email_id
        gt  = self.ground_truth.get(eid, {})
        gt_label = gt.get("label", "")
        reward = 0.0

        if action.type == ActionType.label:
            try:
                label_val = Label(action.value).value
            except ValueError:
                return -0.3  # completely invalid label string

            if label_val == gt_label:
                reward = 0.2
                # Bonus for escalation (hard task)
                if label_val == "escalate":
                    reward = 0.4
            else:
                reward = -0.1

        elif action.type == ActionType.delete:
            if gt_label == "spam":
                reward = 0.3
            else:
                reward = -0.4  

        elif action.type == ActionType.escalate:
            if gt_label == "escalate":
                reward = 0.4
            else:
                reward = -0.2

        elif action.type == ActionType.draft:
            draft_text = (action.value or "").lower()
            kws = gt.get("reply_keywords", [])
            if gt_label == "spam":
                reward = -0.2  # drafting reply to spam is waste
            elif not draft_text:
                reward = -0.1
            else: 
                # Keyword match scoring
                hits = sum(1 for kw in kws if kw.lower() in draft_text)
                ratio = hits / len(kws) if kws else 0.0
                length_ok = 10 <= len(draft_text.split()) <= 80
                reward = 0.1 + (0.2 * ratio) + (0.05 if length_ok else 0)

        elif action.type == ActionType.archive:
            if gt_label in ("low", "med"):
                reward = 0.05  # archiving low/med is acceptable
            elif gt_label == "spam":
                reward = 0.1   # archiving spam is ok (not as good as delete)
            else:
                reward = -0.1  # archiving high/escalate is wrong

        # Progress shaped bonus
        processed = len(self.state.labels) + len(self.state.deleted) + len(self.state.archived)
        total = len(self.state.inbox)
        progress = processed / total if total else 0.0
        reward += 0.05 * progress

        return round(reward, 4)

    def _is_done(self) -> bool:
        processed = (
            set(self.state.labels.keys()) |
            set(self.state.deleted) |
            set(self.state.archived)
        )
        all_processed = processed >= set(self.state.inbox.keys())
        return all_processed or self.state.step_count >= MAX_STEPS

    def _get_obs(self, done: bool = False) -> Observation:
        processed = (
            set(self.state.labels.keys()) |
            set(self.state.deleted) |
            set(self.state.archived)
        )

        email_views = []
        for eid, email in self.state.inbox.items():
            is_labeled = eid in self.state.labels
            email_views.append(EmailView(
                id=email.id,
                subject=email.subject,
                body=email.body,
                sender=email.sender,
                labeled=is_labeled,
                label=self.state.labels.get(eid),
            ))

        stats = Stats(
            total=len(self.state.inbox),
            unread=len(self.state.inbox) - len(processed),
            labeled=len(self.state.labels),
            deleted=len(self.state.deleted),
            escalated=sum(1 for v in self.state.labels.values() if v == "escalate"),
            drafts=len(self.state.replies),
        )

        return Observation(
            current_emails=email_views,
            stats=stats,
            history=self.action_history[-10:],  # Last 10 actions
            done=done,
            step=self.state.step_count,
        )

    # ──────────────── Grader ──────────────────────

    def _strict_clamp(self, val: float) -> float:
        """Ensures val is strictly in (0.01, 0.99) range."""
        return round(max(0.01, min(0.99, float(val))), 4)

    def grader(self) -> GraderResult:
        gt = self.ground_truth
        total = len(gt)
        # Default breakdown with keys from openenv.yaml
        breakdown = {
            "label_accuracy": 0.05,
            "spam_recall": 0.05,
            "reply_relevance": 0.05,
            "escalation_recall": 0.05,
            "inbox_cleared": 0.05,
        }
        
        if total == 0:
            return GraderResult(score=0.1, breakdown=breakdown)

        cfg = TASK_CONFIG.get(self.current_task, TASK_CONFIG["easy"])

        # 1. Label accuracy
        label_correct = sum(
            1 for eid, info in gt.items()
            if self.state.labels.get(eid) == info["label"]
        )
        label_acc_raw = label_correct / total
        label_acc = 0.02 + (label_acc_raw * 0.96)  # Scale [0,1] -> [0.02, 0.98]

        # 2. Spam recall
        spam_ids = [eid for eid, info in gt.items() if info["label"] == "spam"]
        if not spam_ids:
            spam_recall = 0.95  # No spam: pass with room for improvement
        else:
            spam_correct = sum(
                1 for eid in spam_ids
                if eid in self.state.deleted or self.state.labels.get(eid) == "spam"
            )
            spam_recall_raw = spam_correct / len(spam_ids)
            spam_recall = 0.05 + (spam_recall_raw * 0.90) # [0.05, 0.95]

        # 3. Reply relevance
        legit_ids = [eid for eid, info in gt.items() if info["label"] != "spam"]
        reply_score = 0.05
        if legit_ids:
            hits_sum = 0.0
            for eid in legit_ids:
                if eid in self.state.replies:
                    reply_text = self.state.replies[eid].lower()
                    kws = gt[eid].get("reply_keywords", [])
                    if kws:
                        hits = sum(1 for kw in kws if kw.lower() in reply_text)
                        hits_sum += hits / len(kws)
            reply_score_raw = hits_sum / len(legit_ids)
            reply_score = 0.05 + (reply_score_raw * 0.92)

        # 4. Escalation recall
        escalate_ids = [eid for eid, info in gt.items() if info["label"] == "escalate"]
        if not escalate_ids:
            escalate_recall = 0.95
        else:
            esc_correct = sum(
                1 for eid in escalate_ids
                if self.state.labels.get(eid) == "escalate"
            )
            escalate_recall_raw = esc_correct / len(escalate_ids)
            escalate_recall = 0.02 + (escalate_recall_raw * 0.96)

        # 5. Inbox cleared (Progress)
        processed = (
            set(self.state.labels.keys()) |
            set(self.state.deleted) |
            set(self.state.archived)
        )
        progress = len(processed) / total
        if progress >= 1.0:
            inbox_cleared = 0.95
        elif progress >= 0.5:
            inbox_cleared = 0.5 + (progress - 0.5) * 0.4  # 0.5 to 0.9
        else:
            inbox_cleared = 0.05 + progress * 0.45      # 0.05 to 0.5

        # Weighted final score
        raw_weighted_score = (
            cfg.get("label_w", 0)    * label_acc +
            cfg.get("spam_w", 0)     * spam_recall +
            cfg.get("reply_w", 0)    * reply_score +
            cfg.get("escalate_w", 0) * escalate_recall +
            cfg.get("empty_w", 0)    * inbox_cleared
        )
        
        score = self._strict_clamp(raw_weighted_score)
        
        return GraderResult(
            score=score,
            breakdown={
                "label_accuracy":    self._strict_clamp(label_acc),
                "spam_recall":       self._strict_clamp(spam_recall),
                "reply_relevance":   self._strict_clamp(reply_score),
                "escalation_recall": self._strict_clamp(escalate_recall),
                "inbox_cleared":     self._strict_clamp(inbox_cleared),
            },
            details={
                "task": self.current_task,
                "steps_used": str(self.state.step_count),
                "emails_labeled": str(len(self.state.labels)),
                "emails_deleted": str(len(self.state.deleted)),
                "drafts_written": str(len(self.state.replies)),
            }
        )

    # ──────────────── Public Accessors ────────────

    def get_state(self) -> State:
        return self.state

    def get_tasks(self) -> list[dict]:
        return [
            {
                "id": "easy",
                "description": "Binary classify 5 emails as spam or legit (low/med/high). Delete the spam.",
                "difficulty": "easy",
                "email_count": len(self._raw_data["easy"]["emails"]),
                "grader_weights": TASK_CONFIG["easy"],
            },
            {
                "id": "medium",
                "description": "Classify and prioritize 10 mixed emails. Delete spam, label by urgency.",
                "difficulty": "medium",
                "email_count": len(self._raw_data["medium"]["emails"]),
                "grader_weights": TASK_CONFIG["medium"],
            },
            {
                "id": "hard",
                "description": (
                    "Full triage of 15 emails: classify, draft replies for legit emails, "
                    "delete spam, and escalate P0/critical issues."
                ),
                "difficulty": "hard",
                "email_count": len(self._raw_data["hard"]["emails"]),
                "grader_weights": TASK_CONFIG["hard"],
            },
        ]
