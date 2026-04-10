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
    "expert": {"label_w": 0.2, "reply_w": 0.2, "empty_w": 0.1, "spam_w": 0.2, "escalate_w": 0.3},
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

    def _safe_normalize(self, score: float, epsilon: float = 0.01) -> float:
        """Clamp ANY score to strict open interval (epsilon, 1-epsilon).
        Guaranteed: result is always strictly > 0 and strictly < 1.
        """
        try:
            v = float(score)
            if v != v:  # NaN
                v = 0.5
        except Exception:
            v = 0.5
        clamped = max(epsilon, min(1.0 - epsilon, v))
        # Extra safety: should never be exactly 0 or 1, but guard anyway
        if clamped <= 0:
            clamped = epsilon
        if clamped >= 1:
            clamped = 1 - epsilon
        return round(clamped, 4)

    # ──────────────── Step ────────────────────────

    def step(self, action_json: str) -> StepResult:
        # Parse action
        try:
            action = Action.model_validate_json(action_json)
        except Exception as e:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._safe_normalize(-0.5),
                done=False,
                info={"error": f"Invalid action JSON: {e}"},
                reasoning="Failed to parse action JSON."
            )

        email_id = action.email_id

        # Validate email_id exists
        if email_id not in self.state.inbox:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._safe_normalize(-0.2),
                done=False,
                info={"error": f"Email '{email_id}' not found in inbox"},
                reasoning=f"Email ID '{email_id}' does not exist."
            )

        # Prevent re-acting on deleted/archived (for state-mutating actions)
        if email_id in self.state.deleted or email_id in self.state.archived:
            obs = self._get_obs()
            return StepResult(
                observation=obs,
                reward=self._safe_normalize(-0.1),
                done=False,
                info={"warning": f"Email '{email_id}' already processed"},
                reasoning="Cannot act on archived/deleted emails."
            )

        raw_reward, reason = self._compute_reward(action)
        
        # Sparse reward scale [-1.0, 1.0] -> [0, 1]
        sparse_norm = max(0.0, min(1.0, (raw_reward + 1.0) / 2.0))
        
        # Progressive shaping
        trajectory_progress = len(self.action_history) / MAX_STEPS
        shaped_reward = 0.1 * sparse_norm + 0.9 * trajectory_progress
        
        # Safety clamp
        reward = self._safe_normalize(shaped_reward)

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
            },
            reasoning=reason
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

    def _compute_reward(self, action: Action) -> Tuple[float, str]:
        eid = action.email_id
        gt  = self.ground_truth.get(eid, {})
        gt_label = gt.get("label", "")
        reward = 0.0
        reason = "Neutral action"

        # Redundancy check
        prev_actions = [a for a in self.action_history if a.email_id == eid and a.action_type == action.type]
        if prev_actions:
            return -0.2, f"Redundant {action.type}: You already performed this action on this email."

        if action.type == ActionType.label:
            try:
                label_val = Label(action.value).value
            except ValueError:
                return -0.3, f"Invalid label '{action.value}'. Must be one of {list(Label)}."

            if label_val == gt_label:
                reward = 0.3
                reason = f"Correctly labeled email as '{label_val}'."
                if label_val == "escalate":
                    reward = 0.6
                    reason = "EXCELLENT: Critical issue correctly identified for escalation."
            else:
                reward = -0.2
                reason = f"Incorrect label. Expected '{gt_label}', got '{label_val}'."

        elif action.type == ActionType.delete:
            if gt_label == "spam":
                reward = 0.4
                reason = "Correctly deleted spam email."
            elif gt_label == "escalate":
                reward = -1.0
                reason = "CRITICAL ERROR: You deleted a high-priority escalation email! This is a catastrophic failure."
            else:
                reward = -0.5
                reason = f"Improper deletion: You deleted a legitimate '{gt_label}' priority email."

        elif action.type == ActionType.escalate:
            if gt_label == "escalate":
                reward = 0.6
                reason = "Correct escalation of a critical P0 incident."
            else:
                reward = -0.3
                reason = f"False alarm: Escalated a '{gt_label}' priority email that didn't require P0 response."

        elif action.type == ActionType.draft:
            draft_text = (action.value or "").lower()
            kws = gt.get("reply_keywords", [])
            
            # Contextual check: labeling before drafting is better
            labeled = eid in self.state.labels
            label_bonus = 0.05 if labeled else -0.05
            
            if gt_label == "spam":
                reward = -0.3
                reason = "Waste of time: Drafting a reply to a known spam email."
            elif not draft_text:
                reward = -0.1
                reason = "Empty draft: No content provided."
            else: 
                hits = sum(1 for kw in kws if kw.lower() in draft_text)
                ratio = hits / len(kws) if kws else 1.0 # 1.0 if no keywords expected
                length_ok = 10 <= len(draft_text.split()) <= 80
                
                reward = 0.2 + (0.3 * ratio) + (0.1 if length_ok else -0.1) + label_bonus
                reason = f"Draft relevance score: {ratio:.1%}. "
                if not length_ok: reason += "Refine length (10-80 words). "
                if not labeled: reason += "Note: Identify the priority before drafting."

        elif action.type == ActionType.archive:
            if gt_label in ("low", "med"):
                reward = 0.1
                reason = f"Correctly archived {gt_label} priority email."
            elif gt_label == "spam":
                reward = 0.1
                reason = "Archiving spam is acceptable, though deletion is preferred."
            else:
                reward = -0.3
                reason = f"Improper archive: A '{gt_label}' email requires active response, not archiving."

        # Progress shaped bonus
        processed = len(set(self.state.labels.keys()) | set(self.state.deleted) | set(self.state.archived))
        total = len(self.state.inbox)
        progress = processed / total if total else 0.0
        reward += 0.1 * progress

        return round(reward, 4), reason

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

    def _strict_clamp(self, val) -> float:
        """Ensures val is strictly in (0.01, 0.99) range. Never 0.0 or 1.0."""
        try:
            v = float(val)
            if v != v:  # NaN guard
                v = 0.5
        except Exception:
            v = 0.5
        result = round(max(0.01, min(0.99, v)), 4)
        # Double-check bounds (floating point paranoia)
        if result <= 0.0:
            result = 0.01
        if result >= 1.0:
            result = 0.99
        return result

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

        # Hard final guard: score MUST be strictly between 0 and 1
        assert 0.0 < score < 1.0, f"Grader produced boundary score: {score}"

        breakdown = {
            "label_accuracy":    self._strict_clamp(label_acc),
            "spam_recall":       self._strict_clamp(spam_recall),
            "reply_relevance":   self._strict_clamp(reply_score),
            "escalation_recall": self._strict_clamp(escalate_recall),
            "inbox_cleared":     self._strict_clamp(inbox_cleared),
        }
        # Guard all breakdown values too
        for k, v in breakdown.items():
            assert 0.0 < v < 1.0, f"Breakdown '{k}' = {v} is out of range"

        return GraderResult(
            score=score,
            breakdown=breakdown,
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
            {
                "id": "expert",
                "description": (
                    "Advanced triage of 5 critical edge cases: detect sophisticated phishing, "
                    "legal subpoenas, and major infra cost anomalies. High stakes."
                ),
                "difficulty": "expert",
                "email_count": len(self._raw_data["expert"]["emails"]),
                "grader_weights": TASK_CONFIG["expert"],
            },
        ]
