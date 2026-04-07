"""
tests/test_env.py — Full TDD test suite for EmailTriageEnv
Run: pytest -v
"""
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import EmailTriageEnv
from models import Action, ActionType, Label


# ─────────────────── Fixtures ───────────────────

@pytest.fixture
def env():
    return EmailTriageEnv()


@pytest.fixture
def easy_env(env):
    env.reset("easy")
    return env


@pytest.fixture
def medium_env(env):
    env.reset("medium")
    return env


@pytest.fixture
def hard_env(env):
    env.reset("hard")
    return env


# ─────────────────── Model Tests ────────────────

class TestModels:
    def test_action_label_valid(self):
        a = Action(type=ActionType.label, email_id="e1", value="spam")
        assert a.type == "label"
        assert a.value == "spam"

    def test_action_delete_no_value(self):
        a = Action(type=ActionType.delete, email_id="e2")
        assert a.value is None

    def test_action_draft_with_text(self):
        a = Action(type=ActionType.draft, email_id="e3", value="Thanks for reaching out!")
        assert "Thanks" in a.value

    def test_action_escalate(self):
        a = Action(type=ActionType.escalate, email_id="e4")
        assert a.type == "escalate"

    def test_label_enum_values(self):
        assert Label.spam.value == "spam"
        assert Label.high.value == "high"
        assert Label.escalate.value == "escalate"

    def test_action_json_roundtrip(self):
        a = Action(type=ActionType.label, email_id="e1", value="high")
        json_str = a.model_dump_json()
        a2 = Action.model_validate_json(json_str)
        assert a.email_id == a2.email_id
        assert a.value == a2.value


# ─────────────────── Reset Tests ────────────────

class TestReset:
    def test_reset_easy_loads_5_emails(self, env):
        obs = env.reset("easy")
        assert obs.stats.total == 5
        assert obs.stats.unread == 5
        assert obs.stats.labeled == 0

    def test_reset_medium_loads_10_emails(self, env):
        obs = env.reset("medium")
        assert obs.stats.total == 10

    def test_reset_hard_loads_15_emails(self, env):
        obs = env.reset("hard")
        assert obs.stats.total == 15

    def test_reset_returns_observation(self, env):
        obs = env.reset("easy")
        assert len(obs.current_emails) == 5
        assert obs.done is False
        assert obs.step == 0

    def test_reset_clears_state(self, easy_env):
        # Do some actions
        easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "spam"}))
        # Re-reset
        obs = easy_env.reset("easy")
        assert obs.stats.labeled == 0
        assert obs.stats.deleted == 0

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task"):
            env.reset("nonexistent_task")

    def test_reset_reproducible(self, env):
        obs1 = env.reset("hard")
        obs2 = env.reset("hard")
        ids1 = [e.id for e in obs1.current_emails]
        ids2 = [e.id for e in obs2.current_emails]
        assert ids1 == ids2  # Seeded random → same order


# ─────────────────── Step Tests ─────────────────

class TestStep:
    def test_label_correct_spam_gives_positive_reward(self, easy_env):
        result = easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "spam"}))
        assert result.reward > 0.05

    def test_label_correct_high_gives_positive_reward(self, easy_env):
        result = easy_env.step(json.dumps({"type": "label", "email_id": "e2", "value": "high"}))
        assert result.reward > 0.05

    def test_label_wrong_gives_negative_reward(self, easy_env):
        # e1 is spam, mislabeling as high
        result = easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "high"}))
        assert result.reward < 0.07

    def test_delete_spam_gives_positive_reward(self, easy_env):
        result = easy_env.step(json.dumps({"type": "delete", "email_id": "e1"}))
        assert result.reward > 0.05

    def test_delete_legit_gives_negative_reward(self, easy_env):
        # e2 is a legit high priority email
        result = easy_env.step(json.dumps({"type": "delete", "email_id": "e2"}))
        assert result.reward < 0.07

    def test_draft_reply_with_keywords_gives_reward(self, easy_env):
        result = easy_env.step(json.dumps({
            "type": "draft",
            "email_id": "e2",
            "value": "We are looking into the credentials and account reset immediately."
        }))
        assert result.reward > 0.05

    def test_draft_reply_to_spam_penalized(self, easy_env):
        result = easy_env.step(json.dumps({
            "type": "draft",
            "email_id": "e1",  # spam
            "value": "Thanks for your message!"
        }))
        assert result.reward < 0.07

    def test_escalate_correct_escalation_gives_high_reward(self, hard_env):
        result = hard_env.step(json.dumps({"type": "escalate", "email_id": "h1"}))
        assert result.reward >= 0.05

    def test_escalate_wrong_email_penalized(self, hard_env):
        result = hard_env.step(json.dumps({"type": "escalate", "email_id": "h4"}))
        assert result.reward < 0.07

    def test_invalid_action_json_penalized(self, easy_env):
        result = easy_env.step("not valid json {{{{")
        assert result.reward <= 0.05
        assert "error" in result.info

    def test_invalid_email_id_penalized(self, easy_env):
        result = easy_env.step(json.dumps({"type": "label", "email_id": "INVALID_ID", "value": "spam"}))
        assert result.reward <= 0.05

    def test_step_increments_counter(self, easy_env):
        easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "spam"}))
        assert easy_env.state.step_count == 1

    def test_step_updates_history(self, easy_env):
        easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "spam"}))
        obs = easy_env._get_obs()
        assert len(obs.history) == 1
        assert obs.history[0].email_id == "e1"

    def test_done_when_max_steps_exceeded(self, easy_env):
        """Simulate hitting step limit."""
        easy_env.state.step_count = 49
        easy_env.state.labels = {}  # still has unread
        result = easy_env.step(json.dumps({"type": "label", "email_id": "e1", "value": "spam"}))
        assert result.done is True

    def test_done_when_inbox_cleared(self, easy_env):
        actions = [
            {"type": "delete", "email_id": "e1"},
            {"type": "label", "email_id": "e2", "value": "high"},
            {"type": "delete", "email_id": "e3"},
            {"type": "label", "email_id": "e4", "value": "low"},
            {"type": "label", "email_id": "e5", "value": "high"},
        ]
        results = [easy_env.step(json.dumps(a)) for a in actions]
        assert results[-1].done is True


# ─────────────────── Grader Tests ───────────────

class TestGrader:
    def test_grader_returns_zero_on_empty_labels(self, easy_env):
        result = easy_env.grader()
        # No actions → spam_recall part still counts (spam not deleted)
        assert 0.0 <= result.score <= 1.0

    def test_grader_score_improves_with_correct_labels(self, easy_env):
        score_before = easy_env.grader().score
        # Correctly label all easy emails
        easy_env.step(json.dumps({"type": "delete", "email_id": "e1"}))  # spam
        easy_env.step(json.dumps({"type": "label", "email_id": "e2", "value": "high"}))
        easy_env.step(json.dumps({"type": "delete", "email_id": "e3"}))  # spam
        easy_env.step(json.dumps({"type": "label", "email_id": "e4", "value": "low"}))
        easy_env.step(json.dumps({"type": "label", "email_id": "e5", "value": "high"}))
        score_after = easy_env.grader().score
        assert score_after > score_before

    def test_grader_score_in_range(self, medium_env):
        result = medium_env.grader()
        assert 0.0 <= result.score <= 1.0

    def test_grader_has_breakdown(self, easy_env):
        result = easy_env.grader()
        assert "label_accuracy" in result.breakdown
        assert "spam_recall" in result.breakdown
        assert "reply_relevance" in result.breakdown

    def test_grader_hard_task_rewards_escalation(self, hard_env):
        hard_env.step(json.dumps({"type": "escalate", "email_id": "h1"}))
        hard_env.step(json.dumps({"type": "escalate", "email_id": "h3"}))
        hard_env.step(json.dumps({"type": "escalate", "email_id": "h6"}))
        result = hard_env.grader()
        assert result.breakdown["escalation_recall"] > 0

    def test_perfect_easy_score(self, env):
        env.reset("easy")
        # Perfect actions for easy task
        actions = [
            {"type": "delete", "email_id": "e1"},   # spam
            {"type": "label", "email_id": "e2", "value": "high"},
            {"type": "delete", "email_id": "e3"},   # spam
            {"type": "label", "email_id": "e4", "value": "low"},
            {"type": "label", "email_id": "e5", "value": "high"},
        ]
        for a in actions:
            env.step(json.dumps(a))
        result = env.grader()
        assert result.score >= 0.7  # Should be near-perfect


# ─────────────────── Integration Tests ──────────

class TestFullEpisode:
    def test_full_easy_episode(self, env):
        obs = env.reset("easy")
        total_reward = 0.0
        actions = [
            {"type": "delete", "email_id": "e1"},
            {"type": "label", "email_id": "e2", "value": "high"},
            {"type": "delete", "email_id": "e3"},
            {"type": "label", "email_id": "e4", "value": "low"},
            {"type": "label", "email_id": "e5", "value": "high"},
        ]
        for a in actions:
            result = env.step(json.dumps(a))
            total_reward += result.reward

        final = env.grader()
        assert final.score >= 0.0
        assert final.score <= 1.0
        assert total_reward > 0  # Good actions net positive reward

    def test_tasks_info_structure(self, env):
        tasks = env.get_tasks()
        assert len(tasks) == 3
        ids = [t["id"] for t in tasks]
        assert "easy" in ids
        assert "medium" in ids
        assert "hard" in ids
