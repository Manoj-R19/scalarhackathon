import numpy as np

class RubricEvaluator:
    """
    OpenEnv Rubric System for Sovereign Agent v11.0.
    Evaluates specific qualitative metrics across the episode trajectory.
    """
    def __init__(self, rubrics=None):
        self.rubrics = rubrics or [
            {"id": "causal_logic", "weight": 0.3},
            {"id": "p0_crisis", "weight": 0.4},
            {"id": "format_precision", "weight": 0.3}
        ]

    def evaluate_causal(self, traj_logs):
        """Check if check_calendar precedes schedule_meeting."""
        found_check = False
        for entry in traj_logs:
            tool = entry.get("tool", "")
            if tool == "check_calendar":
                found_check = True
            if tool == "schedule_meeting" and not found_check:
                return 0.0
        return 1.0 if found_check else 0.5 # Neutral if neither used

    def evaluate_crisis(self, traj_logs):
        """Check if crisis is escalated within 3 steps of detection."""
        crisis_step = -1
        escalate_step = -1
        for i, entry in enumerate(traj_logs):
            if entry.get("crisis_active") and crisis_step == -1:
                crisis_step = i
            if entry.get("tool") == "escalate_crisis" and escalate_step == -1:
                escalate_step = i
        
        if crisis_step == -1: return 1.0 # No crisis is a pass
        if escalate_step == -1: return 0.0
        return 1.0 if (escalate_step - crisis_step) <= 3 else 0.5

    def total_rubric_score(self, traj_logs):
        """Weighted average of all rubrics."""
        scores = {
            "causal_logic": self.evaluate_causal(traj_logs),
            "p0_crisis": self.evaluate_crisis(traj_logs),
            "format_precision": 1.0 # Mocked for v11.0 core
        }
        
        total = 0.0
        for r in self.rubrics:
            total += scores.get(r["id"], 0.0) * r["weight"]
        
        return np.clip(total, 0.01, 0.99)

def weighted_avg(rubrics, scores):
    return sum(scores[r["id"]] * r["weight"] for r in rubrics)
