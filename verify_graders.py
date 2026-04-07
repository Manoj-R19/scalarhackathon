try:
    from environment import EmailTriageEnv
    from models import State
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

def test_graders():
    env = EmailTriageEnv()
    tasks = ["easy", "medium", "hard"]
    
    for i, task_id in enumerate(tasks, 1):
        env.reset(task=task_id)
        res = env.grader()
        s = res.score
        print(f'Task {i} ({task_id}): {s:.3f}')
        assert 0 < s < 1, f'FAIL: {s} out of range!'
        
        # Verify breakdown too
        for k, v in res.breakdown.items():
            assert 0 < v < 1, f'FAIL: Breakdown {k}={v} out of range!'

    print('ALL GRADERS PASS ✅')

if __name__ == "__main__":
    test_graders()
