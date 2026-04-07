#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import ValidationError
from graders import *  # Adjust import

def exhaustive_test():
    """Tests 100% edge cases"""
    
    # Edge case states
    test_cases = [
        {},                           # Empty → 0.01 min
        {"progress": 0.0},            # Zero progress  
        {"progress": 0.5},            # Half
        {"progress": 1.0},            # Perfect → 0.99 max
        {"score": -1.0},              # Negative
        {"score": 2.0},               # Overshoot
    ]
    
    graders_dict = {
        "easy": grader_easy,
        "medium": grader_medium, 
        "hard": grader_hard,
    }
    
    all_safe = True
    for name, grader in graders_dict.items():
        print(f"\n=== Testing {name} ===")
        for i, state in enumerate(test_cases):
            try:
                score = grader(state)
                status = "✅" if 0.0 < score < 1.0 else "❌"
                print(f"  Case {i}: score={score:.3f} {status}")
                if not (0.0 < score < 1.0):
                    all_safe = False
            except Exception as e:
                print(f"  Case {i}: CRASH {e}")
                all_safe = False
    
    # Final verdict
    print("\n" + ("🎉 PHASE 2 READY!" if all_safe else "🔥 FIX GRADERS!"))
    return all_safe

if __name__ == "__main__":
    exhaustive_test()
