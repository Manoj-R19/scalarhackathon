"""Quick test: verify inference format and _clamp edge cases."""
import sys
sys.path.insert(0, ".")

from inference import log_start, log_step, log_end, _clamp

print("=== Format Output Tests ===")
log_start("easy", "email_triage", "google/gemma-2-9b-it")
log_step(1, '{"type":"label"}', 0.55, False, None)
log_step(2, '{"type":"delete"}', 0.72, True, None)
log_end(True, 2, 0.65, [0.55, 0.72])

print()
print("=== _clamp Edge Cases ===")
tests = [0.0, 1.0, -5.0, 999.0, float("nan"), float("inf"), 0.5, 0.01, 0.99]
all_ok = True
for val in tests:
    result = _clamp(val)
    ok = 0.0 < result < 1.0
    print(f"  _clamp({val!r:10}) = {result:.4f}  -> {'OK' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

# Also test non-numeric
for val in [None, "bad", [], {}]:
    result = _clamp(val)
    ok = 0.0 < result < 1.0
    print(f"  _clamp({val!r:10}) = {result:.4f}  -> {'OK' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

print()
print("=== RESULT:", "ALL OK" if all_ok else "SOME FAILED", "===")
sys.exit(0 if all_ok else 1)
