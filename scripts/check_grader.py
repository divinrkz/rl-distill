import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_distill.grading import is_correct

GOLD = "\\left( 3, \\frac{\\pi}{2} \\right)"

cases = [
    # (pred, gold, expected)
    ("The answer is \\boxed{(3, \\frac{\\pi}{2})}.", GOLD, True),  # full completion
    ("(3, \\frac{\\pi}{2})", GOLD, True),                         # bare extracted answer
    ("\\boxed{(3, \\frac{3\\pi}{2})}", GOLD, False),              # wrong angle
    ("42", "42", True),
    ("\\boxed{7}", "8", False),
]

ok = True
for pred, gold, expected in cases:
    got = is_correct(pred, gold)
    status = "OK " if got == expected else "FAIL"
    if got != expected:
        ok = False
    print(f"{status} expected={expected} got={got}  pred={pred!r}")

sys.exit(0 if ok else 1)
