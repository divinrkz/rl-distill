"""Answer correctness via math-verify. Never string equality.

math-verify needs delimited LaTeX (or a \\boxed{}) to parse reliably; bare
strings like `(3, \\frac{\\pi}{2})` silently fail to verify. `_as_latex` wraps a
bare answer, and leaves a full completion (which already contains \\boxed{})
untouched so math-verify can extract the answer itself.
"""

from math_verify import parse, verify


def _as_latex(s):
    if "$" in s or "\\boxed" in s or "\\(" in s:
        return s
    return f"${s}$"


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        gold_parsed = parse(_as_latex(gold))
        pred_parsed = parse(_as_latex(pred))
        if not gold_parsed or not pred_parsed:
            return False
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False
