"""Answer extraction from raw CoT text. Format-agnostic string parsing only."""

import re


def extract_boxed(text):
    """Return the content of the last \\boxed{...} in text, or None."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def extract_between(text, open_tag, close_tag):
    """Return content between the last open_tag and its following close_tag, or None."""
    start = text.rfind(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return text[start:]
    return text[start:end]
