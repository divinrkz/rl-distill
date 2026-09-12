"""Sentence-category labeling with a local instruct model via vLLM.

Venhoff et al. 6-category taxonomy (same scheme used for the Exp 2 steering
vectors, so Exp 1 and Exp 2 stay comparable). One labeler call per CoT returns a
category for every sentence.
"""

import json
import re

CATEGORIES = {
    "initializing": "restating or setting up the problem; understanding what is asked",
    "deduction": "deriving results or doing the core reasoning/computation toward the answer",
    "adding-knowledge": "recalling a fact, formula, definition, or known result",
    "example-testing": "trying concrete numbers or examples to check or explore",
    "uncertainty-estimation": "expressing doubt or confidence about correctness",
    "backtracking": "abandoning or revising a previous approach; going back",
}
FALLBACK = "unknown"


def _system_prompt():
    lines = ["You label reasoning sentences by their function. Categories:"]
    for name, desc in CATEGORIES.items():
        lines.append(f'- "{name}": {desc}')
    lines.append(
        "\nReturn ONLY a JSON object mapping each sentence index (as a string) to "
        "exactly one category name. No prose, no code fences."
    )
    return "\n".join(lines)


def _user_prompt(problem, sentences):
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return f"Problem:\n{problem}\n\nSentences:\n{numbered}"


def build_label_prompt(problem, sentences, tokenizer):
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(problem, sentences)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_labels(text, n_sentences):
    """Parse the labeler's JSON into a list of n_sentences category names."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    labels = [FALLBACK] * n_sentences
    if not match:
        return labels
    try:
        mapping = json.loads(match.group(0))
    except json.JSONDecodeError:
        return labels
    for k, v in mapping.items():
        try:
            i = int(k)
        except (ValueError, TypeError):
            continue
        if 0 <= i < n_sentences and v in CATEGORIES:
            labels[i] = v
    return labels
