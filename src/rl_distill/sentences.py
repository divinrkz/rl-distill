"""Sentence splitting for CoT traces.

`string_to_sentences` / `process_text_segment` / `clean_python_string_literal`
are adapted from thought-anchors
(masking_graphs/resample/sentence_splitter.py), with the pkld cache decorator
and the unused paragraph/token-range helpers removed.

`split_cot` is the format-aware entry point: it isolates the reasoning region
via the model's format adapter (which handles both `<think>` and
`<|begin_of_thought|>` styles) and then splits that region. No per-format logic
lives in the splitter itself.
"""

import re
from typing import List


def split_cot(text, fmt):
    """Split a raw completion into reasoning sentences, format-agnostic."""
    thinking = fmt.extract_thinking(text)
    sentences, _ = string_to_sentences(thinking)
    return sentences


def split_cot_with_offsets(text, fmt):
    """Return (thinking_text, [(sentence, start_offset), ...]).

    Offsets index into thinking_text, so resampling can reconstruct exact
    prefixes by slicing rather than re-joining sentences.
    """
    thinking = fmt.extract_thinking(text)
    sentences, positions = string_to_sentences(thinking)
    return thinking, list(zip(sentences, positions))


def clean_python_string_literal(text: str) -> str:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^["\']', "", line)
        line = re.sub(r'["\']$', "", line)
        line = re.sub(r'^["\']', "", line)
        line = re.sub(r'["\']$', "", line)
        cleaned_lines.append(line)

    result = " ".join(cleaned_lines)
    result = result.replace("\\n", " ")
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace("\\\\", "\\")
    result = re.sub(r"\s+", " ", result).strip()
    return result


def string_to_sentences(text: str, drop_post_think=False):
    """Split text into (sentences, start positions). Handles headers,
    abbreviations, newlines, and Python string literals."""
    if not text or not isinstance(text, str):
        return [], []

    original_text = text
    line_segments = []
    current_pos = 0

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            line_pos = original_text.find(line, current_pos)
            if line_pos == -1:
                line_pos = current_pos

            line_stripped = line.strip()
            is_header = False
            if (
                (line_stripped.startswith("**") and line_stripped.endswith("**"))
                or (line_stripped.startswith("#"))
                or (line_stripped.endswith(":") and len(line_stripped) < 100)
                or (
                    len(line_stripped) < 80
                    and not any(line_stripped.endswith(p) for p in [".", "!", "?", '"', "'"])
                )
            ):
                is_header = True

            if drop_post_think:
                if i > 0 and i < len(lines) - 1:
                    if (not lines[i - 1].strip() or lines[i - 1].strip().endswith("</think>")) and (
                        not lines[i + 1].strip() or lines[i + 1].strip().startswith("<")
                    ):
                        is_header = True

            line_segments.append({"text": line, "position": line_pos, "is_header": is_header})

        current_pos += len(line) + 1

    all_sentences = []
    all_positions = []
    for segment in line_segments:
        if segment["is_header"]:
            sentence = segment["text"].strip()
            if sentence and len(sentence) >= 4:
                all_sentences.append(sentence)
                all_positions.append(segment["position"])
        else:
            segment_sentences, segment_positions = process_text_segment(
                segment["text"], segment["position"]
            )
            all_sentences.extend(segment_sentences)
            all_positions.extend(segment_positions)

    return all_sentences, all_positions


ABBREVIATIONS = {
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e.",
    "cf.", "al.", "Inc.", "Corp.", "Ltd.", "Co.", "U.S.", "U.K.", "Ph.D.",
    "M.D.", "B.A.", "M.A.", "B.S.", "M.S.", "M.Ed.", "M.S.Ed.", "J.D.", "LL.B.",
    "LL.M.", "M.B.A.",
}

DEGREE_ABBREVIATIONS = {
    "Ph.D.", "M.D.", "B.A.", "M.A.", "B.S.", "M.S.", "Ed.", "M.Ed.", "M.S.Ed.",
    "J.D.", "LL.B.", "LL.M.", "M.B.A.", "PhD.", "MD.", "BA.", "MA.", "BS.",
    "MS.", "MBA.", "JD.", "LLB.", "LLM.",
}


def process_text_segment(text: str, start_position: int = 0):
    if not text or not isinstance(text, str):
        return [], []

    text = clean_python_string_literal(text)
    original_text = text

    protected_text = text
    abbrev_placeholders = {}
    for i, abbrev in enumerate(ABBREVIATIONS):
        if abbrev in protected_text:
            pattern = r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)"
            if re.search(pattern, protected_text):
                placeholder = f"__ABBREV_{i}__"
                abbrev_placeholders[placeholder] = abbrev
                protected_text = re.sub(pattern, placeholder, protected_text)

    # Protect decimal points (period between digits) so math like 0.125 or
    # 1.64493 is not split mid-number. Restored with the abbreviations below.
    protected_text = re.sub(r"(?<=\d)\.(?=\d)", "__DECIMAL__", protected_text)

    sentence_pattern = r'([.!?])\s*(?=[A-Z]|["\']\s*[A-Z]|\d|$)'
    parts = re.split(sentence_pattern, protected_text)

    sentences = []
    current_sentence = ""
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            current_sentence += parts[i] + parts[i + 1]
            sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 2
        else:
            current_sentence += parts[i]
            i += 1
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    restored_sentences = []
    for sentence in sentences:
        for placeholder, abbrev in abbrev_placeholders.items():
            sentence = sentence.replace(placeholder, abbrev)
        sentence = sentence.replace("__DECIMAL__", ".")
        restored_sentences.append(sentence)

    final_sentences = []
    for sentence in restored_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        split_positions = []
        for abbrev in DEGREE_ABBREVIATIONS:
            pattern = re.escape(abbrev) + r"\s+(?=[A-Z0-9])"
            for match in re.finditer(pattern, sentence):
                split_positions.append(match.start() + len(abbrev))

        if split_positions:
            split_positions.sort()
            last_pos = 0
            for pos in split_positions:
                part = sentence[last_pos:pos].strip()
                if part:
                    final_sentences.append(part)
                last_pos = pos
            if last_pos < len(sentence):
                remaining = sentence[last_pos:].strip()
                if remaining:
                    final_sentences.append(remaining)
        else:
            final_sentences.append(sentence)

    final_sentences_clean: List[str] = []
    final_positions: List[int] = []
    for sentence in final_sentences:
        sentence = sentence.strip()
        if len(sentence) < 4:
            continue

        search_start = 0
        if final_positions:
            prev_pos_in_segment = final_positions[-1] - start_position
            search_start = prev_pos_in_segment + len(final_sentences_clean[-1])

        pos = original_text.find(sentence, search_start)
        if pos == -1:
            pos = original_text.find(sentence.strip(), search_start)
        if pos == -1:
            normalized_sentence = " ".join(sentence.split())
            normalized_original = " ".join(original_text.split())
            norm_pos = normalized_original.find(normalized_sentence, search_start)
            if norm_pos != -1:
                pos = norm_pos

        if pos != -1:
            final_sentences_clean.append(sentence)
            final_positions.append(start_position + pos)

    return final_sentences_clean, final_positions
