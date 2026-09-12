"""Difference-of-means steering vectors from labeled reasoning traces.

For each category: vec[cat][layer] = mean(residual at that category's tokens)
                                     - mean(residual at all reasoning tokens),
averaged over traces. Requires a fast tokenizer (offset mapping) to map sentence
character spans to token positions.
"""

import torch

from .hooks import capture, layers_of

OVERALL = "__overall__"


def token_positions(offsets, char_start, char_end):
    """Token indices whose character span overlaps [char_start, char_end)."""
    return [k for k, (a, b) in enumerate(offsets) if b > char_start and a < char_end]


def extract_from_traces(model, tokenizer, traces, categories):
    """traces: list of dicts with keys prompt, completion, thinking, sentences,
    offsets, labels. Returns ({cat: tensor[n_layers, hidden]}, token counts)."""
    device = model.device
    n_layers = len(layers_of(model))
    hidden = model.config.hidden_size
    keys = list(categories) + [OVERALL]
    sums = {c: torch.zeros(n_layers, hidden, dtype=torch.float32, device=device) for c in keys}
    counts = {c: 0 for c in keys}

    model.eval()
    for tr in traces:
        full_text = tr["prompt"] + tr["completion"]
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        offsets = enc["offset_mapping"][0].tolist()
        thinking_start = len(tr["prompt"]) + tr["completion"].find(tr["thinking"])

        with torch.no_grad(), capture(model) as store:
            model(input_ids)
        stacked = torch.stack([store[i][0] for i in range(n_layers)], dim=0)  # [L, seq, H]

        all_pos = []
        for text, off, label in zip(tr["sentences"], tr["offsets"], tr["labels"]):
            cstart = thinking_start + off
            pos = token_positions(offsets, cstart, cstart + len(text))
            if not pos:
                continue
            all_pos.extend(pos)
            if label in sums:
                sums[label] += stacked[:, pos, :].sum(dim=1).float()
                counts[label] += len(pos)
        if all_pos:
            sums[OVERALL] += stacked[:, all_pos, :].sum(dim=1).float()
            counts[OVERALL] += len(all_pos)
        del stacked

    overall_mean = sums[OVERALL] / max(counts[OVERALL], 1)
    vectors = {}
    for c in categories:
        if counts[c] == 0:
            continue
        vectors[c] = (sums[c] / counts[c] - overall_mean).cpu()
    return vectors, counts
