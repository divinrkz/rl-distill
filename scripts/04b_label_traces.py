"""Step 4b — split and label trace sentences (labeler, vLLM).

Adds thinking/sentences/offsets/labels to each trace for extraction.

    python scripts/04b_label_traces.py configs/steer.yaml
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.labeling import build_label_prompt, parse_labels
from rl_distill.runio import load_config, save_json
from rl_distill.sentences import split_cot_with_offsets


def main(config_path):
    config = load_config(config_path)
    run_dir = Path(config["out_base"]) / config["run_name"]
    traces = json.loads((run_dir / "traces.json").read_text())
    fmt = load_formats(config["formats"])[config["target"]["format"]]

    split = []
    for tr in traces:
        thinking, offs = split_cot_with_offsets(tr["completion"], fmt)
        split.append(
            {**tr, "thinking": thinking, "sentences": [s for s, _ in offs],
             "offsets": [o for _, o in offs]}
        )

    lab_llm = load_llm(config["labeler"]["model_name"], config["labeler"]["vllm"])
    lab_tok = lab_llm.get_tokenizer()
    prompts = [build_label_prompt(s["problem"], s["sentences"], lab_tok) for s in split]
    raws = generate(lab_llm, prompts, config["labeler"]["generation"])

    tally = Counter()
    for s, raw in zip(split, raws):
        s["labels"] = parse_labels(raw[0], len(s["sentences"]))
        tally.update(s["labels"])

    save_json(run_dir / "traces_labeled.json", split)
    print(f"labeled {len(split)} traces -> {run_dir / 'traces_labeled.json'}")
    for cat in config["categories"] + ["unknown"]:
        print(f"  {cat:24s} {tally.get(cat, 0)} sentences")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/steer.yaml")
