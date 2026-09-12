"""Step 3b — labeling + category profile (labeler model only).

Reads generation.json from 03a, loads ONLY the labeler, assigns a category to
each sentence, computes the category profile, and writes result.json.

    python scripts/03b_exp1_label.py configs/exp1.yaml
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_distill.generate import generate, load_llm
from rl_distill.importance import category_profile
from rl_distill.labeling import build_label_prompt, parse_labels
from rl_distill.runio import load_config, save_json


def main(config_path):
    config = load_config(config_path)
    run_dir = Path(config["out_base"]) / config["run_name"]
    gen = json.loads((run_dir / "generation.json").read_text())

    sentences = gen["sentences"]
    importance = gen["importance"]
    n = len(sentences)

    t0 = time.time()
    lab_llm = load_llm(config["labeler"]["model_name"], config["labeler"]["vllm"])
    lab_tok = lab_llm.get_tokenizer()
    label_prompt = build_label_prompt(gen["problem"], sentences, lab_tok)
    label_raw = generate(lab_llm, [label_prompt], config["labeler"]["generation"])[0][0]
    labels = parse_labels(label_raw, n)
    t_end = time.time()

    profile, signed, counts = category_profile(labels, importance, config["categories"])

    per_sentence = [
        {
            "idx": i,
            "sentence": sentences[i],
            "category": labels[i],
            "importance": importance[i],
        }
        for i in range(n)
    ]
    result = {
        **{k: gen[k] for k in ("problem", "gold", "target", "base_correct", "n_sentences")},
        "category_profile": profile,
        "category_signed": signed,
        "category_counts": counts,
        "per_sentence": per_sentence,
        "labeler_raw": label_raw,
        "label_time_s": round(t_end - t0, 1),
    }
    save_json(run_dir / "result.json", result)

    print(f"labeled {n} sentences in {t_end - t0:.1f}s")
    print(f"unknown labels: {labels.count('unknown')}/{n}")
    print("category profile (importance share):")
    for c in config["categories"]:
        print(f"  {c:24s} {profile[c]:.3f}  (n={counts[c]})")
    print(f"wrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp1.yaml")
