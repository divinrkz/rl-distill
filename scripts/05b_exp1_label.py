"""Step 5b — label all problem sentences for one model (labeler, batched).

    python scripts/05b_exp1_label.py configs/exp1_scale.yaml --model-id orz
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.generate import generate, load_llm
from rl_distill.labeling import build_label_prompt, parse_labels
from rl_distill.runio import load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["out_base"]) / config["run_name"] / args.model_id
    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])

    files = sorted(out_dir.glob("problem_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    records = [json.loads(f.read_text()) for f in files]
    todo = [(f, r) for f, r in zip(files, records)
            if r["base_correct"] and r["sentences"] and "labels" not in r]
    print(f"{args.model_id}: {len(todo)} problems to label")
    if not todo:
        return

    lab_llm = load_llm(config["labeler"]["model_name"], config["labeler"]["vllm"])
    lab_tok = lab_llm.get_tokenizer()
    prompts = [
        build_label_prompt(ds[r["index"]]["problem"], r["sentences"], lab_tok)
        for _, r in todo
    ]
    raws = generate(lab_llm, prompts, config["labeler"]["generation"])

    for (f, r), raw in zip(todo, raws):
        r["labels"] = parse_labels(raw[0], len(r["sentences"]))
        save_json(f, r)
        print(f"  idx={r['index']} labeled {len(r['sentences'])} "
              f"(unknown {r['labels'].count('unknown')})", flush=True)


if __name__ == "__main__":
    main()
