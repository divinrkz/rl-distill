"""Step 5a — Exp 1 generation across problems (one model, resident, resume-safe).

Loads one model, loops the selected problems doing base + resample + importance,
saves one file per problem so a run can resume after interruption.

    python scripts/05a_exp1_generate.py configs/exp1_scale.yaml --model-id orz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.grading import is_correct
from rl_distill.importance import (
    accuracies,
    build_prefixes,
    select_boundaries,
    sentence_importance,
)
from rl_distill.runio import load_config, make_run_dir, save_config, save_json
from rl_distill.sentences import split_cot_with_offsets


def get_model(config, model_id):
    for m in config["models"]:
        if m["id"] == model_id:
            return m
    raise SystemExit(f"model-id {model_id} not in config['models']")


def pick_base(completions, gold):
    for c in completions:
        if is_correct(c, gold):
            return c, True
    return completions[0], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    run_dir = make_run_dir(config["out_base"], config["run_name"])
    save_config(run_dir, config)
    model_cfg = get_model(config, args.model_id)
    fmt = load_formats(config["formats"])[model_cfg["format"]]
    out_dir = run_dir / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    indices = config["indices"]
    todo = [i for i in indices if not (out_dir / f"problem_{i}.json").exists()]
    print(f"{args.model_id}: {len(indices)} problems, {len(todo)} to do")
    if not todo:
        return

    llm = load_llm(model_cfg["model_name"], config["vllm"])
    tok = llm.get_tokenizer()

    for i in todo:
        problem = ds[i]
        gold = problem["answer"]
        base_prompt = fmt.build_prompt(problem["problem"], tok)
        base_samples = generate(llm, [base_prompt], config["base_generation"])[0]
        base_text, base_correct = pick_base(base_samples, gold)

        thinking, offsets = split_cot_with_offsets(base_text, fmt)
        n_full = len(offsets)
        offsets = select_boundaries(offsets, config.get("max_prefixes"))
        sentences = [s for s, _ in offsets]

        record = {"index": i, "base_correct": base_correct, "n_sentences": len(sentences),
                  "n_sentences_full": n_full, "sentences": sentences}
        if base_correct and sentences:
            prefixes = build_prefixes(thinking, offsets)
            prompts = [fmt.build_resample_prompt(problem["problem"], p, tok) for p in prefixes]
            rollouts = generate(llm, prompts, config["resample"])
            graded = [[is_correct(r, gold) for r in rs] for rs in rollouts]
            acc = accuracies(graded)
            record["acc"] = acc
            record["importance"] = sentence_importance(acc)

        save_json(out_dir / f"problem_{i}.json", record)
        print(f"  idx={i} base_correct={base_correct} n_sentences={len(sentences)}", flush=True)


if __name__ == "__main__":
    main()
