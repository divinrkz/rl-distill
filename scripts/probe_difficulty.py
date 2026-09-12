"""Find Exp 1 problems with a useful difficulty for the target model.

Resampling importance is only informative when the from-scratch pass rate is
intermediate (not ~0 or ~1). This samples base solutions for a set of problems
and prints pass rates, so you can pick an index before paying for resampling.

    python scripts/probe_difficulty.py configs/exp1.yaml --level 5 --n-problems 8 --samples 16
    python scripts/probe_difficulty.py configs/exp1.yaml --indices 3 17 42 --samples 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.grading import is_correct
from rl_distill.runio import load_config


def pick_indices(ds, args):
    if args.indices:
        return args.indices
    idxs = [i for i in range(len(ds)) if str(ds[i].get("level", "")).endswith(str(args.level))]
    return idxs[: args.n_problems]


def get_target(config, model_id):
    if model_id and "models" in config:
        for m in config["models"]:
            if m["id"] == model_id:
                return m
    return config["target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model-id", default=None, help="pick from config['models']")
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--save", default=None, help="write {idx: pass_rate} json")
    args = ap.parse_args()

    config = load_config(args.config)
    target = get_target(config, args.model_id)
    fmt = load_formats(config["formats"])[target["format"]]
    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])

    indices = pick_indices(ds, args)
    problems = [ds[i] for i in indices]

    llm = load_llm(target["model_name"], config["vllm"])
    tok = llm.get_tokenizer()
    prompts = [fmt.build_prompt(p["problem"], tok) for p in problems]

    gen_cfg = dict(config["base_generation"])
    gen_cfg["n"] = args.samples
    outs = generate(llm, prompts, gen_cfg)

    rows = []
    for idx, prob, comps in zip(indices, problems, outs):
        n_ok = sum(is_correct(c, prob["answer"]) for c in comps)
        rows.append((idx, prob.get("level", "?"), n_ok / len(comps)))

    print(f"{'idx':>5}  {'level':<8}  pass_rate  (n={args.samples})")
    for idx, level, rate in sorted(rows, key=lambda r: abs(r[2] - 0.5)):
        flag = "  <- good" if 0.0 < rate < 1.0 else ""
        print(f"{idx:>5}  {str(level):<8}  {rate:>6.2f}{flag}")

    if args.save:
        import json

        with open(args.save, "w") as f:
            json.dump({int(idx): rate for idx, _, rate in rows}, f, indent=2)
        print(f"\nwrote pass rates -> {args.save}")
    print("\nPick indices with intermediate pass rate for BOTH models (see 05_select).")


if __name__ == "__main__":
    main()
