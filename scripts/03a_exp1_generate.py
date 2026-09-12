"""Step 3a — generation + importance (target model only).

Loads ONLY the target model, generates the base solution and resample rollouts,
grades them, computes resampling importance, and writes generation.json. The
labeler runs in a separate process (03b) so this process fully frees the GPU on
exit — vLLM does not reliably release its KV-cache reservation within a process.

    python scripts/03a_exp1_generate.py configs/exp1.yaml
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.grading import is_correct
from rl_distill.importance import accuracies, build_prefixes, sentence_importance
from rl_distill.runio import load_config, make_run_dir, save_config, save_json
from rl_distill.sentences import split_cot_with_offsets


def load_problem(cfg):
    ds = load_dataset(cfg["name"], split=cfg["split"])
    return ds[cfg["index"]]


def pick_base(completions, gold):
    for c in completions:
        if is_correct(c, gold):
            return c, True
    return completions[0], False


def main(config_path):
    config = load_config(config_path)
    run_dir = make_run_dir(config["out_base"], config["run_name"])
    save_config(run_dir, config)

    formats = load_formats(config["formats"])
    fmt = formats[config["target"]["format"]]
    problem = load_problem(config["dataset"])
    gold = problem["answer"]

    t0 = time.time()
    llm = load_llm(config["target"]["model_name"], config["vllm"])
    tok = llm.get_tokenizer()

    base_prompt = fmt.build_prompt(problem["problem"], tok)
    base_samples = generate(llm, [base_prompt], config["base_generation"])[0]
    base_text, base_correct = pick_base(base_samples, gold)

    thinking, offsets = split_cot_with_offsets(base_text, fmt)
    sentences = [s for s, _ in offsets]
    n = len(sentences)

    prefixes = build_prefixes(thinking, offsets)
    resample_prompts = [
        fmt.build_resample_prompt(problem["problem"], p, tok) for p in prefixes
    ]
    t_gen0 = time.time()
    rollouts = generate(llm, resample_prompts, config["resample"])
    t_gen1 = time.time()

    graded = [[is_correct(r, gold) for r in rs] for rs in rollouts]
    acc = accuracies(graded)
    importance = sentence_importance(acc)
    t_end = time.time()

    save_json(
        run_dir / "generation.json",
        {
            "problem": problem["problem"],
            "gold": gold,
            "target": config["target"]["id"],
            "base_correct": base_correct,
            "base_answer": fmt.extract_answer(base_text),
            "n_sentences": n,
            "sentences": sentences,
            "acc": acc,
            "importance": importance,
            "acc_full_prefix": acc[-1],
            "timing": {
                "total_s": round(t_end - t0, 1),
                "resample_gen_s": round(t_gen1 - t_gen0, 1),
                "n_resample_prompts": len(prefixes),
                "rollouts_per_prompt": config["resample"]["n"],
            },
        },
    )
    save_json(
        run_dir / "base.json",
        {"prompt": base_prompt, "completion": base_text, "thinking": thinking},
    )

    print(f"base_correct={base_correct}  n_sentences={n}")
    print(
        f"resample gen: {t_gen1 - t_gen0:.1f}s for {len(prefixes)} prefixes "
        f"x {config['resample']['n']} rollouts"
    )
    print(f"generation total: {t_end - t0:.1f}s")
    print(f"wrote {run_dir / 'generation.json'}  -> now run 03b to label")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp1.yaml")
