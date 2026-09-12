"""Step 4a — generate reasoning traces for vector extraction (target, vLLM).

    python scripts/04a_gen_traces.py configs/steer.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.runio import load_config, make_run_dir, save_config, save_json


def main(config_path):
    config = load_config(config_path)
    run_dir = make_run_dir(config["out_base"], config["run_name"])
    save_config(run_dir, config)

    fmt = load_formats(config["formats"])[config["target"]["format"]]
    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    problems = [ds[i] for i in config["trace_indices"]]

    llm = load_llm(config["target"]["model_name"], config["vllm"])
    tok = llm.get_tokenizer()
    prompts = [fmt.build_prompt(p["problem"], tok) for p in problems]
    completions = generate(llm, prompts, config["trace_generation"])

    traces = [
        {"index": i, "problem": p["problem"], "prompt": pr, "completion": c[0]}
        for i, p, pr, c in zip(config["trace_indices"], problems, prompts, completions)
    ]
    save_json(run_dir / "traces.json", traces)
    print(f"wrote {len(traces)} traces -> {run_dir / 'traces.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/steer.yaml")
