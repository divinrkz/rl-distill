"""Step 4d — quantify the steering control (labeler, vLLM).

For each steered output, label its sentences and measure the fraction that are
the steered category. A working control shows that fraction rising with the
coefficient (dose-response).

    python scripts/04d_measure_control.py configs/steer.yaml
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import generate, load_llm
from rl_distill.labeling import build_label_prompt, parse_labels
from rl_distill.runio import load_config, save_json
from rl_distill.sentences import split_cot


def main(config_path):
    config = load_config(config_path)
    run_dir = Path(config["out_base"]) / config["run_name"]
    outputs = json.loads((run_dir / "control_outputs.json").read_text())
    fmt = load_formats(config["formats"])[config["target"]["format"]]

    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    problem_by_index = {i: ds[i]["problem"] for i in config["steer"]["eval_indices"]}

    for o in outputs:
        o["sentences"] = split_cot(o["text"], fmt)

    lab_llm = load_llm(config["labeler"]["model_name"], config["labeler"]["vllm"])
    lab_tok = lab_llm.get_tokenizer()
    prompts = [
        build_label_prompt(problem_by_index[o["index"]], o["sentences"] or ["(empty)"], lab_tok)
        for o in outputs
    ]
    raws = generate(lab_llm, prompts, config["labeler"]["generation"])

    fracs = defaultdict(list)
    for o, raw in zip(outputs, raws):
        labels = parse_labels(raw[0], len(o["sentences"]))
        frac = labels.count(o["category"]) / len(labels) if labels else 0.0
        o["target_fraction"] = frac
        fracs[(o["category"], o["coeff"])].append(frac)

    summary = {}
    print("steered-category sentence fraction (mean over eval problems):")
    for cat in config["steer"]["categories"]:
        print(f"  {cat}:")
        for coeff in config["steer"]["coeffs"]:
            vals = fracs[(cat, coeff)]
            mean = sum(vals) / len(vals) if vals else 0.0
            summary[f"{cat}@{coeff}"] = mean
            print(f"    coeff {coeff}: {mean:.3f}")

    save_json(run_dir / "control_measured.json", {"summary": summary, "outputs": outputs})
    print(f"wrote {run_dir / 'control_measured.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/steer.yaml")
