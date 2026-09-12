"""Step 4c — extract difference-of-means vectors and run the within-model control.

Loads the target via HF (for residual access), extracts a vector per steer
category, then generates steered vs unsteered on the eval problems across a
coefficient sweep. Outputs are labeled/measured in 04d.

    python scripts/04c_extract_and_steer.py configs/steer.yaml
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.hf import load_hf
from rl_distill.runio import load_config, save_json
from rl_distill.steering.apply import generate_steered, layer_vectors, resolve_layers
from rl_distill.steering.extract import extract_from_traces


def main(config_path):
    config = load_config(config_path)
    run_dir = Path(config["out_base"]) / config["run_name"]
    labeled = json.loads((run_dir / "traces_labeled.json").read_text())
    fmt = load_formats(config["formats"])[config["target"]["format"]]
    steer_cfg = config["steer"]

    model, tok = load_hf(config["target"]["model_name"], config["hf"]["dtype"])

    vectors_path = run_dir / "vectors.pt"
    if vectors_path.exists():
        vectors = torch.load(vectors_path)
        print(f"loaded cached vectors from {vectors_path}")
    else:
        vectors, counts = extract_from_traces(model, tok, labeled, steer_cfg["categories"])
        torch.save(vectors, vectors_path)
        print("extraction token counts:")
        for c in steer_cfg["categories"] + ["__overall__"]:
            print(f"  {c:24s} {counts.get(c, 0)}")
    for c, v in vectors.items():
        print(f"  |{c}| per-layer norm mean: {v.norm(dim=1).mean().item():.3f}")

    n_layers = model.config.num_hidden_layers
    layers = resolve_layers(steer_cfg["layers"], n_layers)

    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    eval_problems = [ds[i] for i in steer_cfg["eval_indices"]]

    total = len(eval_problems) * len(steer_cfg["categories"]) * len(steer_cfg["coeffs"])
    outputs = []
    done = 0
    for idx, prob in zip(steer_cfg["eval_indices"], eval_problems):
        prompt = fmt.build_prompt(prob["problem"], tok)
        for cat in steer_cfg["categories"]:
            if cat not in vectors:
                continue
            vbl = layer_vectors(vectors[cat], layers, model.device)
            for coeff in steer_cfg["coeffs"]:
                text = generate_steered(model, tok, prompt, steer_cfg["gen"], vbl, coeff)
                outputs.append(
                    {"index": idx, "category": cat, "coeff": coeff, "text": text}
                )
                done += 1
                print(f"  [{done}/{total}] idx={idx} {cat} coeff={coeff}", flush=True)

    save_json(run_dir / "control_outputs.json", outputs)
    print(f"wrote {len(outputs)} control outputs -> {run_dir / 'control_outputs.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/steer.yaml")
