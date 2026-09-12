"""Step 4e — cross-model transfer.

Steer the target model (ORZ) with vectors extracted from another model
(OpenThinker), generating steered vs unsteered outputs. Measure with 04d.

    python scripts/04e_transfer.py configs/transfer.yaml
    python scripts/04d_measure_control.py configs/transfer.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.hf import load_hf
from rl_distill.runio import load_config, make_run_dir, save_config, save_json
from rl_distill.steering.apply import generate_steered, layer_vectors, resolve_layers


def main(config_path):
    config = load_config(config_path)
    run_dir = make_run_dir(config["out_base"], config["run_name"])
    save_config(run_dir, config)

    fmt = load_formats(config["formats"])[config["target"]["format"]]
    steer_cfg = config["steer"]
    vectors = torch.load(steer_cfg["vectors_from"])
    print(f"loaded source vectors: {list(vectors.keys())} from {steer_cfg['vectors_from']}")

    model, tok = load_hf(config["target"]["model_name"], config["hf"]["dtype"])
    n_layers = model.config.num_hidden_layers
    layers = resolve_layers(steer_cfg["layers"], n_layers)

    ds = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    eval_problems = [ds[i] for i in steer_cfg["eval_indices"]]

    total = len(eval_problems) * len(steer_cfg["categories"]) * len(steer_cfg["coeffs"])
    outputs, done = [], 0
    for idx, prob in zip(steer_cfg["eval_indices"], eval_problems):
        prompt = fmt.build_prompt(prob["problem"], tok)
        for cat in steer_cfg["categories"]:
            if cat not in vectors:
                continue
            vbl = layer_vectors(vectors[cat], layers, model.device)
            for coeff in steer_cfg["coeffs"]:
                text = generate_steered(model, tok, prompt, steer_cfg["gen"], vbl, coeff)
                outputs.append({"index": idx, "category": cat, "coeff": coeff, "text": text})
                done += 1
                print(f"  [{done}/{total}] idx={idx} {cat} coeff={coeff}", flush=True)

    save_json(run_dir / "control_outputs.json", outputs)
    print(f"wrote {len(outputs)} outputs -> {run_dir / 'control_outputs.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/transfer.yaml")
