"""Step 2 — sentence-split inspection.

Splits two traces per model from the smoke results and prints numbered
sentences for eye inspection. CPU-only (no model loading).

    python scripts/02_split_check.py configs/smoke.yaml
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_distill.formats import load_formats
from rl_distill.runio import load_config
from rl_distill.sentences import split_cot

N_TRACES = 2


def main(config_path):
    config = load_config(config_path)
    formats = load_formats(config["formats"])
    run_dir = Path(config["out_base"]) / config["run_name"]

    for model_cfg in config["models"]:
        fmt = formats[model_cfg["format"]]
        results_path = run_dir / model_cfg["id"] / "results.json"
        if not results_path.exists():
            print(f"[skip] {results_path} not found")
            continue

        records = json.loads(results_path.read_text())
        print(f"\n{'=' * 70}\nMODEL: {model_cfg['id']}  (format: {fmt.name})\n{'=' * 70}")

        for r_i, record in enumerate(records[:N_TRACES]):
            sentences = split_cot(record["completion"], fmt)
            thinking_len = len(fmt.extract_thinking(record["completion"]))
            print(f"\n--- trace {r_i}  |  {len(sentences)} sentences  "
                  f"|  {thinking_len} thinking chars ---")
            for s_i, s in enumerate(sentences):
                print(f"[{s_i:02d}] {s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml")
