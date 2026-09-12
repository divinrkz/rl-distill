"""Step 5c — aggregate per-model category profiles across problems (CPU).

Produces Figure 1 data: mean +/- std of the noise-thresholded importance profile
per category, per model, plus the raw (unthresholded) profile for comparison.

    python scripts/05c_exp1_aggregate.py configs/exp1_scale.yaml
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_distill.importance import aggregate_profiles, category_profile, threshold_importance
from rl_distill.runio import load_config, save_json


def model_profiles(run_dir, model_id, categories, threshold):
    out_dir = run_dir / model_id
    raw_profiles, thr_profiles, n_used = [], [], 0
    for f in sorted(out_dir.glob("problem_*.json")):
        r = json.loads(f.read_text())
        if not r.get("base_correct") or "labels" not in r or not r.get("importance"):
            continue
        n_used += 1
        labels, imp = r["labels"], r["importance"]
        raw_profiles.append(category_profile(labels, imp, categories)[0])
        thr = threshold_importance(imp, threshold)
        thr_profiles.append(category_profile(labels, thr, categories)[0])
    raw_mean, raw_std = aggregate_profiles(raw_profiles, categories)
    thr_mean, thr_std = aggregate_profiles(thr_profiles, categories)
    return {"n_problems": n_used, "raw_mean": raw_mean, "thresholded_mean": thr_mean,
            "thresholded_std": thr_std}


def main(config_path):
    config = load_config(config_path)
    run_dir = Path(config["out_base"]) / config["run_name"]
    categories = config["categories"]
    threshold = config["noise_threshold"]

    profiles = {}
    for m in config["models"]:
        profiles[m["id"]] = model_profiles(run_dir, m["id"], categories, threshold)

    save_json(run_dir / "profiles.json", profiles)

    ids = list(profiles)
    print(f"category profiles (thresholded importance share, |imp|>{threshold})")
    print(f"  n_problems: " + ", ".join(f"{i}={profiles[i]['n_problems']}" for i in ids))
    header = "  " + "category".ljust(24) + "  ".join(f"{i:>16}" for i in ids)
    print(header)
    for c in categories:
        cells = "  ".join(
            f"{profiles[i]['thresholded_mean'][c]:.3f}+/-{profiles[i]['thresholded_std'][c]:.3f}".rjust(16)
            for i in ids
        )
        print("  " + c.ljust(24) + cells)
    print(f"\nwrote {run_dir / 'profiles.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp1_scale.yaml")
