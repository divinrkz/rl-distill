"""Build the two write-up figures.

Figure 1: Exp 1 category profiles (ORZ vs OpenThinker), thresholded importance.
Figure 2: Exp 2 steering dose-response (within-model control, cross-model
transfer, random-direction control) for backtracking and uncertainty.

    python scripts/06_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path("experiments")
OUT = Path("figures")


def figure1():
    data = json.loads((EXP / "exp1_scale" / "profiles.json").read_text())
    models = list(data)
    cats = list(next(iter(data.values()))["thresholded_mean"])
    x = range(len(cats))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, m in enumerate(models):
        means = [data[m]["thresholded_mean"][c] for c in cats]
        stds = [data[m]["thresholded_std"][c] for c in cats]
        ax.bar([i + k * width for i in x], means, width, yerr=stds, capsize=3, label=m)
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in x])
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("importance share (|imp| > threshold)")
    ax.set_title("Exp 1: causal importance by reasoning category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure1_profiles.png", dpi=150)
    print("wrote figures/figure1_profiles.png")


def _curve(run, category):
    path = EXP / run / "control_measured.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())["summary"]
    pts = sorted(
        (float(k.split("@")[1]), v) for k, v in summary.items() if k.startswith(category + "@")
    )
    return [c for c, _ in pts], [v for _, v in pts]


def figure2():
    runs = [
        ("steer_openthinker", "OpenThinker (within-model)"),
        ("transfer_ot_to_orz", "ORZ (OT vector, transfer)"),
        ("transfer_random_to_orz", "ORZ (random control)"),
    ]
    cats = ["backtracking", "uncertainty-estimation"]
    fig, axes = plt.subplots(1, len(cats), figsize=(11, 4.5), sharey=True)
    for ax, cat in zip(axes, cats):
        for run, label in runs:
            cur = _curve(run, cat)
            if cur:
                ax.plot(cur[0], cur[1], marker="o", label=label)
        ax.set_title(cat)
        ax.set_xlabel("steering coefficient")
    axes[0].set_ylabel("steered-category sentence fraction")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Exp 2: steering dose-response")
    fig.tight_layout()
    fig.savefig(OUT / "figure2_steering.png", dpi=150)
    print("wrote figures/figure2_steering.png")


def main():
    OUT.mkdir(exist_ok=True)
    figure1()
    figure2()


if __name__ == "__main__":
    sys.exit(main())
