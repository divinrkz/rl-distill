"""Step 5 — select problems both models solve at intermediate difficulty (CPU).

Intersect two probe outputs (probe_difficulty.py --save) and keep indices where
both models' pass rate is in [lo, hi].

    python scripts/05_select.py experiments/exp1_scale/probe_orz.json experiments/exp1_scale/probe_openthinker.json --out experiments/exp1_scale/selected.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_files", nargs="+")
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rates = [json.load(open(f)) for f in args.probe_files]
    common = set(rates[0])
    for r in rates[1:]:
        common &= set(r)
    selected = sorted(
        int(i) for i in common if all(args.lo <= r[i] <= args.hi for r in rates)
    )
    print(f"both-solvable, intermediate (in [{args.lo},{args.hi}]): {selected}")
    print(f"({len(selected)} of {len(common)} shared problems)")
    if args.out:
        json.dump(selected, open(args.out, "w"))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
