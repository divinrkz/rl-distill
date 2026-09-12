"""Random-direction control vectors with per-layer norms matched to real ones.

If a matched-norm random vector steers ORZ as much as the real vector, ORZ's
apparent response is nonspecific degradation, not transfer.

    python scripts/make_random_vectors.py experiments/steer_openthinker/vectors.pt experiments/random_vectors.pt
"""

import sys

import torch


def main(src, dst):
    real = torch.load(src)
    out = {}
    for cat, vec in real.items():
        norms = vec.norm(dim=1, keepdim=True)  # [n_layers, 1]
        r = torch.randn_like(vec)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8) * norms
        out[cat] = r
    torch.save(out, dst)
    print(f"wrote random matched-norm vectors {list(out.keys())} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
