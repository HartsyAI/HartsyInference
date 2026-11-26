"""Compare two 8-bit alpha PNGs (same source size) and print the parity metrics.

Used for all three checks:
  (1) engine alpha vs diffusers reference alpha (manifest.json lists the files)
  (2) engine CPU alpha vs engine CUDA alpha, same input and noise
  (3) rmbg3-int8 alpha vs rmbg3 bf16 alpha (informational: quantisation loss)

Usage:
  python compare_alpha.py A.alpha.png B.alpha.png [--json]

Metrics
  max_abs_levels   largest |A-B| in 8-bit levels (0..255)
  mean_abs_levels  mean |A-B| in 8-bit levels
  pearson          correlation of the two alphas (1.0 = identical up to scale)

Proposed pass lines (NOT yet validated on real weights; set by the parity owner before the run):
  diffusers vs engine (check 1):  pearson >= 0.999, mean_abs <= 1.0, max_abs <= 8
  CPU vs CUDA (check 2):          pearson >= 0.999, mean_abs <= 1.0, max_abs <= 8
  int8 vs bf16 (check 3):         reported only; the 12 GB check is the VRAM line, not this one
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image


def metrics(path_a: str, path_b: str) -> dict:
    a = Image.open(path_a)
    b = Image.open(path_b)
    if a.size != b.size:
        sys.exit(f"size mismatch: {a.size} vs {b.size}")
    if a.mode != "L" or b.mode != "L":
        sys.exit(f"both must be 8-bit grey (mode L); got {a.mode} and {b.mode}")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    diff = np.abs(x - y)
    corr = float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "a": path_a,
        "b": path_b,
        "size": list(a.size),
        "max_abs_levels": float(diff.max()),
        "mean_abs_levels": float(diff.mean()),
        "pearson": corr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    m = metrics(args.a, args.b)
    if args.json:
        print(json.dumps(m))
    else:
        print(
            f"max_abs={m['max_abs_levels']:.0f} mean_abs={m['mean_abs_levels']:.3f} "
            f"pearson={m['pearson']:.6f} size={m['size'][0]}x{m['size'][1]}"
        )


if __name__ == "__main__":
    main()
