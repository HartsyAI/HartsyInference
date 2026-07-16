#!/usr/bin/env python3
"""Stage-by-stage diff of the C# lineart Generator dump vs the controlnet_aux reference.

Run order:
    1. python dump_lineart.py --variant realistic --ckpt .../sk_model.pth
    2. dotnet test ... --filter LineartParityTests   (writes Output/lineart_csharp_dump/<variant>)
    3. python diff_lineart.py --variant realistic
"""
import argparse
import os

import numpy as np

STAGES = ["m0", "m1", "m2", "m3", "line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="realistic", choices=["realistic", "coarse"])
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ref_dir = os.path.join(here, "lineart_reference_tensors", args.variant)
    cs_dir = os.path.join(repo, "Output", "lineart_csharp_dump", args.variant)

    print(f"{'stage':<6} {'avg err':>12} {'max err':>12} {'rel avg':>12}")
    worst = 0.0
    for stage in STAGES:
        rp = os.path.join(ref_dir, stage + ".bin")
        cp = os.path.join(cs_dir, stage + ".bin")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            print(f"{stage:<6} {'missing':>12}")
            continue
        ref = np.fromfile(rp, dtype=np.float32)
        cs = np.fromfile(cp, dtype=np.float32)
        if ref.size != cs.size:
            print(f"{stage:<6} SIZE MISMATCH ref={ref.size} cs={cs.size}")
            continue
        err = np.abs(ref - cs)
        rel = err.mean() / max(np.abs(ref).mean(), 1e-12)
        print(f"{stage:<6} {err.mean():>12.3e} {err.max():>12.3e} {rel:>12.3e}")
        worst = max(worst, err.mean())
    print(f"\nworst stage avg err: {worst:.3e} — target < 1e-3 on line")


if __name__ == "__main__":
    main()
