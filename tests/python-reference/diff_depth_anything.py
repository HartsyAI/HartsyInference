#!/usr/bin/env python3
"""Stage-by-stage diff of the C# Depth-Anything-V2 dump vs the official Python reference.

Run order:
    1. python dump_depth_anything.py --encoder vits --ckpt .../depth_anything_v2_vits.pth
    2. dotnet test ... --filter DepthAnythingParityTests   (writes Output/depth_anything_csharp_dump/<encoder>)
    3. python diff_depth_anything.py --encoder vits
"""
import argparse
import os

import numpy as np

STAGES = [
    "feat_0", "feat_1", "feat_2", "feat_3",
    "layer_1", "layer_2", "layer_3", "layer_4",
    "layer_1_rn", "layer_2_rn", "layer_3_rn", "layer_4_rn",
    "path_4", "path_3", "path_2", "path_1",
    "depth",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ref_dir = os.path.join(here, "depth_anything_reference_tensors", args.encoder)
    cs_dir = os.path.join(repo, "Output", "depth_anything_csharp_dump", args.encoder)

    print(f"{'stage':<12} {'avg err':>12} {'max err':>12} {'rel avg':>12}")
    worst = 0.0
    for stage in STAGES:
        rp = os.path.join(ref_dir, stage + ".bin")
        cp = os.path.join(cs_dir, stage + ".bin")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            print(f"{stage:<12} {'missing':>12}")
            continue
        ref = np.fromfile(rp, dtype=np.float32)
        cs = np.fromfile(cp, dtype=np.float32)
        if ref.size != cs.size:
            print(f"{stage:<12} SIZE MISMATCH ref={ref.size} cs={cs.size}")
            continue
        err = np.abs(ref - cs)
        rel = err.mean() / max(np.abs(ref).mean(), 1e-12)
        print(f"{stage:<12} {err.mean():>12.3e} {err.max():>12.3e} {rel:>12.3e}")
        worst = max(worst, err.mean())
    print(f"\nworst stage avg err: {worst:.3e} — target < 1e-3 on depth")


if __name__ == "__main__":
    main()
