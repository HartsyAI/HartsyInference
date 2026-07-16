#!/usr/bin/env python3
"""Stage-by-stage diff of the C# ControlNetHED dump vs the controlnet_aux reference.

Run order:
    1. python dump_hed.py --ckpt .../ControlNetHED.pth
    2. dotnet test ... --filter HedParityTests   (writes Output/hed_csharp_dump)
    3. python diff_hed.py
"""
import os

import numpy as np

STAGES = ["proj_1", "proj_2", "proj_3", "proj_4", "proj_5", "edge"]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ref_dir = os.path.join(here, "hed_reference_tensors")
    cs_dir = os.path.join(repo, "Output", "hed_csharp_dump")

    print(f"{'stage':<8} {'avg err':>12} {'max err':>12} {'rel avg':>12}")
    worst = 0.0
    for stage in STAGES:
        rp = os.path.join(ref_dir, stage + ".bin")
        cp = os.path.join(cs_dir, stage + ".bin")
        if not (os.path.exists(rp) and os.path.exists(cp)):
            print(f"{stage:<8} {'missing':>12}")
            continue
        ref = np.fromfile(rp, dtype=np.float32)
        cs = np.fromfile(cp, dtype=np.float32)
        if ref.size != cs.size:
            print(f"{stage:<8} SIZE MISMATCH ref={ref.size} cs={cs.size}")
            continue
        err = np.abs(ref - cs)
        rel = err.mean() / max(np.abs(ref).mean(), 1e-12)
        print(f"{stage:<8} {err.mean():>12.3e} {err.max():>12.3e} {rel:>12.3e}")
        worst = max(worst, err.mean())
    print(f"\nworst stage avg err: {worst:.3e} — target < 1e-3 on edge")


if __name__ == "__main__":
    main()
