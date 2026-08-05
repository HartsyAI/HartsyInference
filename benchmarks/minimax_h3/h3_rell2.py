#!/usr/bin/env python3
"""Relative L2 between two MiniMax-H3 latent dumps (HARTSY_H3_DUMP=<dir>).

Usage:  h3_rell2.py <dir_a> <dir_b>

WHY THIS AND NOT SSIM. End-to-end SSIM cannot grade a numerical change to a 30-step sampler: tiny
per-step differences pick a different trajectory and compound. Measured on this model, the production
fp8 path against an F32-GEMM gold scores SSIM 0.795 at 3 steps and 0.674 at 30 -- the extra loss is
divergence, not extra error. So SSIM is only meaningful COMPARATIVELY (variant A vs variant B, both
against the same reference), never as an absolute threshold.

Run both configs at --steps 1 and point this at their dumps. One step means one DiT forward, so the
final latent is a direct function of that forward and relL2 measures the numerical difference with no
compounding at all. That is the primary gate; keep SSIM as the end-to-end sanity check.
"""
import array
import math
import os
import sys


def load(path):
    with open(path, "rb") as f:
        a = array.array("f")
        a.frombytes(f.read())
    return a


def rel_l2(a, b):
    if len(a) != len(b):
        raise SystemExit(f"length mismatch: {len(a)} vs {len(b)}")
    num = 0.0
    den = 0.0
    max_abs = 0.0
    nonfinite = 0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            nonfinite += 1
            continue
        d = x - y
        num += d * d
        den += y * y
        if abs(d) > max_abs:
            max_abs = abs(d)
    return math.sqrt(num / den) if den > 0 else float("nan"), max_abs, nonfinite


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    da, db = sys.argv[1], sys.argv[2]
    names = ["video_latent_final.bin", "audio_latent_final.bin"]
    print(f"{'tensor':22s} {'relL2':>12s} {'max_abs':>12s} {'n':>10s}  nonfinite")
    worst = 0.0
    for n in names:
        pa, pb = os.path.join(da, n), os.path.join(db, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{n:22s} {'MISSING':>12s}")
            continue
        a, b = load(pa), load(pb)
        r, m, nf = rel_l2(a, b)
        worst = max(worst, r)
        print(f"{n:22s} {r:12.3e} {m:12.3e} {len(a):10d}  {nf}")
    print(f"\nworst relL2 {worst:.3e}")
    # Anchors from the repo's own parity tolerances (MiniMaxH3BackendParityTests): f32 5e-3, bf16 1.5e-2.
    print("within f32 parity tolerance (5e-3)" if worst < 5e-3 else
          "within bf16 parity tolerance (1.5e-2)" if worst < 1.5e-2 else
          f"ABOVE both parity tolerances")


if __name__ == "__main__":
    main()
