"""Per-step comparison of two LongCat dumps (engine or reference), with the same file names: call{n:02d}_step{i:02d}_{t,noise,latent}.npy.

Engine _noise holds the noisy frames only; the reference _noise holds every frame. The noisy frames are the LAST frames of the
reference, so the reference is sliced to the engine's frame count from the end. Latents hold every frame on both sides and are
compared whole. Relative RMS = sqrt(mean((a - b)^2)) / sqrt(mean(b^2)), with b the reference (or the second run).

Modes:
  python compare_steps.py <a_dir> <b_dir> [--bound 2e-2] [--call 0]     per-step verdict against the bound (exit 1 on FAIL)
  python compare_steps.py <a_dir> <b_dir> --frames                      per-frame PSNR of PNG sequences frame_NNN.png

Bounds are fixed before the run (see the spec); this script does not adjust them.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np


def rel_rms(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.sqrt(np.mean(b * b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom) if denom > 0 else float("nan")


def steps_in(directory, call):
    pattern = os.path.join(directory, f"call{call:02d}_step*_latent.npy")
    return sorted(int(re.search(r"_step(\d+)_latent", p).group(1)) for p in glob.glob(pattern))


def compare_dumps(a_dir, b_dir, call, bound):
    steps = [s for s in steps_in(a_dir, call) if s in set(steps_in(b_dir, call))]
    if not steps:
        raise SystemExit(f"no common steps for call {call} in {a_dir} and {b_dir}")
    failed = []
    print(f"{'step':>4} {'t_a':>10} {'t_b':>10} {'noise relRMS':>13} {'latent relRMS':>14}")
    for s in steps:
        def load(d, name):
            return np.load(os.path.join(d, f"call{call:02d}_step{s:02d}_{name}.npy"))
        t_a = float(load(a_dir, "t"))
        t_b = float(load(b_dir, "t"))
        na, nb = load(a_dir, "noise"), load(b_dir, "noise")
        if nb.shape[2] != na.shape[2]:
            nb = nb[:, :, nb.shape[2] - na.shape[2]:]
        la, lb = load(a_dir, "latent"), load(b_dir, "latent")
        if la.shape != lb.shape:
            raise SystemExit(f"step {s}: latent shapes differ {la.shape} vs {lb.shape}")
        rn, rl = rel_rms(na, nb), rel_rms(la, lb)
        print(f"{s:>4} {t_a:>10.3f} {t_b:>10.3f} {rn:>13.3e} {rl:>14.3e}")
        if t_a != t_b:
            failed.append((s, "timestep differs"))
        if not (rn <= bound) or not (rl <= bound):
            failed.append((s, f"noise {rn:.3e} latent {rl:.3e}"))
    if failed:
        print(f"RESULT: FAIL (bound {bound:g}); first failing step {failed[0][0]}: {failed[0][1]}")
        return 1
    print(f"RESULT: PASS (every step within bound {bound:g}; {len(steps)} steps)")
    return 0


def compare_frames(a_dir, b_dir):
    from PIL import Image

    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a_dir, "frame_*.png")))
    names = [n for n in names if os.path.exists(os.path.join(b_dir, n))]
    if not names:
        raise SystemExit("no common frame_*.png")
    psnrs = []
    for n in names:
        a = np.asarray(Image.open(os.path.join(a_dir, n)).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(os.path.join(b_dir, n)).convert("RGB"), dtype=np.float64)
        mse = np.mean((a - b) ** 2)
        psnrs.append(float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        print(f"{n} psnr={psnrs[-1]:.2f} dB")
    finite = [p for p in psnrs if np.isfinite(p)]
    print(f"frames={len(names)} mean PSNR={np.mean(finite) if finite else float('inf'):.2f} dB min={min(psnrs):.2f} dB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir")
    ap.add_argument("b_dir")
    ap.add_argument("--bound", type=float, default=2e-2)
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--frames", action="store_true")
    args = ap.parse_args()
    if args.frames:
        return compare_frames(args.a_dir, args.b_dir)
    return compare_dumps(args.a_dir, args.b_dir, args.call, args.bound)


if __name__ == "__main__":
    sys.exit(main())
