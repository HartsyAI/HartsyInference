"""Per-probe comparison of two LongCat DiT probe dumps (diagnostics.longcatProbeDump), with the same file names: call{n}_probe_{name}.npy.

Prints the relative RMS of b against a for every probe name present in both, in the order the DiT emits them (embedders, then blocks in
index order, then each block's stages), and the first probe whose relative RMS exceeds --bound. Relative RMS = sqrt(mean((a - b)^2)) /
sqrt(mean(b^2)), with b the second directory. Bounds are fixed before the run; this script does not adjust them.

Usage:
  python compare_probes.py <a_dir> <b_dir> [--call 0] [--bound 1e-4]
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
    if a.shape != b.shape:
        return float("nan")
    denom = np.sqrt(np.mean(b * b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom) if denom > 0 else float("nan")


def order_key(name):
    # embedders and RoPE first; then block{i}.<stage>; the _k<n> repeats keep their place after the first pass
    match = re.match(r"block(\d+)\.(.+?)(?:_k(\d+))?$", name)
    if match:
        return (1, int(match.group(1)), int(match.group(3) or 0), match.group(2))
    return (0, 0, int(re.search(r"_k(\d+)$", name).group(1)) if re.search(r"_k(\d+)$", name) else 0, name)


def probe_names(directory, call):
    prefix = f"call{call:02d}_probe_"
    return {os.path.basename(p)[len(prefix):-len(".npy")] for p in glob.glob(os.path.join(directory, f"{prefix}*.npy"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir")
    ap.add_argument("b_dir")
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--bound", type=float, default=1e-4)
    args = ap.parse_args()

    common = sorted(probe_names(args.a_dir, args.call) & probe_names(args.b_dir, args.call), key=order_key)
    if not common:
        raise SystemExit(f"no common call{args.call:02d}_probe_*.npy in {args.a_dir} and {args.b_dir}")
    first_over = None
    print(f"{'probe':<40} {'relRMS':>12}")
    for name in common:
        stem = f"call{args.call:02d}_probe_{name}.npy"
        a = np.load(os.path.join(args.a_dir, stem))
        b = np.load(os.path.join(args.b_dir, stem))
        r = rel_rms(a, b)
        print(f"{name:<40} {r:>12.3e}")
        if first_over is None and not (r <= args.bound):
            first_over = (name, r)
    if first_over is None:
        print(f"RESULT: PASS (every probe within {args.bound:g}; {len(common)} probes)")
        return 0
    print(f"RESULT: FAIL (bound {args.bound:g}); first probe over: {first_over[0]} at {first_over[1]:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
