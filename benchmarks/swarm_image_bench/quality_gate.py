#!/usr/bin/env python3
"""
Cheap content-quality gate for image-benchmark outputs.

HTTP-200 + a plausible wall-clock is NOT proof of a good image (see the audio
sweep's ACE-Step "mostly silence" lesson — content-gate every output, not just
the transport). This flags:
  - near-black / near-white (mean outside [lo, hi])
  - near-uniform (std below a floor -> flat color, no structure)
  - saturated noise (std pinned near max AND mean near mid-grey -> static)

Not a substitute for eyeballing images, just a cheap automatic tripwire.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def classify(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    per_channel_std = arr.reshape(-1, 3).std(axis=0)
    flags = []
    if mean < 8:
        flags.append("near-black")
    if mean > 247:
        flags.append("near-white")
    if std < 3:
        flags.append("near-uniform")
    # Random per-pixel noise loses most of its variance under block-averaging
    # (iid noise std drops ~sqrt(N) per NxN block); real photos keep most of
    # their variance at low frequency (large shapes, gradients, contrast).
    # A photo that's actually static/garbage keeps almost no variance after
    # a 16x16 box-downsample -> low downsampled/original std ratio.
    w, h = img.size
    small = img.resize((max(1, w // 16), max(1, h // 16)), Image.BOX)
    small_std = np.asarray(small, dtype=np.float64).std()
    variance_retained = (small_std / std) if std > 1e-6 else 1.0
    if std > 15 and variance_retained < 0.2:
        flags.append("possible-random-noise")
    return {"path": str(path), "mean": round(mean, 2), "std": round(std, 2),
            "per_channel_std": [round(x, 2) for x in per_channel_std],
            "variance_retained_16x_downsample": round(variance_retained, 3), "flags": flags}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="PNG/JPG paths to check")
    ap.add_argument("--out", help="write JSON report here")
    args = ap.parse_args()
    results = []
    for p in args.images:
        p = Path(p)
        if not p.exists():
            results.append({"path": str(p), "error": "file not found"})
            continue
        try:
            results.append(classify(p))
        except Exception as e:
            results.append({"path": str(p), "error": repr(e)})
    flagged = [r for r in results if r.get("flags")]
    errored = [r for r in results if r.get("error")]
    print(f"checked={len(results)} flagged={len(flagged)} errored={len(errored)}", file=sys.stderr)
    for r in flagged + errored:
        print(f"  {r['path']}: {r.get('flags', r.get('error'))}", file=sys.stderr)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
    sys.exit(1 if flagged or errored else 0)

if __name__ == "__main__":
    main()
