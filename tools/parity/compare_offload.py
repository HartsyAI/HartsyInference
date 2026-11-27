"""Compares two reference dumps (io.safetensors) made with and without block offload.

Bound, fixed before the run: the relative RMS of the latents and of the pixels must each be at most ``BOUND``
(1e-6). The offload only moves weights between host and device, so the arithmetic should be identical; the
bound allows a last-bit difference from a different allocation and nothing more.

Usage: python compare_offload.py <io-off.safetensors> <io-on.safetensors>
"""
import sys

import numpy as np
from safetensors.numpy import load_file

BOUND = 1e-6


def rel_rms(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.sqrt(np.sum((a - b) ** 2) / max(np.sum(b ** 2), 1e-30)))


def main(off_path, on_path):
    off = load_file(off_path)
    on = load_file(on_path)
    worst = {}
    for key in ("latents", "pixels", "features", "modulation"):
        a, b = off[key], on[key]
        if a.shape != b.shape:
            print(f"{key}: shape differs {a.shape} vs {b.shape}")
            sys.exit(1)
        frames = a.shape[0]
        per = [rel_rms(a[i], b[i]) for i in range(frames)]
        worst[key] = max(per)
        print(f"{key}: worst relative RMS {worst[key]:.3e} over {frames} (per frame: {', '.join(f'{x:.3e}' for x in per)}); max abs diff {float(np.max(np.abs(a - b))):.3e}")
    ok = worst["latents"] <= BOUND and worst["pixels"] <= BOUND
    print(f"bound {BOUND:.0e} on latents and pixels: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
