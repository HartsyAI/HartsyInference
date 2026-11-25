"""Compares two Depth Pro 2 runs: a CLI output folder or a reference dump (dump_depth_pro2_sizes.py), in any pairing.

A CLI folder holds one ``*-vision-NNN`` subfolder with ``depth.npy`` (one image) or ``view-NN.depth.npy`` / ``frame-NNNNN.depth.npy`` plus the
matching ``*.result.json`` (camera.fx is the focal length in pixels, camera.width the image width). A reference dump holds ``<name>_depth.npy``,
``<name>_f_px.npy`` for ``--name`` image, views, buffered or unbuffered. Metric depth is ``f_px / (W * inverse_depth)``, so the comparison is on
inverse depth (infinite depth is 0), which is what the network predicts and what metres amplify near the sky; metric depth over the pixels that
are finite in both runs and the focal length are reported next to it. Pure numpy, host Python is fine.

  python depth_pro2_compare.py --a <cli run dir> --b <cli run dir or ref dump> [--name image]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(depth [N,H,W] metres, f_px [N])."""
    ref = path / f"{name}_depth.npy"
    if ref.exists():
        depth = np.load(ref).astype(np.float64)
        f = np.load(path / f"{name}_f_px.npy").astype(np.float64)
        return depth, np.repeat(f, depth.shape[0])
    run = sorted(p for p in path.iterdir() if p.is_dir() and "-vision-" in p.name)
    if len(run) != 1:
        raise SystemExit(f"{path}: expected one *-vision-* folder, found {len(run)}")
    run = run[0]
    single = run / "depth.npy"
    if single.exists():
        files = [(single, run / "result.json")]
    else:
        files = [(p, p.with_name(p.name.replace(".depth.npy", ".result.json"))) for p in sorted(run.glob("*.depth.npy"))]
    depth = np.stack([np.load(d).astype(np.float64) for d, _ in files])
    f = np.array([json.loads(r.read_text())["camera"]["fx"] for _, r in files])
    return depth, f


def inverse(depth: np.ndarray, f: np.ndarray) -> np.ndarray:
    width = depth.shape[-1]
    far = np.isinf(depth) | (depth <= 0)
    return np.where(far, 0.0, f.reshape(-1, 1, 1) / (width * np.where(far, 1.0, depth)))


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path, help="the run under test")
    parser.add_argument("--b", required=True, type=Path, help="the lane it is compared against")
    parser.add_argument("--name", default="image")
    args = parser.parse_args()
    da, fa = load(args.a, args.name)
    db, fb = load(args.b, args.name)
    if da.shape != db.shape:
        raise SystemExit(f"shape {da.shape} vs {db.shape}")
    ia, ib = inverse(da, fa), inverse(db, fb)
    finite = np.isfinite(da) & np.isfinite(db)
    per_frame = [rel_rms(ia[i], ib[i]) for i in range(ia.shape[0])]
    pixel = np.abs(ia - ib) / np.maximum(np.abs(ib), 1e-12)
    result = {
        "frames": int(ia.shape[0]),
        "inverse_depth_relrms": rel_rms(ia, ib),
        "inverse_depth_relrms_worst_frame": max(per_frame),
        "inverse_depth_per_pixel_median": float(np.median(pixel[ib > 0])),
        "depth_relrms_finite": rel_rms(da[finite], db[finite]),
        "f_px_rel_max": float(np.max(np.abs(fa - fb) / fb)),
        "f_px_b": float(fb[0]),
        "infinite_pixels_a": int(np.isinf(da).sum()),
        "infinite_pixels_b": int(np.isinf(db).sum()),
        "infinite_mismatch": int((np.isinf(da) != np.isinf(db)).sum()),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
