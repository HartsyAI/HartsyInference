"""Compares two Depth Pro 2 runs: a CLI output folder or a reference dump (dump_depth_pro2_sizes.py), in any pairing.

A CLI folder holds one ``*-vision-NNN`` subfolder with ``depth.npy`` (one image) or ``view-NN.depth.npy`` / ``frame-NNNNN.depth.npy`` plus the
matching ``*.result.json`` (camera.fx is the focal length in pixels, camera.width the image width). A reference dump holds ``<name>_depth.npy``,
``<name>_f_px.npy`` for ``--name`` image, views, buffered or unbuffered. Metric depth is ``f_px / (W * inverse_depth)``, so the comparison is on
inverse depth (infinite depth is 0), which is what the network predicts and what metres amplify near the sky; metric depth over the pixels that
are finite and above zero in both runs, and the focal length, are reported next to it. Any other depth (+inf, NaN, zero or below) is far;
``far_mismatch``, the NaN counts and ``frames_far_in_b`` show what the comparison skipped. A figure with nothing to compare is null. That metre
figure is dominated by the farthest pixels (a depth is a reciprocal, so a fixed inverse-depth difference grows with the square of the distance);
``--bins`` shows where it comes from: the error by the reference's depth, under distance cutoffs, and at the worst pixel. Pure numpy, host Python
is fine.

  python depth_pro2_compare.py --a <cli run dir> --b <cli run dir or ref dump> [--name image] [--bins]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BIN_EDGES_M = (0, 10, 30, 100, 300, 1000, 10000, np.inf)
CUTOFFS_M = (80, 250, 1000)


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


def far(depth: np.ndarray) -> np.ndarray:
    """Pixels with no usable depth: infinitely far, NaN, or not above zero."""
    return ~np.isfinite(depth) | (depth <= 0)


def inverse(depth: np.ndarray, f: np.ndarray) -> np.ndarray:
    width = depth.shape[-1]
    skipped = far(depth)
    return np.where(skipped, 0.0, f.reshape(-1, 1, 1) / (width * np.where(skipped, 1.0, depth)))


def rel_rms(a: np.ndarray, b: np.ndarray) -> float | None:
    """None when ``b`` is empty or all zero (a frame that is far everywhere)."""
    norm = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / norm if norm else None


def compared(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    """The pixels whose metric depth can be compared: far in neither run."""
    return ~(far(da) | far(db))


def by_depth(da: np.ndarray, db: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict:
    """Where the metric-depth difference sits, binned by the depth of ``db``. Shares are of the compared pixels (see ``compared``)."""
    mask = compared(da, db)
    if not mask.any():
        return {"depth_max_b_m": None, "depth_bins": [], "depth_under": [], "worst_pixel": None, "inverse_abs_diff_max": None,
                "inverse_b_median": None}
    a, b = da[mask], db[mask]
    error = np.abs(a - b)
    relative = error / b
    total = float(np.sum(error ** 2)) or 1.0
    bins = []
    for low, high in zip(BIN_EDGES_M[:-1], BIN_EDGES_M[1:]):
        inside = (b >= low) & (b < high)
        if inside.any():
            bins.append({"from_m": low, "to_m": None if np.isinf(high) else high, "pixels": int(inside.sum()),
                         "max_rel": float(relative[inside].max()), "max_abs_m": float(error[inside].max()),
                         "share_of_squared_error": float(np.sum(error[inside] ** 2) / total)})
    under = []
    for cutoff in CUTOFFS_M:
        inside = b < cutoff
        if inside.any():
            under.append({"under_m": cutoff, "share_of_pixels": float(inside.mean()), "relrms": rel_rms(a[inside], b[inside]),
                          "max_rel": float(relative[inside].max())})
    worst = int(np.argmax(relative))
    inverse_diff = np.abs(ia - ib)[mask]
    return {"depth_max_b_m": float(b.max()), "depth_bins": bins, "depth_under": under,
            "worst_pixel": {"a_m": float(a[worst]), "b_m": float(b[worst]), "rel": float(relative[worst]),
                            "inverse_abs_diff": float(inverse_diff[worst])},
            "inverse_abs_diff_max": float(inverse_diff.max()), "inverse_b_median": float(np.median(ib[mask]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=Path, help="the run under test")
    parser.add_argument("--b", required=True, type=Path, help="the lane it is compared against")
    parser.add_argument("--name", default="image")
    parser.add_argument("--bins", action="store_true", help="also report the metric-depth error by distance")
    args = parser.parse_args()
    da, fa = load(args.a, args.name)
    db, fb = load(args.b, args.name)
    if da.shape != db.shape:
        raise SystemExit(f"shape {da.shape} vs {db.shape}")
    ia, ib = inverse(da, fa), inverse(db, fb)
    mask = compared(da, db)
    per_frame = [error for error in (rel_rms(ia[i], ib[i]) for i in range(ia.shape[0])) if error is not None]
    pixel = np.abs(ia - ib) / np.maximum(np.abs(ib), 1e-12)
    result = {
        "frames": int(ia.shape[0]),
        "inverse_depth_relrms": rel_rms(ia, ib),
        "inverse_depth_relrms_worst_frame": max(per_frame, default=None),
        "inverse_depth_per_pixel_median": float(np.median(pixel[ib > 0])) if (ib > 0).any() else None,
        "depth_relrms_finite": rel_rms(da[mask], db[mask]),
        "f_px_rel_max": float(np.max(np.abs(fa - fb) / fb)),
        "f_px_b": float(fb[0]),
        "infinite_pixels_a": int(np.isinf(da).sum()),
        "infinite_pixels_b": int(np.isinf(db).sum()),
        "infinite_mismatch": int((np.isinf(da) != np.isinf(db)).sum()),
        "far_mismatch": int((far(da) != far(db)).sum()),
        "nan_pixels_a": int(np.isnan(da).sum()),
        "nan_pixels_b": int(np.isnan(db).sum()),
        "frames_far_in_b": int(ia.shape[0]) - len(per_frame),
    }
    if args.bins:
        result.update(by_depth(da, db, ia, ib))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
