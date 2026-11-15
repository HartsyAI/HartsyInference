#!/usr/bin/env python3
"""Compare the engine's per-layer residual stream (ERNIE_DUMP_LAYERS) with reference.py's .npy output.

For every decoder layer present in both folders, it reports over all tokens and over the last token:
max absolute difference, relative error ||a - b|| / ||b||, and cosine similarity. It flags the first layer
whose cosine falls below --threshold (default 0.999). With --strict it exits 1 when any layer is flagged
or a layer is missing, so a pod run fails on a divergence.

Engine files are named layer{LL}_pos{startPos}_t{tokens}_hc{hc}_h{hidden}.f32 (little-endian float32,
row-major [tokens, hc, hidden]). Only the prefill dump (startPos 0) is compared; the reference is one prefill.

Usage:
  python compare.py --ref DIR --engine DIR [--threshold 0.999] [--strict]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ENGINE_RE = re.compile(r"^layer(\d{2})_pos(\d+)_t(\d+)_hc(\d+)_h(\d+)\.f32$")
REF_RE = re.compile(r"^layer_(\d{2})_all\.npy$")


def load_engine(folder: Path) -> dict[tuple[int, int], tuple[Path, int, int, int, int]]:
    """(layer, startPos) -> (path, tokens, hc, hidden)."""
    out: dict[tuple[int, int], tuple[Path, int, int, int, int]] = {}
    for p in folder.iterdir():
        m = ENGINE_RE.match(p.name)
        if m:
            layer, pos, tokens, hc, hidden = (int(g) for g in m.groups())
            out[(layer, pos)] = (p, tokens, hc, hidden)
    return out


def load_ref(folder: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in folder.iterdir():
        m = REF_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """max |a - b|, relative error, cosine similarity (float64)."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = a - b
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = float(np.linalg.norm(b))
    rel = float(np.linalg.norm(diff)) / denom if denom > 0 else float("inf")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cos = float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return max_abs, rel, cos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference.py output folder")
    p.add_argument("--engine", required=True, help="ERNIE_DUMP_LAYERS folder")
    p.add_argument("--threshold", type=float, default=0.999)
    p.add_argument("--strict", action="store_true", help="exit 1 on a flagged or missing layer")
    args = p.parse_args()

    ref = load_ref(Path(args.ref))
    eng = load_engine(Path(args.engine))
    if not ref:
        print(f"no layer_LL_all.npy files in {args.ref}", file=sys.stderr)
        return 1
    if not any(pos == 0 for (_, pos) in eng):
        print(f"no prefill (pos0) dump files in {args.engine}", file=sys.stderr)
        return 1

    print(f"{'layer':>5}  {'max_abs':>12}  {'rel_err':>12}  {'cosine':>12}  {'last_cos':>12}  {'last_rel':>12}  note")
    first_flag: int | None = None
    missing = 0
    for layer in sorted(ref):
        if (layer, 0) not in eng:
            print(f"{layer:>5}  missing engine dump")
            missing += 1
            continue
        path, tokens, hc, hidden = eng[(layer, 0)]
        a = np.fromfile(path, dtype="<f4").reshape(tokens, hc, hidden)
        b = np.load(ref[layer]).astype(np.float32)
        if b.shape != a.shape:
            print(f"{layer:>5}  shape mismatch: engine {a.shape} vs reference {b.shape}")
            missing += 1
            continue
        max_abs, rel, cos = metrics(a, b)
        _, last_rel, last_cos = metrics(a[-1], b[-1])
        note = ""
        if not (cos >= args.threshold):  # also flags NaN
            note = "FLAG cosine below threshold"
            if first_flag is None:
                first_flag = layer
                note += " (first)"
        print(f"{layer:>5}  {max_abs:>12.6g}  {rel:>12.6g}  {cos:>12.8f}  {last_cos:>12.8f}  {last_rel:>12.6g}  {note}")

    if first_flag is None and missing == 0:
        print(f"all {len(ref)} layers have cosine >= {args.threshold}")
    elif first_flag is not None:
        print(f"first layer below cosine {args.threshold}: {first_flag}")
    if args.strict and (first_flag is not None or missing):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
