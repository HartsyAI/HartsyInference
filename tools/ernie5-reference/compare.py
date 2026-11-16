#!/usr/bin/env python3
"""Compare the engine's per-layer residual stream (ERNIE_DUMP_LAYERS) with reference.py's .npy output.

The engine writes one file per forward call, so a prefill and N decode steps give one prefill file and N one-token files per
layer. They are stitched by position into one [tokens, hc, hidden] stream (a gap, an overlap or a repeated position is refused)
and compared with the reference's causal forward over the same tokens: incremental decode through the KV caches must equal the
full-sequence forward at every position.

For every decoder layer present in both folders, it reports over the compared positions: max absolute difference, relative
error ||a - b|| / ||b||, cosine similarity, the prefill block's cosine, and the worst single position's cosine (named). A layer is
flagged when the whole-stream cosine or any single position's cosine falls below --threshold (default 0.999).

The reference defines the positions compared: the first T engine positions, where T is the reference's token count. An engine dump
that is shorter than the reference is an error. When the engine folder holds tokens.json (the harness writes it), its first T
tokens must equal the reference's meta.json tokens, and when the reference ran with --logits over exactly the fed sequence, the
reference's last-position argmax must equal the engine's argmax after its last fed token.

With --strict it exits 1 on a flagged layer, a layer-set mismatch, a stitching or length error, a token mismatch or an argmax
mismatch. --expect-layers N requires exactly layers 0..N-1 on both sides; --expect-tokens N requires the reference to cover N
positions (use the prompt plus decode length, so a prefill-only reference cannot pass a decode check).

--tie-margin M is a diagnostic for a flagged position. It reads the reference's index_gaps.json (per DSA query, the gap between the
last key the indexer kept and the first it dropped, over the spread of that query's candidate scores) and leaves out of the
per-position check every position whose gap is below M in a DSA layer at or before the layer being checked: there a rounding
difference can pick a different key. The positions left out are printed per layer, and the whole-stream cosine still covers them.
When more than --max-tie-fraction (default 0.05) of the selecting positions would be left out, it is an error: the margin would
skip the selection test instead of explaining a few positions.

Engine files are named layer{LL}_pos{startPos}_t{tokens}_hc{hc}_h{hidden}.f32 (little-endian float32, row-major
[tokens, hc, hidden]).

Usage:
  python compare.py --ref DIR --engine DIR [--threshold 0.999] [--strict] [--expect-layers N] [--expect-tokens N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ENGINE_RE = re.compile(r"^layer(\d{2})_pos(\d+)_t(\d+)_hc(\d+)_h(\d+)\.f32$")
REF_RE = re.compile(r"^layer_(\d{2})_all\.npy$")


class StitchError(Exception):
    pass


def load_engine(folder: Path) -> dict[int, list[tuple[int, Path, int, int, int]]]:
    """layer -> [(startPos, path, tokens, hc, hidden)] sorted by startPos."""
    out: dict[int, list[tuple[int, Path, int, int, int]]] = {}
    for p in folder.iterdir():
        m = ENGINE_RE.match(p.name)
        if m:
            layer, pos, tokens, hc, hidden = (int(g) for g in m.groups())
            out.setdefault(layer, []).append((pos, p, tokens, hc, hidden))
    for parts in out.values():
        parts.sort(key=lambda part: part[0])
    return out


def stitch(parts: list[tuple[int, Path, int, int, int]]) -> tuple[np.ndarray, int]:
    """The layer's stream over positions 0..T-1 and the prefill length (the first file's token count)."""
    if not parts or parts[0][0] != 0:
        raise StitchError("no prefill (pos0) file")
    hc, hidden = parts[0][3], parts[0][4]
    rows, expected = [], 0
    for pos, path, tokens, part_hc, part_hidden in parts:
        if pos != expected:
            kind = "gap" if pos > expected else "overlap or repeated position"
            raise StitchError(f"{kind}: a file starts at position {pos}, expected {expected}")
        if (part_hc, part_hidden) != (hc, hidden):
            raise StitchError(f"file at position {pos} is hc{part_hc}_h{part_hidden}, not hc{hc}_h{hidden}")
        rows.append(np.fromfile(path, dtype="<f4").reshape(tokens, hc, hidden))
        expected = pos + tokens
    return np.concatenate(rows, axis=0), parts[0][2]


def load_ref(folder: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in folder.iterdir():
        m = REF_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def near_ties(ref_dir: Path, margin: float) -> tuple[dict[int, set[int]], set[int]]:
    """(DSA layer -> query positions whose indexer kept and dropped a key within `margin` of each other, every selecting position)."""
    path = ref_dir / "index_gaps.json"
    if not path.exists():
        raise FileNotFoundError(f"--tie-margin needs {path}; rerun reference.py, which writes it")
    gaps = json.loads(path.read_text(encoding="utf-8"))
    selecting = {int(pos) for by_pos in gaps.values() for pos in by_pos}
    return {int(layer): {int(pos) for pos, gap in by_pos.items() if gap < margin} for layer, by_pos in gaps.items()}, selecting


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


def layer_set_errors(ref_layers: set[int], eng_layers: set[int], expect: int | None) -> list[str]:
    """Two-way layer-set check: layers present on one side only, and gaps against 0..expect-1."""
    errors: list[str] = []
    only_eng = sorted(eng_layers - ref_layers)
    only_ref = sorted(ref_layers - eng_layers)
    if only_eng:
        errors.append(f"layers in the engine dump but missing from the reference: {only_eng}")
    if only_ref:
        errors.append(f"layers in the reference but missing from the engine dump: {only_ref}")
    if expect is not None:
        want = set(range(expect))
        for name, have in (("reference", ref_layers), ("engine", eng_layers)):
            if have != want:
                errors.append(f"{name} holds {len(have)} layers; expected exactly 0..{expect - 1} "
                              f"(missing {sorted(want - have)}, extra {sorted(have - want)})")
    return errors


def sequence_errors(ref_dir: Path, eng_dir: Path, ref_tokens: int, exact_length: bool) -> tuple[list[str], list[str]]:
    """Token and argmax checks against tokens.json and meta.json; returns (errors, notes). With exact_length (--expect-tokens),
    a fed sequence longer or shorter than the reference is an error rather than a note."""
    errors: list[str] = []
    notes: list[str] = []
    fed_path, meta_path = eng_dir / "tokens.json", ref_dir / "meta.json"
    if not fed_path.exists():
        notes.append("no tokens.json in the engine folder: the fed tokens were not checked against the reference")
        return errors, notes
    fed = json.loads(fed_path.read_text(encoding="utf-8"))
    fed_tokens = list(fed.get("tokens") or [])
    if exact_length and len(fed_tokens) != ref_tokens:
        errors.append(f"the engine was fed {len(fed_tokens)} tokens, the reference ran on {ref_tokens}")
    if meta_path.exists():
        ref_ids = list(json.loads(meta_path.read_text(encoding="utf-8")).get("tokens") or [])
        if fed_tokens[:ref_tokens] != ref_ids:
            errors.append(f"token mismatch: the engine was fed {fed_tokens[:ref_tokens]}, the reference ran on {ref_ids}")
    else:
        notes.append("no meta.json in the reference folder: the tokens were not checked")
    logits_path = ref_dir / "logits_last.npy"
    if not logits_path.exists():
        notes.append("argmax not compared: the reference has no logits_last.npy (run reference.py with --logits)")
    elif "argmax_after_last" not in fed:
        notes.append("argmax not compared: tokens.json has no argmax_after_last")
    elif len(fed_tokens) != ref_tokens:
        notes.append(f"argmax not compared: the engine was fed {len(fed_tokens)} tokens, the reference ran on {ref_tokens}")
    else:
        ref_argmax = int(np.argmax(np.load(logits_path)))
        eng_argmax = int(fed["argmax_after_last"])
        if ref_argmax == eng_argmax:
            notes.append(f"argmax after the last position: {eng_argmax} on both sides")
        else:
            errors.append(f"argmax mismatch after the last position: engine {eng_argmax}, reference {ref_argmax}")
    return errors, notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference.py output folder")
    p.add_argument("--engine", required=True, help="ERNIE_DUMP_LAYERS folder")
    p.add_argument("--threshold", type=float, default=0.999)
    p.add_argument("--strict", action="store_true", help="exit 1 on a flagged layer or any set, length, token or argmax error")
    p.add_argument("--expect-layers", type=int, default=None,
                   help="require both folders to hold exactly layers 0..N-1 (43 for the real checkpoint)")
    p.add_argument("--expect-tokens", type=int, default=None,
                   help="require every reference layer to cover exactly N positions")
    p.add_argument("--position-threshold", type=float, default=None,
                   help="cosine every single position must reach (default: --threshold). One position holds only hc*hidden values, "
                        "so a looser bound can be set here without loosening the whole-stream check")
    p.add_argument("--tie-margin", type=float, default=None,
                   help="leave out of the per-position check the positions where a DSA indexer at or before the layer kept and dropped "
                        "keys within this gap, as a fraction of the query's candidate-score spread (from the reference's "
                        "index_gaps.json); a diagnostic for a flagged position, and the positions left out are printed")
    p.add_argument("--max-tie-fraction", type=float, default=0.05,
                   help="with --tie-margin, an error when more than this fraction of the selecting positions would be left out")
    args = p.parse_args()
    position_threshold = args.threshold if args.position_threshold is None else args.position_threshold

    ref_dir, eng_dir = Path(args.ref), Path(args.engine)
    ties: dict[int, set[int]] = {}
    tie_notes: list[str] = []
    tie_errors: list[str] = []
    if args.tie_margin is not None:
        try:
            ties, selecting = near_ties(ref_dir, args.tie_margin)
        except FileNotFoundError as ex:
            print(ex, file=sys.stderr)
            return 1
        tied = set().union(*ties.values()) if ties else set()
        tie_notes.append(f"near ties below {args.tie_margin}: {len(tied)} of {len(selecting)} selecting positions")
        if selecting and len(tied) > args.max_tie_fraction * len(selecting):
            tie_errors.append(f"{len(tied)} of {len(selecting)} selecting positions are near ties, more than --max-tie-fraction "
                              f"{args.max_tie_fraction}: the margin would skip the selection check")
    ref = load_ref(ref_dir)
    eng = load_engine(eng_dir)
    if not ref:
        print(f"no layer_LL_all.npy files in {args.ref}", file=sys.stderr)
        return 1
    if not any(parts and parts[0][0] == 0 for parts in eng.values()):
        print(f"no prefill (pos0) dump files in {args.engine}", file=sys.stderr)
        return 1

    errors: list[str] = []
    print(f"{'layer':>5}  {'max_abs':>12}  {'rel_err':>12}  {'cosine':>12}  {'prefill_cos':>12}  {'worst_pos_cos':>14}  {'at':>4}  note")
    first_flag: int | None = None
    ref_tokens = None
    prefill_len = None
    ref_lengths: dict[int, int] = {}
    engine_longest = 0
    for layer in sorted(ref):
        b = np.load(ref[layer]).astype(np.float32)
        ref_tokens = b.shape[0]
        ref_lengths[layer] = ref_tokens
        if layer not in eng:
            print(f"{layer:>5}  missing engine dump")
            errors.append(f"layer {layer}: no engine dump")
            continue
        try:
            a, prefill_len = stitch(eng[layer])
        except StitchError as ex:
            print(f"{layer:>5}  {ex}")
            errors.append(f"layer {layer}: {ex}")
            continue
        engine_longest = max(engine_longest, a.shape[0])
        if a.shape[0] < b.shape[0]:
            msg = f"engine covers {a.shape[0]} positions, the reference {b.shape[0]}"
            print(f"{layer:>5}  {msg}")
            errors.append(f"layer {layer}: {msg}")
            continue
        a = a[:b.shape[0]]
        if a.shape != b.shape:
            msg = f"shape mismatch: engine {a.shape} vs reference {b.shape}"
            print(f"{layer:>5}  {msg}")
            errors.append(f"layer {layer}: {msg}")
            continue
        max_abs, rel, cos = metrics(a, b)
        pre = min(prefill_len, b.shape[0])
        _, _, pre_cos = metrics(a[:pre], b[:pre])
        skipped = set().union(*(pos for dsa, pos in ties.items() if dsa <= layer)) & set(range(b.shape[0]))
        if skipped:
            tie_notes.append(f"layer {layer}: left out {len(skipped)} near-tie positions {sorted(skipped)}")
        per_pos = [np.inf if t in skipped else metrics(a[t], b[t])[2] for t in range(b.shape[0])]
        worst = int(np.nanargmin(per_pos)) if not any(np.isnan(per_pos)) else int(np.argmax(np.isnan(per_pos)))
        note = ""
        if not (cos >= args.threshold) or not all(c >= position_threshold for c in per_pos):  # also flags NaN
            note = "FLAG cosine below threshold"
            if first_flag is None:
                first_flag = layer
                note += " (first)"
        print(f"{layer:>5}  {max_abs:>12.6g}  {rel:>12.6g}  {cos:>12.8f}  {pre_cos:>12.8f}  {per_pos[worst]:>14.8f}  {worst:>4}  {note}")

    errors += tie_errors
    errors += layer_set_errors(set(ref), set(eng), args.expect_layers)
    # Every reference layer must cover the same positions: an interrupted shorter rerun into the same folder leaves a mix, and the
    # shorter layers would compare fewer engine positions.
    if len(set(ref_lengths.values())) > 1:
        by_length: dict[int, list[int]] = {}
        for layer, length in ref_lengths.items():
            by_length.setdefault(length, []).append(layer)
        errors.append("reference layers cover different numbers of positions: "
                      + "; ".join(f"{length} positions in layers {layers}" for length, layers in sorted(by_length.items())))
    if args.expect_tokens is not None:
        short = sorted(layer for layer, length in ref_lengths.items() if length != args.expect_tokens)
        if short:
            errors.append(f"reference layers {short} do not cover {args.expect_tokens} positions")
    if ref_tokens is not None:
        seq_errors, notes = sequence_errors(ref_dir, eng_dir, ref_tokens, exact_length=args.expect_tokens is not None)
        if engine_longest > ref_tokens:
            notes.append(f"the engine covers {engine_longest} positions; compared the first {ref_tokens}")
        errors += seq_errors
        decode = ref_tokens - (prefill_len or ref_tokens)
        print(f"compared positions 0..{ref_tokens - 1}: prefill {min(prefill_len or ref_tokens, ref_tokens)}, decode {max(decode, 0)}")
        for note in notes + tie_notes:
            print(note)
    for err in errors:
        print(err)
    if first_flag is None and not errors:
        print(f"all {len(ref)} layers have cosine >= {args.threshold}, and >= {position_threshold} at every position")
    elif first_flag is not None:
        print(f"first layer below cosine {args.threshold}: {first_flag}")
    if args.strict and (first_flag is not None or errors):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
