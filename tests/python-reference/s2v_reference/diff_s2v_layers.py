#!/usr/bin/env python3
"""Per-stage numeric diff between the C# WAN_DEBUG_DIR dumps and the s2v python reference dumps.

Walks the names common to both `shapes.txt` sidecars (in forward order), loads the raw-F32 blobs flat
(the C# side drops the B=1 axis on token tensors — element counts still match), and prints per stage:
  relL2      = ||cs − ref|| / (||ref|| + 1e-12)
  mean ratio = mean(cs) / mean(ref)   (magnitude bugs are invisible to correlation — check std/mean!)
  std ratio  = std(cs) / std(ref)
Flags every stage with relL2 > threshold and names the FIRST divergent stage — that is where the C#
port departs from ComfyUI.

Usage:
  venv/bin/python diff_s2v_layers.py <WAN_DEBUG_DIR> [<ref_dir>] [--tag cond] [--threshold 1e-2]
  (ref_dir defaults to {WAN_DEBUG_DIR}/ref; --tag '' diffs every common name)
"""
import argparse
import os
import sys

import numpy as np

# Forward-order ranking so "first divergence" means first in the computation, not alphabetical.
STAGE_ORDER = [
    "audio_features", "audio_global", "audio_local",
    "latent_in", "text_embeds", "timesteps",
    "post_patchify", "post_condmask", "ref_tokens", "joined_tokens",
    "temb", "timestep_proj", "text_proj",
    "block_0", "audio_delta_inj0", "block_20", "block_39", "audio_delta_inj11",
    "pre_unpatchify", "velocity_out", "cfg_combined",
]


def load_shapes(d):
    shapes = {}
    path = os.path.join(d, "shapes.txt")
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                shapes[parts[0]] = [int(x) for x in parts[1].split(",")]
    return shapes


def load_bin(d, name):
    return np.fromfile(os.path.join(d, "layers", name + ".bin"), dtype=np.float32)


def stage_rank(name, tag):
    base = name[len(tag) + 1:] if tag and name.startswith(tag + "_") else name
    return (STAGE_ORDER.index(base), name) if base in STAGE_ORDER else (len(STAGE_ORDER), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cs_dir", help="the C# WAN_DEBUG_DIR")
    ap.add_argument("ref_dir", nargs="?", default=None, help="python reference dir (default {cs_dir}/ref)")
    ap.add_argument("--tag", default="", help="only diff names with this prefix (cond/uncond); empty = all common")
    ap.add_argument("--threshold", type=float, default=1e-2)
    args = ap.parse_args()

    ref_dir = args.ref_dir or os.path.join(args.cs_dir, "ref")
    cs_shapes, ref_shapes = load_shapes(args.cs_dir), load_shapes(ref_dir)
    common = sorted(set(cs_shapes) & set(ref_shapes), key=lambda n: stage_rank(n, args.tag))
    if args.tag:
        common = [n for n in common if n.startswith(args.tag + "_")]
    if not common:
        sys.exit(f"no common stage names between {args.cs_dir} and {ref_dir} (tag='{args.tag}')")

    print(f"{'stage':34s} {'elems':>10s} {'relL2':>10s} {'mean_ratio':>11s} {'std_ratio':>10s}  flag")
    first_bad = None
    for name in common:
        cs, ref = load_bin(args.cs_dir, name), load_bin(ref_dir, name)
        if cs.size != ref.size:
            print(f"{name:34s} SIZE MISMATCH cs={cs.size} ref={ref.size} "
                  f"(shapes {cs_shapes[name]} vs {ref_shapes[name]})")
            if first_bad is None:
                first_bad = name
            continue
        cs64, ref64 = cs.astype(np.float64), ref.astype(np.float64)
        rel_l2 = np.linalg.norm(cs64 - ref64) / (np.linalg.norm(ref64) + 1e-12)
        rm = ref64.mean()
        mean_ratio = cs64.mean() / rm if abs(rm) > 1e-12 else float("nan")
        rs = ref64.std()
        std_ratio = cs64.std() / rs if rs > 1e-12 else float("nan")
        bad = rel_l2 > args.threshold
        if bad and first_bad is None:
            first_bad = name
        print(f"{name:34s} {cs.size:>10d} {rel_l2:>10.3e} {mean_ratio:>11.4f} {std_ratio:>10.4f}"
              f"  {'<<< DIVERGES' if bad else 'ok'}")

    print()
    if first_bad is None:
        print(f"ALL {len(common)} stages within relL2 {args.threshold:g} — C# matches the ComfyUI reference.")
    else:
        print(f"FIRST DIVERGENCE: {first_bad} (relL2 > {args.threshold:g}) — the C# bug is at or before this stage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
