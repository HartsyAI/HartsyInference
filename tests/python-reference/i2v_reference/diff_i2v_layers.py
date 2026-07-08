#!/usr/bin/env python3
"""Per-stage numeric diff between the C# WAN_DEBUG_DIR dumps and the i2v python reference dumps.

Same method as ../s2v_reference/diff_s2v_layers.py: walks names common to both shapes.txt sidecars in
forward order, compares flat (C# drops the B=1 axis), prints relL2 / mean-ratio / std-ratio per stage,
and names the FIRST divergent stage — that's where the C# port departs from the reference math.

Usage: venv/bin/python diff_i2v_layers.py <WAN_DEBUG_DIR> [<ref_dir>] [--tag cond] [--threshold 1e-2]
"""
import argparse
import os
import sys

import numpy as np

STAGE_ORDER = [
    "latent_in", "in_encoder", "clip_embeds", "timesteps",
    "patch_embed", "cond_temb", "cond_timestepProj", "cond_textProj", "cond_imgProj", "cond_encoderProj",
    "b0_scaleMsa", "b0_gateMsa", "b0_n1", "b0_attn1", "b0_xattn_text", "b0_xattn_img", "b0_attn2", "b0_ff",
] + [f"blocks_{i}" for i in range(40)] + ["pre_unpatchify", "velocity_out"]


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


def order_key(name, tag):
    base = name[len(tag) + 1:] if tag and name.startswith(tag + "_") else name
    try:
        return (0, STAGE_ORDER.index(base))
    except ValueError:
        return (1, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("ref_dir", nargs="?", default=None)
    ap.add_argument("--tag", default="cond")
    ap.add_argument("--threshold", type=float, default=1e-2)
    args = ap.parse_args()
    ref_dir = args.ref_dir or os.path.join(args.dump_dir, "ref")

    cs_shapes = load_shapes(args.dump_dir)
    ref_shapes = load_shapes(ref_dir)
    common = [n for n in cs_shapes if n in ref_shapes]
    if args.tag:
        common = [n for n in common if n.startswith(args.tag + "_")]
    common.sort(key=lambda n: order_key(n, args.tag))
    if not common:
        sys.exit("no common stage names")

    first_bad = None
    print(f"{'stage':34s} {'relL2':>10s} {'mean_ratio':>11s} {'std_ratio':>10s}")
    for name in common:
        cs = np.fromfile(os.path.join(args.dump_dir, "layers", name + ".bin"), dtype=np.float32)
        rf = np.fromfile(os.path.join(ref_dir, "layers", name + ".bin"), dtype=np.float32)
        if cs.size != rf.size:
            print(f"{name:34s}  SIZE MISMATCH cs={cs.size} ref={rf.size}")
            if first_bad is None:
                first_bad = name
            continue
        rel = np.linalg.norm(cs - rf) / (np.linalg.norm(rf) + 1e-12)
        mr = (cs.mean() / rf.mean()) if abs(rf.mean()) > 1e-12 else float("nan")
        sr = (cs.std() / rf.std()) if rf.std() > 1e-12 else float("nan")
        flag = "  <-- DIVERGES" if rel > args.threshold else ""
        print(f"{name:34s} {rel:10.3e} {mr:11.4f} {sr:10.4f}{flag}")
        if flag and first_bad is None:
            first_bad = name
    print()
    print(f"FIRST DIVERGENT STAGE: {first_bad}" if first_bad else "ALL STAGES WITHIN THRESHOLD")


if __name__ == "__main__":
    main()
