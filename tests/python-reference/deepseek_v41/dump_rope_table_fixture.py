"""Dump upstream precompute_freqs_cis (plain and YaRN) as cos/sin tables into fixtures/rope_table.json.

Usage: python dump_rope_table_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
"""
import argparse
import json
import os
import sys

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod
    cases = []
    # (rotary dim, length, original_seq_len, theta, factor, beta_fast, beta_slow)
    for name, args in {
        "plain": (64, 40, 0, 10000.0, 16.0, 32, 1),
        "yarn": (64, 300, 128, 160000.0, 16.0, 32, 1),
        "yarn_small": (32, 96, 64, 40000.0, 40.0, 32, 1),
    }.items():
        f = mod.precompute_freqs_cis(*args)
        cases.append({
            "name": name, "rotaryDim": args[0], "length": args[1], "originalSeqLen": args[2], "theta": args[3],
            "factor": args[4], "betaFast": args[5], "betaSlow": args[6],
            "cos": [float(v) for v in f.real.reshape(-1)], "sin": [float(v) for v in f.imag.reshape(-1)],
        })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "rope_table.json"), "w") as fh:
        json.dump({"cases": cases}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
