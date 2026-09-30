"""Dump mHC mixing, offset interleaved RoPE and sliding-window index cases from the UNMODIFIED upstream model.py.

hc_pre / hc_post / apply_rotary_emb / get_window_topk_idxs are upstream functions; hc_split_sinkhorn is the pure-torch
port of the tilelang kernel (kernel_ports.py). Outputs are committed Unit fixtures under fixtures/.
Usage: python dump_hc_rope_window_fixtures.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
"""
import argparse
import json
import math
import os
import sys
from typing import List

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports


def flat(t: torch.Tensor) -> List[float]:
    return ["NaN" if math.isnan(float(v)) else float(v) for v in t.detach().reshape(-1)]


def write(out_dir: str, name: str, obj: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")


def dump_hc(model_mod, out_dir: str) -> None:
    g = torch.Generator().manual_seed(21)
    hc, tokens, dim = 4, 5, 16
    width = (2 + hc) * hc
    mixes = torch.randn(tokens, width, generator=g) * 2
    scale = torch.tensor([0.7, 1.3, 0.9])
    base = torch.randn(width, generator=g) * 0.5
    streams = torch.randn(tokens, hc, dim, generator=g)
    x = torch.randn(tokens, dim, generator=g)
    out: dict = {"hc": hc, "tokens": tokens, "dim": dim, "eps": 1e-6, "mixes": flat(mixes), "scale": flat(scale),
                 "bias": flat(base), "streams": flat(streams), "x": flat(x), "cases": []}
    for iters in (1, 3, 20):
        pre, post, comb = kernel_ports.hc_split_sinkhorn(mixes[None], scale, base, hc, iters, 1e-6)
        collapsed = model_mod.Block.hc_pre(None, streams[None], pre)
        expanded = model_mod.Block.hc_post(None, x[None], streams[None], post, comb)
        out["cases"].append({"iters": iters, "pre": flat(pre), "post": flat(post), "comb": flat(comb),
                             "preMix": flat(collapsed), "postMix": flat(expanded)})
    write(out_dir, "hc_mix.json", out)


def dump_rope(model_mod, out_dir: str) -> None:
    g = torch.Generator().manual_seed(23)
    length, heads, dim, rd = 5, 2, 16, 8
    angle = torch.arange(length).unsqueeze(1) * (10000.0 ** (-torch.arange(0, rd, 2).float() / rd)).unsqueeze(0)
    freqs = torch.polar(torch.ones_like(angle), angle)
    q = torch.randn(1, length, heads, dim, generator=g)
    k = torch.randn(1, length, dim, generator=g)
    out: dict = {"length": length, "heads": heads, "dim": dim, "rotaryDim": rd, "dimOffset": dim - rd,
                 "cos": flat(freqs.real), "sin": flat(freqs.imag), "q": flat(q), "k": flat(k)}
    for name, t in (("q", q), ("k", k)):
        fwd = t.clone()
        model_mod.apply_rotary_emb(fwd[..., -rd:], freqs)
        inv = fwd.clone()
        model_mod.apply_rotary_emb(inv[..., -rd:], freqs, True)
        out[name + "Rotated"] = flat(fwd)
        out[name + "Inverse"] = flat(inv)
    write(out_dir, "rope_interleaved_offset.json", out)


def dump_window(model_mod, out_dir: str) -> None:
    cases = []
    for window, seqlen, start in ((8, 5, 0), (8, 8, 0), (8, 13, 0), (4, 3, 0), (8, 1, 3), (8, 1, 7), (8, 1, 8),
                                  (8, 1, 15), (8, 1, 21), (4, 1, 1), (1, 1, 5)):
        idx = model_mod.get_window_topk_idxs(window, 1, seqlen, start)[0]
        cases.append({"window": window, "seqLen": seqlen, "startPos": start, "rows": idx.shape[0], "cols": idx.shape[1],
                      "indices": [int(v) for v in idx.reshape(-1)]})
    write(out_dir, "window_indices.json", {"cases": cases})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as model_mod

    dump_hc(model_mod, a.out_dir)
    dump_rope(model_mod, a.out_dir)
    dump_window(model_mod, a.out_dir)


if __name__ == "__main__":
    main()
