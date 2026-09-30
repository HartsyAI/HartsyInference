"""Dump latent-cache quantization, sparse latent attention and indexer-score cases as committed Unit fixtures.

Quantization bytes come from the pure-torch ports of act_quant / fp4_act_quant (UE8M0 scale bytes are re-derived by
casting the fp32 scale to float8_e8m0fnu). Attention uses the float64 sparse_attn_exact over the dequantized rows;
the indexer scores replay the upstream `Indexer.forward` expression with its compress_lens mask.
Usage: python dump_latent_fixtures.py [--out-dir fixtures]
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports as kp  # noqa: E402

E4M3: torch.dtype = torch.float8_e4m3fn
E8M0: torch.dtype = torch.float8_e8m0fnu


def flat(t: torch.Tensor) -> List[float]:
    return ["NaN" if math.isnan(float(v)) else float(v) for v in t.detach().reshape(-1)]


def ints(t: torch.Tensor) -> List[int]:
    return [int(v) for v in t.detach().reshape(-1)]


def quantize(x: torch.Tensor, encoding: str) -> Dict[str, List]:
    """Codes, scale bytes and the in-place dequantized values of x [rows, dim] for one latent encoding."""
    if encoding == "fp8":
        q, s = kp.act_quant(x, 32, "ue8m0", torch.float32)
        codes, scales = q.view(torch.uint8), s.to(E8M0).view(torch.uint8)
        deq = kp.act_quant(x.clone(), 32, "ue8m0", torch.float32, True)
    elif encoding == "fp4e4m3":
        codes, s = kp.fp4_act_quant(x, 16, False, E4M3)
        scales = s.view(torch.uint8)
        deq = kp.fp4_act_quant(x.clone(), 16, True, E4M3)
    else:
        codes, s = kp.fp4_act_quant(x, 32, False, E8M0)
        scales = s.view(torch.uint8)
        deq = kp.fp4_act_quant(x.clone(), 32, True, E8M0)
    return {"codes": ints(codes), "scales": ints(scales), "dequant": flat(deq)}


def edge_rows(dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    rows: List[torch.Tensor] = [
        torch.zeros(dim),
        torch.full((dim,), 1e-6),
        torch.cat([torch.tensor([-0.0, 0.0, 1.0, -1.0]), torch.zeros(dim - 4)]),
        torch.full((dim,), 1e4),
        torch.full((dim,), 448.0),
        torch.full((dim,), 6.0),
        torch.full((dim,), -3.0),
    ]
    # values on the e2m1 rounding midpoints of a known power-of-two scale (ties-to-even) and just off them
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    for scale in (1.0, 0.125, 64.0):
        tie = (mids * scale).repeat(dim // 8)
        rows += [tie, -tie, tie * (1 + 2.0**-20), tie * (1 - 2.0**-20)]
    for _ in range(12):
        mag = 10.0 ** (torch.rand(dim // 16, generator=g) * 6 - 3)
        rows.append((torch.randn(dim, generator=g).reshape(-1, 16) * mag.unsqueeze(-1)).reshape(dim))
    return torch.stack(rows)


def dump_quant(out_dir: str) -> None:
    dim = 64
    x = edge_rows(dim)
    out = {"dim": dim, "rows": x.shape[0], "input": flat(x)}
    for name in ("fp8", "fp4e4m3", "fp4e8m0"):
        out[name] = quantize(x, name)
    write(out_dir, "latent_quant_bytes.json", out)


def dump_attention(out_dir: str) -> None:
    g = torch.Generator().manual_seed(11)
    dim, win_rows, main_rows, tokens, heads, k = 64, 8, 12, 6, 3, 10
    win, main = torch.randn(win_rows, dim, generator=g), torch.randn(main_rows, dim, generator=g) * 2
    wq, mq = quantize(win, "fp8"), quantize(main, "fp4e4m3")
    kv = torch.cat([torch.tensor(wq["dequant"]).reshape(win_rows, dim), torch.tensor(mq["dequant"]).reshape(main_rows, dim)])
    q = torch.randn(tokens, heads, dim, generator=g)
    sink = torch.randn(heads, generator=g)
    idx = torch.randint(-1, win_rows + main_rows, (tokens, k), generator=g, dtype=torch.int32)
    idx[0] = -1                       # no valid key: zeros
    idx[1] = -1
    idx[1, 4] = win_rows + 3          # a single main row
    idx[2, :3] = torch.tensor([2, 2, win_rows])  # repeated slot plus the first main row
    scale = dim ** -0.5
    expected = kp.sparse_attn_exact(q[None], kv[None], sink, idx[None], scale)[0]
    write(out_dir, "sparse_latent_attention.json", {
        "dim": dim, "windowSlots": win_rows, "mainRows": main_rows, "tokens": tokens, "heads": heads, "k": k,
        "scale": scale, "window": {"codes": wq["codes"], "scales": wq["scales"]},
        "main": {"codes": mq["codes"], "scales": mq["scales"]}, "query": flat(q), "sink": flat(sink),
        "indices": ints(idx), "expected": flat(expected),
    })


def dump_indexer(out_dir: str) -> None:
    g = torch.Generator().manual_seed(13)
    dim, heads, seqlen, ratio = 64, 4, 12, 2
    n = seqlen // ratio
    keys = torch.randn(n, dim, generator=g)
    kq = quantize(keys, "fp4e8m0")
    index_k = torch.tensor(kq["dequant"]).reshape(n, dim)
    q = torch.randn(seqlen, heads, dim, generator=g)
    q = kp.fp4_act_quant(q.clone(), 32, True)
    softmax_scale = dim ** -0.5
    weights = torch.randn(seqlen, heads, generator=g) * (softmax_scale * heads ** -0.5)
    score = torch.einsum("shd,td->sht", q, index_k)
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=1)
    lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
    masked = torch.arange(n).unsqueeze(0) >= lens
    score = score.masked_fill(masked, 0.0)
    write(out_dir, "indexer_scores.json", {
        "dim": dim, "heads": heads, "tokens": seqlen, "ratio": ratio, "keys": n,
        "keyCodes": kq["codes"], "keyScales": kq["scales"], "query": flat(q), "headWeights": flat(weights),
        "compressLens": ints(lens), "masked": [int(v) for v in masked.reshape(-1)], "scores": flat(score),
    })


def write(out_dir: str, name: str, obj: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=None, separators=(",", ":"))
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = ap.parse_args()
    dump_quant(a.out_dir)
    dump_attention(a.out_dir)
    dump_indexer(a.out_dir)


if __name__ == "__main__":
    main()
