"""Dump upstream Block.hc_mixes / hc_pre / hc_post (mHC around a sublayer) into fixtures/hyper_connection.json.

The methods are called unbound with a stand-in carrying the Block attributes they read, so no attention or MoE is built.
Usage: python dump_hyper_connection_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_ports  # noqa: E402

sys.modules["kernel"] = kernel_ports


def flat(t: torch.Tensor):
    return [float(v) for v in t.detach().float().reshape(-1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    cases = []
    for name, hc, dim, tokens, iters in [("hc4", 4, 8, 5, 20), ("hc2", 2, 6, 3, 3), ("hc3_one_iter", 3, 4, 4, 1)]:
        torch.manual_seed(900 + hc * 10 + dim)
        mix_hc = (2 + hc) * hc
        blk = SimpleNamespace(norm_eps=1e-6, hc_mult=hc, hc_sinkhorn_iters=iters, hc_eps=1e-6)
        fn = torch.randn(mix_hc, hc * dim) * 0.3
        scale = torch.tensor([0.8, 1.1, 0.6])
        base = torch.randn(mix_hc) * 0.4
        x = torch.randn(1, tokens, hc, dim)
        sub = torch.randn(1, tokens, dim)
        pre, post, comb = mod.Block.hc_mixes(blk, x, fn, scale, base)
        y = mod.Block.hc_pre(blk, x, pre)
        z = mod.Block.hc_post(blk, sub, x, post, comb)
        cases.append({
            "name": name, "hc": hc, "dim": dim, "tokens": tokens, "iters": iters, "normEps": 1e-6, "hcEps": 1e-6,
            "fn": flat(fn), "scale": flat(scale), "base": flat(base), "x": flat(x), "sub": flat(sub),
            "pre": flat(pre), "post": flat(post), "comb": flat(comb), "collapsed": flat(y), "expanded": flat(z),
        })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "hyper_connection.json"), "w") as fh:
        json.dump({"cases": cases}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
