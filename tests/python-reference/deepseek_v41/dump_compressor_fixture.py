"""Dump upstream Compressor prefill + decode pooling (norm bypassed) into fixtures/compressor_pool.json.

The projected kv/score fed to the pooling are recorded too, so the C# side tests only the pooling and state handling.
Usage: python dump_compressor_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
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
    for name, ratio, prefill, decode in [("r4_remainder", 4, 11, 6), ("r2_odd", 2, 5, 5), ("r4_short", 4, 3, 6), ("r4_exact", 4, 8, 4)]:
        torch.manual_seed(7 + ratio + prefill)
        dim, head = 8, 16
        args = mod.ModelArgs(dim=dim, head_dim=head, compress_ratios=(ratio,), max_batch_size=1, max_seq_len=64)
        comp = mod.Compressor(args, 0)
        torch.nn.init.normal_(comp.wkv.weight, std=0.5)
        torch.nn.init.normal_(comp.wgate.weight, std=0.5)
        comp.norm = torch.nn.Identity()
        x = torch.randn(1, prefill + decode, dim)
        steps = []
        out = comp(x[:, :prefill], 0)
        steps.append({"startPos": 0, "len": prefill, "out": None if out is None else flat(out), "rows": 0 if out is None else out.shape[1]})
        for i in range(decode):
            pos = prefill + i
            out = comp(x[:, pos:pos + 1], pos)
            steps.append({"startPos": pos, "len": 1, "out": None if out is None else flat(out), "rows": 0 if out is None else out.shape[1]})
        cases.append({
            "name": name, "ratio": ratio, "dim": dim, "headDim": head,
            "wkv": flat(comp.wkv.weight), "wgate": flat(comp.wgate.weight), "x": flat(x), "steps": steps,
        })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "compressor_pool.json"), "w") as fh:
        json.dump({"cases": cases}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
