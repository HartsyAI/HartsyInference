"""Dump the upstream MoE layer (gate + routed experts + shared expert) with float32 weights into fixtures/moe_exec.json.

Records the gate's weights, chosen experts and combine weights, so the C# side can test expert execution alone or the whole layer.
Usage: python dump_moe_exec_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
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

    mod.default_dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    cases = []
    for name, dim, inter, experts, topk, limit, tokens in [
        ("limit", 8, 6, 6, 2, 1.5, 7), ("nolimit", 8, 6, 5, 3, 0.0, 4), ("top1", 4, 4, 3, 1, 2.0, 5)
    ]:
        torch.manual_seed(100 + dim + experts)
        args = mod.ModelArgs(dim=dim, moe_inter_dim=inter, n_routed_experts=experts, n_activated_experts=topk,
                             n_shared_experts=1, swiglu_limit=limit, expert_dtype=None, dtype="bf16", vocab_size=16)
        moe = mod.MoE(0, args)
        with torch.no_grad():
            for prm in moe.parameters():
                prm.normal_(0, 1.0)
        x = torch.randn(1, tokens, dim) * 2.0
        with torch.no_grad():
            weights, indices = moe.gate(x.view(-1, dim))
            y = moe(x)
        cases.append({
            "name": name, "dim": dim, "inter": inter, "experts": experts, "topk": topk, "limit": limit, "tokens": tokens,
            "x": flat(x), "gateWeight": flat(moe.gate.weight), "gateBias": flat(moe.gate.bias), "indices": [int(v) for v in indices.reshape(-1)], "weights": flat(weights),
            "w1": [flat(e.w1.weight) for e in moe.experts], "w2": [flat(e.w2.weight) for e in moe.experts],
            "w3": [flat(e.w3.weight) for e in moe.experts],
            "sw1": flat(moe.shared_experts.w1.weight), "sw2": flat(moe.shared_experts.w2.weight),
            "sw3": flat(moe.shared_experts.w3.weight), "y": flat(y),
        })
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "moe_exec.json"), "w") as fh:
        json.dump({"cases": cases}, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
