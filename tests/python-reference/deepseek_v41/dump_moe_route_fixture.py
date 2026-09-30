"""Dump a small sqrtsoftplus routing case from the UNMODIFIED upstream Gate (bias, image-token bias_vl, gate_temp, route_scale).

Output is a committed Unit fixture: fixtures/moe_route_sqrtsoftplus.json (logits are x @ W^T before gate_temp).
Usage: python dump_moe_route_fixture.py [--upstream ~/dsv41-ref/upstream] [--out fixtures/moe_route_sqrtsoftplus.json]
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


def flat(t: torch.Tensor) -> list:
    return [float(v) for v in t.detach().reshape(-1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "moe_route_sqrtsoftplus.json"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as model_mod

    tokens, dim, experts, topk = 8, 16, 16, 4
    args = model_mod.ModelArgs(dim=dim, n_routed_experts=experts, n_activated_experts=topk, score_func="sqrtsoftplus",
                               gate_temp=1.7, norm_topk_prob=True, route_scale=1.5, vision_n_layers=1)
    gate = model_mod.Gate(0, args)
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(experts, dim, generator=g))
        gate.bias.copy_(torch.randn(experts, generator=g) * 0.3)
        gate.bias_vl.copy_(torch.randn(experts, generator=g) * 0.3)
    x = torch.randn(tokens, dim, generator=g)
    image_mask = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1], dtype=torch.bool)
    with torch.no_grad():
        logits = x.float() @ gate.weight.float().t()
        weights, indices = gate(x, image_mask)
    out = {
        "tokens": tokens, "experts": experts, "topk": topk, "gateTemp": 1.7, "routeScale": 1.5, "renormEpsilon": 1e-20,
        "logits": flat(logits), "bias": flat(gate.bias), "biasVl": flat(gate.bias_vl),
        "tokenKinds": [int(v) for v in image_mask],
        "indices": [int(v) for v in indices.reshape(-1)], "weights": flat(weights),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
