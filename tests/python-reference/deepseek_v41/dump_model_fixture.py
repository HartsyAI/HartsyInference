"""Dump a small upstream Transformer (all layer modes, float32 weights, no Engram/vision/draft) into fixtures/model_forward.json.

Records every parameter, then per step the input ids, each block's output stream, the final normed hidden state of every position and
the last position's logits, through an 11-token prefill and six single-token decode steps. Engram has its own fixtures and needs the
real tokenizer's token map, so it is left out here.
Usage: python dump_model_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
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
    # nine significant digits round-trip a float32 exactly and keep the fixture small
    return [float(f"{float(v):.9g}") for v in t.detach().float().reshape(-1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    # exact softmax instead of the port's bf16 probabilities, so the host reference can be compared tightly and a gap means a real defect
    mod.sparse_attn = kernel_ports.sparse_attn_exact

    torch.manual_seed(2468)
    cfg = dict(
        dim=16, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=16, o_groups=2, o_lora_rank=8, window_size=8,
        index_n_heads=8, index_head_dim=32, index_topk=3, hc_mult=2, hc_sinkhorn_iters=4, norm_eps=1e-6, hc_eps=1e-6,
        max_seq_len=64, max_batch_size=1, n_layers=6, n_mtp_layers=0, dtype="bf16", expert_dtype=None, vocab_size=32,
        moe_inter_dim=8, n_routed_experts=4, n_activated_experts=2, route_scale=1.5, swiglu_limit=3.0,
        compress_ratios=(0, 2, 2, 1, 1, 1), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4), candidate_source_layer=3,
        candidate_topk_blocks=3, candidate_block_size=2, original_seq_len=16, rope_factor=8, compress_rope_theta=40000.0,
        rope_theta=10000.0, temperature=0.0,
    )
    args = mod.ModelArgs(**cfg)
    model = mod.Transformer(args, None)
    torch.set_default_dtype(torch.float32)
    for name, prm in model.named_parameters():
        prm.data = prm.data.float()
        leaf = name.rsplit(".", 1)[-1]
        if leaf.endswith("fn") and leaf.startswith("hc_"):
            prm.data.normal_(0, 0.1)
        elif leaf.endswith("scale") and leaf.startswith("hc_"):
            prm.data.copy_(0.5 + 0.1 * torch.randn_like(prm))
        elif leaf.endswith("base") and leaf.startswith("hc_"):
            prm.data.normal_(0, 0.2)
        elif name.endswith("norm.weight"):
            prm.data.copy_(1.0 + 0.1 * torch.randn_like(prm))
        elif leaf == "bias":
            prm.data.normal_(0, 0.1)
        elif leaf == "attn_sink":
            prm.data.normal_(0, 0.5)
        else:
            prm.data.normal_(0, 0.4)
    model.eval()

    blocks, finals = [], []
    for blk in model.layers:
        blk.register_forward_hook(lambda _m, _i, out: blocks.append(flat(out[0])))
    model.norm.register_forward_hook(lambda _m, _i, out: finals.append(flat(out)))

    ids = torch.randint(0, cfg["vocab_size"], (1, 17))
    steps = []
    for start, length in [(0, 11)] + [(11 + i, 1) for i in range(6)]:
        blocks.clear()
        finals.clear()
        _, logits, _ = model(ids[:, start:start + length], start)
        steps.append({"start": start, "len": length, "ids": [int(v) for v in ids[0, start:start + length]],
                      "blocks": list(blocks), "final": finals[0], "logits": flat(logits)})
    out = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           "params": {n: flat(p) for n, p in model.named_parameters()}, "steps": steps}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "model_forward.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")


if __name__ == "__main__":
    main()
