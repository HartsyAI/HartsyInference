"""Dump a stack of upstream Attention layers (prefill then decode) with float32 weights into fixtures/attention_stack.json.

The layers cover every mode: window only, a ratio-2 compressed-KV + indexer source, a ratio-2 layer that reuses its shared KV
and indices, a ratio-1 candidate source, an index-only layer using candidates, and a ratio-1 reuse layer. Each layer gets its own
random input per step, so the shared compressed KV / index keys / top-k indices / candidates flow exactly as in the model.
Candidate blocks cover more positions than index_topk and the indexer has enough heads that ReLU almost never zeroes a whole score, so no
top-k pick falls among ties (torch leaves their order unspecified).
Usage: python dump_attention_fixture.py [--upstream ~/dsv41-ref/upstream] [--out-dir fixtures]
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


def named(module: torch.nn.Module):
    return {n: flat(p) for n, p in module.named_parameters()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default=os.path.expanduser("~/dsv41-ref/upstream"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "fixtures"))
    a = p.parse_args()
    sys.path.insert(0, os.path.join(a.upstream, "inference"))
    import model as mod

    # exact softmax instead of the port's bf16 probabilities, so the host reference can be compared tightly and a gap means a real defect
    mod.sparse_attn = kernel_ports.sparse_attn_exact

    mod.default_dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(1234)
    cfg = dict(
        dim=16, n_heads=2, head_dim=32, rope_head_dim=16, q_lora_rank=16, o_groups=2, o_lora_rank=8, window_size=8,
        index_n_heads=8, index_head_dim=32, index_topk=3, hc_mult=2, norm_eps=1e-6, max_seq_len=64, max_batch_size=1,
        n_layers=6, n_mtp_layers=0, dtype="bf16", expert_dtype=None, vocab_size=16, moe_inter_dim=8, n_routed_experts=4,
        n_activated_experts=2,
        compress_ratios=(0, 2, 2, 1, 1, 1), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
        candidate_source_layer=3, candidate_topk_blocks=3, candidate_block_size=2,
        original_seq_len=16, rope_factor=8, compress_rope_theta=40000.0, rope_theta=10000.0,
    )
    args = mod.ModelArgs(**cfg)
    layers = []
    for i in range(args.n_layers):
        attn = mod.Attention(i, args)
        for prm in attn.parameters():
            prm.data = prm.data.float()
            prm.data.normal_(0, 0.4)
        for name, prm in attn.named_parameters():
            if name.endswith("norm.weight") or name.endswith("k_norm.weight"):
                prm.data.fill_(1.0).add_(torch.randn_like(prm) * 0.1)
        layers.append(attn)
    prefill, decode = 11, 6
    steps = []
    with torch.no_grad():
        for start, length in [(0, prefill)] + [(prefill + i, 1) for i in range(decode)]:
            ins, outs, picks = [], [], []
            for i, attn in enumerate(layers):
                x = torch.randn(1, length, args.dim)
                ins.append(flat(x))
                outs.append(flat(attn(x, start)))
                # the indices an index-source layer just chose, offset past the window rows (exact, unlike the outputs)
                picks.append([int(v) for v in mod.shared_attn.topk_idxs.reshape(-1)] if attn.is_index_source else None)
            steps.append({"start": start, "len": length, "x": ins, "y": outs, "topk": picks})
    out = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "layers": [named(l) for l in layers], "steps": steps,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "attention_stack.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")
    print("layers", [sorted(n for n in named(l)) for l in layers[:2]])


if __name__ == "__main__":
    main()
