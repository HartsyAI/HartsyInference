#!/usr/bin/env python3
"""Tiny random Flint reference for the CPU unit lane.

Builds a FlintForCausalLM from transformers' own modeling code with a shrunken config that keeps every structural trait of
the released 40B-A2.5B checkpoint: two leading dense layers, then sparse layers (sigmoid router with normalized top-k and an
ungated shared expert), per-head QK RMSNorm, learned per-head attention sinks, sliding-window layers with RoPE, and
full-attention layers without any positional encoding. Every parameter is re-drawn so each path carries signal, then the
weights (HF on-disk names, F32), config.json and the float32 reference logits are written to
tests/HartsyInference.LLM.Tests/Fixtures/Flint/.

Run: python -I tests/python-reference/dump_flint_tiny_reference.py  (transformers >= 5.22 with flint)
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import FlintConfig, FlintForCausalLM

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HartsyInference.LLM.Tests", "Fixtures", "Flint")
SEQ_LEN = 12

config = FlintConfig(
    vocab_size=64, hidden_size=64, intermediate_size=48, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, sliding_window=5, moe_intermediate_size=16, shared_expert_intermediate_size=24, num_experts=8,
    num_experts_per_tok=3, tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    layer_types=["full_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                 "full_attention", "sliding_attention", "full_attention"])

torch.manual_seed(0)
model = FlintForCausalLM(config).float().eval()
gen = torch.Generator().manual_seed(1234)


def draw(shape, std, mean=0.0):
    return mean + std * torch.randn(shape, generator=gen)


with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(draw(p.shape, 0.15, 1.0))
        elif name.endswith("self_attn.sinks"):
            p.copy_(draw(p.shape, 1.0))
        elif name.endswith("mlp.gate.weight"):
            p.copy_(draw(p.shape, 0.5))
        elif name.endswith("embed_tokens.weight"):
            p.copy_(draw(p.shape, 1.0))
        else:
            p.copy_(draw(p.shape, 0.25))

ids = torch.randint(3, config.vocab_size, (1, SEQ_LEN), generator=gen)
with torch.no_grad():
    logits = model(input_ids=ids).logits[0]

os.makedirs(OUT_DIR, exist_ok=True)
save_file({k: v.contiguous().float() for k, v in model.state_dict().items()}, os.path.join(OUT_DIR, "flint_tiny.safetensors"))
save_file({"input_ids": ids[0].to(torch.int32).contiguous(), "logits": logits.contiguous().float()},
          os.path.join(OUT_DIR, "flint_tiny_expected.safetensors"))
with open(os.path.join(OUT_DIR, "flint_tiny_config.json"), "w") as f:
    json.dump(config.to_dict(), f, indent=1)
print("wrote", OUT_DIR, "logit range", float(logits.min()), float(logits.max()))
