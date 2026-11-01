#!/usr/bin/env python3
"""Tiny random MiniMax-M3 text reference for the CPU unit lane.

Builds a MiniMaxM3VLForCausalLM from transformers' own modeling code with a shrunken text config that keeps every structural
trait of MiniMax-M3.1-Flash: Gemma-form RMSNorm, per-head QK norm, partial RoPE, three dense layers then sparse MoE layers (sigmoid
router with a selection-only bias, routed scaling, one shared expert) using the clamped alpha-SwiGLU, and MiniMax Sparse Attention on
every layer (a block-max lightning indexer with its own key cache). The indexer blocks are 4 keys wide and keep 2 plus the local block,
so the 24-token prompt really prunes (6 blocks). Every parameter is re-drawn so each path carries signal, then the weights (HF on-disk
names, F32), config.json and the float32 reference logits are written to tests/HartsyInference.LLM.Tests/Fixtures/MiniMaxM3/.

Run: python -I tests/python-reference/dump_minimax_m3_tiny_reference.py  (transformers >= 5.22 with minimax_m3_vl)
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HartsyInference.LLM.Tests", "Fixtures", "MiniMaxM3")
SEQ_LEN = 24
LAYERS = 6

config = MiniMaxM3VLTextConfig(
    vocab_size=64, hidden_size=64, intermediate_size=16, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, num_local_experts=8, num_experts_per_tok=3, dense_intermediate_size=48, shared_intermediate_size=24,
    routed_scaling_factor=2.0, index_n_heads=2, index_head_dim=16, index_block_size=4, index_topk_blocks=2,
    index_local_blocks=1, pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
    layer_types=["minimax_m3_sparse"] * LAYERS,
    mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5})

torch.manual_seed(0)
model = MiniMaxM3VLForCausalLM(config).float().eval()
gen = torch.Generator().manual_seed(1234)


def draw(shape, std, mean=0.0):
    return mean + std * torch.randn(shape, generator=gen)


with torch.no_grad():
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if name.endswith("norm.weight"):
            p.copy_(draw(p.shape, 0.15))          # Gemma form: the stored value is the offset from 1
        elif name.endswith("e_score_correction_bias"):
            p.copy_(draw(p.shape, 0.3))
        elif name.endswith("mlp.gate.weight"):
            p.copy_(draw(p.shape, 0.5))
        elif name.endswith("embed_tokens.weight"):
            p.copy_(draw(p.shape, 1.0))
        elif "inv_freq" in name:
            continue
        else:
            p.copy_(draw(p.shape, 0.3))

ids = torch.randint(3, config.vocab_size, (1, SEQ_LEN), generator=gen)
with torch.no_grad():
    logits = model(input_ids=ids).logits[0]

os.makedirs(OUT_DIR, exist_ok=True)
state = {k: v.contiguous().float() for k, v in model.state_dict().items() if "inv_freq" not in k}
save_file(state, os.path.join(OUT_DIR, "minimax_m3_tiny.safetensors"))
save_file({"input_ids": ids[0].to(torch.int32).contiguous(), "logits": logits.contiguous().float()},
          os.path.join(OUT_DIR, "minimax_m3_tiny_expected.safetensors"))
with open(os.path.join(OUT_DIR, "minimax_m3_tiny_config.json"), "w") as f:
    json.dump(config.to_dict(), f, indent=1)
print("wrote", OUT_DIR, "logit range", float(logits.min()), float(logits.max()))
