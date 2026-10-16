#!/usr/bin/env python3
"""Tiny random Muse Spark reference for the CPU unit lane.

Builds a MuseSparkForConditionalGeneration from transformers' own modeling code with a shrunken text config that keeps
every structural trait of the released 816B checkpoint (centered and input-scaled RMSNorms, weightless embedding /
post-attention / per-head output norms, QK-norm with the MetaP Q scale, sigmoid attention output gate, norm-preserving
residual gates, sliding-window RoPE layers with a NoPE full-attention layer every fourth layer counted from the last,
the latent MoE with its pre/post expert norms, sigmoid routing with a selection-only bias, and the scaled final
softcap). Every parameter is re-drawn so each path carries signal, then the text weights (HF on-disk names, F32),
config.json and the float32 reference logits/hidden states are written to
tests/HartsyInference.LLM.Tests/Fixtures/MuseSpark/.

Run: python -I tests/python-reference/dump_muse_spark_tiny_reference.py  (transformers >= 5.21 with muse_spark)
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import (MuseSparkAudioConfig, MuseSparkConfig, MuseSparkForConditionalGeneration,
                          MuseSparkTextConfig, MuseSparkVisionConfig)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HartsyInference.LLM.Tests", "Fixtures",
                       "MuseSpark")
SEQ_LEN = 12

text = MuseSparkTextConfig(
    vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, sliding_window=6, num_local_experts=8, num_experts_per_tok=3,
    moe_hidden_size=32, moe_intermediate_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2)
vision = MuseSparkVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
audio = MuseSparkAudioConfig(hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1], out_hidden_size=text.hidden_size)
config = MuseSparkConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=audio.to_dict(),
                         out_hidden_size=32, projector_hidden_size=16)

torch.manual_seed(0)
model = MuseSparkForConditionalGeneration(config).float().eval()
gen = torch.Generator().manual_seed(1234)


def draw(shape, std, mean=0.0):
    return mean + std * torch.randn(shape, generator=gen)


OFFSET_NORMS = ("input_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "pre_expert_norm.weight", "post_expert_norm.weight")
with torch.no_grad():
    for name, p in model.named_parameters():
        if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
            continue
        if name.endswith(OFFSET_NORMS):
            p.copy_(draw(p.shape, 0.1))
        elif name.endswith("language_model.norm.weight"):
            p.copy_(draw(p.shape, 0.1, 1.0))
        elif name.endswith("_residual_gate.gate"):
            p.copy_(draw(p.shape, 0.8))
        elif name.endswith("mlp.gate.weight"):
            p.copy_(draw(p.shape, 0.3))
        elif name.endswith("embed_tokens.weight"):
            p.copy_(draw(p.shape, 1.0))
        elif name == "lm_head.weight":
            p.copy_(draw(p.shape, 1.5))
        elif ".experts." in name:
            p.copy_(draw(p.shape, 0.15))
        else:
            p.copy_(draw(p.shape, 0.08))
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(draw(b.shape, 0.05))

input_ids = torch.randint(3, text.vocab_size, (1, SEQ_LEN), generator=gen)
with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

os.makedirs(OUT_DIR, exist_ok=True)
state = model.state_dict()
weights = {k: v.detach().float().contiguous() for k, v in state.items()
           if k.startswith("model.language_model.") or k == "lm_head.weight"}
save_file(weights, os.path.join(OUT_DIR, "muse_spark_tiny.safetensors"))
expected = {
    "input_ids": input_ids[0].to(torch.int32).contiguous(),
    "logits": out.logits[0].float().contiguous(),
}
for i, h in enumerate(out.hidden_states):
    expected[f"hidden_{i}"] = h[0].float().contiguous()
save_file(expected, os.path.join(OUT_DIR, "muse_spark_tiny_expected.safetensors"))
with open(os.path.join(OUT_DIR, "muse_spark_tiny_config.json"), "w", encoding="utf-8") as f:
    json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    f.write("\n")
print(f"wrote {len(weights)} weights, logits {tuple(out.logits.shape)}, {len(out.hidden_states)} hidden states")
