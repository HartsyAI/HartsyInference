#!/usr/bin/env python3
"""Synthetic GLM-4 parity reference: a tiny random-weight Glm4ForCausalLM (real HF architecture, real
partial-rotary RoPE, real sandwich norms, real 8:1 GQA ratio matching production's 32:2), run on a short
sequence to expose position-dependent RoPE drift and GQA-broadcast bugs. Dumps every weight (renamed to the
GgufLanguageModel/Glm4KeyMapper target key convention) plus per-layer hidden states and final logits to raw
little-endian float32 .bin files + a text manifest, for a C# test to load into GenericTransformer and diff.
No download needed -- this validates the ARCHITECTURE (RoPE pairing + partial-rotary + sandwich norm
placement + GQA broadcast), not a specific checkpoint's numerics.
"""
import os
import numpy as np
import torch
from transformers.models.glm4.configuration_glm4 import Glm4Config
from transformers.models.glm4.modeling_glm4 import Glm4ForCausalLM

torch.manual_seed(1234)

OUT_DIR = os.path.join(os.path.dirname(__file__), "glm4_ref")
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN = 64
HEADS = 16
KV_HEADS = 2          # 8:1 GQA ratio -- same order of magnitude as production's 32:2 (16:1)
HEAD_DIM = 8
INTERMEDIATE = 96
LAYERS = 2
VOCAB = 64
SEQ_LEN = 16  # long enough to expose position-dependent RoPE drift (position 0 hides the bug)

config = Glm4Config(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    intermediate_size=INTERMEDIATE,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    attention_bias=True,
    pad_token_id=None,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
)
model = Glm4ForCausalLM(config)
model.eval()

# Re-init every parameter with small, deterministic, non-degenerate values (default init can be too close
# to zero/identity to expose a wrong-pairing or wrong-broadcast bug -- want every dim to carry real signal).
gen = torch.Generator().manual_seed(42)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + 0.05 * torch.randn(p.shape, generator=gen))
        else:
            p.copy_(0.08 * torch.randn(p.shape, generator=gen))

input_ids = torch.randint(0, VOCAB, (1, SEQ_LEN), generator=gen)

with torch.no_grad():
    out = model(input_ids=input_ids, output_hidden_states=True)

logits = out.logits[0].numpy()  # [seq, vocab]
hidden_states = [h[0].numpy() for h in out.hidden_states]  # (layers+1) x [seq, hidden]

weights = {}
sd = model.state_dict()


def w(dst, src):
    weights[dst] = sd[src].numpy()


w("model.embed_tokens.weight", "model.embed_tokens.weight")
w("model.norm.weight", "model.norm.weight")
w("lm_head.weight", "lm_head.weight")
for i in range(LAYERS):
    p = f"model.layers.{i}."
    w(f"model.layers.{i}.input_layernorm.weight", p + "input_layernorm.weight")
    w(f"model.layers.{i}.self_attn.q_proj.weight", p + "self_attn.q_proj.weight")
    w(f"model.layers.{i}.self_attn.q_proj.bias", p + "self_attn.q_proj.bias")
    w(f"model.layers.{i}.self_attn.k_proj.weight", p + "self_attn.k_proj.weight")
    w(f"model.layers.{i}.self_attn.k_proj.bias", p + "self_attn.k_proj.bias")
    w(f"model.layers.{i}.self_attn.v_proj.weight", p + "self_attn.v_proj.weight")
    w(f"model.layers.{i}.self_attn.v_proj.bias", p + "self_attn.v_proj.bias")
    w(f"model.layers.{i}.self_attn.o_proj.weight", p + "self_attn.o_proj.weight")
    # HF's post_self_attn_layernorm (norm applied to attn OUTPUT, pre-residual) -> our post_attention_layernorm slot
    w(f"model.layers.{i}.post_attention_layernorm.weight", p + "post_self_attn_layernorm.weight")
    # HF's post_attention_layernorm (functionally the PRE-FFN norm) -> our pre_feedforward_layernorm slot
    w(f"model.layers.{i}.pre_feedforward_layernorm.weight", p + "post_attention_layernorm.weight")
    w(f"model.layers.{i}.post_feedforward_layernorm.weight", p + "post_mlp_layernorm.weight")
    # HF's fused gate_up_proj [2*inter, hidden] is already gate-then-up row-concatenated (see Glm4MLP.forward
    # chunk(2, dim=-1) on the OUTPUT, which for a Linear means rows [0:inter)=gate, [inter:2inter)=up).
    w(f"model.layers.{i}.mlp.gate_up_proj.weight", p + "mlp.gate_up_proj.weight")
    w(f"model.layers.{i}.mlp.down_proj.weight", p + "mlp.down_proj.weight")

manifest_lines = []


def dump(name, arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    fname = name.replace("/", "_").replace(".", "_") + ".bin"
    arr.tofile(os.path.join(OUT_DIR, fname))
    shape_str = ",".join(str(d) for d in arr.shape)
    manifest_lines.append(f"{name}\t{shape_str}\t{fname}")


dump("input_ids", input_ids.numpy().astype(np.float32))
dump("logits", logits)
for i, h in enumerate(hidden_states):
    dump(f"hidden_{i}", h)
for k, v in weights.items():
    dump(f"w::{k}", v)

meta = dict(
    hidden=HIDDEN, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, intermediate=INTERMEDIATE,
    layers=LAYERS, vocab=VOCAB, seq_len=SEQ_LEN, rope_theta=10000.0, rotary_dim=int(HEAD_DIM * 0.5),
    rms_eps=1e-6,
)
with open(os.path.join(OUT_DIR, "manifest.tsv"), "w") as f:
    f.write("\n".join(manifest_lines) + "\n")
with open(os.path.join(OUT_DIR, "meta.tsv"), "w") as f:
    for k, v in meta.items():
        f.write(f"{k}\t{v}\n")

print("wrote", OUT_DIR)
print("logits[-1][:8] =", logits[-1][:8])
print("hidden_states count:", len(hidden_states), "shapes:", [h.shape for h in hidden_states])
