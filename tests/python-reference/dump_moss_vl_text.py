"""Reference dump for the MOSS-VL-Realtime text decoder's first layers (float32 numpy transcription).

Transcribes the text forward of OpenMOSS's modeling_moss_vl.py (MossVLTextSelfAttention, MossVLTextCrossAttention,
the decoder layers, MossVLTextRotaryEmbedding with interleaved M-RoPE) and the single-image position rules of
MossVLModel.compute_position_ids / compute_vision_position_ids, over the real checkpoint's weights. Nothing from the
checkpoint's code is executed and no pickle is loaded: weights come through safetensors.

The prompt is fixed below (one image placeholder); the image's merged tokens come from the vision dump's merged.npy
(so run dump_moss_vl_vision.py first). Only the first LAYERS decoder layers run, which includes a gated cross-attention
layer, and the output is the hidden state after the last of them, before the final norm.

Usage: python dump_moss_vl_text.py <out_dir> <merged.npy> [<checkpoint_dir>]
"""

import json
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

IMAGE_TOKEN_ID = 151655
PROMPT_IDS = [151644, 872, 198, 151652, 151655, 151653, 9707, 1879]
MERGED_H, MERGED_W = 8, 12           # the vision dump's 16 x 24 patch grid, merged 2 x 2
LAYERS = 3                           # layers 0 (self), 1 (self), 2 (gated cross-attention)
CROSS_LAYERS = (2,)                  # within the first LAYERS layers; the checkpoint's cross layers are 2, 6, 10, ...
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 4096, 32, 8, 128
THETA = 5_000_000.0
SECTION = (24, 20, 20)
EPS = 1e-6
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "..", "Models", "LLM", "moss-vl-realtime")


class Weights:
    """Reads the checkpoint's tensors through its index, widened to float32."""

    def __init__(self, root):
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.root = root
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            with safe_open(os.path.join(self.root, self.weight_map[name]), framework="pt") as f:
                self.cache[name] = f.get_tensor(name).to(torch.float32).numpy()
        return self.cache[name]


def rms_norm(x, w):
    var = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
    return w * (x / np.sqrt(var + EPS)).astype(np.float32)


def linear(x, w):
    return x @ w.T


def silu(x):
    return x / (1.0 + np.exp(-x))


def positions(ids, image_token, merged_h, merged_w):
    """Text positions [3, T] and the image's vision positions [3, merged_h*merged_w + 1] (the last column is the separator)."""
    t = len(ids)
    text = np.zeros((3, t), dtype=np.int64)
    vision = np.zeros((3, merged_h * merged_w + 1), dtype=np.int64)
    regular_before, shift = 0, 0
    for i, tok in enumerate(ids):
        if tok != image_token:
            text[:, i] = regular_before + shift
            regular_before += 1
            continue
        t0 = regular_before + shift
        span = max(merged_h, merged_w)
        text[:, i] = t0 + span
        for y in range(merged_h):
            for x in range(merged_w):
                vision[:, y * merged_w + x] = (t0, t0 + y, t0 + x)
        vision[:, merged_h * merged_w] = t0 + span
        shift += span + 1
    return text, vision


def mrope_cos_sin(pos):
    """Interleaved M-RoPE: frequency pair j reads the temporal axis unless j % 3 selects H or W inside its section."""
    inv = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    freqs = pos[:, :, None].astype(np.float64) * inv[None, None, :]       # [3, n, 64]
    merged = freqs[0].copy()
    for dim, offset in ((1, 1), (2, 2)):
        length = SECTION[dim] * 3
        merged[:, offset:length:3] = freqs[dim][:, offset:length:3]
    emb = np.concatenate([merged, merged], axis=-1)                       # [n, 128]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    """x is [heads, n, 128]; cos and sin are [n, 128]."""
    return x * cos[None] + rotate_half(x) * sin[None]


def attention(q, k, v, causal):
    """q [HEADS, n, 128], k and v [KV_HEADS, m, 128]: grouped-query attention with the head-to-kv map of repeat_kv."""
    rep = HEADS // KV_HEADS
    k = np.repeat(k, rep, axis=0)
    v = np.repeat(v, rep, axis=0)
    scores = (q @ np.transpose(k, (0, 2, 1))) / math.sqrt(HEAD_DIM)
    if causal:
        n, m = scores.shape[1], scores.shape[2]
        mask = np.triu(np.ones((n, m), dtype=bool), k=1 + (m - n))
        scores = np.where(mask[None], -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs @ v


def self_layer(w, x, i, cos, sin):
    p = f"model.language_model.layers.{i}."
    n = x.shape[0]
    h = rms_norm(x, w.get(p + "input_layernorm.weight"))
    q = linear(h, w.get(p + "self_attn.q_proj.weight")).reshape(n, HEADS, HEAD_DIM)
    k = linear(h, w.get(p + "self_attn.k_proj.weight")).reshape(n, KV_HEADS, HEAD_DIM)
    v = linear(h, w.get(p + "self_attn.v_proj.weight")).reshape(n, KV_HEADS, HEAD_DIM)
    q = rms_norm(q, w.get(p + "self_attn.q_norm.weight")).transpose(1, 0, 2)
    k = rms_norm(k, w.get(p + "self_attn.k_norm.weight")).transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    out = attention(q, k, v, causal=True).transpose(1, 0, 2).reshape(n, HEADS * HEAD_DIM)
    x = x + linear(out, w.get(p + "self_attn.o_proj.weight"))
    h = rms_norm(x, w.get(p + "post_attention_layernorm.weight"))
    mlp = linear(silu(linear(h, w.get(p + "mlp.gate_proj.weight"))) * linear(h, w.get(p + "mlp.up_proj.weight")),
                 w.get(p + "mlp.down_proj.weight"))
    return x + mlp


def cross_layer(w, x, i, vision, text_cos, text_sin, vision_cos, vision_sin, row_mask):
    p = f"model.language_model.layers.{i}."
    n = x.shape[0]
    h = rms_norm(x, w.get(p + "input_layernorm.weight"))
    q = linear(h, w.get(p + "cross_attn.q_proj.weight")).reshape(n, HEADS, HEAD_DIM)
    q = rms_norm(q, w.get(p + "cross_attn.q_norm.weight")).transpose(1, 0, 2)
    q = apply_rope(q, text_cos, text_sin)
    m = vision.shape[0]
    k = linear(vision, w.get(p + "cross_attn.k_proj.weight")).reshape(m, KV_HEADS, HEAD_DIM)
    k = rms_norm(k, w.get(p + "cross_attn.k_norm.weight")).transpose(1, 0, 2)
    k = apply_rope(k, vision_cos, vision_sin)
    v = linear(vision, w.get(p + "cross_attn.v_proj.weight")).reshape(m, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
    out = attention(q, k, v, causal=False).transpose(1, 0, 2).reshape(n, HEADS * HEAD_DIM)
    attn = linear(out, w.get(p + "cross_attn.o_proj.weight")) * row_mask[:, None]
    x = x + np.tanh(w.get(p + "cross_attn_attn_gate")) * attn
    h = rms_norm(x, w.get(p + "post_attention_layernorm.weight"))
    mlp = linear(silu(linear(h, w.get(p + "mlp.gate_proj.weight"))) * linear(h, w.get(p + "mlp.up_proj.weight")),
                 w.get(p + "mlp.down_proj.weight")) * row_mask[:, None]
    return x + np.tanh(w.get(p + "cross_attn_mlp_gate")) * mlp


def main():
    out_dir = sys.argv[1]
    merged_path = sys.argv[2]
    root = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CHECKPOINT
    os.makedirs(out_dir, exist_ok=True)
    w = Weights(root)

    text_pos, vision_pos = positions(PROMPT_IDS, IMAGE_TOKEN_ID, MERGED_H, MERGED_W)
    text_cos, text_sin = mrope_cos_sin(text_pos)
    vision_cos, vision_sin = mrope_cos_sin(vision_pos)

    first_image = PROMPT_IDS.index(IMAGE_TOKEN_ID)
    row_mask = np.array([1.0 if i >= first_image else 0.0 for i in range(len(PROMPT_IDS))], dtype=np.float32)

    merged = np.load(merged_path).astype(np.float32)
    assert merged.shape == (MERGED_H * MERGED_W, HIDDEN), merged.shape
    vision = np.concatenate([merged, w.get("model.separator_token")[None]], axis=0)
    assert vision.shape[0] == vision_pos.shape[1]

    x = w.get("model.language_model.embed_tokens.weight")[np.array(PROMPT_IDS)].astype(np.float32)
    cross_input = None
    for i in range(LAYERS):
        if i in CROSS_LAYERS:
            cross_input = x
            x = cross_layer(w, x, i, vision, text_cos, text_sin, vision_cos, vision_sin, row_mask)
        else:
            x = self_layer(w, x, i, text_cos, text_sin)
    # The gated cross-attention output is small (gates ~1e-2), so the C# test checks the layer's own delta too.
    np.save(os.path.join(out_dir, "text_hidden.npy"), x.astype(np.float32))
    np.save(os.path.join(out_dir, "cross_input.npy"), cross_input.astype(np.float32))
    with open(os.path.join(out_dir, "prompt_ids.txt"), "w") as f:
        f.write(" ".join(str(t) for t in PROMPT_IDS))
    meta = {
        "prompt_ids": PROMPT_IDS, "image_token_id": IMAGE_TOKEN_ID, "merged_h": MERGED_H, "merged_w": MERGED_W,
        "layers": LAYERS, "text_positions": text_pos.tolist(), "vision_length": int(vision.shape[0]),
        "row_mask": row_mask.tolist(), "seed_source": "vision merged.npy",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.join(out_dir, "text_hidden.npy"), x.shape)


if __name__ == "__main__":
    main()
