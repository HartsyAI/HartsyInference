"""Reference dump for the MOSS-VL-Realtime vision tower (float32 numpy transcription).

Transcribes the vision forward of OpenMOSS's modeling_moss_vl.py (MossVLVisionModel, MossVLVisionBlock,
MossVLVisionPatchMerger, fast_pos_embed_interpolate, rot_pos_emb) over the real checkpoint's weights, without
executing any checkpoint code: weights are read with safetensors and nothing is unpickled. Used by the C# parity
test; the outputs are not committed.

Usage: python dump_moss_vl_vision.py <out_dir> [<checkpoint_dir>]
"""

import json
import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

GRID_T, GRID_H, GRID_W = 1, 16, 24          # patch-grid (multiples of the merge size 2)
SEED = 20261108
EPS = 1e-6
MERGE = 2
PATCH = 16
IN_CH = 3
TEMPORAL = 1
HEADS = 16
DEEPSTACK = [8, 16, 24]


class Weights:
    """Reads the vision tower's tensors from the sharded checkpoint via its index (no unpickling)."""

    def __init__(self, root):
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.root = root
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            shard = self.weight_map[name]
            with safe_open(os.path.join(self.root, shard), framework="pt") as f:
                self.cache[name] = f.get_tensor(name).to(torch.float32).numpy()
        return self.cache[name]


def layer_norm(x, w, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * w + b


def linear(x, w, b):
    return x @ w.T + (b if b is not None else 0.0)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    erf = np.vectorize(math.erf, otypes=[np.float32])
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def pos_embed_interpolate(table, t, h, w):
    """fast_pos_embed_interpolate for one image: bilinear over the learned grid, in merge-block token order."""
    side = int(math.sqrt(table.shape[0]))
    hs = np.linspace(0, side - 1, h, dtype=np.float32)
    ws = np.linspace(0, side - 1, w, dtype=np.float32)
    hf, wf = hs.astype(np.int64), ws.astype(np.int64)
    hc = np.clip(hf + 1, 0, side - 1)
    wc = np.clip(wf + 1, 0, side - 1)
    dh, dw = (hs - hf)[:, None], (ws - wf)[None, :]
    out = np.zeros((h, w, table.shape[1]), dtype=np.float32)
    for hi, wi, hw in ((hf, wf, (1 - dh) * (1 - dw)), (hf, wc, (1 - dh) * dw),
                       (hc, wf, dh * (1 - dw)), (hc, wc, dh * dw)):
        idx = (hi[:, None] * side + wi[None, :])
        out += table[idx.reshape(-1)].reshape(h, w, -1) * hw.reshape(h, w, 1)
    # raster -> merge-block order, repeated over frames
    out = np.tile(out.reshape(h, w, -1)[None], (t, 1, 1, 1))
    out = out.reshape(t, h // MERGE, MERGE, w // MERGE, MERGE, -1).transpose(0, 1, 3, 2, 4, 5)
    return out.reshape(t * h * w, -1)


def rotary_cos_sin(t, h, w, head_dim):
    """rot_pos_emb + cat(emb, emb): cos/sin of shape [tokens, head_dim] in merge-block token order."""
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, 2, dtype=np.float32) / half))
    max_hw = max(h, w)
    freq = np.outer(np.arange(max_hw, dtype=np.float32), inv_freq)
    coords = []
    for _ in range(t):
        for hb in range(h // MERGE):
            for wb in range(w // MERGE):
                for mh in range(MERGE):
                    for mw in range(MERGE):
                        coords.append((hb * MERGE + mh, wb * MERGE + mw))
    coords = np.array(coords, dtype=np.int64)
    emb = np.concatenate([freq[coords[:, 0]], freq[coords[:, 1]]], axis=-1)
    emb = np.concatenate([emb, emb], axis=-1)
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def attention(x, weights, cos, sin):
    seq = x.shape[0]
    qkv = linear(x, weights["qkv.w"], weights["qkv.b"]).reshape(seq, 3, HEADS, -1)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    c, s = cos[:, None, :], sin[:, None, :]
    q = q * c + rotate_half(q) * s
    k = k * c + rotate_half(k) * s
    q, k, v = (a.transpose(1, 0, 2) for a in (q, k, v))            # [heads, seq, hd]
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = (probs @ v).transpose(1, 0, 2).reshape(seq, -1)
    return linear(out, weights["proj.w"], weights["proj.b"])


def block_forward(x, bw, cos, sin):
    x = x + attention(layer_norm(x, bw["norm1.w"], bw["norm1.b"]), bw, cos, sin)
    h = linear(layer_norm(x, bw["norm2.w"], bw["norm2.b"]), bw["fc1.w"], bw["fc1.b"])
    return x + linear(gelu_tanh(h), bw["fc2.w"], bw["fc2.b"])


def main():
    out_dir = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "..", "..", "Models", "LLM", "moss-vl-realtime")
    os.makedirs(out_dir, exist_ok=True)
    W = Weights(root)
    V = "model.visual"

    rng = np.random.default_rng(SEED)
    seq = GRID_T * GRID_H * GRID_W
    flat_dim = IN_CH * TEMPORAL * PATCH * PATCH
    patches = rng.uniform(-1.0, 1.0, size=(seq, flat_dim)).astype(np.float32)   # block-ordered, (c, t, kh, kw)

    proj = W.get(f"{V}.patch_embed.proj.weight").reshape(1152, flat_dim)
    x = linear(patches, proj, W.get(f"{V}.patch_embed.proj.bias"))
    x = x + pos_embed_interpolate(W.get(f"{V}.pos_embed.weight"), GRID_T, GRID_H, GRID_W)
    cos, sin = rotary_cos_sin(GRID_T, GRID_H, GRID_W, 72)
    np.save(os.path.join(out_dir, "patches.npy"), patches)
    np.save(os.path.join(out_dir, "h0.npy"), x)

    taps = {}
    for i in range(27):
        bw = {
            "norm1.w": W.get(f"{V}.blocks.{i}.norm1.weight"), "norm1.b": W.get(f"{V}.blocks.{i}.norm1.bias"),
            "norm2.w": W.get(f"{V}.blocks.{i}.norm2.weight"), "norm2.b": W.get(f"{V}.blocks.{i}.norm2.bias"),
            "qkv.w": W.get(f"{V}.blocks.{i}.attn.qkv.weight"), "qkv.b": W.get(f"{V}.blocks.{i}.attn.qkv.bias"),
            "proj.w": W.get(f"{V}.blocks.{i}.attn.proj.weight"), "proj.b": W.get(f"{V}.blocks.{i}.attn.proj.bias"),
            "fc1.w": W.get(f"{V}.blocks.{i}.mlp.linear_fc1.weight"), "fc1.b": W.get(f"{V}.blocks.{i}.mlp.linear_fc1.bias"),
            "fc2.w": W.get(f"{V}.blocks.{i}.mlp.linear_fc2.weight"), "fc2.b": W.get(f"{V}.blocks.{i}.mlp.linear_fc2.bias"),
        }
        x = block_forward(x, bw, cos, sin)
        if i in DEEPSTACK:
            taps[i] = x
    np.save(os.path.join(out_dir, "block_last.npy"), x)

    # Merger: each stream is normalized per token (norms[0] = last block, norms[1..3] = taps 8, 16, 24),
    # the four streams are concatenated per token, then four consecutive tokens (one 2x2 merge block) form one row.
    streams = [x] + [taps[i] for i in DEEPSTACK]
    normed = [layer_norm(s, W.get(f"{V}.merger.norms.{n}.weight"), W.get(f"{V}.merger.norms.{n}.bias"))
              for n, s in enumerate(streams)]
    per_token = np.concatenate(normed, axis=-1)                              # [seq, 4 * 1152]
    grouped = per_token.reshape(seq // (MERGE * MERGE), -1)                  # [seq/4, 18432]
    h = gelu_erf(linear(grouped, W.get(f"{V}.merger.linear_fc1.weight"), W.get(f"{V}.merger.linear_fc1.bias")))
    merged = linear(h, W.get(f"{V}.merger.linear_fc2.weight"), W.get(f"{V}.merger.linear_fc2.bias"))
    np.save(os.path.join(out_dir, "merged.npy"), merged.astype(np.float32))

    meta = {"grid": [GRID_T, GRID_H, GRID_W], "seed": SEED, "checkpoint": os.path.abspath(root),
            "merged_shape": list(merged.shape), "torch": torch.__version__, "numpy": np.__version__}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
