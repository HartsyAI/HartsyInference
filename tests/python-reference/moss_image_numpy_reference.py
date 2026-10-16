"""
Builds the tiny MOSS-Image transformer fixture for MossImageParityTests from a numpy transcription of the
reference module, for boxes without torch.

The transcription follows OpenMOSS's ComfyUI-native module (MOSS-Image comfyui/moss-image/model.py, the same
math as SGLang's runtime/models/dits/moss_image.py) line for line, in float64 over weights rounded to F16:
  - 2 biased text-refiner blocks (fused qkv, per-head Q/K RMSNorm, SiTU-GLU), RoPE on the text slice only;
  - joint blocks over [image ; text] under ONE modulation shared by every block (no per-block adaLN);
  - SiTU(hidden, gate) = (4*tanh(gate/4)*sigmoid(gate)) * (25*tanh(hidden/25));
  - 4-axis (r, t, h, w) rotate-half RoPE: head_dim/8 bands per axis, image t = text length, h/w centred;
  - the timestep sinusoid in float64 as [cos, sin]; RoPE phases in float32, as the module builds them.
dump_moss_image_reference.py re-runs the stored weights and inputs through the module itself (torch + ComfyUI)
and checks these outputs against it, so the transcription is verified wherever torch is available.

Cases (keys prefixed in the output file):
  full.*     every block, timestep 500
  skip.*     joint block 1 skipped (the reference's uncond_skip_layers), timestep 937
  text.*     the refined text tokens (timestep-independent)

Usage: python -I moss_image_numpy_reference.py <output .safetensors>
Requires: numpy, safetensors.
"""
import sys

import numpy as np
from safetensors.numpy import save_file

DIM = 64
HEADS = 2
FFN = 96
LAYERS = 3
TEXT_DIM = 48
FREQ_DIM = 32
LATENT = 16
ROPE_BASE = 10000.0
EPS = 1e-6
TEXT_TOKENS = 5
GRID = (3, 4)
SEED = 20261016


def weights(rng):
    head = DIM // HEADS
    shapes = {
        "video_embedder.weight": (DIM, LATENT), "video_embedder.bias": (DIM,),
        "text_embedder.proj.weight": (DIM, TEXT_DIM), "text_embedder.proj.bias": (DIM,),
        "time_embedding.0.weight": (DIM, FREQ_DIM), "time_embedding.0.bias": (DIM,),
        "time_embedding.2.weight": (DIM, DIM), "time_embedding.2.bias": (DIM,),
        "shared_block_modulation.weight": (6 * DIM, DIM), "shared_block_modulation.bias": (6 * DIM,),
        "final_norm_video.weight": (DIM,),
        "final_linear_video.weight": (LATENT, DIM),
        "video_head_modulation.weight": (2 * DIM, DIM), "video_head_modulation.bias": (2 * DIM,),
    }
    for b in range(2):
        p = f"text_embedder.blocks.{b}"
        shapes.update({
            f"{p}.norm1.weight": (DIM,), f"{p}.norm2.weight": (DIM,),
            f"{p}.q_norm.weight": (head,), f"{p}.k_norm.weight": (head,),
            f"{p}.qkv.weight": (3 * DIM, DIM), f"{p}.qkv.bias": (3 * DIM,),
            f"{p}.proj.weight": (DIM, DIM), f"{p}.proj.bias": (DIM,),
            f"{p}.ffn_gate.weight": (FFN, DIM), f"{p}.ffn_gate.bias": (FFN,),
            f"{p}.ffn_up.weight": (FFN, DIM), f"{p}.ffn_up.bias": (FFN,),
            f"{p}.ffn_down.weight": (DIM, FFN), f"{p}.ffn_down.bias": (DIM,),
        })
    for i in range(LAYERS):
        p = f"layers.{i}"
        shapes.update({
            f"{p}.norm1.weight": (DIM,), f"{p}.norm2.weight": (DIM,),
            f"{p}.q_norm.weight": (head,), f"{p}.k_norm.weight": (head,),
            f"{p}.proj.q.weight": (DIM, DIM), f"{p}.proj.k.weight": (DIM, DIM),
            f"{p}.proj.v.weight": (DIM, DIM), f"{p}.proj.o.weight": (DIM, DIM),
            f"{p}.proj.ffn_up_gate.weight": (2 * FFN, DIM), f"{p}.proj.ffn_down.weight": (DIM, FFN),
        })
    out = {}
    for name, shape in shapes.items():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("_norm.weight") \
                or name == "final_norm_video.weight":
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith(".bias"):
            value = 0.1 * rng.standard_normal(shape)
        else:
            value = rng.standard_normal(shape) / np.sqrt(shape[-1])
        out[name] = value.astype(np.float16)
    return out


def linear(w, x, name, bias=True):
    y = x @ w[f"{name}.weight"].astype(np.float64).T
    if bias:
        y = y + w[f"{name}.bias"].astype(np.float64)
    return y


def rms(x, weight):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * weight.astype(np.float64)


def situ(hidden, gate):
    return (4.0 * np.tanh(gate / 4.0) / (1.0 + np.exp(-gate))) * (25.0 * np.tanh(hidden / 25.0))


def rope_table(height, width, text_length):
    head = DIM // HEADS
    bands_n = head // 8
    bands = (1.0 / (np.float32(ROPE_BASE) ** (np.arange(bands_n, dtype=np.float32) / np.float32(bands_n)))).astype(np.float32)
    h, w = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    image = np.stack((np.zeros_like(h), np.full_like(h, text_length), h, w), -1).reshape(-1, 4).astype(np.float32)
    image[:, 2] -= np.float32((height - 1) / 2)
    image[:, 3] -= np.float32((width - 1) / 2)
    text = np.zeros((text_length, 4), dtype=np.float32)
    text[:, 1] = np.arange(text_length, dtype=np.float32)
    phase = (np.concatenate((image, text), 0)[:, :, None] * bands).astype(np.float32)   # [N, 4, bands]
    sin = np.sin(phase.astype(np.float64)).reshape(len(phase), -1)                       # [N, head/2]
    cos = np.cos(phase.astype(np.float64)).reshape(len(phase), -1)
    return sin, cos


def rotate(x, sin, cos):
    # x [S, heads, head]; tables [S, head/2] shared by every head (rotate-half).
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    s, c = sin[:, None, :], cos[:, None, :]
    return np.concatenate((a * c - b * s, b * c + a * s), -1)


def attention(q, k, v, sin, cos):
    seq = q.shape[0]
    head = DIM // HEADS
    q = rotate(q.reshape(seq, HEADS, head), sin, cos).transpose(1, 0, 2)
    k = rotate(k.reshape(seq, HEADS, head), sin, cos).transpose(1, 0, 2)
    v = v.reshape(seq, HEADS, head).transpose(1, 0, 2)
    logits = q @ k.transpose(0, 2, 1) / np.sqrt(head)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    return (p @ v).transpose(1, 0, 2).reshape(seq, DIM)


def head_norm(x, weight):
    seq = x.shape[0]
    head = DIM // HEADS
    return rms(x.reshape(seq, HEADS, head), weight).reshape(seq, DIM)


def text_block(w, p, x, sin, cos):
    qkv = linear(w, rms(x, w[f"{p}.norm1.weight"]), f"{p}.qkv")
    q, k, v = np.split(qkv, 3, -1)
    q = head_norm(q, w[f"{p}.q_norm.weight"])
    k = head_norm(k, w[f"{p}.k_norm.weight"])
    x = x + linear(w, attention(q, k, v, sin, cos), f"{p}.proj")
    y = rms(x, w[f"{p}.norm2.weight"])
    return x + linear(w, situ(linear(w, y, f"{p}.ffn_up"), linear(w, y, f"{p}.ffn_gate")), f"{p}.ffn_down")


def joint_block(w, p, x, sin, cos, mod):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = rms(x, w[f"{p}.norm1.weight"]) * (1 + scale_a) + shift_a
    q = head_norm(linear(w, y, f"{p}.proj.q", bias=False), w[f"{p}.q_norm.weight"])
    k = head_norm(linear(w, y, f"{p}.proj.k", bias=False), w[f"{p}.k_norm.weight"])
    v = linear(w, y, f"{p}.proj.v", bias=False)
    x = x + linear(w, attention(q, k, v, sin, cos), f"{p}.proj.o", bias=False) * gate_a
    y = rms(x, w[f"{p}.norm2.weight"]) * (1 + scale_f) + shift_f
    hidden, gate = np.split(linear(w, y, f"{p}.proj.ffn_up_gate", bias=False), 2, -1)
    return x + linear(w, situ(hidden, gate), f"{p}.proj.ffn_down", bias=False) * gate_f


def time_vector(w, timestep):
    half = FREQ_DIM // 2
    freq = 10000.0 ** (-np.arange(half, dtype=np.float64) / half)
    phase = np.float64(np.float32(timestep)) * freq
    embedding = np.concatenate((np.cos(phase), np.sin(phase))).astype(np.float32).astype(np.float64)
    hidden = linear(w, embedding, "time_embedding.0")
    hidden = hidden / (1.0 + np.exp(-hidden))
    return linear(w, hidden, "time_embedding.2")


def refine_text(w, context, sin, cos, image_tokens):
    text = linear(w, context, "text_embedder.proj")
    for b in range(2):
        text = text_block(w, f"text_embedder.blocks.{b}", text, sin[image_tokens:], cos[image_tokens:])
    return text


def forward(w, latent, timestep, context, skip=()):
    height, width = GRID
    tokens = height * width
    time = time_vector(w, timestep)
    mod = linear(w, time, "shared_block_modulation").reshape(6, DIM)
    head = linear(w, time, "video_head_modulation").reshape(2, DIM)
    sin, cos = rope_table(height, width, context.shape[0])
    text = refine_text(w, context, sin, cos, tokens)
    x = np.concatenate((linear(w, latent, "video_embedder"), text), 0)
    for i in range(LAYERS):
        if i not in skip:
            x = joint_block(w, f"layers.{i}", x, sin, cos, mod)
    x = rms(x[:tokens], w["final_norm_video.weight"]) * (1 + head[1]) + head[0]
    return linear(w, x, "final_linear_video", bias=False), text


def main():
    output = sys.argv[1]
    rng = np.random.default_rng(SEED)
    w = weights(rng)
    tokens = GRID[0] * GRID[1]
    latent = rng.standard_normal((tokens, LATENT)).astype(np.float32)
    context = rng.standard_normal((TEXT_TOKENS, TEXT_DIM)).astype(np.float32)
    full, text = forward(w, latent.astype(np.float64), 500.0, context.astype(np.float64))
    skip, _ = forward(w, latent.astype(np.float64), 937.0, context.astype(np.float64), skip=(1,))

    tensors = {f"weights.{name}": value for name, value in w.items()}
    head = DIM // HEADS
    bands_n = head // 8
    tensors["weights.rope.bands"] = (1.0 / (np.float32(ROPE_BASE) ** (np.arange(bands_n, dtype=np.float32)
                                                                        / np.float32(bands_n)))).astype(np.float32)
    tensors["inputs.latent"] = latent[None]
    tensors["inputs.context"] = context[None]
    tensors["full.velocity"] = full.astype(np.float32)[None]
    tensors["skip.velocity"] = skip.astype(np.float32)[None]
    tensors["text.refined"] = text.astype(np.float32)[None]
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, output)
    for key in sorted(k for k in tensors if not k.startswith("weights.")):
        t = tensors[key].astype(np.float64)
        print(f"{key}: {t.shape} mean={t.mean():.6f} std={t.std():.6f}")


if __name__ == "__main__":
    main()
