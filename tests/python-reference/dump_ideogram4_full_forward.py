"""
Ideogram 4 transformer — Python step-by-step parity reference.

This is a SELF-CONTAINED reference for the C# `Ideogram4Transformer` (see
src/HartsyInference.Diffusion/Models/Denoisers/Ideogram4Transformer.cs). It does NOT
need the gated 9.3B checkpoint or the upstream `ideogram-oss/ideogram4` package: it
builds a TINY synthetic transformer (same architecture, small dims), runs a forward
pass in idiomatic PyTorch, and dumps the same tap points the C# `Ideogram4DebugDump`
writes. An independent re-implementation of the same documented spec is what makes the
diff meaningful — a transcription bug in EITHER side (wrong chunk order, wrong
rotate_half, double-silu, MRoPE interleave slicing, masking, permute) shows up as a
layer where the error jumps from ~1e-6 to ~1e-3+.

What it validates: the C# transformer math matches an independent implementation of the
spec on identical weights + inputs. What it does NOT validate: whether the spec's
ambiguous points are faithful to real upstream (norm_final affine? block eps? the
final-layer double-silu?). Those are exposed as the SPEC_* toggles below and must be
confirmed against real weights via Ideogram4GenerationTests producing a sensible image.

Pipeline:
  1. build tiny synthetic weights (seeded) -> save as bare-key safetensors
  2. generate fixed inputs (x, llm_features, position_ids, indicator, t) -> save .bin
  3. run the reference forward, capturing taps -> save .bin + index.json

The C# side (Ideogram4DiffTests) loads the SAME synthetic checkpoint + inputs, runs
Ideogram4Transformer.Forward with IDEOGRAM4_DEBUG_DIR set, and diff_ideogram4_layers.py
compares. Everything is CPU + float32, so it runs anywhere (no big GPU needed).

Run:
  python3 tests/python-reference/dump_ideogram4_full_forward.py
"""
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(1234)
np.random.seed(1234)

# ── Tiny config (mirrors Ideogram4Config defaults, scaled down) ────────────────────
EMB         = 64
LAYERS      = 2
HEADS       = 2
HEAD_DIM    = EMB // HEADS          # 32
FFN         = 128
ADALN       = 32
IN_CH       = 16
LLM_DIM     = 48
THETA       = 5_000_000
SECTION     = (6, 5, 5)            # sum == HEAD_DIM/2 == 16
# eps values: matched to the C# implementation-as-written so a correct port diffs ~1e-6.
BLOCK_EPS    = 1e-5                # Ideogram4Config.NormEps  (block RMS + QK norm)
LLM_NORM_EPS = 1e-6               # hard-coded in Ideogram4Transformer for llm_cond_norm
FINAL_LN_EPS = 1e-6              # ApplyFinalLayer LayerNormNoAffine

# ── Spec toggles (the doc's open questions). Defaults MATCH the C# port. Flip one and
#    re-run + re-diff to test that hypothesis against real upstream behavior. ─────────
SPEC_NORM_FINAL_AFFINE = False    # C# uses LayerNormNoAffine and DROPS final_layer.norm_final.{weight,bias}.
                                  #   The arch doc lists norm_final as an AFFINE LayerNorm. If upstream applies
                                  #   affine, set True (and the C# port has a bug at output_velocity).
SPEC_FINAL_DOUBLE_SILU = True     # C# ApplyFinalLayer applies silu() to the (already-silu'd) adaln_input again
                                  #   before final_layer.adaln_modulation. Blocks do NOT re-silu. Suspicious asymmetry.

# ── Fixed inputs ───────────────────────────────────────────────────────────────────
NUM_TEXT = 5
GRID_H, GRID_W = 3, 3
NUM_IMG = GRID_H * GRID_W          # 9
L = NUM_TEXT + NUM_IMG             # 14
TIMESTEP = 0.7
IMAGE_POSITION_OFFSET = 65536
LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ideogram4_reference_tensors")
FWD_DIR = os.path.join(OUT_DIR, "full_forward")
INP_DIR = os.path.join(FWD_DIR, "inputs")
os.makedirs(INP_DIR, exist_ok=True)


def save_bin(path, t):
    """Raw little-endian float32, C-contiguous (matches the C# DebugDump)."""
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    a.tofile(path)


def save_bin_i32(path, a):
    np.asarray(a, dtype=np.int32).ravel().tofile(path)


# ── Synthetic weights (bare post-converter keys consumed by Ideogram4Transformer.LoadWeights) ──
def randn(*shape, scale=0.02):
    return (torch.randn(*shape) * scale).to(torch.float32)


def build_weights():
    w = {}
    w["input_proj.weight"] = randn(EMB, IN_CH)
    w["input_proj.bias"]   = randn(EMB)
    w["llm_cond_norm.weight"] = (1.0 + randn(LLM_DIM))
    w["llm_cond_proj.weight"] = randn(EMB, LLM_DIM)
    w["llm_cond_proj.bias"]   = randn(EMB)
    w["t_embedding.mlp_in.weight"]  = randn(EMB, EMB)
    w["t_embedding.mlp_in.bias"]    = randn(EMB)
    w["t_embedding.mlp_out.weight"] = randn(EMB, EMB)
    w["t_embedding.mlp_out.bias"]   = randn(EMB)
    w["adaln_proj.weight"] = randn(ADALN, EMB)
    w["adaln_proj.bias"]   = randn(ADALN)
    w["embed_image_indicator.weight"] = randn(2, EMB)
    w["final_layer.linear.weight"] = randn(IN_CH, EMB)
    w["final_layer.linear.bias"]   = randn(IN_CH)
    w["final_layer.adaln_modulation.weight"] = randn(EMB, ADALN)
    w["final_layer.adaln_modulation.bias"]   = randn(EMB)
    # Present in the checkpoint per the arch doc; the C# port currently ignores them.
    w["final_layer.norm_final.weight"] = (1.0 + randn(EMB))
    w["final_layer.norm_final.bias"]   = randn(EMB)
    for i in range(LAYERS):
        p = f"layers.{i}"
        w[f"{p}.attention_norm1.weight"] = (1.0 + randn(EMB))
        w[f"{p}.attention_norm2.weight"] = (1.0 + randn(EMB))
        w[f"{p}.ffn_norm1.weight"]       = (1.0 + randn(EMB))
        w[f"{p}.ffn_norm2.weight"]       = (1.0 + randn(EMB))
        w[f"{p}.attention.qkv.weight"]    = randn(3 * EMB, EMB)
        w[f"{p}.attention.o.weight"]      = randn(EMB, EMB)
        w[f"{p}.attention.norm_q.weight"] = (1.0 + randn(HEAD_DIM))
        w[f"{p}.attention.norm_k.weight"] = (1.0 + randn(HEAD_DIM))
        w[f"{p}.feed_forward.w1.weight"] = randn(FFN, EMB)
        w[f"{p}.feed_forward.w3.weight"] = randn(FFN, EMB)
        w[f"{p}.feed_forward.w2.weight"] = randn(EMB, FFN)
        w[f"{p}.adaln_modulation.weight"] = randn(4 * EMB, ADALN)
        w[f"{p}.adaln_modulation.bias"]   = randn(4 * EMB)
    return w


# ── Reference math (mirrors the C# routines exactly, in float32) ───────────────────
def rms_norm(x, weight, eps):
    # NormKernels.RmsNorm: x * rsqrt(mean(x^2,-1)+eps) * weight
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def layernorm_no_affine(x, eps):
    # DiTUtils.LayerNormNoAffine: population variance, no affine.
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)


def build_sinusoidal(t, dim):
    # Ideogram4Transformer.BuildSinusoidal: scaled=1e4*t, freq_k=exp(-k*ln(1e4)/(half-1)),
    # layout [sin(scaled*freq) , cos(scaled*freq)].
    half = dim // 2
    scaled = 1e4 * t
    log_scale = math.log(1e4) / (half - 1)
    out = torch.zeros(dim, dtype=torch.float32)
    for k in range(half):
        freq = math.exp(-k * log_scale)
        ang = scaled * freq
        out[k] = math.sin(ang)
        out[half + k] = math.cos(ang)
    return out.unsqueeze(0)  # [1, dim]


def build_mrope_cos_sin(position_ids):
    # Ideogram4Mrope.BuildCosSin. position_ids: [1, L, 3] (t,h,w). Output cos/sin: [1, L, HEAD_DIM].
    half = HEAD_DIM // 2
    inv_freq = [1.0 / (THETA ** ((2 * k) / HEAD_DIM)) for k in range(half)]
    cos = torch.zeros(1, L, HEAD_DIM, dtype=torch.float32)
    sin = torch.zeros(1, L, HEAD_DIM, dtype=torch.float32)
    for s in range(L):
        pt = float(position_ids[0, s, 0]); ph = float(position_ids[0, s, 1]); pw = float(position_ids[0, s, 2])
        freqs = [pt * inv_freq[k] for k in range(half)]      # default temporal
        for j in range(SECTION[1]):                          # height -> k = 1,4,...
            k = 1 + 3 * j
            if k >= half: break
            freqs[k] = ph * inv_freq[k]
        for j in range(SECTION[2]):                          # width -> k = 2,5,...
            k = 2 + 3 * j
            if k >= half: break
            freqs[k] = pw * inv_freq[k]
        for k in range(half):
            c = math.cos(freqs[k]); sn = math.sin(freqs[k])
            cos[0, s, k] = c; cos[0, s, half + k] = c
            sin[0, s, k] = sn; sin[0, s, half + k] = sn
    return cos, sin


def apply_rotary(x, cos, sin):
    # x: [1, L, HEADS, HEAD_DIM]; cos/sin: [1, L, HEAD_DIM] broadcast over heads.
    # out[i]=x[i]*cos[i]+rotate_half(x)[i]*sin[i], rotate_half=cat(-x[half:],x[:half]).
    half = HEAD_DIM // 2
    c = cos.unsqueeze(2); s = sin.unsqueeze(2)
    lower = x[..., :half]; upper = x[..., half:]
    rot = torch.cat([-upper, lower], dim=-1)
    return x * c + rot * s


def attention(block, x):
    # fused QKV -> split -> per-head QK-RMSNorm -> MRoPE -> SDPA -> o proj
    qkv = F.linear(x, block["attention.qkv.weight"])               # [1,L,3*EMB]
    q, k, v = qkv.split(EMB, dim=-1)
    q = q.view(1, L, HEADS, HEAD_DIM)
    k = k.view(1, L, HEADS, HEAD_DIM)
    v = v.view(1, L, HEADS, HEAD_DIM)
    q = rms_norm(q, block["attention.norm_q.weight"], BLOCK_EPS)
    k = rms_norm(k, block["attention.norm_k.weight"], BLOCK_EPS)
    q = apply_rotary(q, MROPE_COS, MROPE_SIN)
    k = apply_rotary(k, MROPE_COS, MROPE_SIN)
    qh = q.permute(0, 2, 1, 3)  # [1,H,L,D]
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)                                  # [1,H,L,D]
    out = out.permute(0, 2, 1, 3).reshape(1, L, EMB)
    return F.linear(out, block["attention.o.weight"])


def block_forward(block, x, adaln_input):
    proj = F.linear(adaln_input, block["adaln_modulation.weight"], block["adaln_modulation.bias"])  # [1,4*EMB]
    scale_msa = 1.0 + proj[:, 0 * EMB:1 * EMB]
    gate_msa  = torch.tanh(proj[:, 1 * EMB:2 * EMB])
    scale_mlp = 1.0 + proj[:, 2 * EMB:3 * EMB]
    gate_mlp  = torch.tanh(proj[:, 3 * EMB:4 * EMB])
    scale_msa = scale_msa.unsqueeze(1); gate_msa = gate_msa.unsqueeze(1)
    scale_mlp = scale_mlp.unsqueeze(1); gate_mlp = gate_mlp.unsqueeze(1)

    # Attention sublayer (sandwich norm, scale-only, tanh-gated residual).
    norm1 = rms_norm(x, block["attention_norm1.weight"], BLOCK_EPS) * scale_msa
    attn = attention(block, norm1)
    attn_normed = rms_norm(attn, block["attention_norm2.weight"], BLOCK_EPS)
    after_attn = x + gate_msa * attn_normed

    # MLP sublayer (SwiGLU, sandwich norm).
    norm2 = rms_norm(after_attn, block["ffn_norm1.weight"], BLOCK_EPS) * scale_mlp
    gate = F.silu(F.linear(norm2, block["feed_forward.w1.weight"]))
    up   = F.linear(norm2, block["feed_forward.w3.weight"])
    mlp  = F.linear(gate * up, block["feed_forward.w2.weight"])
    mlp_normed = rms_norm(mlp, block["ffn_norm2.weight"], BLOCK_EPS)
    return after_attn + gate_mlp * mlp_normed


def main():
    weights = build_weights()
    save_file({k: v.clone() for k, v in weights.items()}, os.path.join(OUT_DIR, "transformer.safetensors"))

    # ── Inputs ──
    x = randn(1, L, IN_CH, scale=1.0)
    llm_features = randn(1, L, LLM_DIM, scale=1.0)
    position_ids = torch.zeros(1, L, 3, dtype=torch.float32)
    indicator = np.zeros(L, dtype=np.int32)
    for t in range(NUM_TEXT):
        position_ids[0, t, :] = float(t)
        indicator[t] = LLM_TOKEN_INDICATOR
    for idx in range(NUM_IMG):
        row, col = idx // GRID_W, idx % GRID_W
        s = NUM_TEXT + idx
        position_ids[0, s, 0] = IMAGE_POSITION_OFFSET
        position_ids[0, s, 1] = IMAGE_POSITION_OFFSET + row
        position_ids[0, s, 2] = IMAGE_POSITION_OFFSET + col
        indicator[s] = OUTPUT_IMAGE_INDICATOR

    save_bin(os.path.join(INP_DIR, "x.bin"), x)
    save_bin(os.path.join(INP_DIR, "llm_features.bin"), llm_features)
    save_bin(os.path.join(INP_DIR, "position_ids.bin"), position_ids)
    save_bin_i32(os.path.join(INP_DIR, "indicator.bin"), indicator)
    with open(os.path.join(INP_DIR, "timestep.bin"), "wb") as f:
        f.write(struct.pack("<f", TIMESTEP))
    meta = dict(emb=EMB, layers=LAYERS, heads=HEADS, head_dim=HEAD_DIM, ffn=FFN, adaln=ADALN,
                in_ch=IN_CH, llm_dim=LLM_DIM, theta=THETA, section=list(SECTION),
                block_eps=BLOCK_EPS, llm_norm_eps=LLM_NORM_EPS, final_ln_eps=FINAL_LN_EPS,
                num_text=NUM_TEXT, grid_h=GRID_H, grid_w=GRID_W, L=L, timestep=TIMESTEP,
                image_position_offset=IMAGE_POSITION_OFFSET,
                spec_norm_final_affine=SPEC_NORM_FINAL_AFFINE, spec_final_double_silu=SPEC_FINAL_DOUBLE_SILU)
    with open(os.path.join(INP_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── MRoPE (also dumped standalone so RoPE bugs localize without running the whole model) ──
    global MROPE_COS, MROPE_SIN
    MROPE_COS, MROPE_SIN = build_mrope_cos_sin(position_ids)

    index = []

    def tap(name, t):
        save_bin(os.path.join(FWD_DIR, f"{name.replace('.', '_')}.bin"), t)
        index.append(dict(name=name, shape=list(t.shape), file=f"{name.replace('.', '_')}.bin"))

    tap("mrope_cos", MROPE_COS)
    tap("mrope_sin", MROPE_SIN)

    text_mask = torch.tensor((indicator == LLM_TOKEN_INDICATOR), dtype=torch.float32).view(1, L, 1)
    img_mask  = torch.tensor((indicator == OUTPUT_IMAGE_INDICATOR), dtype=torch.float32).view(1, L, 1)

    # Masked image embedding: x *= img; input_proj; *= img
    x_proj = F.linear(x * img_mask, weights["input_proj.weight"], weights["input_proj.bias"]) * img_mask
    tap("input_proj", x_proj)

    # Masked text embedding: llm *= text; llm_cond_norm; llm_cond_proj; *= text
    llm_norm = rms_norm(llm_features * text_mask, weights["llm_cond_norm.weight"], LLM_NORM_EPS)
    llm_proj = F.linear(llm_norm, weights["llm_cond_proj.weight"], weights["llm_cond_proj.bias"]) * text_mask

    # h = image_emb + text_emb + image_indicator(row 1 image / row 0 else)
    h = x_proj + llm_proj
    ind_rows = torch.where(img_mask.bool(), weights["embed_image_indicator.weight"][1],
                           weights["embed_image_indicator.weight"][0])
    h = h + ind_rows
    tap("hidden_in", h)

    # Shared AdaLN conditioning: adaln_input = silu(adaln_proj(silu(mlp_out(silu(mlp_in(sinusoid(t)))))))
    sin_emb = build_sinusoidal(TIMESTEP, EMB)
    m1 = F.silu(F.linear(sin_emb, weights["t_embedding.mlp_in.weight"], weights["t_embedding.mlp_in.bias"]))
    t_cond = F.linear(m1, weights["t_embedding.mlp_out.weight"], weights["t_embedding.mlp_out.bias"])
    adaln_input = F.silu(F.linear(t_cond, weights["adaln_proj.weight"], weights["adaln_proj.bias"]))
    tap("adaln_input", adaln_input)

    # Blocks
    cur = h
    for i in range(LAYERS):
        block = {k[len(f"layers.{i}."):]: v for k, v in weights.items() if k.startswith(f"layers.{i}.")}
        cur = block_forward(block, cur, adaln_input)
        tap(f"layers.{i}", cur)

    # Final layer
    fin_in = F.silu(adaln_input) if SPEC_FINAL_DOUBLE_SILU else adaln_input
    scale = F.linear(fin_in, weights["final_layer.adaln_modulation.weight"], weights["final_layer.adaln_modulation.bias"])
    normed = layernorm_no_affine(cur, FINAL_LN_EPS)
    if SPEC_NORM_FINAL_AFFINE:
        normed = normed * weights["final_layer.norm_final.weight"] + weights["final_layer.norm_final.bias"]
    modulated = normed * (1.0 + scale.unsqueeze(1))
    out_vel = F.linear(modulated, weights["final_layer.linear.weight"], weights["final_layer.linear.bias"])
    tap("output_velocity", out_vel)

    with open(os.path.join(FWD_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote synthetic checkpoint + inputs + {len(index)} reference taps to:")
    print(f"  {FWD_DIR}")
    print(f"  config: emb={EMB} layers={LAYERS} heads={HEADS} head_dim={HEAD_DIM} L={L}")
    print(f"  toggles: norm_final_affine={SPEC_NORM_FINAL_AFFINE} final_double_silu={SPEC_FINAL_DOUBLE_SILU}")
    print(f"  output_velocity stats: mean={out_vel.mean():.6f} abs_mean={out_vel.abs().mean():.6f} std={out_vel.std():.6f}")


if __name__ == "__main__":
    main()
