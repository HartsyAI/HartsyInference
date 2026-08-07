# F-Lite Architecture

> **Source of truth:** [`fal-ai/f-lite/f_lite/model.py`](https://github.com/fal-ai/f-lite/blob/main/f_lite/model.py) and [`pipeline.py`](https://github.com/fal-ai/f-lite/blob/main/f_lite/pipeline.py). Config from [`Freepik/F-Lite/dit_model/config.json`](https://huggingface.co/Freepik/F-Lite/blob/main/dit_model/config.json).
> **License:** CreativeML OpenRAIL-M (Apache-compatible for inference).
> **Variants:** F-Lite (10B), F-Lite-7B (distilled), F-Lite-Texture.

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## TL;DR

F-Lite is a single-stream cross-attention DiT — closer to a vanilla DiT-XL than to Flux. 40 layers, hidden=3072, 12 heads (head_dim=256), patch_size=2 over a 16-channel Flux-VAE latent. Text encoder is T5-XXL with hidden states pulled from layer 17 (re-passed through final_layer_norm + dropout). 16 learnable register tokens are prepended to image tokens. AdaLN-Zero modulation produces 9 outputs per block (shift/scale/gate × {self-attn, cross-attn, MLP}). RoPE is 2D rotary precomputed on a 512×512 grid; the per-axis dim is `head_dim/2 = 128`. A V-residual mechanism (`lambda * v + (1-lambda) * v_0`) carries V from block 0 forward as a residual into every subsequent self-attn V.

## Config (canonical 10B)

```json
{
  "in_channels": 16, "patch_size": 2,
  "hidden_size": 3072, "depth": 40, "num_heads": 12,
  "mlp_ratio": 4.0, "cross_attn_input_size": 4096,
  "residual_v": true, "use_rope": true, "rope_base": 10000,
  "train_bias_and_rms": false, "dynamic_softmax_temperature": false
}
```

**Implications:**
- `train_bias_and_rms=false` — **no biases** on Q/K/V/proj/MLP linears; **RMSNorm has no learned scale** (pure normalization). The AdaLN linear is the only place a bias exists, by design.
- `residual_v=true` — adds a V-residual across blocks with a learnable scalar `lambda_param` (init 0.5). Block 0 produces `v_0`; blocks 1..N do `v = lambda * v_current + (1-lambda) * v_0` before SDPA.
- `head_dim = hidden_size / num_heads = 256`.
- `mlp_hidden = hidden_size * mlp_ratio = 12288`.
- `proj_out_dim = patch_size² * in_channels = 64`.

## Block forward (DiTBlock)

```
shift_sa, scale_sa, gate_sa,
shift_ca, scale_ca, gate_ca,
shift_mlp, scale_mlp, gate_mlp = AdaLN(SiLU(temb)).chunk(9, dim=-1)

# Self-attn
norm_x = RMSNorm(x)                                        # no learned scale
norm_x = norm_x * (1 + scale_sa) + shift_sa                # broadcast over seq
attn_out, v = self_attn(norm_x, v_0=v_0, rope=rope)
x = x + gate_sa * attn_out

# Cross-attn (over T5 context)
norm_x = RMSNorm(x)
norm_x = norm_x * (1 + scale_ca) + shift_ca
x = x + gate_ca * cross_attn(norm_x, context)

# MLP
norm_x = RMSNorm(x)
norm_x = norm_x * (1 + scale_mlp) + shift_mlp
x = x + gate_mlp * (Linear(GELU(Linear(norm_x))))
```

**Self-attn fused QKV:** `nn.Linear(hidden, 3*hidden, bias=False)` → reshape `b l (3 h d) -> 3 b h l d` → unbind. With `residual_v=true`: block 0 emits `v_0`; blocks 1+ apply `v = lambda * v + (1-lambda) * v_0` before RoPE+QK-norm. RoPE rotates Q and K (not V); QK-norm is per-head RMSNorm.

**Cross-attn:** Q from x via `nn.Linear(hidden, hidden, bias=False)`; K/V from context via `nn.Linear(4096, 2*hidden, bias=False)` then chunked. QK-norm applied; **no RoPE on cross-attn**.

## Top-level transformer forward

```
x = PatchEmbed(latent)                                     # Conv2d(16, 3072, k=2, s=2, bias=True)  — note bias=True here
x = concat([register_tokens.repeat(B), x], dim=1)          # 16 + N image tokens
rope = TwoDimRotary(x, hw, extend_with_register_tokens=16) # cos/sin tables

t_emb = timestep_embedding(t * 1000, hidden)               # sinusoidal cos+sin halves
t_emb = Linear(hidden, 4*hidden)(t_emb); SiLU; Linear(4*hidden, hidden)

v_0 = None
for block in blocks:
    x, v = block(x, context, t_emb, v_0, rope)
    if v_0 is None: v_0 = v                                # capture from block 0

x = x[:, 16:, :]                                           # drop register tokens
final_shift, final_scale = (Linear(SiLU(t_emb))).chunk(2)
x = RMSNorm(x) * (1 + final_scale) + final_shift
x = Linear(hidden, patch_size² * in_channels)(x)           # bias=True (zero-init)
return rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)')
```

## TwoDimRotary (specifics)

Per `model.py:242-291` — precomputes `freqs_hw_cos` / `freqs_hw_sin` of shape `[H_max, W_max, head_dim]` for `H_max = W_max = 512`. At runtime:

1. Slice `[0:h, 0:w]` and reshape to `[h*w, head_dim]`.
2. Prepend `extend_with_register_tokens=16` rows: `cos = ones`, `sin = zeros` (so register tokens get the identity rotation — they sit at "position 0").
3. Return shape `[1, 1, T+16, head_dim]`.

**`apply_rotary_emb`** (line 294) is **non-interleaved** half-split: `x1, x2 = x[..., :d], x[..., d:]`; output `[x1*cos + x2*sin, -x1*sin + x2*cos]`. Different from Flux's pairwise `[2i, 2i+1]` rotation — this is the "GPT-NeoX style" half-rotation.

## Pipeline + scheduler

T5 encode at layer 17 (i.e. `hidden_states[-8]` of 24 total encoder layers): pull layer 17, **re-apply** `text_encoder.encoder.final_layer_norm` then `text_encoder.encoder.dropout` (eval mode → no-op for dropout). Negative prompt either explicit or zeros.

**Dynamic-shift flow-match scheduler** (no diffusers Scheduler class — the loop is inline):

```
alpha = 2 * sqrt(image_token_size / (64 * 64))   # default; user can override
acc_latents = latents.clone()
for i in range(N, 0, -1):
    t       = i      / N
    t_next  = (i-1) / N
    t       = t      * alpha / (1 + (alpha-1) * t)
    t_next  = t_next * alpha / (1 + (alpha-1) * t_next)
    dt      = t - t_next
    velocity = transformer(latents, context, t)
    acc_latents = acc_latents + dt * velocity
    latents = acc_latents.clone()
```

CFG via dual-batch concat (uncond + cond → single forward → chunk). APG (Adaptive Projected Guidance) is optional — splits the CFG delta into parallel + orthogonal components and clamps the orthogonal one. Defer APG to v2.

VAE: AutoencoderKL (Flux Schnell VAE, 16-ch). Apply `latents = latents / scaling_factor + shift_factor` before decode.

## Weight key naming

Diffusers `dit_model/diffusion_pytorch_model.safetensors` keys, observed from the source:

```
patch_embed.patch_proj.{weight, bias}
register_tokens                        # [1, 16, hidden]
time_embed.0.{weight, bias}            # Linear (hidden → 4*hidden)
time_embed.2.{weight, bias}            # Linear (4*hidden → hidden)

blocks.{i}.norm1.weight                # only present if train_bias_and_rms=true (i.e. NOT in F-Lite 10B)
blocks.{i}.self_attn.qkv.weight        # [3*hidden, hidden]   no bias
blocks.{i}.self_attn.proj.weight       # [hidden, hidden]     no bias
blocks.{i}.self_attn.qk_norm.query_norm.weight   # only if trainable
blocks.{i}.self_attn.qk_norm.key_norm.weight     # only if trainable
blocks.{i}.self_attn.lambda_param      # [1] (only if residual_v)

blocks.{i}.norm2.weight                # only if trainable
blocks.{i}.cross_attn.q.weight         # [hidden, hidden]
blocks.{i}.cross_attn.context_kv.weight    # [2*hidden, 4096]
blocks.{i}.cross_attn.proj.weight
blocks.{i}.cross_attn.qk_norm.{query, key}_norm.weight  # only if trainable

blocks.{i}.norm3.weight                # only if trainable
blocks.{i}.mlp.0.weight                # Linear (hidden → mlp_hidden)
blocks.{i}.mlp.0.bias                  # only if train_bias_and_rms=true
blocks.{i}.mlp.2.weight                # Linear (mlp_hidden → hidden)
blocks.{i}.mlp.2.bias                  # only if train_bias_and_rms=true

blocks.{i}.adaLN_modulation.1.{weight, bias}     # always has bias (the "AdaLN-Zero" path)

# RoPE (rope.freqs_hw_cos, rope.freqs_hw_sin) — NOT IN CHECKPOINT (registered as buffers, recomputable)

final_modulation.1.{weight, bias}      # Linear (hidden → 2*hidden)
final_norm.weight                      # only if trainable
final_proj.{weight, bias}
```

For F-Lite-10B with `train_bias_and_rms=false`, the norms have no `.weight` keys, the QKV/MLP linears have no `.bias` keys, and the qk_norm sub-modules have no weights. The only biases in the entire transformer are `patch_embed.patch_proj.bias`, `time_embed.{0,2}.bias`, all `adaLN_modulation.1.bias`, `final_modulation.1.bias`, and `final_proj.bias`. Verify this against the actual safetensors at load time and route the load accordingly.

## VRAM (12 GB GPU target)

| Component | F16 | FP8 |
|---|---|---|
| Transformer (10B) | ~20 GB | ~10 GB |
| T5-XXL | ~9.5 GB | ~4.7 GB |
| VAE (Flux) | ~84 MB | n/a — keep at F16 |

10B at FP8 + T5 at FP8 + eviction discipline ≈ ~10.5 GB peak. Tight on a 12 GB GPU but feasible. The 7B distilled variant is more comfortable (~7 GB at FP8).

## Implementation notes for the C# port

- **Reuse `FluxVae`** — F-Lite uses Flux Schnell's 16-ch VAE verbatim. `VaeConfig.Flux` is correct.
- **Reuse `T5TextEncoder` with `T5TextEncoderConfig.Xxl`** — text encoder is HF T5-XXL. The "layer 17" extraction needs an `EncodeAtLayer(int layerIndex)` API on `T5TextEncoder` if it doesn't already exist; otherwise the simplest path is to add an option to return the requested intermediate hidden state.
- **No-affine RMSNorm** path needs adding (or fold it into the existing `RmsNorm` with a `weight=null` branch; existing project pattern uses `DiTUtils.LayerNormNoAffine` for the LayerNorm equivalent).
- **2D RoPE non-interleaved** — distinct from `FluxRope` (interleaved pairwise) and `ZImageRope` (multi-axis). New file: `FLiteRope.cs`.
- **V-residual** — block 0 returns its V; transformer caches and feeds it into blocks 1..N's self-attn. This is a **stateful** pass dependency, not a per-block thing.
- **CFG via batch-of-2 forward** — simpler than Flux/SD3 dual-pass; pack `[neg, pos]` into one batch and chunk after. Saves a forward pass at the cost of 2× peak activation memory (acceptable on 12 GB at 1024×1024 if FP8).
- **Scheduler is inline** — F-Lite doesn't use a diffusers `Scheduler` class, just a simple `t = t * α / (1 + (α-1) * t)` integrator in the pipeline. Implement directly in `FLitePipeline.GenerateFromTokens` rather than through `IScheduler` (avoids a 3-line scheduler shell).
- **Final layer is zero-init** at training; for inference this is irrelevant — load the trained weights as-is.
