# Microsoft Lens Architecture — Research Notes

> **Status:** Complete (read-only from upstream code; no checkpoint inspected on disk yet) | **Last Updated:** 2026-05-27 | **Needed Before:** `LensTransformer`, `LensGptOssEncoder`, `LensPipeline` implementation
>
> **Sources of truth:**
> - GitHub: [microsoft/Lens](https://github.com/microsoft/Lens) — `lens/transformer.py` (LensTransformer2DModel, ~700 lines), `lens/pipeline.py` (LensPipeline, ~580 lines), `lens/text_encoder.py` (LensGptOssEncoder, ~130 lines), `lens/resolution.py`, `lens/reasoner.py`
> - HuggingFace: [microsoft/Lens](https://huggingface.co/microsoft/Lens) (RL-tuned, 30.7 GB), [microsoft/Lens-Turbo](https://huggingface.co/microsoft/Lens-Turbo) (4-step distilled), `microsoft/Lens-Base` (supervised baseline)
> - `transformer/config.json` and `text_encoder/config.json` extracted verbatim below
>
> **License:** MIT (for both DiT weights and the inference code). The GPT-OSS text encoder it depends on is licensed separately as Apache-2.0 by OpenAI under `openai/gpt-oss-20b` upstream — Microsoft re-publishes a Lens-trimmed copy alongside Lens.

## Summary

Lens is a **3.8B-parameter dual-stream MMDiT** image generator from Microsoft Research. Released 2026-05-25. Architecturally similar to Flux's double-stream block, but at one-fifth the parameter count: 48 layers × hidden=1536 × 24 heads × 64 head-dim, with **3-axis complex-polar RoPE** (frame=8, h=28, w=28; total 64 = head_dim) and a **SwiGLU MLP** (hidden=4096). The headline trick is the text encoder: it does **not** use T5, CLIP, or any diffusion-tuned LLM — instead it concatenates four layer hidden states (layers 5, 11, 17, 23) from a frozen **GPT-OSS** MoE causal LM, normalizes each layer with its own RMSNorm, and projects the concat (4×2880 = 11520) down to 1536. This is what Microsoft means by "massive text encoder training on GPT image outputs" — quality scales because the conditioning signal is much richer than a single T5/CLIP pool, while inference stays cheap because the DiT itself is small.

VAE is the same **Flux.2 semantic VAE** (`AutoencoderKLFlux2`) already implemented for Flux.2 Klein in this codebase: 16× spatial downsample × 4× channel patchify = 32-channel latent → 128-channel transformer input. Scheduler is the standard `FlowMatchEulerDiscreteScheduler` with an **empirical mu** computed per-resolution. CFG is a dual-pass batch-of-2 with **norm-rescaling** (Microsoft's twist — combined prediction is rescaled to match the conditional branch's L2 norm per token). Default sampling is **20 steps, CFG 5.0** for the RL-tuned variant; **4 steps, CFG 1.0** for Turbo; **50 steps, CFG 5.0** for Base.

For HartsyInference the genuinely new piece is **GPT-OSS as a text encoder** — Mixture-of-Experts (32 local experts, 4 active per token, MXFP4-native), GQA (64 query heads : 8 KV heads), alternating sliding/full attention (window 128) — none of which the existing `LlamaStyleEncoder` supports. The transformer block itself is close enough to Flux's double-stream that ~70% of the block code can be reused (modulation, fused QKV, joint attention, AdaLN-Continuous final). RoPE is its own thing (complex-polar with scale_rope=True centered around zero), but mathematically identical to Qwen-Image's 3-axis RoPE just with different per-axis dims.

## Detailed Findings

### Model variants

| Variant | Repo | Steps | CFG | Training | Size (BF16) | Use case |
|---|---|---|---|---|---|---|
| **Lens** (default) | `microsoft/Lens` | 20 | 5.0 | Pre-train → SFT → RL | 30.7 GB total (~7.6 GB DiT + ~13 GB GPT-OSS MXFP4 + ~10 GB Flux.2 VAE FP16) | Production quality |
| **Lens-Turbo** | `microsoft/Lens-Turbo` | 4 | 1.0 | Above + distillation | same | Fast inference; **no CFG needed** |
| **Lens-Base** | `microsoft/Lens-Base` | 50 | 5.0 | Pre-train + SFT only | same | Reference / fair-comparison baseline |

All three share the **exact same transformer architecture** — only the weights and the recommended sampler settings differ. One C# `LensConfig` covers all three; the variant is just a checkpoint selection plus a default `{steps, cfg}` tuple.

### Transformer config (`transformer/config.json`, verbatim)

```json
{
  "_class_name": "LensTransformer2DModel",
  "_diffusers_version": "0.37.1",
  "attention_head_dim": 64,
  "axes_dims_rope": [8, 28, 28],
  "enc_hidden_dim": 2880,
  "gate_mlp": true,
  "in_channels": 128,
  "inner_dim": 1536,
  "multi_layer_encoder_feature": true,
  "num_attention_heads": 24,
  "num_layers": 48,
  "out_channels": 32,
  "patch_size": 2,
  "rms_norm": true,
  "selected_layer_index": [5, 11, 17, 23]
}
```

**Derived constants:**
- `inner_dim` = 24 × 64 = **1536** (note: smaller than head*dim would suggest for a 3.8B model — most parameters are in the 48 layers, not the width)
- `mlp_hidden` = `int(1536 / 3 * 8)` = **4096** (SwiGLU 8/3 ratio)
- `mlp_param_count` per layer = 3 × 1536 × 4096 × 2 streams = **75.5M params** (largest single contributor)
- `qkv_param_count` per layer = 2 × 3 × 1536² × 2 streams = **28.3M params**
- `proj_out` produces `patch_size² × out_channels` = 4 × 32 = **128**, exactly matching `in_channels`. The image stream's I/O channels match because the transformer denoises a packed-patch latent (`out_channels=32` is the VAE channel count; the 4× comes from the 2×2 patch unpack).
- Latent channels (transformer I/O) = **128** = 4 × 32 (after pipeline-side patchify of the 32-ch Flux.2 VAE latent into 2×2 patches).

### Top-level transformer structure (`LensTransformer2DModel`)

State-dict layout (these are the exact PyTorch parameter names that will be in `model.safetensors`):

```
pos_embed                                  LensEmbedRope  (no learnable params — complex tensors are NOT saved as buffers)
time_text_embed.time_proj                  Timesteps      (no learnable params; sinusoidal)
time_text_embed.timestep_embedder.linear_1 Linear(256 → 1536)
time_text_embed.timestep_embedder.linear_2 Linear(1536 → 1536)

txt_norm.0.weight                          RMSNorm(2880)  — one per selected layer
txt_norm.1.weight                          RMSNorm(2880)
txt_norm.2.weight                          RMSNorm(2880)
txt_norm.3.weight                          RMSNorm(2880)
txt_in.weight, txt_in.bias                 Linear(11520 → 1536)   (4 × 2880 = 11520)

img_in.weight, img_in.bias                 Linear(128 → 1536)

transformer_blocks.{i}.attn.norm_q.weight        RMSNorm(64)
transformer_blocks.{i}.attn.norm_k.weight        RMSNorm(64)
transformer_blocks.{i}.attn.norm_added_q.weight  RMSNorm(64)
transformer_blocks.{i}.attn.norm_added_k.weight  RMSNorm(64)
transformer_blocks.{i}.attn.img_qkv.{weight,bias} Linear(1536 → 4608) — fused Q/K/V, bias=True
transformer_blocks.{i}.attn.txt_qkv.{weight,bias} Linear(1536 → 4608) — fused Q/K/V, bias=True
transformer_blocks.{i}.attn.to_out.0.{weight,bias} Linear(1536 → 1536)
transformer_blocks.{i}.attn.to_add_out.{weight,bias} Linear(1536 → 1536)
transformer_blocks.{i}.img_mod.1.{weight,bias}   Linear(1536 → 9216) — Sequential(SiLU, Linear). 6× = 9216
transformer_blocks.{i}.txt_mod.1.{weight,bias}   Linear(1536 → 9216)
transformer_blocks.{i}.img_norm1.weight          RMSNorm(1536)
transformer_blocks.{i}.img_norm2.weight          RMSNorm(1536)
transformer_blocks.{i}.txt_norm1.weight          RMSNorm(1536)
transformer_blocks.{i}.txt_norm2.weight          RMSNorm(1536)
transformer_blocks.{i}.img_mlp.w1.weight         Linear(1536 → 4096, bias=False)  — SwiGLU
transformer_blocks.{i}.img_mlp.w3.weight         Linear(1536 → 4096, bias=False)
transformer_blocks.{i}.img_mlp.w2.weight         Linear(4096 → 1536, bias=False)
transformer_blocks.{i}.txt_mlp.w1.weight         Linear(1536 → 4096, bias=False)
transformer_blocks.{i}.txt_mlp.w3.weight         Linear(1536 → 4096, bias=False)
transformer_blocks.{i}.txt_mlp.w2.weight         Linear(4096 → 1536, bias=False)

norm_out.linear.weight, norm_out.linear.bias    Linear(1536 → 3072) — diffusers AdaLayerNormContinuous (shift+scale)
proj_out.{weight, bias}                          Linear(1536 → 128)   (patch_size² × out_channels = 4 × 32)
```

**Total params estimate:** 48 layers × (6.6M attn + 75.5M MLP + small norms+mods) + I/O ≈ **3.8B** as advertised. (MLP dominates at ~70% of parameter mass.)

### Block forward pass (`LensTransformerBlock.forward`)

```
# Modulation: each stream gets 6 chunks (shift1, scale1, gate1, shift2, scale2, gate2)
img_mod1, img_mod2 = Linear(SiLU(temb)).chunk(2, dim=-1)   # split into two halves of 3 each
txt_mod1, txt_mod2 = Linear(SiLU(temb)).chunk(2, dim=-1)

# Pre-attn modulation
img_modulated, img_gate1 = _modulate(img_norm1(hidden), img_mod1)
txt_modulated, txt_gate1 = _modulate(txt_norm1(encoder), txt_mod1)
   where _modulate(x, mod) splits mod into (shift, scale, gate) along dim=-1
   and returns (x * (1 + scale) + shift, gate)

# Joint attention
img_attn, txt_attn = LensJointAttention(img_modulated, txt_modulated, rope, mask)
hidden  += img_gate1 * img_attn
encoder += txt_gate1 * txt_attn

# Pre-MLP modulation
img_modulated2, img_gate2 = _modulate(img_norm2(hidden),  img_mod2)
hidden  += img_gate2 * img_mlp(img_modulated2)
txt_modulated2, txt_gate2 = _modulate(txt_norm2(encoder), txt_mod2)
encoder += txt_gate2 * txt_mlp(txt_modulated2)

return encoder, hidden    # NOTE the return order: text first, image second
```

**Distinguishing details vs Flux double-stream:**
1. **Order of modulation outputs.** Lens splits the 6×inner-dim mod output into two halves (one for attn, one for MLP), then `_modulate` splits each half into `(shift, scale, gate)` via a third chunk. Flux's `FluxDoubleStreamBlock` uses a single 6-output chunk and reads in `(shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)` order. **Lens's order is `(shift1, scale1, gate1, shift2, scale2, gate2)` where 1=attn and 2=mlp — same semantic mapping but chunked differently.** Watch the C# extraction order against the dump harness.
2. **Per-stream norms inside the attention.** Lens normalises Q and K separately per stream (`norm_q`/`norm_k` for image, `norm_added_q`/`norm_added_k` for text) **before** RoPE and **before** the concat. Flux merges them. Bias=True on every QKV — Flux uses bias=False on most paths.
3. **The block returns `(encoder, hidden)`, not `(hidden, encoder)`.** Top-level loop unpacks accordingly.

### Joint attention (`LensJointAttention`)

```
img_qkv: Linear(1536 → 4608, bias=True)
txt_qkv: Linear(1536 → 4608, bias=True)
norm_q, norm_k, norm_added_q, norm_added_k: RMSNorm(64) each, eps=1e-5

img_qkv = img_qkv(img_modulated).view(B, S_img, 3, 24, 64).unbind(dim=2)  # → img_q, img_k, img_v
txt_qkv = txt_qkv(txt_modulated).view(B, S_txt, 3, 24, 64).unbind(dim=2)  # → txt_q, txt_k, txt_v

# Per-head RMS norm on Q and K (NOT V)
img_q = norm_q(img_q);     img_k = norm_k(img_k)
txt_q = norm_added_q(txt_q); txt_k = norm_added_k(txt_k)

# Apply complex-polar RoPE BEFORE concat (different freqs per stream)
img_freqs, txt_freqs = pos_embed(img_shapes, [S_txt], device)   # [S_img, 32 complex], [S_txt, 32 complex]
img_q = apply_rotary_emb_lens(img_q, img_freqs[:S_img])
img_k = apply_rotary_emb_lens(img_k, img_freqs[:S_img])
if S_txt > 0:
    txt_q = apply_rotary_emb_lens(txt_q, txt_freqs[:S_txt])
    txt_k = apply_rotary_emb_lens(txt_k, txt_freqs[:S_txt])

# Concat into one joint sequence, SDPA in [B, H, S, D] layout
q = cat([img_q, txt_q], dim=1).transpose(1, 2)   # [B, 24, S_img+S_txt, 64]
k = cat([img_k, txt_k], dim=1).transpose(1, 2)
v = cat([img_v, txt_v], dim=1).transpose(1, 2)

# Additive mask: [B, 1, 1, S_img+S_txt]. Image always valid; text positions follow encoder_mask.
attn_mask: float tensor with -inf where text was padded, 0 elsewhere

out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
out = out.transpose(1, 2).reshape(B, S_img+S_txt, 1536)
img_out = to_out[0](out[:, :S_img, :])      # to_out[1] = Identity
txt_out = to_add_out(out[:, S_img:, :])
return img_out, txt_out
```

### RoPE (`LensEmbedRope.apply_rotary_emb_lens`)

Complex-polar (Qwen-Image style), **NOT** the half-rotation form used by Flux/F-Lite. Per-axis dims `(8, 28, 28)` sum to 64 = head_dim. The rotation is applied by interleaving pairs as complex numbers and multiplying by `e^(iθ)`:

```
x_complex = view_as_complex(x.reshape(..., 32, 2))   # [B, S, H, 32 complex]
freqs_cis = freqs.unsqueeze(1)                       # broadcast over heads
x_out = view_as_real(x_complex * freqs_cis).flatten(3)
```

**Frequency table construction (in `__init__`):**
- `pos_index = arange(4096)`, `neg_index = arange(4096).flip(0) * -1 - 1` (negative indices from -1 down to -4096)
- For each axis dim d, `freqs[i, k] = 1 / θ^(2k/d)` with θ=10000, then `torch.polar(ones, outer(index, 1/θ))` → complex of unit modulus rotated by phase
- `pos_freqs` and `neg_freqs` are pre-computed once for both index ranges and held as plain attributes (NOT buffers, because `register_buffer` strips imaginary components on safetensors save/load — important note in upstream code)
- **`scale_rope=True`** (default in `LensTransformer2DModel.__init__`): per-axis frequencies for height and width are split — negative indices for the top/left half, positive indices for the bottom/right half. This centers the rotation around the image's middle pixel instead of the top-left. Frame axis always uses positive frequencies (it's just `[0]` for a still image).

**Concretely for a 1024×1024 image:** latent is `64×64`. RoPE freqs are computed once per `(frame, h_lat, w_lat)` triple and cached in `rope_cache[f"{idx}_{h}_{w}"]`. Text RoPE positions start at `max(h//2, w//2)` (with scale_rope=True), giving text tokens positions that don't collide with image tokens.

### Multi-layer text feature mixing (top-level forward)

```
# encoder_hidden_states is a List[Tensor] when multi_layer_encoder_feature=True
normed = [txt_norm[i](encoder_hidden_states[i]) for i in range(4)]  # each [B, S_txt, 2880]
encoder_hidden_states = cat(normed, dim=-1)                          # [B, S_txt, 11520]
encoder_hidden_states = txt_in(encoder_hidden_states)                # [B, S_txt, 1536]
```

The four hidden states come from GPT-OSS layers `[5, 11, 17, 23]` (0-indexed, so layers 6/12/18/24 in 1-indexed counting — last is the final transformer layer because GPT-OSS has 24 layers total). Each is RMSNorm'd by its own learnable scale, then concatenated channel-wise. This is the "massive text encoder training on GPT image outputs" trick — Microsoft trained the DiT with this layer-concat conditioning so the projection learns to mix coarse-to-fine semantic information from the LM.

### Final layer

```
hidden = norm_out(hidden, temb)   # diffusers AdaLayerNormContinuous: hidden * (1 + scale(temb)) + shift(temb)
return proj_out(hidden)            # [B, S_img, 128]
```

`AdaLayerNormContinuous(elementwise_affine=False, eps=1e-6)` is a single `Linear(1536 → 3072)` that produces `[shift, scale]` from `temb`, applied to a LayerNorm-without-affine of the input. **The chunk order is `[shift, scale]`** (matching diffusers — the `Sd3.5` AdaLN-Continuous final layer in this codebase already uses this exact pattern, see `Sd3Transformer.cs` and `QwenImageTransformer.cs`).

Output `[B, S_img, 128]` is the **packed** prediction. Unpacking happens in the pipeline:

```python
rearrange(out, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=2, p2=2, h=latent_h, w=latent_w)
```

→ `[B, 32, h_lat * 2, w_lat * 2]` → Flux.2 VAE-shaped latent.

### Pipeline forward (`LensPipeline.__call__`)

```
1. resolve_resolution(base_resolution, aspect_ratio)         # 1024 or 1440 base × 9 aspect ratios
2. (optional) PromptReasoner.refine(prompt)                  # LLM-rewriting via the same GPT-OSS, OFF by default
3. encode_prompt: build chat template, tokenize, GPT-OSS.encode_layers → 4 layer hidden states + mask
   - chat template:
     [system]    "Describe the image by detailing the color, shape, size, texture, quantity,
                  text, spatial relationships of the objects and background."
     [user]      <prompt>
     [assistant] thinking="Need to generate one image according to the description.", content=""
     (rendered text is split at "<|return|>"; everything before kept)
   - tokenized with right-padding (max_length=512), GPT-OSS BPE
   - 97-token offset is stripped from the front of the hidden states (the system + chat-template
     wrapper occupies exactly 97 tokens — this is precomputed and stored as DEFAULT_TXT_OFFSET)
4. align text features: pad pos/neg to common S_txt
5. prepare_latents: shape = (B, latent_h * latent_w, 128) — i.e. PACKED token sequence, NOT [B,C,H,W]
6. scheduler.set_timesteps(sigmas=linspace(1.0, 1.0/N, N), mu=empirical_mu(seq_len, N))
7. denoise loop (CFG via batch-of-2 duplicate):
   for t in scheduler.timesteps:
       hidden = latents.repeat(2, 1, 1)                       # [2B, S_img, 128]
       noise = transformer(hidden, encoder=encoder_features, mask=..., timestep=t/1000, img_shapes)
       cond, uncond = noise.chunk(2)
       comb = uncond + cfg * (cond - uncond)
       # Norm rescaling: comb is rescaled so |comb_token| matches |cond_token|
       cond_norm = norm(cond, dim=-1, keepdim=True)
       comb_norm = norm(comb, dim=-1, keepdim=True)
       scale = where(comb_norm > 0, cond_norm / clamp_min(comb_norm, 1e-12), 1)
       noise_pred = comb * scale
       latents = scheduler.step(noise_pred, t, latents)
8. decode:
   x = rearrange(latents, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=2, p2=2)  # → [B, 32, h, w]
   bn = vae.bn
   shift = (-bn.running_mean).view(1, 32, 1, 1)
   scale = (1 / sqrt(bn.running_var + eps)).view(1, 32, 1, 1)
   x = patchify(x)             # 2x2 spatial→channel
   x = x / scale - shift       # reverse BN normalization
   x = unpatchify(x)
   image = vae.decode(x).sample
   image.clamp(-1, 1); (image + 1) * 127.5 → uint8
```

**Two pipeline-level details worth pinning:**

1. **Norm-rescaled CFG.** Standard CFG is `pred = uncond + cfg * (cond - uncond)`. Microsoft adds a per-token rescaling step that brings the L2 norm of each predicted token back down to the conditional branch's norm. This prevents the high-CFG over-saturation seen in plain CFG and is why CFG=5.0 stays clean. The C# port should match this exactly — even at CFG=1.0 (where the rescale is a no-op in expectation) the multiplication is still performed.

2. **The `latents` tensor never leaves packed token form during the denoise loop.** Shape is `(B, latent_h * latent_w, 128)` — a token sequence of length `H/16 * W/16` (because VAE downsamples 16× and the pipeline-side patchify does another 2×2 = 4× on the channel dim). Only at decode time is it rearranged into `[B, 32, H/16, W/16]` for the Flux.2 VAE.

### Empirical-mu schedule

```python
def compute_empirical_mu(image_seq_len, num_steps):
    a1, b1 = 8.73809524e-05, 1.89833333    # calibration for low step counts
    a2, b2 = 0.00016927, 0.45666666        # calibration for high step counts (long sequences)
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2
    m_200 = a2 * image_seq_len + b2
    m_10  = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * num_steps + b
```

For a 1024×1024 image with the Flux.2 VAE: `seq_len = 64 * 64 = 4096`, just below the 4300 threshold. At `num_steps = 20`: `m_200 = 0.00016927*4096 + 0.45666666 ≈ 1.1499`, `m_10 = 8.73809524e-5*4096 + 1.89833333 ≈ 2.2562`. `a = (1.1499 - 2.2562) / 190 ≈ -0.005823`, `b = 1.1499 - 200*a ≈ 2.3146`. `mu = -0.005823 * 20 + 2.3146 ≈ 2.198`. For 1440×1440: `seq_len = 90 * 90 = 8100` — above threshold, so `mu = 0.00016927 * 8100 + 0.45666666 ≈ 1.828`. Sigmas are `linspace(1.0, 1.0/num_steps, num_steps)`, then time-shifted by the scheduler using `mu`.

This **must match the diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas, mu)` implementation exactly** — particularly the `time_shift` function it uses internally:

```python
sigma_shifted = exp(mu) / (exp(mu) + (1/sigma - 1) ** 1.0)
```

This is the standard SD3-style shifted-sigma form. The existing C# `FlowMatchEulerDiscreteScheduler` (used by Flux, SD3.5, Z-Image) already implements this; we just need to plumb `mu` through.

### GPT-OSS text encoder (`text_encoder/config.json`)

```json
{
  "model_type": "gpt_oss",
  "hidden_size": 2880,
  "num_hidden_layers": 24,
  "vocab_size": 201088,
  "intermediate_size": 2880,
  "hidden_act": "silu",
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "sliding_window": 128,
  "num_local_experts": 32,
  "num_experts_per_tok": 4,
  "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention", ...]
   (alternates over 24 layers, starting with sliding_attention)
}
```

**This is the upstream `openai/gpt-oss-20b` model** (or a very close cousin), trimmed by Lens to never run beyond layer 23. The total parameter count is large (~20B) but each token only activates 4 of 32 experts per MoE layer → ~3.6B activated params per forward pass. **MXFP4** native packing keeps the on-disk size around 12-13 GB.

**Forward path (Lens-specific, in `LensGptOssEncoder.forward`):**

```
input_ids, attention_mask → embed_tokens → position_ids = arange(S)
position_embeddings = rotary_emb(hidden, position_ids)   # RoPE for GPT-OSS itself
causal_mask_mapping = {"full_attention": create_causal_mask(...),
                       "sliding_attention": create_sliding_window_causal_mask(...)}
captured: List[Tensor] = [None, None, None, None]
for i in range(min(24, max(selected_layer_index)+1)):     # early exit after layer 23
    hidden = decoder_layer[i](hidden, attn_mask=causal_mask_mapping[layer_types[i]],
                              position_embeddings=position_embeddings,
                              past_key_values=None, use_cache=False)
    if i in {5, 11, 17, 23}:
        captured[selected.index(i)] = hidden
return captured   # list of 4 tensors, each [B, S_txt, 2880]
```

**Net-new infra required for this in HartsyInference:**

1. **MoE FFN with sparse routing.** Top-4-of-32 expert selection, sparse dispatch through SwiGLU MLPs (each expert is a SwiGLU pair). HiDream already has a `NumRoutedExperts`/`NumActivatedExperts` config but the routing/dispatch primitive itself in HiDream is "single-expert fallback" today (see `MODEL_STATUS.md`) — we'd need the real top-k router. This is the single biggest infrastructure ask.

2. **Grouped-Query Attention (64 Q heads : 8 KV heads).** `LlamaStyleEncoder` already supports GQA, so this is a config flag.

3. **Alternating sliding-window vs full causal attention.** Layers 0, 2, 4, ... use a 128-token sliding window; layers 1, 3, 5, ... use unrestricted causal. The existing `LlamaStyleEncoder` does **not** support per-layer attention masks — it picks one and applies it uniformly. This needs a small config addition.

4. **MXFP4 weight unpack.** The text encoder ships as MXFP4-packed safetensors. **Two options:**
   - **(a) Dequant at load to F16/BF16** (~24 GB; same trade-off as the FP8 mix in Z-Image and SD3.5). Simple but doubles encoder VRAM cost.
   - **(b) Native MXFP4 GEMM** — would require a new dequant kernel similar to GGUF Q4_K's, deployed inside `CudaBackend.Linear`'s hot path. Bigger lift but keeps encoder ~12 GB.

   For the first cut, **dequant-at-load is the right call** — the encoder's whole job is one forward pass per generation (it doesn't run inside the denoise loop), so the activations cost is the bottleneck, not the weight footprint. Native MXFP4 can be a Phase 4 follow-up.

5. **GPT-OSS BPE tokenizer.** Vocab 201,088 tokens, ChatML-style template with `<|return|>` markers, system+user+assistant role tags. The `Microsoft.ML.Tokenizers.BpeTokenizer` already powers `Qwen3Tokenizer` and `ClipTokenizer` — same machinery, new merges/vocab files (`tokenizer.json` from `text_encoder/`). Need to verify the special-token set: `<|start|>`, `<|end|>`, `<|message|>`, `<|return|>`, `<|channel|>`, `<|constrain|>` are the GPT-OSS markers.

6. **The 97-token system-prompt offset.** Hard-coded constant in the pipeline — the chat-template wrapper (system message + assistant thinking) always tokenizes to exactly 97 tokens, and Microsoft strips those tokens from the encoder output. **This is brittle:** the offset is template-specific, and any future tokenizer update would change it. The C# port should re-verify on first run by running the chat template through the tokenizer and counting the tokens before the user content begins.

### Flux.2 semantic VAE — reused as-is

`AutoencoderKLFlux2` is already loaded by `Flux2Pipeline.cs` in this codebase. Lens uses **identical** loading code and the same BN-style un-normalization at the pipeline boundary (`shift = -bn.running_mean`, `scale = 1/sqrt(bn.running_var + eps)`). The Lens pipeline's `_decode` is essentially a copy of the Flux.2 decode path (compare against `Flux2Pipeline.cs:290` — the 2×2 unpatchify is the same op).

**No new VAE code needed.** The existing `Flux2CheckpointConverter` and `VaeDecoder` cover this entirely.

### Resolution buckets (`lens/resolution.py`)

Two base resolutions × nine aspect ratios = eighteen pre-defined `(height, width)` pairs:

| Aspect | 1024 base | 1440 base | Latent (H/16 × W/16) | Tokens (latent_h × latent_w) |
|---|---|---|---|---|
| 1:2  | (1472, 736)  | (2080, 1040) | 92×46 / 130×65 | 4232 / 8450 |
| 9:16 | (1376, 768)  | (1936, 1088) | 86×48 / 121×68 | 4128 / 8228 |
| 2:3  | (1248, 832)  | (1760, 1168) | 78×52 / 110×73 | 4056 / 8030 |
| 3:4  | (1152, 864)  | (1616, 1216) | 72×54 / 101×76 | 3888 / 7676 |
| 1:1  | (1024, 1024) | (1440, 1440) | 64×64 / 90×90 | **4096 / 8100** |
| 4:3  | (864, 1152)  | (1216, 1616) | (transpose) | (same totals) |
| 3:2  | (832, 1248)  | (1168, 1760) | (transpose) | (same totals) |
| 16:9 | (768, 1376)  | (1088, 1936) | (transpose) | (same totals) |
| 2:1  | (736, 1472)  | (1040, 2080) | (transpose) | (same totals) |

**All values divisible by 16** so they tile cleanly into Flux.2 VAE latents.

## Key Numbers / Constants

| Constant | Value | Source |
|---|---|---|
| Parameters (DiT) | 3.8 B | README |
| Layers | 48 | `num_layers` |
| Hidden (inner) | 1536 | `inner_dim` |
| Attention heads | 24 | `num_attention_heads` |
| Head dim | 64 | `attention_head_dim` |
| MLP hidden (SwiGLU) | 4096 | `int(1536 / 3 * 8)` |
| Patch size | 2 | `patch_size` |
| In channels (transformer) | 128 | `in_channels` (= 4× of 32 after pipeline patchify) |
| Out channels (transformer) | 32 | `out_channels` |
| Text encoder hidden | 2880 | `enc_hidden_dim`; matches `openai/gpt-oss-20b` |
| Text encoder layers used | [5, 11, 17, 23] | `selected_layer_index`; 0-indexed |
| Text input projection dim | 4 × 2880 = 11520 | concat width |
| Modulation outputs per stream | 6 × 1536 = 9216 | `Linear(SiLU(temb))` width |
| RoPE θ | 10000 | `LensEmbedRope.__init__` |
| RoPE per-axis dims | (8, 28, 28) | `axes_dims_rope` |
| RoPE scale_rope | True | `LensTransformer2DModel.__init__` (hardcoded, not a config field) |
| Max position-grid pre-compute | 4096 | `pos_index = arange(4096)` |
| Scheduler | FlowMatchEulerDiscreteScheduler | `model_index.json` |
| Empirical mu — low-seq slope (a1) | 8.73809524e-5 | `compute_empirical_mu` |
| Empirical mu — low-seq intercept (b1) | 1.89833333 | " |
| Empirical mu — high-seq slope (a2) | 0.00016927 | " |
| Empirical mu — high-seq intercept (b2) | 0.45666666 | " |
| Sigmas | `linspace(1.0, 1.0/N, N)` | pipeline |
| Default CFG (Lens / Lens-Base) | 5.0 | README, pipeline default 4.0 (overridden in CLI) |
| Default steps (Lens / Lens-Base) | 20 / 50 | README |
| Default steps (Lens-Turbo) | 4 | README |
| Default CFG (Lens-Turbo) | 1.0 | README |
| Tokenizer pad token | EOS (`tokenizer.eos_token`) | pipeline |
| Padding side | right | pipeline |
| Max sequence length (text) | 512 | pipeline default |
| `txt_offset` (system-prompt strip) | 97 | `DEFAULT_TXT_OFFSET` |
| VAE downsample × patchify | 16 × (2×2) = 16× spatial, 4× channels | `vae_scale_factor=16`, pipeline patchify |
| Default sample size | 1024 | pipeline (used when no base_resolution / aspect_ratio) |
| Total repo size | 30.7 GB | HF page |

### GPT-OSS encoder

| Constant | Value |
|---|---|
| Hidden size | 2880 |
| Layers (model has) | 24 |
| Layers (Lens uses) | up to 23 (last selected layer) |
| Heads (Q) | 64 |
| Heads (KV) | 8 (GQA 8:1) |
| Intermediate (per expert) | 2880 |
| Vocab | 201,088 |
| Activation | SiLU (SwiGLU FFN per expert) |
| Local experts | 32 |
| Active experts per token | 4 |
| Sliding window | 128 tokens |
| Attention pattern | alternating sliding/full per layer |
| Native dtype | MXFP4 (4-bit packed with 32-element block scales) |

## Data Layouts / Formats

### Latent tensor shapes through the pipeline

```
RGB image                          [B, 3, H, W]                   uint8 [0, 255]
↓ VAE encode (decode is symmetric — Lens is t2i so we only decode)
VAE latent                         [B, 32, H/16, W/16]            F32/BF16
↓ pipeline 2×2 patchify (channel)
Packed latent (transformer input)  [B, (H/16)·(W/16), 128]        BF16
                                   = (B, S_img, in_channels)

img_in projection                  [B, S_img, 1536]
+ 48× blocks (joint with txt)      [B, S_img, 1536]   stays put; encoder grows/shrinks alongside
proj_out                           [B, S_img, 128]
↓ pipeline rearrange + unpatchify + BN un-normalize
                                   [B, 32, H/16, W/16]            BF16
↓ VAE decode
RGB image                          [B, 3, H, W]                   F32 [-1, 1]
```

### Text feature shapes

```
prompt → chat-templated text → tokenize → input_ids [B, S_padded]
↓ GPT-OSS forward, capture layers [5,11,17,23]
List[4] of [B, S_padded, 2880]
↓ strip first 97 tokens (system + chat template wrapper)
List[4] of [B, S_txt, 2880]      where S_txt = S_padded - 97
↓ per-layer RMSNorm + channel-concat + Linear
[B, S_txt, 1536]   (now joins the image stream in each block)
```

### Safetensors file layout (microsoft/Lens)

```
transformer/diffusion_pytorch_model.safetensors      ~7.6 GB BF16   (all LensTransformer2DModel keys)
text_encoder/model.safetensors                       ~12-13 GB MXFP4 packed (LensGptOssEncoder weights — full GPT-OSS layers)
text_encoder/model.safetensors.index.json            (multi-shard pointer if sharded)
vae/diffusion_pytorch_model.safetensors              ~5-10 GB FP16   (AutoencoderKLFlux2; same file as Flux.2 ships)
tokenizer/tokenizer.json + tokenizer_config.json     GPT-OSS BPE
scheduler/scheduler_config.json                      FlowMatchEulerDiscreteScheduler config
model_index.json                                     pipeline manifest (415 B)
```

### ComfyUI distribution (`Comfy-Org/Lens`) — actual checkpoints in use

The diffusers `microsoft/Lens` repo (above) is the reference, but the **checkpoints we actually load** are the ComfyUI-repackaged ones at [`huggingface.co/Comfy-Org/Lens`](https://huggingface.co/Comfy-Org/Lens) (MIT, ungated). These differ from diffusers in three load-bearing ways, all handled by `LensCheckpointConverter.ConvertComfy*` + the `Mxfp8Codec` / `Nvfp4Codec`:

| File | Size | Format | Notes |
|---|---|---|---|
| `diffusion_models/lens_bf16.safetensors` / `lens_turbo_bf16` | 8.2 GB | plain **BF16** | diffusers-native key names, **no `transformer.` prefix** (`transformer_blocks.{i}.attn.img_qkv`, `img_mlp.w1/w2/w3`, `norm_out.linear`, `proj_out`, `time_text_embed.timestep_embedder`). Loads through the existing converter passthrough + fused-QKV split. |
| `diffusion_models/lens_mxfp8.safetensors` / `lens_turbo_mxfp8` | 5.5 GB | **MXFP8** (`mxfp8_block32`) | per-Linear: `{name}.weight` F8E4M3 `[out,in]` + `{name}.weight_scale` U8 (E8M0, group 32 along in, **swizzled**) + `{name}.comfy_quant` JSON blob. Dequant `w = decode_e4m3(weight)·2^(scale-127)` → BF16. No transpose. |
| `text_encoders/gpt_oss_20b_nvfp4.safetensors` | 13.2 GB | **NVFP4** (`nvfp4`) | GPT-OSS-20B, **no `model.` prefix** (`layers.{i}.…`, `embed_tokens.weight`, `norm.weight`, plus an embedded `tokenizer_json` blob). MoE experts only are quantized: `experts.{gate_up,down}_proj.weight` U8 (FP4 E2M1, **high nibble = even elem**) + `.weight_scale` F8E4M3 (group 16, **swizzled**) + `.weight_scale_2` F32 per-expert global + `.comfy_quant`. Attn/router/embed/norm stay BF16; biases use `gate_up_proj.bias` (renamed to HF `gate_up_proj_bias`). Dequant `w = e2m1(nibble)·global·decode_e4m3(block_scale)`, then transpose `[E,out,in]→[E,in,out]` to the runtime layout. |
| `vae/flux2-vae.safetensors` | 336 MB | FP16/BF16 | the Flux.2 semantic VAE, reused as-is. |

**Swizzled block scales.** Both MXFP8 and NVFP4 store their per-block scale tensors in NVIDIA's cuBLAS "blocked" layout (ComfyUI's `comfy.float.to_blocked`): the logical `[out, in/group]` scale matrix is zero-padded to `[128·ceil(out/128), 4·ceil((in/group)/4)]` and permuted. `BlockScaleSwizzle.SwizzledIndex(row, blockCol, paddedCols)` inverts the permutation (verified by an exact swizzle round-trip against `to_blocked`). This is why NVFP4 `down_proj.weight_scale` shows the padded `[32, 2944, 180]` shape (out 2880 → 2944).

**MXFP4 vs MXFP8/NVFP4.** The earlier `Mxfp4Codec` (E8M0 group-32 FP4, no global) matches the **diffusers** `microsoft/Lens` text encoder. The ComfyUI repo uses MXFP8 (DiT) + NVFP4 (TE) instead. All three codecs coexist.

**Memory caveat (≤12 GB target).** Dequant-at-load of the NVFP4 20B encoder to F32 is large (experts dominate). The DiT (BF16/MXFP8→BF16) fits with the existing eviction discipline; the encoder is best run on CPU (system RAM) before the DiT loads. A per-layer streaming dequant of the encoder experts is the follow-up for tight-RAM hosts.

## Algorithm Steps (pseudocode for the C# port)

```
LensPipeline.GenerateFromTokens(promptIds, negIds, posMask, negMask,
                                height, width, steps, cfgScale, seed):
    (latentH, latentW) = (height / 16, width / 16)
    seqLen = latentH * latentW

    // 1. Run GPT-OSS encoder, capture layers [5, 11, 17, 23]; strip txt_offset=97
    posLayers = textEncoder.EncodeLayers(promptIds, posMask, selected=[5,11,17,23])
    negLayers = textEncoder.EncodeLayers(negIds,    negMask, selected=[5,11,17,23])
    posLayers, posMask = StripOffset(posLayers, posMask, 97)
    negLayers, negMask = StripOffset(negLayers, negMask, 97)
    posLayers, posMask, negLayers, negMask = AlignToSharedSTxt(posLayers, posMask, negLayers, negMask)

    // 2. Concat [pos, neg] along batch for CFG (batch-of-2)
    encoderFeatures = [Concat(pos, neg) for (pos, neg) in zip(posLayers, negLayers)]   // List[4] of [2B, S_txt, 2880]
    encoderMask = ConcatBatch(posMask, negMask)                                          // [2B, S_txt]

    // 3. Initial noise — packed token form
    latents = SeedGenerator.CreateNoise(seed, [B, seqLen, 128], dtype)

    // 4. Empirical mu + scheduler timesteps
    mu = ComputeEmpiricalMu(seqLen, steps)
    sigmas = LinSpace(1.0f, 1.0f / steps, steps)
    scheduler.SetTimesteps(sigmas, mu)

    // 5. Denoise loop
    for t in scheduler.Timesteps:
        hidden = latents.RepeatBatch(2)                                                   // [2B, seqLen, 128]
        timestep = Broadcast(t, 2B) / 1000.0f
        noise = transformer.Forward(hidden, encoderFeatures, encoderMask, timestep, (1, latentH, latentW))
        cond, uncond = noise.SplitBatch(2)
        comb = uncond + cfgScale * (cond - uncond)
        // Norm-rescaling — important for Lens's high-CFG behavior
        condNorm = Norm(cond, dim=-1, keepDim=true)
        combNorm = Norm(comb, dim=-1, keepDim=true)
        scale = Where(combNorm > 0, condNorm / Max(combNorm, 1e-12f), 1.0f)
        noisePred = comb * scale
        latents = scheduler.Step(noisePred, t, latents)

    // 6. Free transformer + encoder before VAE decode (PHASE_3_DEVIATIONS #18 pattern)
    backend.Sync()
    backend.FreeWeights(transformer.EnumerateWeights())
    backend.FreeWeights(textEncoder.EnumerateWeights())

    // 7. Decode
    latent2D = Rearrange(latents, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", 2, 2, latentH, latentW)
    latent2D = PatchifyForBn(latent2D)        // 2x2 spatial→channel for BN un-norm
    latent2D = latent2D / scale - shift       // shift = -bn.running_mean, scale = 1/sqrt(bn.running_var + eps)
    latent2D = UnpatchifyForBn(latent2D)
    rgb = vae.Decode(latent2D)                // [B, 3, H, W] in [-1, 1]
    return ClampToUint8(rgb)
```

## Reference Implementations

- **microsoft/Lens — `lens/transformer.py`** ([github.com/microsoft/Lens/blob/main/lens/transformer.py](https://github.com/microsoft/Lens/blob/main/lens/transformer.py)) — `LensTransformer2DModel`, `LensTransformerBlock`, `LensJointAttention`, `LensEmbedRope`, `apply_rotary_emb_lens`. ~700 lines. **Primary reference for the C# transformer port.**
- **microsoft/Lens — `lens/pipeline.py`** ([github.com/microsoft/Lens/blob/main/lens/pipeline.py](https://github.com/microsoft/Lens/blob/main/lens/pipeline.py)) — `LensPipeline`, `compute_empirical_mu`. ~580 lines. **Primary reference for the C# pipeline port.**
- **microsoft/Lens — `lens/text_encoder.py`** ([github.com/microsoft/Lens/blob/main/lens/text_encoder.py](https://github.com/microsoft/Lens/blob/main/lens/text_encoder.py)) — `LensGptOssEncoder`. ~130 lines. **Subclass of `GptOssForCausalLM` from transformers.**
- **microsoft/Lens — `lens/resolution.py`** ([github.com/microsoft/Lens/blob/main/lens/resolution.py](https://github.com/microsoft/Lens/blob/main/lens/resolution.py)) — 18 fixed resolution buckets.
- **diffusers `AutoencoderKLFlux2`** — same VAE as Flux.2 Klein/Dev. Already implemented in this codebase via `Flux2CheckpointConverter` + `VaeDecoder`.
- **transformers `GptOssForCausalLM`** ([github.com/huggingface/transformers — models/gpt_oss/](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss)) — the base GPT-OSS implementation. **Reference for MoE routing semantics, MXFP4 unpack, sliding/full attention mask construction.**
- **diffusers `FlowMatchEulerDiscreteScheduler`** — existing scheduler (used by Flux, SD3.5, Z-Image, F-Lite). The `set_timesteps(sigmas, mu)` form is what Lens calls.
- **diffusers `AdaLayerNormContinuous`** — `Linear(hidden → 2*hidden) → [shift, scale]`; identical to the version used by SD3.5/Qwen-Image final layers in this codebase.

## Differences Between Implementations

The reference is single-source (microsoft/Lens), but there are a few places where the upstream code diverges from common idioms in this codebase:

1. **Block return order is `(encoder_hidden_states, hidden_states)`** — text first, image second. Flux's `FluxDoubleStreamBlock.forward` returns `(hidden_states, encoder_hidden_states)` (image first). **Don't blindly copy the unpacking pattern.**
2. **`_modulate` chunks the half-mod into `(shift, scale, gate)` along dim=-1.** The chunked tensor has shape `[B, 3·1536]`; chunking yields three `[B, 1536]` tensors, then `.unsqueeze(1)` broadcasts across the sequence dim. Flux uses similar shape mechanics but a different overall chunk order; the C# port should follow Lens's `(shift1, scale1, gate1, shift2, scale2, gate2)` exactly.
3. **RoPE is complex-polar (Qwen-Image style), not pair-rotation (Flux style).** The C# port should re-use Qwen-Image's `RopeApplyComplex` kernel rather than Flux's `RopeApplyPair`.
4. **CFG batch ordering is `[positive, negative]`** in the doubled tensor (concat along dim=0). After the forward, `chunk(2)` gives `(cond, uncond) = (positive_pred, negative_pred)`. **Note that this is opposite to some other pipelines in the codebase that batch as `[uncond, cond]`** — pay attention to this in the C# port.
5. **`scale_rope=True` is hardcoded** in `LensTransformer2DModel.__init__` but **not in the JSON config**. The C# `LensConfig` should hardcode it the same way (don't expose it as a knob unless we ever need scale_rope=False).
6. **`rope_cache` is a runtime cache.** Upstream stores it as a plain dict on the module so the first forward per (h, w) computes freqs and subsequent forwards hit the cache. Important for the C# port to do the same — recomputing 4096-entry tables per step is wasteful.
7. **`pos_freqs` / `neg_freqs` are NOT registered as buffers** — register_buffer strips imaginary parts on safetensors save/load. They live as ordinary tensor attributes that get re-built in `__init__`. The C# port has no such constraint (we use raw `Tensor` with our own dtype handling), so we can construct them once at model creation and treat them as constants.

## Open Questions

- **Exact safetensors tensor names in `transformer/model.safetensors`** — the upstream `LensTransformer2DModel` uses diffusers's `ModelMixin.save_pretrained`, so the names should mirror the PyTorch state-dict keys exactly. First step on download: `safetensors_metadata --print-keys transformer/diffusion_pytorch_model.safetensors` and reconcile against the layout above. **Most likely** there's no key remap needed (diffusers-native).
- **MXFP4 dequant exactness** — does Microsoft pack with the standard MXFP4 (4-bit elements + per-32-element E8M0 scale) or a custom variant? Need to inspect a sample `.safetensors` and verify against `transformers/models/gpt_oss/modeling_gpt_oss.py`'s unpack routine. Most likely standard MXFP4.
- **GPT-OSS MoE router output shape and softmax** — top-k selection: is it softmax over all 32 logits then top-4, or top-4 logits then softmax over those? The exact router order changes numerics — read `modeling_gpt_oss.py:GptOssExperts.forward`.
- **Lens-Turbo CFG=1.0 — does it still apply the norm-rescale?** With cfg=1.0, `comb = uncond + 1*(cond-uncond) = cond`, so `comb_norm = cond_norm` and `scale = 1`. The rescale is a no-op in expectation but the multiplication still happens, which can introduce tiny rounding error. **Probably fine, but flag for first-run verification.**
- **`txt_offset = 97` validity for non-English prompts** — the system prompt is fixed English ("Describe the image by detailing the color, ..."), but the user prompt may be multilingual. The 97 only counts the wrapper, not the user prompt, so this should be tokenizer-language-invariant. Worth confirming by checking the chat-template output for non-ASCII input.
- **PromptReasoner (default OFF)** — uses the same GPT-OSS to rewrite the prompt into a longer, more detailed description before encoding. Probably out of scope for the first cut; deferred.

## Implementation Notes (recommendations for HartsyInference)

### What can be reused

- **`FlowMatchEulerDiscreteScheduler`** — used by Flux, SD3.5, Z-Image, F-Lite. Just plumb `mu` through.
- **`VaeDecoder` (Flux.2 preset)** — already loaded by `Flux2Pipeline.cs`. Lens uses the same decode path with BN un-normalization at the pipeline boundary.
- **`Flux2CheckpointConverter`** — handles the Flux.2 VAE weights file. Already in tree.
- **`AdaLayerNormContinuous` pattern** — same `Linear(1536 → 3072) → [shift, scale]` as SD3.5 and Qwen-Image final layers.
- **`RMSNorm` (with and without learned scale)** — existing primitive. Lens uses learned-scale RMSNorm everywhere (`eps=1e-5` for QK norms, `eps=1e-6` for stream norms).
- **`SwiGLU` MLP (`w1, w2, w3` naming)** — Flux double-stream block uses this same pattern. Bias=False on all three.
- **HiDream `NumRoutedExperts` config plumbing** — start here for the MoE FFN side of the encoder.
- **Qwen-Image complex-polar RoPE** — reuse `RopeApplyComplex` if it exists; otherwise port from `QwenImageRope.cs`.
- **`Microsoft.ML.Tokenizers.BpeTokenizer`** — already powers `Qwen3Tokenizer` and `ClipTokenizer`. Add a new `GptOssTokenizer` class with the appropriate special-token set and chat template.

### What's net-new

| Component | Effort | Why |
|---|---|---|
| **MoE FFN with real top-k routing** | High (~1-2 weeks) | Existing HiDream uses single-expert fallback. Need a proper grouped-by-expert dispatch primitive on CPU and CUDA. Shared with future text-LLM MoE work, so worth doing right. |
| **MXFP4 dequant-at-load** | Medium (~3-4 days) | New dtype handler in `Tensor.LoadAs`. Standard MXFP4 unpack is well documented; we already do FP8 scale-companion folding. |
| **Sliding-window + full-attention alternating mask** | Low (~1 day) | New flag in `LlamaStyleEncoderConfig`: `LayerTypes : string[]` (per-layer "sliding" or "full"). Mask builder branches per layer. |
| **`LensTransformer.cs` + `LensTransformerBlock.cs` + `LensRope.cs`** | Medium (~3-5 days) | Mostly a Flux-double-block clone with the differences flagged above. |
| **`LensGptOssEncoder.cs`** | Medium (~3-5 days) | Subclass `LlamaStyleEncoder` (or write a new `MoeLlamaStyleEncoder`) that captures multi-layer hidden states and exits early after the last selected layer. Reuse GPT-OSS-specific layer math. |
| **`LensPipeline.cs`** | Low (~2 days) | Standard pipeline once the pieces are wired. Norm-rescaled CFG is one extra op. |
| **`LensCheckpointConverter.cs`** | Low (~1 day) | Probably diffusers-naming passthrough for the transformer; MXFP4 unpack for the encoder; Flux.2 VAE converter for the VAE. |
| **`LensGenerationTests.cs`** | Low (~1 day) | Standard test scaffold mirroring `Flux2GenerationTests`. VRAM probe (~12 GB for the DiT at FP16 + MXFP4 encoder; manageable on 12 GB cards with eviction). |
| **`dump_lens_full_forward.py` + `diff_lens_layers.py` + `LensDiffTests.cs`** | Medium (~2-3 days) | Standard layer-by-layer diff harness following the SD3.5 / Z-Image template. |

### VRAM budget on a 12 GB card (RTX 3060 / 4070 / etc.)

- DiT (FP16, 3.8 B) ≈ **7.6 GB** → at FP8 cast-on-load ≈ **3.8 GB**
- GPT-OSS encoder (MXFP4 native) ≈ **12 GB** packed; **24 GB** dequant'd to FP16
- Flux.2 VAE (FP16) ≈ **~5 GB**
- Activations at 1024×1024 (4096 tokens × 1536 dim × batch-of-2 for CFG, 48 layers worth of carryover): ~2-3 GB peak

**Path on 12 GB:** can't fit dequant'd encoder + DiT + VAE simultaneously. Plan:
1. Encode prompt with the encoder loaded; capture 4-layer hidden states (small — `4 × S_txt × 2880` is well under 1 GB).
2. `backend.FreeWeights(textEncoder)` before loading the DiT.
3. Run the denoise loop on the DiT alone (~4-8 GB peak depending on FP8/FP16).
4. `backend.FreeWeights(transformer)` before VAE decode (mirrors PHASE_3_DEVIATIONS #18, #33).
5. VAE decode.

This is the same eviction-discipline pattern Flux/SD3.5/Qwen-Image use today. **Native MXFP4 encoder GEMM would let the encoder stay resident**, but isn't needed for correctness — only throughput on repeated generations.

### Suggested implementation order

1. **GPT-OSS infrastructure** — MXFP4 dequant, MoE FFN routing, alternating attention masks, tokenizer. Land these as additions to the existing `LlamaStyleEncoder` family. This unblocks Lens AND any future MoE-LM text encoder.
2. **`LensConfig` + `LensTransformer` scaffold** — port the transformer block by block from `lens/transformer.py`. Reuse Qwen-Image RoPE, Flux2 VAE.
3. **`LensCheckpointConverter`** — most likely a diffusers-naming passthrough with FP8/MXFP4 scale-companion folding.
4. **`LensPipeline`** — straightforward once the encoder is up.
5. **Validation harness** — first-run debug; expect 1-3 pipeline-level bug iterations per the SD3.5 / Z-Image / Flux first-run pattern.

### What NOT to do

- **Don't try to call into `dotLLM` for the GPT-OSS encoder.** Lens needs the multi-layer hidden states with mid-network capture, which is not a public API of either project. Re-implement the relevant forward as a single self-contained `LensGptOssEncoder` class.
- **Don't fold the norm-rescale into the scheduler step.** Keep it explicit in the pipeline so the layer-diff harness can validate it against the upstream code exactly.
- **Don't expose `selected_layer_index` as a runtime knob.** It's part of how the DiT was trained; changing it means re-training. Hardcode in `LensConfig`.
- **Don't reuse `T5TextEncoder.EncodeAtLayer` semantics.** That helper re-applies the final layer norm; Lens does NOT — each captured hidden state is the raw output of the selected decoder layer (no extra normalization), and the per-layer `txt_norm` in the DiT does the only normalization that matters.
