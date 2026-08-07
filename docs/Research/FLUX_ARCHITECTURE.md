# Flux Architecture — Research Notes

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## 1. Summary

Flux is a 12B-parameter Diffusion Transformer (DiT) from Black Forest Labs that uses **flow matching** instead of DDPM-based diffusion and processes image and text tokens **jointly** through transformer blocks. The architecture has two stages: **double-stream blocks** (19 layers) where image and text tokens have separate projections but share a single joint attention operation, followed by **single-stream blocks** (38 layers) where image and text tokens are concatenated into a single sequence and processed with a unified transformer block.

Key differences from SD1.5/SDXL:
- **Flow matching** replaces DDPM noise schedules with a continuous-time ODE (velocity prediction along straight interpolation paths)
- **DiT** replaces the UNet — no convolutions, skip connections, or downsampling/upsampling stages
- **Dual text encoders**: T5-XXL (up to 512 tokens, 4096-dim) + CLIP-L/14 (77 tokens, 768-dim pooled)
- **16-channel VAE** with 8x spatial compression (vs. 4-channel in SD1.5/SDXL)
- **2x2 patchification** of latents into token sequences
- **RoPE** (Rotary Position Embeddings) for both spatial and text positions
- **Guidance distillation** (dev) or **full distillation** (schnell) eliminates the need for true classifier-free guidance in most cases

---

## 2. Key Numbers/Constants

### Architecture Constants

| Parameter | Value |
|-----------|-------|
| Total parameters | ~12B |
| Hidden size | 3072 |
| Number of attention heads | 24 |
| Head dimension | 128 (= 3072 / 24) |
| MLP ratio | 4.0 |
| MLP hidden dimension | 12288 (= 3072 * 4) |
| Double-stream blocks (depth) | 19 |
| Single-stream blocks (depth_single_blocks) | 38 |
| In channels (latent patch dim) | 64 |
| Out channels | 64 |
| Patch size | 2x2 (applied to 16-channel latents) |
| RoPE theta | 10,000 |
| RoPE axes dimensions | [16, 56, 56] (sum = 128 = head_dim) |
| QKV bias | True |
| LayerNorm epsilon | 1e-6 |
| RMSNorm epsilon | 1e-6 |
| GELU approximation | tanh |

### Text Encoder Constants

| Parameter | T5-XXL | CLIP-L/14 |
|-----------|--------|-----------|
| Output dimension | 4096 | 768 (pooled) |
| Max tokens | 512 (dev) / 256 (schnell) | 77 |
| Usage | Per-token embeddings -> context | Pooled embedding -> vec |
| Projected to | 3072 (context_in_dim -> hidden_size) | Added to timestep embed |
| Approximate size (FP16) | ~5.5 GB | ~250 MB |

### VAE Constants

| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| Spatial compression | 8x |
| Scaling factor | 0.3611 (for Flux VAE) |
| Shift factor | 0.1159 (for Flux VAE) |

### Scheduler Constants (Flow-Matching Euler)

| Parameter | Value |
|-----------|-------|
| num_train_timesteps | 1000 |
| base_shift | 0.5 |
| max_shift | 1.15 |
| base_seq_len | 256 |
| max_seq_len | 4096 |
| time_shift_type | "exponential" (default) |
| Typical inference steps (dev) | 20-50 |
| Typical inference steps (schnell) | 1-4 |
| Default guidance_scale (dev) | 3.5 |

### Variant Differences

| Parameter | dev | schnell | canny/depth | fill |
|-----------|-----|---------|-------------|------|
| in_channels | 64 | 64 | 128 | 384 |
| guidance_embed | True | False | True | True |
| Blocks | 19+38 | 19+38 | 19+38 | 19+38 |
| Hidden size | 3072 | 3072 | 3072 | 3072 |

---

## 3. Data Layouts/Formats

### Latent Packing

Input latents of shape `[B, 16, H_lat, W_lat]` are packed into 2x2 patches:

```python
# Pack: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
latents = latents.view(B, C, H // 2, 2, W // 2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)
latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
# Result: [B, seq_len, 64] where seq_len = H_lat/2 * W_lat/2
```

Unpacking reverses this:
```python
# Unpack: [B, seq_len, C*4] -> [B, C, H, W]
latents = latents.view(B, H // 2, W // 2, C, 2, 2)
latents = latents.permute(0, 3, 1, 4, 2, 5)
latents = latents.reshape(B, C, H, W)
```

### Position ID Layout

```
Text IDs:  [num_text_tokens, 3] — all zeros (no spatial position)
Image IDs: [H_packed * W_packed, 3] — channel 0 = 0, channel 1 = row, channel 2 = col

Combined:  [num_text_tokens + num_image_tokens, 3]
```

For a 1024x1024 image with max 512 text tokens:
- Latent: 128x128, packed: 64x64 = 4096 image tokens
- Text: up to 512 tokens
- Total sequence: up to 4608 tokens

### RoPE Embedding Layout

The `EmbedND` class produces position embeddings of shape `[1, total_seq_len, head_dim/2, 2, 2]`:

```python
# For each axis i:
#   rope(ids[..., i], axes_dim[i], theta)
#   produces shape [total_seq_len, axes_dim[i]//2, 2, 2]
# Concatenated along dim=-3 (the frequency dimension):
#   shape [total_seq_len, 64, 2, 2]  (since 8+28+28=64)
# Then unsqueeze(1) for broadcast across heads:
#   shape [1, total_seq_len, 64, 2, 2]
```

### Attention Tensor Shapes

In double-stream blocks:
```
Q, K, V (per-stream): [B, num_heads, stream_seq_len, head_dim]
Q, K, V (concatenated): [B, num_heads, total_seq_len, head_dim]
Attention output: [B, total_seq_len, hidden_size]
Split back: txt_attn [B, txt_len, hidden_size], img_attn [B, img_len, hidden_size]
```

In single-stream blocks:
```
Input x: [B, total_seq_len, hidden_size]   (text + image concatenated)
Q, K, V: [B, num_heads, total_seq_len, head_dim]
Output:  [B, total_seq_len, hidden_size]
```

### Timestep Embedding

The sinusoidal timestep embedding function:
```python
def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t       # Scale [0,1] timesteps to [0, 1000]
    half = dim // 2
    freqs = exp(-log(max_period) * arange(0, half) / half)
    args = t[:, None] * freqs[None]
    return concat([cos(args), sin(args)], dim=-1)
```

The time_factor=1000 scales the [0,1] flow-matching timestep to the [0, 1000] range expected by the sinusoidal embedding.

---

## 4. Reference Implementations

### Primary Sources

- **BFL official repo**: [github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux) — canonical architecture implementation
  - `src/flux/model.py` — `FluxParams` dataclass, `Flux` class
  - `src/flux/modules/layers.py` — `DoubleStreamBlock`, `SingleStreamBlock`, `EmbedND`, `Modulation`, `QKNorm`
  - `src/flux/math.py` — `rope()`, `apply_rope()`, `attention()`
  - `src/flux/util.py` — model configs for dev/schnell/canny/depth/fill

- **Diffusers**: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
  - `src/diffusers/models/transformers/transformer_flux.py` — `FluxTransformer2DModel`
  - `src/diffusers/pipelines/flux/pipeline_flux.py` — `FluxPipeline`
  - `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` — `FlowMatchEulerDiscreteScheduler`
  - `src/diffusers/loaders/lora_conversion_utils.py` — LoRA key conversion functions

### Papers

- [Esser et al., 2024 — "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2403.03206) — SD3/Flux foundation (MMDiT, flow matching for image generation)
- [Lipman et al., 2023 — "Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747) — Flow matching framework
- [Peebles & Xie, 2023 — "Scalable Diffusion Models with Transformers (DiT)"](https://arxiv.org/abs/2212.09748) — DiT architecture
- [Dehghani et al., 2023 — "Scaling Vision Transformers to 22 Billion Parameters"](https://arxiv.org/abs/2302.05442) — Parallel attention+MLP blocks (used in SingleStreamBlock)
- [Su et al., 2024 — "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) — RoPE
- [Hu et al., 2022 — "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — LoRA

### Other Resources

- [DeepWiki — Flux Model Architecture](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)
- [Demystifying Flux Architecture](https://arxiv.org/html/2507.09595v1)
- [HuggingFace FluxTransformer2DModel docs](https://huggingface.co/docs/diffusers/main/en/api/models/flux_transformer)
- [HuggingFace Flux Pipeline docs](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)
- [FlowMatchEulerDiscreteScheduler docs](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)
- [Nunchaku LoRA Format Conversion](https://deepwiki.com/mit-han-lab/nunchaku/4.1-lora-format-conversion)
- [diffusers issue #9291 — kohya Flux LoRA key compatibility](https://github.com/huggingface/diffusers/issues/9291)

---

## 5. Differences Between Implementations

### BFL vs Diffusers Naming

| Concept | BFL Name | Diffusers Name |
|---------|----------|----------------|
| Image input projection | `img_in` | `x_embedder` |
| Text input projection | `txt_in` | `context_embedder` |
| Timestep embedder | `time_in` | `time_text_embed.timestep_embedder` |
| CLIP vec embedder | `vector_in` | `time_text_embed.text_embedder` |
| Guidance embedder | `guidance_in` | `time_text_embed.guidance_embedder` |
| Double-stream blocks | `double_blocks` | `transformer_blocks` |
| Single-stream blocks | `single_blocks` | `single_transformer_blocks` |
| Image modulation | `double_blocks.{i}.img_mod.lin` | `transformer_blocks.{i}.norm1.linear` |
| Text modulation | `double_blocks.{i}.txt_mod.lin` | `transformer_blocks.{i}.norm1_context.linear` |
| Image QKV (fused) | `double_blocks.{i}.img_attn.qkv` | `transformer_blocks.{i}.attn.to_q/k/v` (split) |
| Text QKV (fused) | `double_blocks.{i}.txt_attn.qkv` | `transformer_blocks.{i}.attn.add_q/k/v_proj` (split) |
| Single fused linear | `single_blocks.{i}.linear1` | split into `attn.to_q/k/v` + `proj_mlp` |
| Output projection | `single_blocks.{i}.linear2` | `single_transformer_blocks.{i}.proj_out` |
| Final norm | `final_layer.norm_final` | `norm_out.norm` |
| Final linear | `final_layer.linear` | `proj_out` |
| Final modulation | `final_layer.adaLN_modulation` | `norm_out.linear` |

### QKV Fusion Difference

BFL stores a **fused QKV** weight `[3 * hidden_size, hidden_size]` in each attention layer. Diffusers stores **separate** Q, K, V weights `[hidden_size, hidden_size]` each. This affects:
- Weight loading (must split/merge during conversion)
- LoRA application (fused LoRA on QKV vs. separate LoRA on Q, K, V)
- Memory layout and matmul patterns

### Single Block linear1 Fusion

BFL `single_blocks.{i}.linear1` is a single weight `[3*3072 + 12288, 3072] = [21504, 3072]` that fuses:
- QKV projection (3 * 3072 = 9216)
- MLP gate projection (12288)

Diffusers splits this into separate `to_q`, `to_k`, `to_v`, and `proj_mlp` weights.

### RoPE Implementation

BFL uses a 2x2 rotation matrix representation and applies it via element-wise operations with reshape tricks. Diffusers uses `apply_rotary_emb` with complex number or cos/sin pair representation. The mathematical result is identical.

---

## 6. Implementation Notes

### Memory Considerations

The 12B parameter model at FP16 requires ~24GB VRAM. Key memory optimization strategies:
- **Attention**: The combined sequence length (text + image) can be large (e.g., 4608 for 1024x1024). Flash attention or memory-efficient attention is essential.
- **Sequential block execution**: Process blocks sequentially (not in parallel) to limit activation memory.
- **Offloading**: T5-XXL can be offloaded to CPU after encoding text (it is only needed once per generation).

### Precision Requirements

- **RoPE frequencies**: Compute in float32 (the `rope()` function explicitly returns `.float()`). Apply in float32, then cast back.
- **RMSNorm**: Cast to float32 for the mean/rsqrt computation, then back to model dtype.
- **Modulation**: The SiLU + linear can stay in model dtype (BF16/FP16).
- **Attention**: Can use FP16/BF16 with flash attention. The QKNorm before attention improves numerical stability.

### C# Implementation Strategy

1. **Shared infrastructure with SD3**: The MMDiT pattern (joint attention with separate streams) is shared. The main difference is SD3 uses 3 text encoders and different block counts.

2. **RoPE as a reusable component**: The `rope()` function is a simple frequency computation + 2x2 matrix construction. Can be precomputed for a given resolution and cached. The `apply_rope()` function is a simple element-wise operation that can be SIMD-vectorized.

3. **Modulation as a pattern**: Every block uses the same modulation pattern (SiLU -> Linear -> chunk -> apply as shift/scale/gate). This should be a shared utility.

4. **Fused vs. split QKV**: For implementation, start with the diffusers-style separate Q/K/V projections for clarity. Optimize to fused QKV later if profiling shows benefit.

5. **Packing/unpacking**: The latent packing is a simple reshape/permute — implement as a view operation if the tensor layout supports it, otherwise a copy.

6. **LoRA support**: Must handle both BFL-format (fused QKV) and diffusers-format (split Q/K/V) LoRA weights. Implement conversion at load time, applying in the diffusers layout.

7. **Weight loading**: Support both BFL safetensors (fused QKV, single linear1) and diffusers safetensors (split projections). Convert BFL format to internal representation at load time by splitting fused weights.

### Key Formulas for SIMD/GPU Kernels

**RMSNorm:**
```
rrms = 1 / sqrt(mean(x^2) + eps)
output = x * rrms * scale
```

**Modulation application:**
```
x_out = (1 + scale) * LayerNorm(x) + shift
```

**RoPE application (per pair of elements):**
```
x_out[2i]   = cos(theta) * x[2i] - sin(theta) * x[2i+1]
x_out[2i+1] = sin(theta) * x[2i] + cos(theta) * x[2i+1]
```

**Flow-matching Euler step (per element):**
```
x_next = x + v * dt
```

### Testing Strategy

1. Load a known BFL safetensors checkpoint, verify weight shapes match expected layout
2. Run single DoubleStreamBlock with known input, compare output against diffusers to within 1e-4
3. Run single SingleStreamBlock with known input, compare output against diffusers to within 1e-4
4. Verify RoPE output matches BFL implementation for a grid of positions
5. Run full pipeline for schnell (1-4 steps) and compare against diffusers output pixel-by-pixel (tolerance ~1e-2 for full pipeline)
6. Test LoRA loading from all three formats (BFL, diffusers, kohya/ComfyUI)
