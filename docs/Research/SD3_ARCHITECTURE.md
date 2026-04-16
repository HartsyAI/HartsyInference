# SD3 Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (SD3)

## Summary

Stable Diffusion 3 (SD3) uses a **Multi-Modal Diffusion Transformer (MMDiT)** architecture that jointly processes image and text tokens through transformer blocks with shared attention. It replaces the U-Net used in SD1.5/SDXL with a pure transformer operating on latent patches. Three text encoders (CLIP-L/14, OpenCLIP bigG/14, T5-v1.1-XXL) provide conditioning. The model uses **rectified flow matching** instead of DDPM noise schedules, with a logit-normal timestep distribution during training. QK-norm via RMSNorm stabilizes attention at scale. The architecture is parameterized by a single depth value `d` from which hidden size and head count are derived (`hidden_size = 64 * d`, `num_heads = d`).

SD3 shares the flow-matching paradigm with Flux but differs significantly in block structure: SD3 uses symmetric "joint blocks" where both modalities have independent weights but share a single concatenated attention operation, whereas Flux uses double-stream blocks (separate attention per modality with cross-attention) followed by single-stream blocks (concatenated sequence with shared weights).

---

## Detailed Findings

### 1. MMDiT Block Structure

Each MMDiT **JointBlock** contains two sub-blocks:

- **Context block** (`context_block`): processes text tokens, configured as `pre_only=True` (no MLP, no post-attention residual for the final block)
- **Image block** (`x_block`): processes image/latent tokens, full transformer block with attention + MLP

The processing flow within each JointBlock:

1. **Modulation (AdaLN-Zero)**: The timestep embedding `c` is passed through `SiLU + Linear` to produce shift, scale, and gate parameters. The formula is: `x_modulated = x * (1 + scale) + shift`
2. **Separate QKV projection**: Each sub-block independently computes Q, K, V from its modulated input
3. **Joint attention**: Q, K, V from both streams are concatenated along the sequence dimension, attention is computed over the combined sequence, then outputs are split back to their respective streams
4. **Gated residual**: Output is gated by a learned parameter: `x = x + gate * attn_output`
5. **MLP (image block only)**: SwiGLU feedforward network: `w2(silu(w1(x)) * w3(x))`
6. **Second gated residual** for MLP output

The final JointBlock has `context_pre_only=True`, meaning the context stream only contributes Q/K/V to the last attention operation but receives no output -- the text conditioning is "consumed" by the last layer.

### 2. Text Encoder Conditioning Pipeline

SD3 uses three text encoders that produce two conditioning signals:

**Encoder outputs:**
| Encoder | Hidden Dim | Context Length | Pooled Output Dim |
|---------|-----------|----------------|-------------------|
| CLIP-L/14 (OpenAI) | 768 | 77 tokens | 768 |
| CLIP-G/14 (OpenCLIP bigG) | 1280 | 77 tokens | 1280 |
| T5-v1.1-XXL | 4096 | 77 tokens (extendable to 256) | N/A (not used) |

**Context vector construction (`c_crossattn`):**
```
1. l_out = CLIP_L.encode(tokens)          # shape [B, 77, 768]
2. g_out = CLIP_G.encode(tokens)          # shape [B, 77, 1280]
3. t5_out = T5_XXL.encode(tokens)         # shape [B, 77, 4096]
4. lg_out = concat(l_out, g_out, dim=-1)  # shape [B, 77, 2048]
5. lg_out = pad(lg_out, (0, 4096-2048))   # shape [B, 77, 4096]  (zero-pad to match T5 dim)
6. context = concat(lg_out, t5_out, dim=-2) # shape [B, 154, 4096]
```

The context tensor is then projected from `joint_attention_dim` (4096) to `caption_projection_dim` (= hidden_size, 1536 for SD3 Medium) via a learned `nn.Linear` layer (`context_embedder`).

**Pooled projection (`y`):**
```
1. l_pooled = CLIP_L.pooled_output()      # shape [B, 768]
2. g_pooled = CLIP_G.pooled_output()      # shape [B, 1280]
3. y = concat(l_pooled, g_pooled, dim=-1) # shape [B, 2048]
```

The pooled vector `y` is combined with the timestep embedding and used to produce the AdaLN modulation parameters (shift/scale/gate) for every block. The `adm_in_channels` for the y-embedder is 2048.

**T5-XXL can be dropped at inference** with minimal quality loss (reduces VRAM by ~10GB). When dropped, the T5 portion of the context tensor is zeroed out.

### 3. QK-Norm (Query-Key Normalization)

SD3 applies **RMSNorm with learnable scale** to Q and K vectors before the attention dot product:

```
q = RMSNorm(q_proj(x))  # per-head normalization
k = RMSNorm(k_proj(x))  # per-head normalization
attn = softmax(q @ k^T / sqrt(d_head)) @ v
```

Key details:
- Applied independently to each attention head
- Uses RMSNorm (root mean square normalization), NOT LayerNorm
- Epsilon = 1e-6
- Includes a **learnable scale parameter** (unlike standard RMSNorm which just normalizes)
- The reference implementation supports `qk_norm="rms"` (RMSNorm) and `qk_norm="ln"` (LayerNorm) but the SD3 paper specifies RMSNorm
- This prevents attention logit explosion at high resolutions and large model scales, enabling stable training without loss divergence
- The normalization makes dot products equivalent to cosine similarity (scaled by the learnable parameter)

Note: The original SD3 Medium release (non-3.5) may not use QK-norm in the released checkpoint. SD3.5 models definitively use QK-norm. The reference code supports it as an optional parameter.

### 4. Positional Encoding

SD3 uses **2D sinusoidal positional embeddings** for image patches:

1. Image latent (e.g., 128x128 at latent resolution for 1024x1024 images) is divided into 2x2 patches, yielding a 64x64 grid of patch tokens
2. Separate sinusoidal embeddings are computed for height and width coordinates
3. Sin and cos components are concatenated across half the embedding dimension
4. Supports dynamic scaling via `pos_embed_scaling_factor` and `pos_embed_offset` for variable resolutions
5. For varying aspect ratios: positions are constructed based on maximum resolution, then center-cropped to match the actual aspect ratio ("bucketed sampling")

The positional embeddings are **added** to patch embeddings (not concatenated).

Text tokens do not receive explicit positional encoding in the MMDiT -- they rely on the positional information already encoded by the text encoders.

### 5. Flow Matching Formulation

SD3 uses **rectified flow** (conditional flow matching with optimal transport):

**Forward process (noising):**
```
x_t = (1 - sigma) * x_0 + sigma * noise
```
where `sigma` is the noise level at time `t`.

**Velocity prediction:** The model predicts the velocity field `v = dx/dt`, and the Euler step is:
```
x_{t-1} = x_t + (sigma_next - sigma_current) * model_output
```

**Timestep shifting (inference):** The sigma schedule is shifted to improve quality:
```
sigma = shift * t / (1 + (shift - 1) * t)
```
where `t` is linearly spaced in [0, 1] and `shift = 3.0` for SD3 Medium.

**Training timestep distribution:** Logit-normal sampling with parameters `mean=0.0, std=1.0`:
```
pi_ln(t; m, s) = 1/(s*sqrt(2*pi)) * 1/(t*(1-t)) * exp(-(logit(t)-m)^2 / (2*s^2))
```
This biases training toward intermediate noise levels where the model learns most effectively.

### 6. Patch Embedding

Images enter the MMDiT through a patch embedding layer:
- `nn.Conv2d(in_channels=16, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)`
- For SD3 Medium: `Conv2d(16, 1536, kernel_size=2, stride=2)`
- The output is flattened from spatial dims to a sequence of patch tokens
- Positional embeddings are then added

### 7. Timestep Embedding

- Sinusoidal positional encoding with `frequency_embedding_size=256`
- Fed through a 2-layer MLP: `Linear(256, hidden_size) -> SiLU -> Linear(hidden_size, hidden_size)`
- Combined with the pooled text projection via `CombinedTimestepTextProjEmbeddings`

### 8. Final Layer (Unpatchify)

The last layer projects the transformer output back to pixel space:
- AdaLN modulation (shift + scale from timestep conditioning)
- `nn.Linear(hidden_size, patch_size * patch_size * out_channels)` = `Linear(1536, 2*2*16)` = `Linear(1536, 64)`
- Reshape from sequence back to spatial: unflatten patches to reconstruct the latent image

---

## Key Numbers/Constants

### SD3 Medium (2B parameters)
| Parameter | Value |
|-----------|-------|
| Depth (num_layers) | 24 |
| Hidden size | 1536 (= 64 * 24) |
| Attention heads | 24 (= depth) |
| Head dimension | 64 |
| MLP hidden dim | 6144 (= 4 * 1536, with SwiGLU) |
| Patch size | 2 |
| Input/output channels | 16 (VAE latent channels) |
| Sample size (latent) | 128 (for 1024px images) |
| Pos embed max size | 192 |
| Joint attention dim | 4096 (context embedder input) |
| Caption projection dim | 1536 (= hidden_size) |
| Pooled projection dim | 2048 (= 768 + 1280) |
| ADM in channels | 2048 |
| Context sequence length | 154 (= 77 CLIP + 77 T5) |
| Latent scale factor | 1.5305 |
| Latent shift factor | 0.0609 |
| Flow matching shift | 3.0 |
| QK-norm | RMSNorm (learnable scale), eps=1e-6 |
| MLP activation | SwiGLU (SiLU gating) |
| AdaLN activation | SiLU |

### SD3.5 Large (8B parameters)
| Parameter | Value |
|-----------|-------|
| Depth (num_layers) | 38 |
| Hidden size | 2432 (= 64 * 38) |
| Attention heads | 38 |
| Head dimension | 64 |
| Architecture variant | MMDiT-X (dual attention) |
| Dual attention layers | First ~13 layers |
| QK-norm | RMSNorm (confirmed) |

### SD3.5 Medium (2.5B parameters)
| Parameter | Value |
|-----------|-------|
| Architecture variant | MMDiT-X (dual attention) |
| Dual attention layers | First ~12 layers |
| QK-norm | RMSNorm |

### Text Encoder Dimensions
| Encoder | Hidden | Layers | Heads | Vocab | Context Len |
|---------|--------|--------|-------|-------|-------------|
| CLIP-L/14 | 768 | 12 | 12 | 49408 | 77 |
| CLIP-G/14 | 1280 | 32 | 20 | 49408 | 77 |
| T5-v1.1-XXL | 4096 | 24 | 64 | 32128 | 77-256 |

### VAE (AutoEncoder)
| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| Downsampling factor | 8 |
| Base channels | 128 |
| Channel multipliers | (1, 2, 4, 4) |
| Resolution blocks per level | 2 |

### Scaling Study Model Sizes (from paper)
| Depth | Params (approx) |
|-------|-----------------|
| 15 | 450M |
| ~24 | ~2B |
| 38 | 8B |

---

## Data Layouts/Formats

### Input Latent
- Shape: `[B, 16, H/8, W/8]` (e.g., `[B, 16, 128, 128]` for 1024x1024)
- After patch embedding: `[B, (H/8/2)*(W/8/2), hidden_size]` = `[B, 4096, 1536]` for 1024x1024

### Context (Text Conditioning)
- Combined context: `[B, 154, 4096]` (77 CLIP tokens + 77 T5 tokens, each 4096-dim after padding)
- After context_embedder projection: `[B, 154, 1536]`

### Pooled Projection
- Shape: `[B, 2048]` (concatenated CLIP-L + CLIP-G pooled outputs)

### Timestep
- Scalar per batch element, range [0, 1] in flow matching (or [0, 1000] in discrete steps)

### Joint Attention Sequence
- Concatenated: `[B, num_image_tokens + num_text_tokens, hidden_size]`
- For 1024x1024: `[B, 4096 + 154, 1536]` = `[B, 4250, 1536]`

---

## Algorithm Steps

### Inference (Text-to-Image)

```
1. Encode text with all three encoders:
   a. l_out, l_pooled = CLIP_L(tokens_l)         # [B,77,768], [B,768]
   b. g_out, g_pooled = CLIP_G(tokens_g)         # [B,77,1280], [B,1280]
   c. t5_out = T5_XXL(tokens_t5)                 # [B,77,4096]
   d. lg_out = cat(l_out, g_out, dim=-1)         # [B,77,2048]
   e. lg_out = pad(lg_out, (0, 2048))            # [B,77,4096]
   f. context = cat(lg_out, t5_out, dim=-2)      # [B,154,4096]
   g. y = cat(l_pooled, g_pooled, dim=-1)        # [B,2048]

2. Project context: context = context_embedder(context)  # [B,154,1536]

3. Initialize latent x ~ N(0,1), shape [B,16,H/8,W/8]

4. Compute sigma schedule:
   a. timesteps = linspace(sigma_max, sigma_min, num_steps)
   b. sigmas = timesteps / 1000
   c. sigmas = shift * sigmas / (1 + (shift-1) * sigmas)   # shift=3.0
   d. Append sigma=0 as terminal

5. For each step i:
   a. Patch-embed x to tokens: [B, N_patches, hidden_size]
   b. Add positional embeddings
   c. Compute timestep + pooled embedding -> modulation params
   d. For each JointBlock (0..23):
      - Apply AdaLN to image tokens and context tokens (separate weights)
      - Compute Q_img, K_img, V_img from image tokens
      - Compute Q_ctx, K_ctx, V_ctx from context tokens
      - (Optional) Apply QK-norm: Q = RMSNorm(Q), K = RMSNorm(K)
      - Concatenate: Q = cat(Q_ctx, Q_img), K = cat(K_ctx, K_img), V = cat(V_ctx, V_img)
      - Attention: out = softmax(Q @ K^T / sqrt(64)) @ V
      - Split output back to context and image portions
      - Apply gated residual connections
      - MLP on image tokens (SwiGLU), gated residual
      - (Final block: context gets no output, only contributes to attention)
   e. Final layer: AdaLN + Linear projection -> [B, N_patches, patch_size^2 * 16]
   f. Unpatchify to [B, 16, H/8, W/8]
   g. Euler step: x = x + (sigma_next - sigma_current) * model_output

6. Decode latent: image = VAE.decode((x - shift_factor) * scale_factor)
   where scale_factor=1.5305, shift_factor=0.0609
```

### Classifier-Free Guidance (CFG)

SD3 supports standard CFG with separate conditional and unconditional forward passes:
```
output = uncond_output + guidance_scale * (cond_output - uncond_output)
```

The unconditional pass uses zeroed-out text embeddings and zero pooled projections.

---

## Reference Implementations

| Source | URL | Notes |
|--------|-----|-------|
| Stability AI SD3 reference | [github.com/Stability-AI/sd3-ref](https://github.com/Stability-AI/sd3-ref) | Original MMDiT implementation (`mmdit.py`, `sd3_impls.py`, `other_impls.py`) |
| Stability AI SD3.5 | [github.com/Stability-AI/sd3.5](https://github.com/Stability-AI/sd3.5) | MMDiT-X variant (`mmditx.py`) with dual attention |
| HuggingFace Diffusers | [SD3Transformer2DModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py) | Production implementation with JointTransformerBlock |
| HuggingFace Diffusers (attention) | [attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py) | JointTransformerBlock class |
| FlowMatchEulerDiscreteScheduler | [scheduling_flow_match_euler_discrete.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) | Flow matching Euler scheduler |
| SD3 Paper | [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206) | "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" |
| SD3 Paper PDF | [Stability AI S3](https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf) | Direct PDF link |
| Lucidrains MMDiT | [github.com/lucidrains/mmdit](https://github.com/lucidrains/mmdit) | Minimal single-layer MMDiT implementation |
| HuggingFace SD3 blog | [huggingface.co/blog/sd3](https://huggingface.co/blog/sd3) | Diffusers integration overview |
| Sayak Paul attention analysis | [sayak.dev/posts/attn-diffusion.html](https://sayak.dev/posts/attn-diffusion.html) | Comparison of attention flavors across diffusion models |
| DeepWiki SD3 text encoders | [deepwiki.com/Stability-AI/sd3-ref/4.4-text-encoders](https://deepwiki.com/Stability-AI/sd3-ref/4.4-text-encoders) | Text encoder combination details |
| SD3 HuggingFace model card | [huggingface.co/stabilityai/stable-diffusion-3-medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | Model card and weights |

---

## Differences Between Implementations

### SD3 MMDiT vs Flux DiT

| Aspect | SD3 (MMDiT) | Flux (DiT) |
|--------|-------------|------------|
| **Block type** | All JointBlocks (symmetric dual-stream) | Double-stream blocks (19) + single-stream blocks (38) |
| **Attention** | Concatenate Q/K/V from both modalities, single attention op | Double-stream: separate self-attn + cross-attn; Single-stream: concat into one sequence |
| **Context handling** | Context contributes to every block's attention, final block is context_pre_only | Context fully participates in double-stream blocks, then merged for single-stream |
| **Positional encoding** | 2D sinusoidal (additive) | RoPE (Rotary Position Embeddings) for both 2D image and text |
| **QK-norm** | RMSNorm (optional in SD3, standard in SD3.5) | RMSNorm (always enabled) |
| **MLP type** | SwiGLU | GELU (approximate=tanh) in some blocks |
| **Text encoders** | CLIP-L + CLIP-G + T5-XXL | CLIP-L + T5-XXL (no CLIP-G) |
| **Pooled conditioning** | CLIP-L + CLIP-G pooled (2048-dim) | CLIP-L pooled (768-dim) |
| **Flow matching shift** | shift=3.0 (static) | Dynamic shifting based on image resolution |
| **Guidance** | Standard CFG (two forward passes) | Guidance-distilled (single pass with guidance embedding) for Schnell |
| **Depth formula** | hidden_size = 64 * depth, heads = depth | Fixed: hidden_size=3072, heads=24 (for Flux-dev/schnell) |
| **Total params** | ~2B (Medium) / ~8B (Large) | ~12B |

### SD3 (Stability AI ref) vs Diffusers Implementation

| Aspect | Stability AI Reference | HuggingFace Diffusers |
|--------|----------------------|----------------------|
| **Class name** | `MMDiT` | `SD3Transformer2DModel` |
| **Block class** | `JointBlock` (wraps `DismantledBlock`) | `JointTransformerBlock` |
| **QK-norm param** | `qk_norm="rms"` or `"ln"` | `qk_norm="rms_norm"` or `"layer_norm"` |
| **Config format** | Derived from weight shapes at load time | Explicit `config.json` |
| **Depth derivation** | `depth = x_embedder.proj.weight.shape[0] // 64` | Explicit `num_layers` parameter |
| **MLP** | `SwiGLUFeedForward` class | Standard `FeedForward` with gelu-approximate |
| **Modulation** | `modulate()` function + manual shift/scale | `AdaLayerNormZero` module |

### SD3 vs SD3.5 (MMDiT vs MMDiT-X)

| Aspect | SD3 (MMDiT) | SD3.5 (MMDiT-X) |
|--------|-------------|------------------|
| **Dual attention** | No | Yes (first ~12-13 layers) |
| **QK-norm** | Optional/absent in released checkpoint | Always enabled (RMSNorm) |
| **AdaLN** | Standard AdaLN-Zero (6 params: shift, scale, gate for norm1 and MLP) | Extended AdaLN with extra modulation params for dual attention |
| **Depth** | 24 (Medium, 2B) | 24 (Medium, 2.5B) / 38 (Large, 8B) |
| **Extra params** | N/A | Second self-attention module per dual-attention layer |

---

## Open Questions

- [x] Exact QK-norm implementation: **RMSNorm with learnable scale, eps=1e-6, applied per-head to Q and K**
- [x] How the three text encoder outputs are combined: **CLIP-L + CLIP-G concatenated along feature dim (2048), zero-padded to 4096, then concatenated with T5 along sequence dim (154 tokens total)**
- [x] Architectural differences from Flux at the block level: **See comparison table above**
- [ ] Whether SD3 Medium's released checkpoint actually includes QK-norm weights (the reference code supports it optionally, but the original 2B release may not have trained with it; SD3.5 definitely uses it)
- [ ] Exact dual_attention_layers tuple for SD3.5 Medium and Large (confirmed to be approximately first 12-13 layers, but exact indices need verification from config.json)
- [ ] Whether the SwiGLU MLP in the Stability AI reference matches the GELU MLP in the diffusers implementation, or if diffusers adapted it

---

## Implementation Notes

### For SharpInference

1. **Code reuse with Flux**: The joint attention mechanism is fundamentally different from Flux's double-stream/single-stream split. However, the following can be shared:
   - Flow matching scheduler (same `FlowMatchEulerDiscreteScheduler`, different shift values)
   - VAE decoder (same 16-channel architecture)
   - T5-XXL text encoder
   - CLIP-L text encoder
   - Basic transformer infrastructure (attention, norms, MLPs)
   - Patch embedding / unpatchify operations

2. **SD3-specific components** that need dedicated implementation:
   - `JointBlock` with symmetric dual-stream attention (different from Flux's asymmetric blocks)
   - Text encoder combination logic (three encoders -> context + pooled)
   - AdaLN-Zero modulation with the specific parameter count
   - Context embedder (Linear projection from 4096 -> hidden_size)
   - 2D sinusoidal positional embeddings (vs RoPE in Flux)
   - QK-norm as optional per-model feature

3. **Memory considerations**:
   - T5-XXL alone is ~10GB in fp16; can be dropped for lighter inference
   - CLIP-L: ~0.5GB, CLIP-G: ~1.5GB
   - SD3 Medium transformer: ~4GB in fp16
   - Total with all encoders: ~16GB in fp16
   - Quantization (fp8 for T5, fp8/int8 for transformer) reduces this significantly

4. **Weight loading**: The Stability AI reference derives all architecture parameters from weight tensor shapes at load time. Key formula: `depth = x_embedder.proj.weight.shape[0] // 64`. This means SharpInference can auto-detect model configuration from safetensors metadata without requiring a separate config file.

5. **Latent scaling**: After VAE decode, apply: `latent_for_decode = (x / 1.5305) + 0.0609`. This is different from SD1.5/SDXL which use a simple scale factor.

6. **Scheduler defaults**: Use `shift=3.0` for SD3 Medium, `num_inference_steps=28` is a common default. The scheduler computes sigmas as: `sigma = 3.0 * t / (1 + 2.0 * t)` where t is linearly spaced.
