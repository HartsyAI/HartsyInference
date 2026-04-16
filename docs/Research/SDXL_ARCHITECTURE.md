# SDXL Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (SDXL)

## Summary

Stable Diffusion XL (SDXL) is a latent diffusion model that significantly extends SD1.5 with a 2.6B-parameter UNet (vs ~860M for SD1.5), dual CLIP text encoders (CLIP-L/14 + OpenCLIP-G/14), micro-conditioning signals for image size/crop/target, and an optional refiner model that operates on partially-denoised latents. The architecture uses three downsampling levels instead of four, a heterogeneous transformer block distribution [1, 2, 10], a 2048-dimensional cross-attention context from concatenated dual encoder outputs, and a VAE with scaling factor 0.13025 (vs 0.18215 for SD1.5). The refiner is a separate UNet with different channel dimensions (384/768/1536/1536), four downsampling levels, only CLIP-G for text encoding (cross_attention_dim=1280), and aesthetic score conditioning instead of target size conditioning.

## Detailed Findings

### 1. UNet Architecture — Base Model

The SDXL base UNet removes the fourth downsampling level present in SD1.5/2.x and uses channel multipliers [1, 2, 4] on a base of 320 channels:

| Level | Resolution (for 1024x1024 input) | Channels | Transformer Blocks | Cross-Attention | Down Block Type |
|-------|----------------------------------|----------|--------------------|-----------------|-----------------|
| 0     | 128x128 (latent)                 | 320      | 0                  | No              | DownBlock2D     |
| 1     | 64x64                            | 640      | 2                  | Yes             | CrossAttnDownBlock2D |
| 2     | 32x32                            | 1280     | 10                 | Yes             | CrossAttnDownBlock2D |
| Mid   | 32x32                            | 1280     | 10                 | Yes             | UNetMidBlock2DCrossAttn |

The up path mirrors the down path symmetrically:

| Level | Channels | Transformer Blocks | Up Block Type |
|-------|----------|--------------------|---------------|
| 0     | 1280     | 10                 | CrossAttnUpBlock2D |
| 1     | 640      | 2                  | CrossAttnUpBlock2D |
| 2     | 320      | 0                  | UpBlock2D     |

Each level has `layers_per_block = 2`, meaning 2 ResNet blocks per level. The CrossAttn levels interleave ResNet blocks with SpatialTransformer blocks. The heterogeneous transformer depth [1, 2, 10] concentrates attention capacity at the lowest spatial resolution, where semantic understanding matters most.

**Attention head configuration:**
- `attention_head_dim` (diffusers naming) = [5, 10, 20] -- these are actually the **number of heads** per level
- Each head has dimension 64 uniformly: 320/5=64, 640/10=64, 1280/20=64
- Stability AI config: `num_head_channels: 64`
- Cross-attention uses the same head structure, with keys/values projected from the 2048-dim text context

**Input/output:**
- in_channels: 4 (latent space)
- out_channels: 4 (latent space)
- sample_size: 128 (for 1024x1024 images with 8x VAE downsampling)

**Other UNet parameters:**
- `act_fn`: silu
- `norm_num_groups`: 32
- `norm_eps`: 1e-5
- `use_linear_projection`: true (linear layers instead of 1x1 conv for attention projections)
- `flip_sin_to_cos`: true
- `freq_shift`: 0

### 2. Dual CLIP Text Encoders

SDXL uses two text encoders processed in parallel:

| Encoder | Model | Hidden Size | Output Used | Max Tokens |
|---------|-------|-------------|-------------|------------|
| text_encoder_1 | OpenAI CLIP ViT-L/14 | 768 | Penultimate hidden state (clip_skip=2) | 77 |
| text_encoder_2 | OpenCLIP ViT-bigG/14 | 1280 | Penultimate hidden state + pooled output | 77 |

**Text conditioning construction:**

1. Both encoders process the same tokenized prompt (77 tokens each, padded/truncated independently)
2. The penultimate hidden states are extracted (layer -2, before the final layer norm)
3. The outputs are **concatenated along the channel/feature dimension**: [batch, 77, 768] + [batch, 77, 1280] = **[batch, 77, 2048]**
4. This 2048-dim context is used as the cross-attention key/value input (`cross_attention_dim: 2048`)
5. The **pooled output** from text_encoder_2 (CLIP-G) is extracted at the EOS token position: **[batch, 1280]**
6. The pooled output is used as part of the ADM conditioning vector (see below)

**Parameter counts:**
- CLIP-L/14: ~123.65M parameters (~250 MB fp16)
- OpenCLIP ViT-bigG/14: ~694.7M parameters (~680 MB fp16)

### 3. Micro-Conditioning (ADM Vector)

SDXL introduces "micro-conditioning" that encodes image metadata as additional conditioning signals injected into the UNet via the timestep embedding path. The system is called `addition_embed_type: "text_time"`.

**Six scalar conditioning parameters (base model):**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `orig_height` | Original training image height | 1024 |
| `orig_width` | Original training image width | 1024 |
| `crop_top` | Top crop coordinate (pixels) | 0 |
| `crop_left` | Left crop coordinate (pixels) | 0 |
| `target_height` | Target generation height | 1024 |
| `target_width` | Target generation width | 1024 |

**Embedding pipeline:**

1. Each of the 6 scalar values is independently embedded using sinusoidal/Fourier positional encoding (same function as timestep embedding) to `addition_time_embed_dim = 256` dimensions
2. The 6 embeddings are concatenated: 6 x 256 = **1536 dimensions**
3. The pooled text embedding (1280-dim from CLIP-G) is concatenated: 1280 + 1536 = **2816 dimensions**
4. This 2816-dim vector is the `projection_class_embeddings_input_dim`
5. A linear projection maps 2816 -> 1280 (the time embedding dimension = 4 * model_channels = 4 * 320 = 1280)
6. The result is **added to the timestep embedding** and injected into the UNet via FiLM conditioning (scale/shift on GroupNorm outputs)

**Stability AI config:**
```yaml
adm_in_channels: 2816
```

**Fourier embedding function** (same as timestep embedding):
```
embedding(x, dim=256):
  half = dim // 2  # 128
  freqs = exp(-ln(10000) * arange(0, half) / half)
  args = x * freqs
  embedding = [cos(args), sin(args)]  # [256]
```

### 4. Timestep Embedding

The timestep embedding follows the same pattern as SD1.5 but with adapted dimensions:

1. Scalar timestep t -> sinusoidal embedding (320-dim, matching model_channels)
2. Linear(320, 1280) -> SiLU -> Linear(1280, 1280)
3. ADM conditioning (from micro-conditioning above) is **added** to this 1280-dim vector
4. The combined embedding is injected into every ResNet block via FiLM conditioning

### 5. Refiner Model Architecture

The SDXL refiner is a second, separate UNet designed to operate on partially-denoised latents from the base model. It forms an "ensemble of expert denoisers."

**Key architectural differences from base:**

| Parameter | Base | Refiner |
|-----------|------|---------|
| model_channels | 320 | 384 |
| channel_mult | [1, 2, 4] | [1, 2, 4, 4] |
| block_out_channels | [320, 640, 1280] | [384, 768, 1536, 1536] |
| transformer_depth | [1, 2, 10] | 4 (uniform) |
| context_dim (cross_attention) | 2048 | 1280 |
| text_encoders | CLIP-L + CLIP-G | CLIP-G only |
| adm_in_channels | 2816 | 2560 |
| attention_head_dim (num_heads) | [5, 10, 20] | [6, 12, 24, 24] |
| head_dim | 64 | 64 |
| down_block_types | DB, CADB, CADB | DB, CADB, CADB, DB |
| up_block_types | CAUB, CAUB, UB | UB, CAUB, CAUB, UB |

The refiner has **4 downsampling levels** (like SD1.5) instead of the base's 3, but uses larger channel counts. Cross-attention occurs only at levels 1 and 2 (the middle two levels).

**Refiner ADM conditioning (2560-dim):**
- Pooled text embedding from CLIP-G: 1280
- orig_height, orig_width: 2 x 256 = 512
- crop_top, crop_left: 2 x 256 = 512
- aesthetic_score: 1 x 256 = 256
- Total: 1280 + 512 + 512 + 256 = **2560**

The refiner replaces target_size conditioning with a single **aesthetic_score** scalar. Default positive aesthetic score: 6.0. Default negative aesthetic score: 2.5 (some implementations use 7.5/2.0).

### 6. Base-to-Refiner Handoff

The refiner is specialized on the first 200 discrete noise scales (out of 1000 total). The handoff works as follows:

1. The base model denoises from timestep 999 down to some cutoff (e.g., timestep 200)
2. The refiner takes the partially-denoised latent and continues denoising from that point to timestep 0
3. In practice, this is controlled by the `high_noise_frac` parameter (default ~0.8):
   - `high_noise_frac = 0.8` means the base handles 80% of denoising, refiner handles last 20%
   - This maps to approximately timestep 200 as the switchover point
4. The refiner operates via SDEdit (img2img) on the base model's output latents
5. Both models operate in the **same latent space** (same VAE)

**Discrete noise schedule:** Both models use the same 1000-step noise schedule. The refiner was fine-tuned from the base model on timesteps 0-199 inclusive.

### 7. VAE (AutoencoderKL)

SDXL uses the same VAE architecture as SD1.5 (KL-regularized autoencoder) but with different training and a different scaling factor.

**VAE config:**

| Parameter | Value |
|-----------|-------|
| scaling_factor | **0.13025** (SD1.5: 0.18215) |
| latent_channels | 4 |
| in_channels / out_channels | 3 (RGB) |
| block_out_channels | [128, 256, 512, 512] |
| layers_per_block | 2 |
| norm_num_groups | 32 |
| act_fn | silu |
| sample_size | 1024 |
| spatial_downscale_factor | 8 |

The VAE was retrained with a larger batch size (256 vs 9 for SD1.5) and exponential moving average (EMA) weight tracking, resulting in better reconstruction quality.

**Encode/decode:**
```
encode: pixel_image [B, 3, H, W] -> latent [B, 4, H/8, W/8]
  latent = encoder(image) * 0.13025

decode: latent [B, 4, H/8, W/8] -> pixel_image [B, 3, H, W]
  image = decoder(latent / 0.13025)
```

For a 1024x1024 image, latent shape is [B, 4, 128, 128].

## Key Numbers/Constants

| Constant | Value | Context |
|----------|-------|---------|
| VAE scaling factor | 0.13025 | Multiply latents after encoding, divide before decoding |
| SD1.5 VAE scaling factor | 0.18215 | For comparison |
| Base UNet parameters | ~2.6B | Total trainable parameters |
| CLIP-L parameters | ~123.65M | First text encoder |
| OpenCLIP-G parameters | ~694.7M | Second text encoder |
| Cross-attention dim (base) | 2048 | 768 + 1280 concatenated |
| Cross-attention dim (refiner) | 1280 | CLIP-G only |
| Pooled text dim | 1280 | From CLIP-G EOS token |
| ADM vector dim (base) | 2816 | 1280 + 6*256 |
| ADM vector dim (refiner) | 2560 | 1280 + 5*256 |
| Time embed dim (base) | 1280 | 4 * model_channels (320) |
| Time embed dim (refiner) | 1536 | 4 * model_channels (384) |
| Addition time embed dim | 256 | Sinusoidal embedding for each size scalar |
| Attention head dim | 64 | Uniform across all levels (both models) |
| Max token length | 77 | Per encoder, including BOS/EOS |
| Noise schedule steps | 1000 | Discrete timesteps 0-999 |
| Refiner timestep range | 0-199 | Last 200 steps of denoising |
| Default aesthetic score (pos) | 6.0 | Refiner positive conditioning |
| Default aesthetic score (neg) | 2.5 | Refiner negative conditioning |
| Default guidance scale | 5.0-7.5 | Typical CFG scale for base |

## Data Layouts/Formats

### Latent tensor
```
[batch, 4, height/8, width/8]
For 1024x1024: [B, 4, 128, 128]
```

### Text encoder outputs (base model)
```
CLIP-L hidden states:   [B, 77, 768]
CLIP-G hidden states:   [B, 77, 1280]
Concatenated context:   [B, 77, 2048]   # cross-attention input
CLIP-G pooled output:   [B, 1280]       # for ADM vector
```

### Text encoder outputs (refiner)
```
CLIP-G hidden states:   [B, 77, 1280]   # cross-attention input (no CLIP-L)
CLIP-G pooled output:   [B, 1280]        # for ADM vector
```

### ADM conditioning vector (base)
```
pooled_text:    [B, 1280]   # CLIP-G pooled
orig_size_emb:  [B, 512]    # sinusoidal(orig_h) ++ sinusoidal(orig_w), each 256-dim
crop_emb:       [B, 512]    # sinusoidal(crop_top) ++ sinusoidal(crop_left), each 256-dim
target_emb:     [B, 512]    # sinusoidal(target_h) ++ sinusoidal(target_w), each 256-dim
--> concat all: [B, 2816]
--> project:    [B, 2816] -> Linear -> [B, 1280]
--> add to timestep embedding
```

### ADM conditioning vector (refiner)
```
pooled_text:    [B, 1280]   # CLIP-G pooled
orig_size_emb:  [B, 512]    # sinusoidal(orig_h) ++ sinusoidal(orig_w)
crop_emb:       [B, 512]    # sinusoidal(crop_top) ++ sinusoidal(crop_left)
aesthetic_emb:  [B, 256]    # sinusoidal(aesthetic_score)
--> concat all: [B, 2560]
--> project:    [B, 2560] -> Linear -> [B, 1536]
--> add to timestep embedding
```

### UNet skip connection shapes (base model, 1024x1024 input)
```
Down path output shapes (stored for skip connections):
  Block 0: [B, 320, 128, 128]   (conv_in)
  Block 1: [B, 320, 128, 128]   (ResBlock)
  Block 2: [B, 320, 128, 128]   (ResBlock)
  Block 3: [B, 320, 64, 64]     (Downsample)
  Block 4: [B, 640, 64, 64]     (ResBlock + 2x Transformer)
  Block 5: [B, 640, 64, 64]     (ResBlock + 2x Transformer)
  Block 6: [B, 640, 32, 32]     (Downsample)
  Block 7: [B, 1280, 32, 32]    (ResBlock + 10x Transformer)
  Block 8: [B, 1280, 32, 32]    (ResBlock + 10x Transformer)

Mid block: [B, 1280, 32, 32]

Up path (concatenates matching skip connection):
  Block 0: cat([B, 1280, 32, 32], skip8) -> [B, 2560, 32, 32] -> ResBlock+10xTrans -> [B, 1280, 32, 32]
  Block 1: cat([B, 1280, 32, 32], skip7) -> [B, 2560, 32, 32] -> ResBlock+10xTrans -> [B, 1280, 32, 32]
  Block 2: cat([B, 1280, 32, 32], skip6) -> [B, 1920, 32, 32] -> ResBlock+10xTrans+Upsample -> [B, 1280, 64, 64]
  Block 3: cat([B, 1280, 64, 64], skip5) -> [B, 1920, 64, 64] -> ResBlock+2xTrans -> [B, 640, 64, 64]
  Block 4: cat([B, 640, 64, 64], skip4) -> [B, 1280, 64, 64] -> ResBlock+2xTrans -> [B, 640, 64, 64]
  Block 5: cat([B, 640, 64, 64], skip3) -> [B, 960, 64, 64] -> ResBlock+2xTrans+Upsample -> [B, 640, 128, 128]
  Block 6: cat([B, 640, 128, 128], skip2) -> [B, 960, 128, 128] -> ResBlock -> [B, 320, 128, 128]
  Block 7: cat([B, 320, 128, 128], skip1) -> [B, 640, 128, 128] -> ResBlock -> [B, 320, 128, 128]
  Block 8: cat([B, 320, 128, 128], skip0) -> [B, 640, 128, 128] -> ResBlock -> [B, 320, 128, 128]

Output: GroupNorm -> SiLU -> Conv2D(320, 4) -> [B, 4, 128, 128]
```

## Algorithm Steps

### Full SDXL Generation Pipeline (Base + Refiner)

```
1. Tokenize prompt with both CLIP-L and CLIP-G tokenizers (max 77 tokens each)
2. Encode tokens through both text encoders:
   a. CLIP-L: extract penultimate hidden state [B, 77, 768]
   b. CLIP-G: extract penultimate hidden state [B, 77, 1280] + pooled [B, 1280]
3. Concatenate hidden states: [B, 77, 768] ++ [B, 77, 1280] = [B, 77, 2048]
4. Build ADM vector:
   a. Embed each of 6 size scalars with sinusoidal encoding (256-dim each)
   b. Concatenate: pooled_text(1280) ++ size_embeds(1536) = [B, 2816]
5. Initialize random latent noise: [B, 4, 128, 128] (for 1024x1024)
6. For each timestep t from T_max down to T_switch (e.g., 999 to 200):
   a. Compute timestep embedding + ADM conditioning
   b. Run base UNet: noise_pred = UNet(latent, t, context_2048, adm_2816)
   c. Apply CFG: noise_pred = uncond + scale * (cond - uncond)
   d. Scheduler step: latent = scheduler.step(noise_pred, t, latent)
7. (Optional) Switch to refiner:
   a. Re-encode prompt with CLIP-G only -> context [B, 77, 1280], pooled [B, 1280]
   b. Build refiner ADM: pooled(1280) ++ size_embeds(1024) ++ aesthetic(256) = [B, 2560]
   c. For each timestep t from T_switch down to 0:
      - Run refiner UNet: noise_pred = RefinerUNet(latent, t, context_1280, adm_2560)
      - Apply CFG and scheduler step
8. Decode final latent: image = VAE.decode(latent / 0.13025)
9. Clip to [0, 1] and convert to uint8
```

### Sinusoidal Size Embedding (per scalar)

```
function embed_scalar(value, dim=256):
  half_dim = dim / 2  # 128
  log_base = ln(10000.0)
  freqs = exp(-log_base * [0, 1, ..., half_dim-1] / half_dim)
  args = value * freqs
  return concat(cos(args), sin(args))  # [256]
```

## Reference Implementations

- **Stability AI generative-models** (original): https://github.com/Stability-AI/generative-models
  - Base config: `configs/inference/sd_xl_base.yaml`
  - Refiner config: `configs/inference/sd_xl_refiner.yaml`
- **HuggingFace diffusers** (most widely used): https://github.com/huggingface/diffusers
  - Pipeline: `src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`
  - UNet: `src/diffusers/models/unets/unet_2d_condition.py`
  - Base config: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json
  - Refiner config: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/unet/config.json
- **stable-diffusion.cpp** (C++ inference): https://github.com/leejet/stable-diffusion.cpp
- **minSDXL** (minimal implementation): https://github.com/cloneofsimo/minSDXL
- **SDXL paper**: https://arxiv.org/abs/2307.01952

## Differences Between Implementations

| Aspect | Stability AI (sgm) | HuggingFace diffusers | ComfyUI |
|--------|--------------------|-----------------------|---------|
| Config format | YAML (ldm-style) | JSON (diffusers-style) | Python dicts |
| `attention_head_dim` naming | `num_head_channels: 64` | `attention_head_dim: [5, 10, 20]` (misleading: these are num_heads) | `num_head_channels` |
| Transformer depth | `transformer_depth: [1, 2, 10]` | `transformer_layers_per_block: [1, 2, 10]` | Same as sgm |
| Channel config | `channel_mult: [1, 2, 4]` on `model_channels: 320` | `block_out_channels: [320, 640, 1280]` | Same as sgm |
| ADM channels | `adm_in_channels: 2816` | `projection_class_embeddings_input_dim: 2816` | `adm_in_channels` |
| Default aesthetic score | 6.0 / 2.5 | 6.0 / 2.5 | User-configurable (often 7.5 / 2.0) |
| Refiner handoff | `high_noise_frac` parameter | `denoising_end` / `denoising_start` | Manual timestep split |
| Weight format | safetensors (single file) | safetensors (sharded) | Both supported |
| clip_skip | Layer index from end | `clip_skip` parameter | `stop_at_clip_layer` |

**Naming confusion note:** In the diffusers config, `attention_head_dim: [5, 10, 20]` is the **number of attention heads** at each level, NOT the dimension per head. The actual head dimension is `block_out_channels[i] / attention_head_dim[i] = 64` uniformly. The Stability AI config uses `num_head_channels: 64` which is the per-head dimension directly.

## Open Questions

- [x] Exact refiner handoff timestep conventions — Refiner trained on timesteps 0-199, base handles 200-999, controlled via `high_noise_frac` or `denoising_end`/`denoising_start`
- [x] Aesthetic score conditioning range and default values — Positive: 6.0, Negative: 2.5 (diffusers defaults)
- [ ] SDXL LoRA weight naming conventions vs SD1.5 — differs due to different block structure and naming; needs separate research
- [ ] Exact behavior of `clip_skip` when using dual encoders — both encoders use penultimate layer by default; does clip_skip apply to both?
- [ ] Whether the refiner's 4-level architecture with uniform transformer_depth=4 was chosen for quality or compatibility reasons
- [ ] Performance impact of removing the 4th downsampling level in the base model

## Implementation Notes

### For SharpInference

1. **Dual encoder management**: Need to load and run both CLIP-L and CLIP-G. The CLIP-G model is ~680 MB fp16 alone. Consider sequential execution to reduce peak memory.

2. **ADM vector assembly**: The conditioning vector must be built differently for base (2816-dim, 6 scalars) vs refiner (2560-dim, 5 scalars with aesthetic_score). This should be a separate utility function.

3. **Attention head dim confusion**: Internally use `head_dim = 64` consistently. Compute `num_heads = channels / 64` at each level. Do not follow the diffusers naming convention.

4. **VAE scaling factor**: Must use 0.13025, NOT 0.18215. Applying the wrong factor produces washed-out or oversaturated images. This is a common integration bug.

5. **Refiner is optional**: Many users skip the refiner entirely. The pipeline should work with base-only mode as the default.

6. **Memory considerations** (fp16):
   - Base UNet: ~5.5 GB
   - CLIP-L: ~250 MB
   - CLIP-G: ~680 MB
   - VAE: ~160 MB
   - Total: ~6.6 GB minimum
   - With refiner loaded simultaneously: add ~6 GB for refiner UNet

7. **Weight key mapping**: SDXL safetensors use different key prefixes than SD1.5. The `model.diffusion_model.` prefix is standard for Stability AI checkpoints. Diffusers format uses different names entirely.

8. **Heterogeneous transformer depth**: The [1, 2, 10] distribution means most compute is at 32x32 resolution. This has implications for tiling/chunking strategies and memory planning.

9. **Linear projection**: SDXL uses `use_linear_projection: true`, meaning attention Q/K/V projections use nn.Linear instead of 1x1 Conv2d. This is functionally identical but may affect weight loading from different formats.

10. **GEGLU activation**: The transformer feedforward blocks use GEGLU (Gated GeLU) activation, which splits the intermediate dimension in half for gating. If the feedforward hidden dim is 4*1280=5120 for level 2, the actual linear layer outputs 2*5120=10240 which is split into two halves for GEGLU.

### Sources

- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis (arXiv)](https://arxiv.org/abs/2307.01952)
- [SDXL Base 1.0 on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL Refiner 1.0 on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- [Stability AI generative-models repo](https://github.com/Stability-AI/generative-models)
- [HuggingFace diffusers SDXL documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl)
- [SDXL UNet architecture gist](https://gist.github.com/w-hc/2c294af8e5db747593cca5149410fdf1)
- [stable-diffusion.cpp SDXL models (DeepWiki)](https://deepwiki.com/leejet/stable-diffusion.cpp/6.1.2-sdxl-models)
- [sd-scripts SDXL Training (DeepWiki)](https://deepwiki.com/sdbds/sd-scripts/4-sdxl-training)
- [The Arrival of SDXL 1.0 (Towards Data Science)](https://towardsdatascience.com/the-arrival-of-sdxl-1-0-4e739d5cc6c7/)
- [The SDXL Model Pipeline (xta0.me)](https://www.xta0.me/2025/01/20/GenAI-Stable-Diffusion-SDXL.html)
- [SDXL VAE config](https://huggingface.co/CaioXapelaum/sdxl/blob/main/vae/config.json)
- [Progressive Knowledge Distillation of SDXL](https://arxiv.org/html/2401.02677v1)
