# Stable Diffusion Architectures — Research Notes

> **Stub.** The narrative walkthrough and restated pseudocode were removed on 2026-08-06 — this model
> is built and verified, so the C# is the source of truth for *how it works*. What remains is what the
> code cannot tell you: upstream provenance, reference constants, and bring-up traps. History is in git.

## SD 1.5

### Summary

Stable Diffusion 1.5 uses a UNet2DConditionModel with ~860M parameters that operates in the 4-channel VAE latent space. The architecture consists of 4 down-blocks, 1 middle block, and 4 up-blocks with channel dimensions 320, 640, 1280, 1280. Cross-attention to CLIP ViT-L/14 text embeddings (768-dim, 77 tokens) is present at the first three levels but not the fourth. The model uses 8 attention heads at all levels, GeGLU feed-forward blocks, GroupNorm with 32 groups, and SiLU activations throughout. Timestep conditioning is injected via sinusoidal embedding followed by an MLP then added into each ResNet block.

The UNet receives a noisy latent tensor of shape (B, 4, 64, 64) for 512x512 images and predicts the noise (epsilon-prediction by default). The down path produces 12 skip connection tensors that are concatenated with the up path hidden states, making up-path ResNet blocks significantly larger (e.g., 2560 to 1280 input channels).

Sources: [CompVis v1-inference.yaml](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml), [diffusers UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py), [CompVis openaimodel.py](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py)

### Model Configuration (Exact Values)

From the CompVis v1-inference.yaml (authoritative config, weights were trained with this):

```yaml
model_channels: 320
in_channels: 4
out_channels: 4
channel_mult: [1, 2, 4, 4]
num_res_blocks: 2
attention_resolutions: [4, 2, 1]  # levels 0, 1, 2
num_heads: 8
use_spatial_transformer: true
transformer_depth: 1
context_dim: 768
```

From the diffusers config.json ([source](https://huggingface.co/runwayml/stable-diffusion-v1-5/raw/main/unet/config.json)):

```json
{
  "in_channels": 4,
  "out_channels": 4,
  "block_out_channels": [320, 640, 1280, 1280],
  "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
  "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
  "layers_per_block": 2,
  "cross_attention_dim": 768,
  "attention_head_dim": 8,
  "norm_num_groups": 32,
  "norm_eps": 1e-05,
  "act_fn": "silu",
  "sample_size": 64,
  "flip_sin_to_cos": true,
  "freq_shift": 0
}
```

**NOTE on attention_head_dim**: In diffusers, `attention_head_dim: 8` caused naming confusion ([Issue #2011](https://github.com/huggingface/diffusers/issues/2011)). The CompVis source is authoritative: **8 attention heads** at every level, with `dim_head = channels / num_heads`.

### Key Numbers / Constants

| Parameter | Value |
|-----------|-------|
| Input/output channels | 4 |
| Model channels | 320 |
| Channel multipliers | [1, 2, 4, 4] |
| Block out channels | [320, 640, 1280, 1280] |
| Layers per down block | 2 |
| Layers per up block | 3 (extra for skip from downsample) |
| Cross-attention levels | 0, 1, 2 (NOT 3) |
| Transformer depth | 1 |
| Attention heads | 8 (all levels) |
| Dim per head | 40, 80, 160 (levels 0, 1, 2) |
| Cross-attention context dim | 768 (CLIP ViT-L/14) |
| Text tokens | 77 |
| Timestep embed dim | 320 sinusoidal to 1280 MLP |
| GroupNorm groups | 32 |
| GroupNorm eps | 1e-5 |
| Activation | SiLU |
| FF multiplier | 4x with GeGLU |
| Downsample | Conv2d stride=2, k=3, pad=1 |
| Upsample | nearest interp 2x + Conv2d k=3, pad=1 |
| Prediction type | epsilon (default) |
| VAE scaling factor | 0.18215 |
| Total UNet params | ~860M |
| Total model | ~1.066B (UNet+CLIP+VAE) |

### Data Layouts / Formats

Major weight tensor shapes at 1280ch (largest):
- attn1.to_q/k/v: (1280, 1280) each
- attn2.to_q: (1280, 1280), attn2.to_k/v: (1280, 768) each
- ff.net.0.proj: (10240, 1280) for GeGLU, ff.net.2: (1280, 5120)
- Up-path ResNet conv1: (1280, 2560, 3, 3) due to skip concat
- conv_shortcut in up-path: (1280, 2560, 1, 1)

Block counts: 22 ResNetBlock2D total (8 down + 2 mid + 12 up), 16 Transformer2DModel (6 down + 1 mid + 9 up), 3 Downsample, 3 Upsample.

### Differences Between Implementations

- CompVis vs diffusers: config field names differ (num_heads:8 vs attention_head_dim:8) but produce identical architectures
- proj_in/proj_out: Conv2d(1x1) or Linear depending on diffusers version, functionally identical
- Some community fine-tunes use v-prediction; UNet weights identical, only scheduler changes

### Implementation Notes

- Start with SD1.5 as simplest UNet, SDXL/Flux build on this
- Up-path ResNets have much larger weights due to skip concatenation (2560 input channels)
- GroupNorm(32) used everywhere, optimize thoroughly
- GeGLU (not GELU) in all FF blocks, the GEGLU linear is 2x wider than standard
- 12 skip connections stored during down pass, significant memory at peak
- Cross-attention K/V projections take 768-dim (CLIP) not channel dim
- Timestep embedding computed once per step, reused across all 22 ResNet blocks

### Reference Implementations

- [CompVis v1-inference.yaml](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml) -- original training config
- [CompVis openaimodel.py](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py) -- original UNet source
- [diffusers UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py), [unet_2d_blocks.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py), [resnet.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py), [embeddings.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py), [attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py)
- [diffusers Issue #2011](https://github.com/huggingface/diffusers/issues/2011) -- attention_head_dim naming confusion
- [labml.ai UNet walkthrough](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)

---

## SDXL

### Summary

Stable Diffusion XL (SDXL) is a latent diffusion model that significantly extends SD1.5 with a 2.6B-parameter UNet (vs ~860M for SD1.5), dual CLIP text encoders (CLIP-L/14 + OpenCLIP-G/14), micro-conditioning signals for image size/crop/target, and an optional refiner model that operates on partially-denoised latents. Key architectural differences from SD1.5:

- Three downsampling levels instead of four (channel_mult [1, 2, 4] vs [1, 2, 4, 4])
- Heterogeneous transformer block distribution [1, 2, 10] instead of uniform depth=1
- 2048-dimensional cross-attention context from concatenated dual encoder outputs (vs 768)
- Uniform 64 dim/head with variable head counts [5, 10, 20] (vs uniform 8 heads with variable dim/head)
- `use_linear_projection: true` (linear layers instead of 1x1 conv for attention projections)
- ADM micro-conditioning via timestep embedding path (not present in SD1.5)
- VAE scaling factor 0.13025 (vs 0.18215 for SD1.5)

### Key Numbers / Constants

| Constant | Value |
|----------|-------|
| Base UNet parameters | ~2.6B |
| CLIP-L parameters | ~123.65M (~250 MB fp16) |
| OpenCLIP-G parameters | ~694.7M (~680 MB fp16) |
| Cross-attention dim (base) | 2048 (768 + 1280 concatenated) |
| Cross-attention dim (refiner) | 1280 (CLIP-G only) |
| Pooled text dim | 1280 (from CLIP-G EOS token) |
| ADM vector dim (base) | 2816 (1280 + 6*256) |
| ADM vector dim (refiner) | 2560 (1280 + 5*256) |
| Time embed dim (base) | 1280 (4 * 320) |
| Time embed dim (refiner) | 1536 (4 * 384) |
| Addition time embed dim | 256 (sinusoidal per scalar) |
| Attention head dim | 64 (uniform, both models) |
| Max token length | 77 (per encoder, including BOS/EOS) |
| Noise schedule steps | 1000 (timesteps 0-999) |
| Refiner timestep range | 0-199 |
| Default aesthetic score (pos/neg) | 6.0 / 2.5 |
| Default guidance scale | 5.0-7.5 |
| VAE scaling factor | 0.13025 |

### Data Layouts / Formats

**Latent tensor:**
```
[batch, 4, height/8, width/8]
For 1024x1024: [B, 4, 128, 128]
```

**Text encoder outputs (base model):**
```
CLIP-L hidden states:   [B, 77, 768]
CLIP-G hidden states:   [B, 77, 1280]
Concatenated context:   [B, 77, 2048]   # cross-attention input
CLIP-G pooled output:   [B, 1280]       # for ADM vector
```

**Text encoder outputs (refiner):**
```
CLIP-G hidden states:   [B, 77, 1280]   # cross-attention input (no CLIP-L)
CLIP-G pooled output:   [B, 1280]       # for ADM vector
```

**ADM conditioning vector (base):**
```
pooled_text:    [B, 1280]   # CLIP-G pooled
orig_size_emb:  [B, 512]    # sinusoidal(orig_h) ++ sinusoidal(orig_w), each 256-dim
crop_emb:       [B, 512]    # sinusoidal(crop_top) ++ sinusoidal(crop_left), each 256-dim
target_emb:     [B, 512]    # sinusoidal(target_h) ++ sinusoidal(target_w), each 256-dim
--> concat all: [B, 2816]
--> project:    [B, 2816] -> Linear -> [B, 1280]
--> add to timestep embedding
```

**ADM conditioning vector (refiner):**
```
pooled_text:    [B, 1280]   # CLIP-G pooled
orig_size_emb:  [B, 512]    # sinusoidal(orig_h) ++ sinusoidal(orig_w)
crop_emb:       [B, 512]    # sinusoidal(crop_top) ++ sinusoidal(crop_left)
aesthetic_emb:  [B, 256]    # sinusoidal(aesthetic_score)
--> concat all: [B, 2560]
--> project:    [B, 2560] -> Linear -> [B, 1536]
--> add to timestep embedding
```

**UNet skip connection shapes (base model, 1024x1024 input):**
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

### Differences Between Implementations

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

### Implementation Notes

1. **Dual encoder management**: Need to load and run both CLIP-L and CLIP-G. The CLIP-G model is ~680 MB fp16 alone. Consider sequential execution to reduce peak memory.
2. **ADM vector assembly**: The conditioning vector must be built differently for base (2816-dim, 6 scalars) vs refiner (2560-dim, 5 scalars with aesthetic_score). This should be a separate utility function.
3. **Attention head dim confusion**: Internally use `head_dim = 64` consistently. Compute `num_heads = channels / 64` at each level. Do not follow the diffusers naming convention.
4. **VAE scaling factor**: Must use 0.13025, NOT 0.18215. Applying the wrong factor produces washed-out or oversaturated images. This is a common integration bug.
5. **Refiner is optional**: Many users skip the refiner entirely. The pipeline should work with base-only mode as the default.
6. **Memory considerations** (fp16): Base UNet ~5.5 GB, CLIP-L ~250 MB, CLIP-G ~680 MB, VAE ~160 MB, Total ~6.6 GB minimum. With refiner loaded simultaneously: add ~6 GB.
7. **Weight key mapping**: SDXL safetensors use different key prefixes than SD1.5. The `model.diffusion_model.` prefix is standard for Stability AI checkpoints. Diffusers format uses different names entirely.
8. **Heterogeneous transformer depth**: The [1, 2, 10] distribution means most compute is at 32x32 resolution. This has implications for tiling/chunking strategies and memory planning.
9. **Linear projection**: SDXL uses `use_linear_projection: true`, meaning attention Q/K/V projections use nn.Linear instead of 1x1 Conv2d. Functionally identical but may affect weight loading from different formats.
10. **GEGLU activation**: The transformer feedforward blocks use GEGLU. If the feedforward hidden dim is 4*1280=5120 for level 2, the actual linear layer outputs 2*5120=10240 which is split into two halves for GEGLU.

### Reference Implementations

- [Stability AI generative-models](https://github.com/Stability-AI/generative-models) -- original, configs at `configs/inference/sd_xl_base.yaml` and `sd_xl_refiner.yaml`
- [HuggingFace diffusers](https://github.com/huggingface/diffusers) -- pipeline, UNet, [base config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json), [refiner config](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/unet/config.json)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) -- C++ inference
- [minSDXL](https://github.com/cloneofsimo/minSDXL) -- minimal implementation
- [SDXL paper (arXiv:2307.01952)](https://arxiv.org/abs/2307.01952)
- [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Refiner 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) on HuggingFace
- [HuggingFace diffusers SDXL docs](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl)
