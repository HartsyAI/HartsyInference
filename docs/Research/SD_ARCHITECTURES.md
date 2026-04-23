# Stable Diffusion Architectures — Research Notes

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

### Channel Dimensions

| Level | Channels | Channel Mult | Spatial (512x512 input) |
|-------|----------|-------------|------------------------|
| 0 | 320 | 1 | 64 x 64 |
| 1 | 640 | 2 | 32 x 32 |
| 2 | 1280 | 4 | 16 x 16 |
| 3 | 1280 | 4 | 8 x 8 |

### Exact UNet Block Structure

#### Input Convolution
- Conv2d(4, 320, kernel_size=3, stride=1, padding=1) converts latent channels to model channels

#### Down Path (4 down-blocks)

**Down Block 0: CrossAttnDownBlock2D (320 ch, 64x64)**
- ResNetBlock2D(320 to 320) + Transformer2DModel(320, heads=8, dim_head=40, context=768)
- ResNetBlock2D(320 to 320) + Transformer2DModel(320, heads=8, dim_head=40, context=768)
- Downsample2D(320, Conv2d stride=2)
- Skip connections: 3 tensors at (B,320,64,64), (B,320,64,64), (B,320,32,32)

**Down Block 1: CrossAttnDownBlock2D (640 ch, 32x32)**
- ResNetBlock2D(320 to 640) + Transformer2DModel(640, heads=8, dim_head=80, context=768)
- ResNetBlock2D(640 to 640) + Transformer2DModel(640, heads=8, dim_head=80, context=768)
- Downsample2D(640, Conv2d stride=2)
- Skip connections: 3 tensors at (B,640,32,32), (B,640,32,32), (B,640,16,16)

**Down Block 2: CrossAttnDownBlock2D (1280 ch, 16x16)**
- ResNetBlock2D(640 to 1280) + Transformer2DModel(1280, heads=8, dim_head=160, context=768)
- ResNetBlock2D(1280 to 1280) + Transformer2DModel(1280, heads=8, dim_head=160, context=768)
- Downsample2D(1280, Conv2d stride=2)
- Skip connections: 3 tensors at (B,1280,16,16), (B,1280,16,16), (B,1280,8,8)

**Down Block 3: DownBlock2D (1280 ch, 8x8) NO ATTENTION**
- ResNetBlock2D(1280 to 1280)
- ResNetBlock2D(1280 to 1280)
- No downsample
- Skip connections: 2 tensors at (B,1280,8,8), (B,1280,8,8)

**Total skip connections: 3+3+3+2 = 11** (plus 1 from conv_in = 12 total)

#### Middle Block: UNetMidBlock2DCrossAttn (1280 ch, 8x8)
- ResNetBlock2D(1280 to 1280)
- Transformer2DModel(1280, heads=8, dim_head=160, context=768)
- ResNetBlock2D(1280 to 1280)

#### Up Path (4 up-blocks)

Each up block pops layers_per_block + 1 skip connections. Skip tensors are concatenated along channel dim before each ResNet block.

**Up Block 0: UpBlock2D (1280 ch, 8x8) NO ATTENTION**
- ResNetBlock2D(2560 to 1280), ResNetBlock2D(2560 to 1280), ResNetBlock2D(2560 to 1280)
- Upsample2D(1280, nearest interp + Conv2d 3x3)

**Up Block 1: CrossAttnUpBlock2D (1280 ch, 16x16)**
- ResNetBlock2D(2560 to 1280) + Transformer2DModel(1280, 8h, 160d, 768ctx)
- ResNetBlock2D(2560 to 1280) + Transformer2DModel(...)
- ResNetBlock2D(1920 to 1280) + Transformer2DModel(...)
- Upsample2D(1280)

**Up Block 2: CrossAttnUpBlock2D (640 ch, 32x32)**
- ResNetBlock2D(1920 to 640) + Transformer2DModel(640, 8h, 80d, 768ctx)
- ResNetBlock2D(1280 to 640) + Transformer2DModel(...)
- ResNetBlock2D(960 to 640) + Transformer2DModel(...)
- Upsample2D(640)

**Up Block 3: CrossAttnUpBlock2D (320 ch, 64x64)**
- ResNetBlock2D(960 to 320) + Transformer2DModel(320, 8h, 40d, 768ctx)
- ResNetBlock2D(640 to 320) + Transformer2DModel(...)
- ResNetBlock2D(640 to 320) + Transformer2DModel(...)
- No upsample

#### Final Output
- GroupNorm(32, 320, eps=1e-5) then SiLU then Conv2d(320, 4, 3, 1, 1)
- Output: (B, 4, 64, 64)

### Attention Configuration

| Level | Channels | Heads | Dim/Head | Cross-Attention |
|-------|----------|-------|----------|----------------|
| 0 | 320 | 8 | 40 | YES |
| 1 | 640 | 8 | 80 | YES |
| 2 | 1280 | 8 | 160 | YES |
| 3 | 1280 | - | - | NO |
| Mid | 1280 | 8 | 160 | YES |

### BasicTransformerBlock (inside Transformer2DModel)

Each Transformer2DModel contains 1 BasicTransformerBlock (transformer_depth=1) with:

1. LayerNorm then Self-Attention (Q,K,V from image features, bias=False for QKV projections)
2. LayerNorm then Cross-Attention (Q from image, K/V from CLIP text encoder 768-dim)
3. LayerNorm then FeedForward (GeGLU: channels to 4x channels with gating)

All with residual connections. The Transformer2DModel wraps this with GroupNorm + proj_in (Conv2d 1x1 or Linear) + [block] + proj_out + residual.

### ResNetBlock2D Structure

```
norm1(GroupNorm 32) -> SiLU -> conv1(3x3)
-> temb_proj(Linear 1280->out_ch) broadcast-add
-> norm2(GroupNorm 32) -> SiLU -> dropout(0.0) -> conv2(3x3)
-> + skip(input)
```

Skip = identity if channels match, else Conv2d(1x1).

### Timestep Embedding

- Sinusoidal: half_dim=160, frequencies=exp(-log(10000)*arange(160)/160), cos-first (flip_sin_to_cos=true), shape (B, 320)
- MLP: Linear(320,1280) then SiLU then Linear(1280,1280), output (B, 1280)
- Each ResNet block projects via Linear(1280, out_channels) and broadcast-adds

### v-prediction vs eps-prediction

SD 1.5 uses **epsilon-prediction** by default. Architecture is identical for both; difference is in scheduler math only. prediction_type is in the scheduler config, not the UNet config.

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

### Algorithm Steps

1. conv_in(latent) produces (B, 320, 64, 64), store as skip[0]
2. Sinusoidal(t) then MLP produces (B, 1280) timestep embedding
3. CLIP encode produces (B, 77, 768) text conditioning
4. Down path: ResNet+Attn per block, store skips, downsample
5. Mid block: ResNet then CrossAttn then ResNet
6. Up path: pop skips, concat, ResNet+Attn, upsample
7. GroupNorm then SiLU then conv_out produces (B, 4, 64, 64)

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

### UNet Architecture -- Base Model

| Level | Resolution (1024x1024 input) | Channels | Transformer Blocks | Cross-Attention | Down Block Type |
|-------|------------------------------|----------|--------------------|-----------------|-----------------|
| 0     | 128x128 (latent)             | 320      | 0                  | No              | DownBlock2D     |
| 1     | 64x64                        | 640      | 2                  | Yes             | CrossAttnDownBlock2D |
| 2     | 32x32                        | 1280     | 10                 | Yes             | CrossAttnDownBlock2D |
| Mid   | 32x32                        | 1280     | 10                 | Yes             | UNetMidBlock2DCrossAttn |

The up path mirrors the down path symmetrically:

| Level | Channels | Transformer Blocks | Up Block Type |
|-------|----------|--------------------|---------------|
| 0     | 1280     | 10                 | CrossAttnUpBlock2D |
| 1     | 640      | 2                  | CrossAttnUpBlock2D |
| 2     | 320      | 0                  | UpBlock2D     |

Each level has `layers_per_block = 2`. The heterogeneous transformer depth [1, 2, 10] concentrates attention capacity at the lowest spatial resolution, where semantic understanding matters most.

**Attention head configuration:**
- `attention_head_dim` (diffusers naming) = [5, 10, 20] -- these are actually the **number of heads** per level
- Each head has dimension 64 uniformly: 320/5=64, 640/10=64, 1280/20=64
- Stability AI config: `num_head_channels: 64`
- Cross-attention uses the same head structure, with keys/values projected from the 2048-dim text context

**Other UNet parameters:**
- in_channels / out_channels: 4
- sample_size: 128 (for 1024x1024 images with 8x VAE downsampling)
- act_fn: silu, norm_num_groups: 32, norm_eps: 1e-5
- flip_sin_to_cos: true, freq_shift: 0

### Dual CLIP Text Encoders

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
6. The pooled output is used as part of the ADM conditioning vector

**Parameter counts:**
- CLIP-L/14: ~123.65M (~250 MB fp16)
- OpenCLIP ViT-bigG/14: ~694.7M (~680 MB fp16)

### Micro-Conditioning (ADM Vector)

SDXL introduces "micro-conditioning" that encodes image metadata as additional conditioning signals injected into the UNet via the timestep embedding path (`addition_embed_type: "text_time"`).

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
6. The result is **added to the timestep embedding** and injected into the UNet via FiLM conditioning

Stability AI config: `adm_in_channels: 2816`

**Sinusoidal size embedding (per scalar):**
```
function embed_scalar(value, dim=256):
  half_dim = dim / 2  # 128
  freqs = exp(-ln(10000) * [0, 1, ..., half_dim-1] / half_dim)
  args = value * freqs
  return concat(cos(args), sin(args))  # [256]
```

### Timestep Embedding (SDXL Differences)

Same pattern as SD1.5 (sinusoidal 320-dim -> MLP -> 1280-dim) but with the ADM conditioning vector **added** to the 1280-dim result before injection into ResNet blocks.

### Refiner Model Architecture

The SDXL refiner is a second, separate UNet designed to operate on partially-denoised latents from the base model ("ensemble of expert denoisers").

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

The refiner has **4 downsampling levels** (like SD1.5) instead of the base's 3, but uses larger channel counts. Cross-attention occurs only at levels 1 and 2.

**Refiner ADM conditioning (2560-dim):**
- Pooled text embedding from CLIP-G: 1280
- orig_height, orig_width: 2 x 256 = 512
- crop_top, crop_left: 2 x 256 = 512
- aesthetic_score: 1 x 256 = 256
- Total: 1280 + 512 + 512 + 256 = **2560**

The refiner replaces target_size conditioning with a single **aesthetic_score** scalar. Default positive aesthetic score: 6.0. Default negative aesthetic score: 2.5 (some implementations use 7.5/2.0).

### Base-to-Refiner Handoff

The refiner is specialized on the first 200 discrete noise scales (out of 1000 total):

1. The base model denoises from timestep 999 down to some cutoff (e.g., timestep 200)
2. The refiner takes the partially-denoised latent and continues denoising from that point to timestep 0
3. In practice, controlled by `high_noise_frac` (default ~0.8): base handles 80% of denoising, refiner handles last 20%
4. The refiner operates via SDEdit (img2img) on the base model's output latents
5. Both models operate in the **same latent space** (same VAE)

### VAE (AutoencoderKL)

SDXL uses the same VAE architecture as SD1.5 but with different training and scaling factor.

| Parameter | SD 1.5 | SDXL |
|-----------|--------|------|
| scaling_factor | 0.18215 | **0.13025** |
| latent_channels | 4 | 4 |
| in/out channels | 3 (RGB) | 3 (RGB) |
| block_out_channels | [128, 256, 512, 512] | [128, 256, 512, 512] |
| layers_per_block | 2 | 2 |
| spatial_downscale_factor | 8 | 8 |
| sample_size | 512 | 1024 |

The SDXL VAE was retrained with a larger batch size (256 vs 9) and EMA weight tracking, resulting in better reconstruction quality.

```
encode: pixel_image [B, 3, H, W] -> latent [B, 4, H/8, W/8]
  latent = encoder(image) * 0.13025

decode: latent [B, 4, H/8, W/8] -> pixel_image [B, 3, H, W]
  image = decoder(latent / 0.13025)
```

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

### Algorithm Steps (Full SDXL Pipeline, Base + Refiner)

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

### Open Questions

- [ ] SDXL LoRA weight naming conventions vs SD1.5 -- differs due to different block structure and naming; needs separate research
- [ ] Exact behavior of `clip_skip` when using dual encoders -- both encoders use penultimate layer by default; does clip_skip apply to both?
- [ ] Whether the refiner's 4-level architecture with uniform transformer_depth=4 was chosen for quality or compatibility reasons
- [ ] Performance impact of removing the 4th downsampling level in the base model

### Reference Implementations

- [Stability AI generative-models](https://github.com/Stability-AI/generative-models) -- original, configs at `configs/inference/sd_xl_base.yaml` and `sd_xl_refiner.yaml`
- [HuggingFace diffusers](https://github.com/huggingface/diffusers) -- pipeline, UNet, [base config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json), [refiner config](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/unet/config.json)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) -- C++ inference
- [minSDXL](https://github.com/cloneofsimo/minSDXL) -- minimal implementation
- [SDXL paper (arXiv:2307.01952)](https://arxiv.org/abs/2307.01952)
- [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Refiner 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) on HuggingFace
- [HuggingFace diffusers SDXL docs](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl)
