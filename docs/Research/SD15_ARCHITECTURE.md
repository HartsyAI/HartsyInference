# SD 1.5 Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (UNet)

## Summary

Stable Diffusion 1.5 uses a UNet2DConditionModel with ~860M parameters that operates in the 4-channel VAE latent space. The architecture consists of 4 down-blocks, 1 middle block, and 4 up-blocks with channel dimensions 320, 640, 1280, 1280. Cross-attention to CLIP ViT-L/14 text embeddings (768-dim, 77 tokens) is present at the first three levels but not the fourth. The model uses 8 attention heads at all levels, GeGLU feed-forward blocks, GroupNorm with 32 groups, and SiLU activations throughout. Timestep conditioning is injected via sinusoidal embedding followed by an MLP then added into each ResNet block.

The UNet receives a noisy latent tensor of shape (B, 4, 64, 64) for 512x512 images and predicts the noise (epsilon-prediction by default). The down path produces 12 skip connection tensors that are concatenated with the up path hidden states, making up-path ResNet blocks significantly larger (e.g., 2560 to 1280 input channels).

Sources: [CompVis v1-inference.yaml](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml), [diffusers UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py), [CompVis openaimodel.py](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py)

## Detailed Findings

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

## Key Numbers / Constants

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
| Total UNet params | ~860M |
| Total model | ~1.066B (UNet+CLIP+VAE) |

## Data Layouts / Formats

Major weight tensor shapes at 1280ch (largest):
- attn1.to_q/k/v: (1280, 1280) each
- attn2.to_q: (1280, 1280), attn2.to_k/v: (1280, 768) each
- ff.net.0.proj: (10240, 1280) for GeGLU, ff.net.2: (1280, 5120)
- Up-path ResNet conv1: (1280, 2560, 3, 3) due to skip concat
- conv_shortcut in up-path: (1280, 2560, 1, 1)

Block counts: 22 ResNetBlock2D total (8 down + 2 mid + 12 up), 16 Transformer2DModel (6 down + 1 mid + 9 up), 3 Downsample, 3 Upsample.

## Algorithm Steps

1. conv_in(latent) produces (B, 320, 64, 64), store as skip[0]
2. Sinusoidal(t) then MLP produces (B, 1280) timestep embedding
3. CLIP encode produces (B, 77, 768) text conditioning
4. Down path: ResNet+Attn per block, store skips, downsample
5. Mid block: ResNet then CrossAttn then ResNet
6. Up path: pop skips, concat, ResNet+Attn, upsample
7. GroupNorm then SiLU then conv_out produces (B, 4, 64, 64)

## Reference Implementations

- [CompVis v1-inference.yaml](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml) original training config
- [CompVis openaimodel.py](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py) original UNet source
- [diffusers UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py)
- [diffusers unet_2d_blocks.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py)
- [diffusers resnet.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py)
- [diffusers embeddings.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py)
- [diffusers attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py)
- [diffusers Issue #2011](https://github.com/huggingface/diffusers/issues/2011) attention_head_dim naming confusion
- [labml.ai UNet walkthrough](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)
- [Stable Diffusion Wikipedia](https://en.wikipedia.org/wiki/Stable_Diffusion)

## Differences Between Implementations

- CompVis vs diffusers: config field names differ (num_heads:8 vs attention_head_dim:8) but produce identical architectures
- proj_in/proj_out: Conv2d(1x1) or Linear depending on diffusers version, functionally identical
- Some community fine-tunes use v-prediction; UNet weights identical, only scheduler changes

## Open Questions

- [x] Attention head count: **8 heads at all levels** (CompVis authoritative)
- [x] v-prediction vs eps-prediction: **identical architecture, different output interpretation**
- [x] Channel multiplier variants: **standard SD1.5 uses [1,2,4,4]**

## Implementation Notes

- Start with SD1.5 as simplest UNet, SDXL/Flux build on this
- Up-path ResNets have much larger weights due to skip concatenation (2560 input channels)
- GroupNorm(32) used everywhere, optimize thoroughly
- GeGLU (not GELU) in all FF blocks, the GEGLU linear is 2x wider than standard
- 12 skip connections stored during down pass, significant memory at peak
- Cross-attention K/V projections take 768-dim (CLIP) not channel dim
- Timestep embedding computed once per step, reused across all 22 ResNet blocks
