# VAE Architecture — Research Notes

## Summary

The Variational Autoencoder (VAE) used in Stable Diffusion pipelines is an `AutoencoderKL` — a convolutional encoder-decoder with KL-divergence regularization that converts between RGB pixel space and a lower-dimensional latent space. All SD-family models (SD1.5, SDXL, SD3, Flux) use **the same fundamental architecture** (identical block types, identical layer structure) but differ in latent channel count (4 vs 16), scaling/shift constants, and whether quant/post-quant convolutions are present. The encoder compresses images by 8x spatially through four downsampling stages; the decoder reverses this with four upsampling stages. A mid-block with self-attention sits at the bottleneck. Tiled VAE decoding allows arbitrarily large images to be decoded within fixed VRAM by splitting the latent tensor into overlapping tiles, decoding each independently, and blending overlapping regions with linear interpolation.

## Detailed Findings

### 1. Overall Pipeline Role

The VAE operates at the boundaries of the diffusion pipeline:
- **Encode** (image-to-latent): Used during img2img, inpainting, and training. Takes an RGB image `[B, 3, H, W]` and produces a latent `[B, C, H/8, W/8]` where C is 4 (SD1.5/SDXL) or 16 (SD3/Flux).
- **Decode** (latent-to-image): Used at the end of every generation. Takes a latent `[B, C, H/8, W/8]` and produces an RGB image `[B, 3, H, W]`.

The encoder outputs a `DiagonalGaussianDistribution` (mean + logvar), from which a latent sample is drawn via the reparameterization trick. The decoder is deterministic.

### 2. Encoder Architecture

The encoder follows this exact sequence:

1. **conv_in**: `Conv2d(3 -> block_out_channels[0], kernel=3, stride=1, padding=1)` — e.g., `Conv2d(3 -> 128)`
2. **down_blocks**: 4x `DownEncoderBlock2D`, each containing:
   - `layers_per_block` (default 2) `ResnetBlock2D` layers
   - A `Downsample2D` (strided Conv2d) at the end of every block **except the last**
3. **mid_block**: `UNetMidBlock2D` containing:
   - 1 `ResnetBlock2D`
   - 1 `Attention` (self-attention) layer
   - 1 `ResnetBlock2D`
4. **conv_norm_out**: `GroupNorm(32, block_out_channels[-1])` — e.g., `GroupNorm(32, 512)`
5. **conv_act**: `SiLU` activation
6. **conv_out**: `Conv2d(block_out_channels[-1] -> 2 * latent_channels, kernel=3, padding=1)` — outputs mean + logvar (e.g., `Conv2d(512 -> 8)` for 4-channel latent, `Conv2d(512 -> 32)` for 16-channel latent)
7. **quant_conv** (SD1.5/SDXL only): `Conv2d(2 * latent_channels -> 2 * latent_channels, kernel=1)` — absent in SD3/Flux (`use_quant_conv=false`)

Channel progression through down_blocks for the standard config `[128, 256, 512, 512]`:

| Block | Input Channels | Output Channels | Spatial Size (512px input) | Downsample? |
|-------|---------------|-----------------|---------------------------|-------------|
| 0     | 128           | 128             | 512 -> 256                | Yes         |
| 1     | 128           | 256             | 256 -> 128                | Yes         |
| 2     | 256           | 512             | 128 -> 64                 | Yes         |
| 3     | 512           | 512             | 64 (no change)            | No          |

### 3. Decoder Architecture

The decoder mirrors the encoder in reverse:

1. **post_quant_conv** (SD1.5/SDXL only): `Conv2d(latent_channels -> latent_channels, kernel=1)` — absent in SD3/Flux (`use_post_quant_conv=false`)
2. **conv_in**: `Conv2d(latent_channels -> block_out_channels[-1], kernel=3, stride=1, padding=1)` — e.g., `Conv2d(4 -> 512)`
3. **mid_block**: `UNetMidBlock2D` — same structure as encoder mid-block (ResNet-Attention-ResNet)
4. **up_blocks**: 4x `UpDecoderBlock2D`, each containing:
   - `layers_per_block + 1` (default 3) `ResnetBlock2D` layers (one extra compared to encoder)
   - An `Upsample2D` at the end of every block **except the last**
5. **conv_norm_out**: `GroupNorm(32, block_out_channels[0])` — e.g., `GroupNorm(32, 128)`
6. **conv_act**: `SiLU` activation
7. **conv_out**: `Conv2d(block_out_channels[0] -> 3, kernel=3, padding=1)` — outputs RGB

Channel progression through up_blocks (reversed order of block_out_channels):

| Block | Input Channels | Output Channels | Spatial Size (64px latent) | Upsample? |
|-------|---------------|-----------------|---------------------------|-----------|
| 0     | 512           | 512             | 64 -> 128                 | Yes       |
| 1     | 512           | 512             | 128 -> 256                | Yes       |
| 2     | 512           | 256             | 256 -> 512                | Yes       |
| 3     | 256           | 128             | 512 (no change)           | No        |

### 4. ResnetBlock2D Internal Structure

Each `ResnetBlock2D` follows this sequence:

```
input_tensor ─────────────────────────────────────────┐ (skip/shortcut)
     │                                                 │
     ├─> GroupNorm(32, in_ch) -> SiLU -> Conv2d(3x3)  │ (if in_ch != out_ch:
     │                                                 │   Conv2d(1x1) shortcut)
     ├─> GroupNorm(32, out_ch) -> SiLU -> Dropout      │
     │       -> Conv2d(3x3)                            │
     │                                                 │
     └─────────────── + ──────────────────────────────┘
                      │
                  output / output_scale_factor
```

Key parameters:
- **groups**: 32 (GroupNorm groups, matching `norm_num_groups`)
- **eps**: 1e-6 (GroupNorm epsilon)
- **non_linearity**: `swish` (SiLU)
- **output_scale_factor**: 1.0 (divides the residual sum)
- **conv_shortcut**: 1x1 `Conv2d` added when `in_channels != out_channels`
- No time embedding in VAE (the ResNet blocks in the VAE have `temb_channels=None`, unlike UNet ResNet blocks)

### 5. Mid-Block Attention

The mid-block's self-attention operates at the lowest spatial resolution (64x64 for 512px input, 128x128 for 1024px input). Structure:

- `Attention(channels=512, heads=1, dim_head=512, residual_connection=True)`
- The attention head count defaults to `channels // attention_head_dim` where `attention_head_dim = output_channel` (512), resulting in **1 attention head** with dimension 512.
- This is computationally expensive at higher resolutions; the gist by madebyollin notes it can be disabled with minimal quality impact.

### 6. DiagonalGaussianDistribution (KL Latent Space)

The encoder's `conv_out` produces `2 * latent_channels` channels, which are split into `mean` and `logvar`:

```python
mean, logvar = torch.chunk(encoder_output, 2, dim=1)
# logvar is clamped to [-30, 20] for numerical stability
std = torch.exp(0.5 * logvar)
# Reparameterization trick:
sample = mean + std * torch.randn_like(mean)
```

**KL divergence** (vs standard normal):
```python
kl = 0.5 * torch.sum(mean^2 + var - 1.0 - logvar, dim=[1, 2, 3])
```

**Mode** (deterministic, used during inference): simply returns `mean`.

During inference, pipelines typically use `.sample()` (with reparameterization) or `.mode()` (deterministic) from the distribution, then multiply by the `scaling_factor`.

### 7. Scaling and Shift Factors

The scaling factor normalizes the latent space to approximately unit variance. The shift factor (SD3/Flux only) centers the distribution. Applied as:

**Encoding** (image to latent for diffusion):
```
latents = (raw_latent - shift_factor) * scaling_factor
```

**Decoding** (latent from diffusion to image):
```
raw_latent = latents / scaling_factor + shift_factor
```

When `shift_factor` is None (SD1.5/SDXL), the formulas simplify to:
```
latents = raw_latent * scaling_factor        # encode
raw_latent = latents / scaling_factor        # decode
```

### 8. Tiled VAE Decoding

Tiled decoding splits the latent tensor into overlapping tiles, decodes each independently, and blends the results. This keeps VRAM usage constant regardless of output image size.

**Tile size computation** (from diffusers `AutoencoderKL.__init__`):
```python
tile_sample_min_size = sample_size                                            # e.g., 512 or 1024
tile_latent_min_size = sample_size / (2 ** (len(block_out_channels) - 1))     # e.g., 512/8 = 64 or 1024/8 = 128
tile_overlap_factor = 0.25
```

**Tiling algorithm** (`tiled_decode`):

```
overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor))
    # e.g., 64 * 0.75 = 48 (latent pixels between tile starts)
blend_extent = int(tile_latent_min_size * tile_overlap_factor)
    # e.g., 64 * 0.25 = 16 (latent pixels of overlap)
row_limit = tile_latent_min_size - blend_extent
    # e.g., 64 - 16 = 48

rows = []
for i in range(0, latent_height, overlap_size):
    row = []
    for j in range(0, latent_width, overlap_size):
        tile = z[:, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
        decoded_tile = decode_single_tile(tile)  # -> [B, 3, tile_sample_min_size, tile_sample_min_size]
        row.append(decoded_tile)
    rows.append(row)

# Blend horizontally within each row
for i, row in enumerate(rows):
    for j in range(1, len(row)):
        row[j] = blend_h(row[j-1], row[j], blend_extent)   # pixel-space blend extent
    rows[i] = [tile[:, :, :, :row_limit] for tile in row[:-1]] + [row[-1]]
    rows[i] = torch.cat(rows[i], dim=3)

# Blend vertically between rows
for i in range(1, len(rows)):
    rows[i] = blend_v(rows[i-1], rows[i], blend_extent)
result_rows = [row[:, :, :row_limit, :] for row in rows[:-1]] + [rows[-1]]
result = torch.cat(result_rows, dim=2)
```

**Blend functions** (linear interpolation):
```python
def blend_v(a, b, blend_extent):
    # Vertical: blend top of b with bottom of a
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for y in range(blend_extent):
        weight = y / blend_extent   # 0.0 at top (favor a) -> 1.0 at bottom (favor b)
        b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - weight) + b[:, :, y, :] * weight
    return b

def blend_h(a, b, blend_extent):
    # Horizontal: blend left of b with right of a
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    for x in range(blend_extent):
        weight = x / blend_extent
        b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - weight) + b[:, :, :, x] * weight
    return b
```

**Important**: The blend_extent computed in latent space gets scaled to pixel space by the VAE's spatial compression factor (8x). So 16 latent pixels of overlap = 128 pixel-space overlap in the decoded tiles.

### 9. SD1.5 vs SDXL VAE: Same Architecture, Different Weights

Confirmed: SDXL VAE has **identical architecture** to SD1.5 VAE. Both use:
- `block_out_channels = [128, 256, 512, 512]`
- `latent_channels = 4`
- `layers_per_block = 2`
- `norm_num_groups = 32`
- 4x `DownEncoderBlock2D` / 4x `UpDecoderBlock2D`
- `quant_conv` and `post_quant_conv` present

The differences are:
- **Training**: SDXL VAE trained with batch size 256 (vs 9 for SD1.5) and uses EMA weight tracking
- **sample_size**: 1024 (SDXL) vs 512 (SD1.5) — affects tiled decode tile sizes
- **scaling_factor**: 0.13025 (SDXL) vs 0.18215 (SD1.5) — reflects different latent distributions
- **Quality**: SDXL VAE produces better high-frequency detail; outperforms SD1.5 VAE on all reconstruction metrics

They are **not interchangeable** without retraining the UNet, because the different scaling factor and latent distribution statistics mean the UNet was trained expecting a specific latent distribution.

## Key Numbers/Constants

| Model | scaling_factor | shift_factor | latent_channels | Compression | sample_size | quant_conv | post_quant_conv |
|-------|---------------|-------------|-----------------|-------------|-------------|------------|-----------------|
| SD 1.5 | 0.18215 | None | 4 | 48x (8x spatial, 0.75x channel) | 512 | Yes | Yes |
| SDXL | 0.13025 | None | 4 | 48x | 1024 | Yes | Yes |
| SD3 | 1.5305 | 0.0609 | 16 | 12x (8x spatial, ~5.3x channel) | 1024 | No | No |
| Flux.1 | 0.3611 | 0.1159 | 16 | 12x | 1024 | No | No |

All models share these architectural constants:
- **block_out_channels**: `[128, 256, 512, 512]`
- **down_block_types**: 4x `DownEncoderBlock2D`
- **up_block_types**: 4x `UpDecoderBlock2D`
- **layers_per_block**: 2 (encoder), 3 (decoder, i.e., layers_per_block + 1)
- **norm_num_groups**: 32
- **act_fn**: `silu` (SiLU / Swish)
- **Spatial compression**: 8x (3 downsamples, each 2x)
- **Mid-block attention heads**: 1 (head_dim = 512)

### Tiled VAE Constants

| Model | tile_sample_min_size | tile_latent_min_size | tile_overlap_factor | overlap_size (latent) | blend_extent (latent) |
|-------|---------------------|---------------------|--------------------|-----------------------|-----------------------|
| SD 1.5 | 512 | 64 | 0.25 | 48 | 16 |
| SDXL | 1024 | 128 | 0.25 | 96 | 32 |
| SD3 | 1024 | 128 | 0.25 | 96 | 32 |
| Flux.1 | 1024 | 128 | 0.25 | 96 | 32 |

Optimal settings (from community testing and defaults in ComfyUI/A1111):
- **Pixel-space tile size**: 512 (default and most common)
- **Pixel-space overlap**: 64-128 pixels (larger = fewer seams, slower)
- **Latent-space tile size**: 64 (for 512px tiles)
- **Overlap factor**: 0.25 is the standard; increasing to 0.5 further reduces seam artifacts at the cost of more tiles

### Total Parameter Counts (approximate)

For the standard `[128, 256, 512, 512]` architecture with 4 latent channels:
- **Encoder**: ~34M parameters
- **Decoder**: ~49M parameters
- **Total VAE**: ~83M parameters (including quant_conv / post_quant_conv)

For 16 latent channels (SD3/Flux), the total is slightly higher (~84M) due to larger conv_out/conv_in.

## Data Layouts/Formats

**Input image**: `[B, 3, H, W]` — float32/float16, values in `[-1, 1]` (normalized from `[0, 255]`)

**Encoder output (raw)**: `[B, 2*C, H/8, W/8]` — split into mean `[B, C, H/8, W/8]` and logvar `[B, C, H/8, W/8]`

**Latent (after sampling + scaling)**: `[B, C, H/8, W/8]` where C=4 (SD1.5/SDXL) or C=16 (SD3/Flux)

**Decoder output**: `[B, 3, H, W]` — float32/float16, values approximately in `[-1, 1]`

**Weight tensor naming** (safetensors keys):
```
encoder.conv_in.weight                          [128, 3, 3, 3]
encoder.down_blocks.{0-3}.resnets.{0-1}.norm1.weight
encoder.down_blocks.{0-3}.resnets.{0-1}.norm1.bias
encoder.down_blocks.{0-3}.resnets.{0-1}.conv1.weight
encoder.down_blocks.{0-3}.resnets.{0-1}.norm2.weight
encoder.down_blocks.{0-3}.resnets.{0-1}.norm2.bias
encoder.down_blocks.{0-3}.resnets.{0-1}.conv2.weight
encoder.down_blocks.{0-2}.downsamplers.0.conv.weight    (blocks 0-2 only)
encoder.mid_block.resnets.{0-1}.norm1/conv1/norm2/conv2
encoder.mid_block.attentions.0.group_norm.weight
encoder.mid_block.attentions.0.to_q/to_k/to_v/to_out.0.weight
encoder.conv_norm_out.weight                    [512]
encoder.conv_out.weight                         [8, 512, 3, 3] or [32, 512, 3, 3]
quant_conv.weight                               [8, 8, 1, 1] or absent
decoder.conv_in.weight                          [512, 4, 3, 3] or [512, 16, 3, 3]
decoder.mid_block.resnets/attentions (same structure as encoder)
decoder.up_blocks.{0-3}.resnets.{0-2}           (3 resnets per up_block)
decoder.up_blocks.{0-2}.upsamplers.0.conv.weight (blocks 0-2 only)
decoder.conv_norm_out.weight                    [128]
decoder.conv_out.weight                         [3, 128, 3, 3]
post_quant_conv.weight                          [4, 4, 1, 1] or absent
```

## Algorithm Steps

### Encoding (Image to Latent)

```
1. Normalize image to [-1, 1]: x = (image / 255.0) * 2.0 - 1.0
2. h = conv_in(x)                                           # [B, 128, H, W]
3. For each down_block (i = 0..3):
     For each resnet in block (j = 0..1):
       h = resnet_block(h)                                  # GroupNorm->SiLU->Conv->GroupNorm->SiLU->Conv + skip
     If i < 3:
       h = downsample(h)                                    # Conv2d(stride=2), halves spatial dims
4. h = mid_block_resnet_0(h)                                # [B, 512, H/8, W/8]
5. h = mid_block_attention(h)                               # Self-attention at bottleneck
6. h = mid_block_resnet_1(h)
7. h = GroupNorm(h) -> SiLU(h)
8. h = conv_out(h)                                          # [B, 2*C, H/8, W/8]
9. If quant_conv present: h = quant_conv(h)                 # 1x1 conv
10. Split h into mean, logvar (along channel dim)
11. Sample: z = mean + exp(0.5 * logvar) * epsilon          # epsilon ~ N(0, 1)
12. Apply scaling: latent = (z - shift_factor) * scaling_factor
```

### Decoding (Latent to Image)

```
1. Undo scaling: z = latent / scaling_factor + shift_factor
2. If post_quant_conv present: z = post_quant_conv(z)       # 1x1 conv
3. h = conv_in(z)                                           # [B, 512, H/8, W/8]
4. h = mid_block_resnet_0(h)
5. h = mid_block_attention(h)
6. h = mid_block_resnet_1(h)
7. For each up_block (i = 0..3):
     For each resnet in block (j = 0..2):                   # 3 resnets per block
       h = resnet_block(h)
     If i < 3:
       h = upsample(h)                                      # nearest-neighbor interpolate + Conv2d, doubles spatial dims
8. h = GroupNorm(h) -> SiLU(h)
9. h = conv_out(h)                                          # [B, 3, H, W]
10. Denormalize: image = ((h + 1.0) / 2.0) * 255.0
```

### Tiled Decoding Algorithm

```
1. Compute tile parameters:
     overlap_size = tile_latent_min_size * (1 - tile_overlap_factor)    # stride between tiles
     blend_extent = tile_latent_min_size * tile_overlap_factor          # overlap region
     row_limit = tile_latent_min_size - blend_extent                    # kept portion per tile

2. Extract tiles in a grid:
     For i = 0, overlap_size, 2*overlap_size, ... < latent_height:
       For j = 0, overlap_size, 2*overlap_size, ... < latent_width:
         tile = latent[:, :, i:i+tile_latent_min_size, j:j+tile_latent_min_size]
         decoded_tile = decode(tile)    # standard decode on single tile
         Store in rows[row_idx][col_idx]

3. Blend horizontally within each row:
     For each pair of adjacent tiles (left, right):
       For x in 0..blend_extent:
         weight = x / blend_extent
         right[:, :, :, x] = left[:, :, :, -blend_extent+x] * (1-weight) + right[:, :, :, x] * weight
     Crop each tile (except last) to row_limit width, concatenate

4. Blend vertically between rows:
     For each pair of adjacent rows (top, bottom):
       For y in 0..blend_extent:
         weight = y / blend_extent
         bottom[:, :, y, :] = top[:, :, -blend_extent+y, :] * (1-weight) + bottom[:, :, y, :] * weight
     Crop each row (except last) to row_limit height, concatenate

5. Result is the full decoded image
```

## Reference Implementations

- **Primary**: [huggingface/diffusers `AutoencoderKL`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py) — the canonical implementation with tiled encode/decode
- **VAE Encoder/Decoder classes**: [huggingface/diffusers `vae.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py) — `Encoder` and `Decoder` class definitions
- **ResnetBlock2D**: [huggingface/diffusers `resnet.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py)
- **Block definitions**: [huggingface/diffusers `unet_2d_blocks.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py) — `DownEncoderBlock2D`, `UpDecoderBlock2D`, `UNetMidBlock2D`
- **SD1.5 VAE config**: [stable-diffusion-v1-5 vae/config.json](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/vae)
- **SDXL VAE config**: [stabilityai/sdxl-vae config.json](https://huggingface.co/stabilityai/sdxl-vae)
- **Flux.1 VAE config**: [black-forest-labs/FLUX.1-schnell vae/config.json](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/vae/config.json)
- **VAE scaling notes**: [madebyollin/notes_on_sd_vae (GitHub Gist)](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9) — comprehensive notes on VAE scaling, compression ratios, and model differences
- **ComfyUI TiledDiffusion**: [shiimizu/ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) — community tiled VAE implementation with configurable tile sizes
- **Original VAE paper**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
- **Latent Diffusion paper**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022) — sections 4.3.2 and D.1 for scaling factor derivation
- **SDXL VAE training details**: [HN discussion](https://news.ycombinator.com/item?id=39218520) — confirms identical architecture, different training (batch 256, EMA)
- **Scaling factor explanation**: [huggingface/diffusers issue #437](https://github.com/huggingface/diffusers/issues/437)
- **SD3 shift factor PR**: [huggingface/diffusers PR #8910](https://github.com/huggingface/diffusers/pull/8910)

## Differences Between Implementations

### SD1.5 / SDXL vs SD3 / Flux

| Feature | SD1.5 / SDXL | SD3 / Flux.1 |
|---------|-------------|-------------|
| Latent channels | 4 | 16 |
| quant_conv | Yes (1x1 Conv2d) | No |
| post_quant_conv | Yes (1x1 Conv2d) | No |
| shift_factor | None (not used) | Present (0.0609 / 0.1159) |
| Encoder conv_out | Conv2d(512 -> 8) | Conv2d(512 -> 32) |
| Decoder conv_in | Conv2d(4 -> 512) | Conv2d(16 -> 512) |
| Compression ratio | 48x | 12x |

### ComfyUI vs Diffusers Tiled VAE

- **Diffusers**: tile_overlap_factor = 0.25 (hardcoded), tile size = sample_size (512 or 1024 pixel-space), linear blending
- **ComfyUI**: configurable tile size (default 512), configurable overlap (default 64), same linear blending approach
- **A1111 TiledVAE extension**: more aggressive tiling with configurable parameters, tile size commonly 512, overlap 64-128

### Flux.2 VAE (Future Reference)

Per the madebyollin notes, Flux.2 introduces a significantly different VAE:
- 32 latent channels (smallest 6x compression factor)
- Modified architecture that encodes normalization scaling factors internally
- Uses RePA-like training scheme
- Different from the standard AutoencoderKL architecture

## Implementation Notes

### For SharpInference C# Implementation

1. **Single VAE class**: One `AutoencoderKL` class can handle all four model families. Use config values to control:
   - `latent_channels` (4 or 16)
   - `scaling_factor` and `shift_factor` (from config)
   - Whether to instantiate `quant_conv` / `post_quant_conv` (based on `use_quant_conv` / `use_post_quant_conv`)

2. **Memory optimization priorities**:
   - Tiled decode is essential for high-res generation
   - The mid-block attention is the most memory-intensive operation (full self-attention at bottleneck resolution); for 1024px input this is attention over 128x128 = 16,384 tokens
   - Consider float16/bfloat16 for all VAE operations (the `force_upcast` flag in diffusers forces float32 for the decoder for numerical stability, but many implementations skip this)

3. **Downsampling**: Implemented as `Conv2d(in_ch, out_ch, kernel=3, stride=2, padding=1)` — a strided convolution, not pooling

4. **Upsampling**: Implemented as nearest-neighbor interpolation (scale_factor=2) followed by `Conv2d(in_ch, out_ch, kernel=3, stride=1, padding=1)`

5. **GroupNorm**: All normalization uses 32 groups with eps=1e-6. The `GROUPNORM_MATH.md` research document covers the math.

6. **Attention in mid-block**: Single-head self-attention with `head_dim=512`. Uses group norm (not layer norm) before projection. Has residual connection. Can be disabled for speed with minimal quality loss.

7. **Weight loading order**: When loading safetensors, the key prefix structure is:
   - `encoder.` for encoder weights
   - `decoder.` for decoder weights
   - `quant_conv.` / `post_quant_conv.` for the 1x1 convolutions (absent in SD3/Flux)

8. **Tiled decode blending**: The linear blend can be precomputed as a weight mask for GPU efficiency rather than using the per-pixel loop shown in the diffusers reference. Create a 1D linear ramp `[0, 1/N, 2/N, ..., 1]` and apply it as a multiplicative mask during tile compositing.

9. **Decoder has more ResNet blocks**: The decoder uses `layers_per_block + 1 = 3` ResNet blocks per up_block (vs 2 per down_block in the encoder). This asymmetry is intentional — the decoder needs more capacity for reconstruction.

10. **Conv2d padding**: All 3x3 convolutions use `padding=1` (same padding). The 1x1 shortcut convolutions use `padding=0`.
