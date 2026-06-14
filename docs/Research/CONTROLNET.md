# ControlNet — Research Notes

## Summary

ControlNet (Zhang et al., 2023) adds spatial conditioning controls to pretrained text-to-image diffusion models by creating a trainable copy of the encoder blocks and connecting them to the frozen model through "zero convolutions" — 1x1 convolutions with weights and biases initialized to zero. This ensures the ControlNet produces zero output at initialization, preventing any distortion to the base model before training begins. Different ControlNet variants (Canny, Depth, OpenPose, Scribble, etc.) share the same architecture but differ only in their input preprocessing. For SD1.5, there are 13 injection points (12 encoder skip connections + 1 mid block). For SDXL, there are 10 injection points due to the smaller encoder. For Flux, ControlNet operates on transformer blocks rather than UNet blocks, injecting residuals into both the joint (double-stream) and single-stream transformer blocks. Multi-ControlNet combines residuals from multiple ControlNets via element-wise addition with per-ControlNet scaling factors.

## Detailed Findings

### Core Architecture

ControlNet creates two copies of the UNet encoder:
1. **Locked copy**: The original pretrained encoder weights, frozen during training
2. **Trainable copy**: An exact copy that learns to process conditional inputs

The trainable copy is connected to the locked model through **zero convolutions** — 1x1 convolutional layers where all weights and biases are initialized to zero. This design means:
- At initialization, the ControlNet outputs are all zeros, so the base model behavior is perfectly preserved
- During training, gradients flow through the locked encoder (which acts as a deep backbone), enabling the trainable copy to learn meaningful control features
- The parameters "progressively grow from zero," ensuring stable training without harmful noise

The original SD encoder does not need to store gradients (it is locked/frozen), so GPU memory overhead is relatively modest despite the doubled encoder parameters.

Sources: [ControlNet paper](https://arxiv.org/abs/2302.05543), [ControlNet GitHub](https://github.com/lllyasviel/ControlNet)

### Input Hint Block (Conditioning Embedding)

The control image is preprocessed by an **input hint block** (`ControlNetConditioningEmbedding`) that progressively downsamples the input by 8x to match the latent resolution:

```
Conv2d(hint_channels -> 16, 3x3, pad=1) -> SiLU ->
Conv2d(16 -> 16, 3x3, pad=1) -> SiLU ->
Conv2d(16 -> 32, 3x3, pad=1, stride=2) -> SiLU ->      # /2
Conv2d(32 -> 32, 3x3, pad=1) -> SiLU ->
Conv2d(32 -> 96, 3x3, pad=1, stride=2) -> SiLU ->       # /4
Conv2d(96 -> 96, 3x3, pad=1) -> SiLU ->
Conv2d(96 -> 256, 3x3, pad=1, stride=2) -> SiLU ->      # /8
zero_conv(256 -> model_channels, 3x3, pad=1)             # final zero conv
```

The `conditioning_embedding_out_channels` defaults to `[16, 32, 96, 256]` for both SD1.5 and SDXL ControlNets. The hint_channels is typically 3 (RGB control image input).

Source: [lllyasviel/ControlNet cldm.py](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py)

### SD1.5 ControlNet: 13 Injection Points

SD1.5 UNet has `block_out_channels = [320, 640, 1280, 1280]` with `layers_per_block = 2` and 4 down-block groups. The ControlNet mirrors the 12 encoder blocks + 1 mid block, producing **13 control outputs** total.

**SD1.5 ControlNet config:**
- `block_out_channels`: `[320, 640, 1280, 1280]`
- `down_block_types`: `["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]`
- `layers_per_block`: 2
- `cross_attention_dim`: 768
- `in_channels`: 4
- `conditioning_embedding_out_channels`: `[16, 32, 96, 256]`
- `attention_head_dim`: 8
- `act_fn`: "silu"
- `norm_num_groups`: 32

**The 13 injection points with channel dimensions:**

| Index | Source Block | Channel Dim | Spatial Resolution (at 512x512, latent 64x64) |
|-------|------------|-------------|-----------------------------------------------|
| 0 | Input conv | 320 | 64x64 |
| 1 | Down block 0, ResBlock 0 | 320 | 64x64 |
| 2 | Down block 0, ResBlock 1 | 320 | 64x64 |
| 3 | Down block 0, Downsample | 320 | 32x32 |
| 4 | Down block 1, ResBlock 0 | 640 | 32x32 |
| 5 | Down block 1, ResBlock 1 | 640 | 32x32 |
| 6 | Down block 1, Downsample | 640 | 16x16 |
| 7 | Down block 2, ResBlock 0 | 1280 | 16x16 |
| 8 | Down block 2, ResBlock 1 | 1280 | 16x16 |
| 9 | Down block 2, Downsample | 1280 | 8x8 |
| 10 | Down block 3, ResBlock 0 | 1280 | 8x8 |
| 11 | Down block 3, ResBlock 1 | 1280 | 8x8 |
| 12 | Mid block (separate) | 1280 | 8x8 |

Each zero_conv is a `Conv2d(ch, ch, kernel_size=1, padding=0)` with all parameters initialized to zero.

**How residuals are injected into the UNet decoder:**

In the original ControlNet implementation (`ControlledUnetModel`):
- **Mid block**: `h += control.pop()` — the mid block control residual is added directly to the UNet mid block output
- **Decoder (output) blocks**: `h = torch.cat([h, hs.pop() + control.pop()], dim=1)` — control residuals are added to the corresponding skip connection before concatenation with the decoder features

In the diffusers implementation:
- The ControlNet `forward()` returns `down_block_res_samples` (list of 12 tensors) and `mid_block_res_sample` (1 tensor)
- These are passed to the UNet, which adds them to the skip connections during the up-block processing

Sources: [ControlNet cldm.py](https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py), [diffusers ControlNetModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/controlnet.py), [control_v11p_sd15_canny config](https://huggingface.co/lllyasviel/control_v11p_sd15_canny)

### SDXL ControlNet: 10 Injection Points

SDXL has a different UNet with only 3 down-block groups (vs 4 in SD1.5), resulting in **10 injection points** (9 encoder + 1 mid block).

**SDXL ControlNet config:**
- `block_out_channels`: `[320, 640, 1280]`
- `down_block_types`: `["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]`
- `layers_per_block`: 2
- `cross_attention_dim`: 2048
- `in_channels`: 4
- `conditioning_embedding_out_channels`: `[16, 32, 96, 256]`
- `attention_head_dim`: `[5, 10, 20]`
- `transformer_layers_per_block`: `[1, 2, 10]`
- `use_linear_projection`: true
- `addition_embed_type`: "text_time"
- `addition_time_embed_dim`: 256
- `projection_class_embeddings_input_dim`: 2816

**The 10 injection points with channel dimensions:**

| Index | Source Block | Channel Dim | Spatial Resolution (at 1024x1024, latent 128x128) |
|-------|------------|-------------|---------------------------------------------------|
| 0 | Input conv | 320 | 128x128 |
| 1 | Down block 0, ResBlock 0 | 320 | 128x128 |
| 2 | Down block 0, ResBlock 1 | 320 | 128x128 |
| 3 | Down block 0, Downsample | 320 | 64x64 |
| 4 | Down block 1, ResBlock 0 | 640 | 64x64 |
| 5 | Down block 1, ResBlock 1 | 640 | 64x64 |
| 6 | Down block 1, Downsample | 640 | 32x32 |
| 7 | Down block 2, ResBlock 0 | 1280 | 32x32 |
| 8 | Down block 2, ResBlock 1 | 1280 | 32x32 |
| 9 | Mid block (separate) | 1280 | 32x32 |

Key SDXL-specific differences:
- `addition_embed_type = "text_time"` means the ControlNet must also receive the SDXL additional conditioning (original_size, crop_coords, target_size) concatenated with the timestep embedding
- `projection_class_embeddings_input_dim = 2816` accounts for the text pooled embedding (1280 from CLIP-G) + time/size embeddings (6 * 256 = 1536), total 2816
- No fourth down-block group, so 3 fewer injection points than SD1.5
- Much larger transformer attention at the deepest level (10 transformer layers per block vs 1)

Sources: [diffusers controlnet-canny-sdxl-1.0 config](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0), [HuggingFace SDXL ControlNet docs](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl)

### Flux ControlNet: Transformer Block Injection

Flux uses a DiT (Diffusion Transformer) architecture instead of a UNet, so ControlNet for Flux operates fundamentally differently. Instead of injecting into skip connections between encoder and decoder, it injects residuals directly into the transformer blocks.

**FluxControlNetModel default config:**
- `patch_size`: 1
- `in_channels`: 64
- `inner_dim`: 3072 (24 heads x 128 dim per head)
- `num_attention_heads`: 24
- `attention_head_dim`: 128
- `joint_attention_dim`: 4096 (T5-XXL text embedding dim)
- `pooled_projection_dim`: 768
- `num_layers` (joint/double-stream blocks): 5 (shallow) to 19 (full)
- `num_single_layers` (single-stream blocks): 0 (shallow) to 38 (full)

**Key architectural differences from UNet ControlNet:**

1. **Zero-initialized linear layers** instead of zero convolutions: `controlnet_blocks` and `controlnet_single_blocks` are `nn.Linear(inner_dim, inner_dim)` wrapped in `zero_module()`, not Conv2d
2. **Control condition injection**: The control image is processed through a `ControlNetConditioningEmbedding` (if present) or directly embedded, then added to `hidden_states` via `controlnet_x_embedder`
3. **Two sets of control outputs**:
   - `controlnet_block_samples`: from the joint (double-stream) transformer blocks, each passed through a zero-initialized linear projection
   - `controlnet_single_block_samples`: from the single-stream transformer blocks, each passed through a zero-initialized linear projection
4. **Shallow ControlNet**: Most Flux ControlNets use fewer layers than the full Flux transformer (e.g., 5 joint blocks instead of 19). The `align_res_stack_to_original_blocks` method distributes outputs across all DiT blocks by repeating them

**Forward pass flow:**
```
1. Embed input: hidden_states = x_embedder(hidden_states)
2. Process control condition through hint block (if available)
3. Fuse control: hidden_states += controlnet_x_embedder(controlnet_cond)
4. Process through joint transformer_blocks -> collect block_samples
5. Apply zero-initialized linear projection to each sample
6. Process through single_transformer_blocks -> collect single_block_samples
7. Apply zero-initialized linear projection to each sample
8. Scale all outputs by conditioning_scale
9. Return (controlnet_block_samples, controlnet_single_block_samples)
```

**Available Flux ControlNet models:**

| Type | Developer | Notes |
|------|-----------|-------|
| Canny | InstantX | High quality |
| Depth | Shakker-Labs | High quality |
| Union | InstantX | Multi-mode (see below) |
| Canny | XLabs-AI | Alternative implementation |
| Depth | XLabs-AI | Alternative implementation |
| HED | XLabs-AI | Edge detection |

**Flux ControlNet Union control_mode mapping (InstantX):**

| Mode | Control Type | Quality |
|------|-------------|---------|
| 0 | Canny | High |
| 1 | Tile | High |
| 2 | Depth | High |
| 3 | Blur | High |
| 4 | Pose (OpenPose) | High |
| 5 | Gray | Low (still training) |
| 6 | Low Quality (upscale) | High |

Sources: [diffusers FluxControlNetModel](https://huggingface.co/docs/diffusers/en/api/models/controlnet_flux), [FluxControlNet pipeline docs](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_flux), [InstantX FLUX.1-dev-Controlnet-Union](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union), [XLabs-AI flux-controlnet-collections](https://huggingface.co/XLabs-AI/flux-controlnet-collections)

### Multi-ControlNet: Combining Multiple Controls

Multiple ControlNets can be applied simultaneously using `MultiControlNetModel` (for UNet-based) or `FluxMultiControlNetModel` (for Flux). The combination mechanism is straightforward element-wise addition:

```python
for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
    down_samples, mid_sample = controlnet(
        sample, timestep, encoder_hidden_states, image, conditioning_scale=scale, ...
    )

    if i == 0:
        down_block_res_samples = down_samples
        mid_block_res_sample = mid_sample
    else:
        down_block_res_samples = [
            samples_prev + samples_curr
            for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
        ]
        mid_block_res_sample += mid_sample
```

Key points:
- Each ControlNet's outputs are **already scaled** by its individual `conditioning_scale` before being summed
- Residuals are **summed, not averaged** — so using two ControlNets at scale 1.0 each doubles the total control signal
- **Best practice**: When combining multiple ControlNets, use lower `controlnet_conditioning_scale` values (e.g., 0.5 each instead of 1.0) and ensure control masks don't overlap spatially
- **Temporal control**: `control_guidance_start` and `control_guidance_end` (floats 0.0-1.0) control what fraction of denoising steps each ControlNet is active for

**Guess mode** (classifier-free ControlNet guidance): When enabled, uses logarithmic scaling across depth levels: `scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1)` — meaning deeper (lower resolution) features get stronger control (scale approaching 1.0) while shallow (higher resolution) features get weaker control (scale approaching 0.1).

Sources: [diffusers MultiControlNetModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/multicontrolnet.py), [HuggingFace ControlNet guide](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet), [DeepWiki diffusers ControlNet](https://deepwiki.com/huggingface/diffusers/3.5-controlnet-and-conditional-generation)

### ControlNet Union: Single Model, Multiple Control Types

ControlNet Union models (xinsir for SDXL, InstantX for Flux) combine multiple control types into a single network, eliminating the need to load separate models for each control type.

**SDXL ControlNet Union (xinsir/controlnet-union-sdxl-1.0):**

Control mode mapping:
| Mode | Control Type |
|------|-------------|
| 0 | OpenPose |
| 1 | Depth |
| 2 | Thick line (Scribble/HED/SoftEdge/TEED-512) |
| 3 | Thin line (Canny/MLSD/LineArt/AnimeLineArt/TEED-1280) |
| 4 | Normal map |
| 5 | Segment map |

ProMax version adds modes 6-10: Tile Deblur, Tile Variation, Tile Super Resolution, Image Inpainting, Image Outpainting.

Key advantage: Reduces VRAM from ~7.5 GB (three separate ControlNets) to ~2.5 GB (one union model). Multi-condition fusion is learned during training, so no hyperparameter tuning is needed for combining conditions.

Sources: [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), [ControlNetPlus GitHub](https://github.com/xinsir6/ControlNetPlus)

### ControlNet Types and Input Preprocessing

All ControlNet variants share the same architecture — they differ only in the preprocessing applied to create the control image (3-channel RGB, same resolution as the generation target). The `controlnet_aux` library provides preprocessors.

| Type | Preprocessor | Input Format | Notes |
|------|-------------|--------------|-------|
| **Canny** | Canny edge detector | Grayscale edges (white on black), 3ch | Params: low_threshold (100), high_threshold (200) |
| **Depth** | MiDaS / DPT / Zoe | Grayscale depth map, 3ch | Closer = lighter, farther = darker |
| **OpenPose** | OpenPose / DWPose | RGB skeleton overlay on black background | Keypoints as colored lines/dots; JSON array format available |
| **Scribble** | HED / PiDiNet / user drawn | Black lines on white background | Inverted from Canny; user can hand-draw |
| **SoftEdge** | HED / PiDiNet / TEED | Soft grayscale edges | More natural edge transitions than Canny |
| **LineArt** | Line art extractor | Black lines on white background | Clean outlines; anime variant available |
| **MLSD** | M-LSD detector | Straight line segments on black | Architectural/structural lines only |
| **Normal** | BAE normal estimator | RGB normal map | R=X, G=Y, B=Z surface normals |
| **Segmentation** | OneFormer / ADE20K | Color-coded semantic regions | Each class has a fixed color |
| **Tile** | Downsample + upsample (or direct) | Full-color image (blurred/downscaled) | Used for upscaling/detail enhancement |
| **Inpaint** | Mask + image composition | Image with masked region | Concatenated: image (3ch) + mask (1ch) or 3ch masked image |
| **Shuffle** | Random pixel/patch shuffle | Shuffled color regions | Transfers color/style without structure |
| **IP2P** | None (instruction-based) | Original image | Instruct Pix2Pix-style editing |

All control images must be:
- RGB format (3 channels), except inpaint which may use 4 channels
- Same resolution as the target generation (or resized to match)
- Values normalized to [0, 1] or [-1, 1] depending on implementation
- Channel order: RGB (configurable via `controlnet_conditioning_channel_order`)

Sources: [ControlNet Complete Guide](https://stable-diffusion-art.com/controlnet/), [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux), [ControlNet v1.1 models](https://comfyui-wiki.com/en/resource/controlnet-models/controlnet-v1-1-sd15-sd2)

## Key Numbers/Constants

| Constant | Value | Context |
|----------|-------|---------|
| SD1.5 injection points | 13 | 12 encoder skip connections + 1 mid block |
| SDXL injection points | 10 | 9 encoder skip connections + 1 mid block |
| SD1.5 block_out_channels | [320, 640, 1280, 1280] | 4 resolution levels |
| SDXL block_out_channels | [320, 640, 1280] | 3 resolution levels |
| SD1.5 layers_per_block | 2 | ResBlocks per resolution level |
| SDXL layers_per_block | 2 | ResBlocks per resolution level |
| SD1.5 cross_attention_dim | 768 | CLIP-L embedding dim |
| SDXL cross_attention_dim | 2048 | CLIP-L + CLIP-G concatenated |
| Flux inner_dim | 3072 | 24 heads x 128 dim |
| Flux joint_attention_dim | 4096 | T5-XXL embedding dim |
| Flux default joint blocks | 5 (shallow) / 19 (full) | Double-stream transformer blocks |
| Flux default single blocks | 0 (shallow) / 38 (full) | Single-stream transformer blocks |
| conditioning_embedding_out_channels | [16, 32, 96, 256] | Input hint block channel progression |
| Default conditioning_scale | 1.0 | Applied to all control outputs |
| control_scales (original) | [1.0] * 13 | Per-injection-point scaling |
| Guess mode scale range | logspace(-1, 0) | 0.1 to 1.0 logarithmic |
| SD1.5 ControlNet params | ~361M | Roughly matches encoder size |
| SDXL UNet params | ~2.6B | Much larger base model |
| Flux base model params | ~12B | Flux.1-dev |
| hint_channels | 3 | RGB control image input |
| Downscale factor (hint block) | 8x | Matches VAE latent compression |

## Data Layouts/Formats

### ControlNet Input Tensor
```
control_image: [batch, 3, height, width]  # RGB, float32, values in [0, 1]
```

### SD1.5 / SDXL ControlNet Output
```
down_block_res_samples: Tuple of 12 (SD1.5) or 9 (SDXL) tensors
  Each: [batch, channels, h, w] where channels and spatial dims vary per block
mid_block_res_sample: [batch, 1280, h_mid, w_mid]
  SD1.5: [batch, 1280, 8, 8] at 512x512
  SDXL:  [batch, 1280, 32, 32] at 1024x1024
```

### Flux ControlNet Output
```
controlnet_block_samples: Tuple of N tensors (one per joint block)
  Each: [batch, seq_len, inner_dim]  where inner_dim=3072
controlnet_single_block_samples: Tuple of M tensors (one per single block)
  Each: [batch, seq_len, inner_dim]  where inner_dim=3072
```

### ControlNet Safetensors Weight Keys (SD1.5 diffusers format)
```
controlnet_cond_embedding.conv_in.weight        # [16, 3, 3, 3]
controlnet_cond_embedding.conv_in.bias           # [16]
controlnet_cond_embedding.blocks.0.weight        # [16, 16, 3, 3]
controlnet_cond_embedding.blocks.1.weight        # [32, 16, 3, 3] (stride=2)
controlnet_cond_embedding.blocks.2.weight        # [32, 32, 3, 3]
controlnet_cond_embedding.blocks.3.weight        # [96, 32, 3, 3] (stride=2)
controlnet_cond_embedding.blocks.4.weight        # [96, 96, 3, 3]
controlnet_cond_embedding.blocks.5.weight        # [256, 96, 3, 3] (stride=2)
controlnet_cond_embedding.conv_out.weight        # [320, 256, 3, 3]
controlnet_cond_embedding.conv_out.bias          # [320]

controlnet_down_blocks.{0..11}.weight            # [ch, ch, 1, 1] zero convs
controlnet_down_blocks.{0..11}.bias              # [ch]
controlnet_mid_block.weight                      # [1280, 1280, 1, 1]
controlnet_mid_block.bias                        # [1280]

down_blocks.{0..3}.resnets.{0..1}.*              # Copied from UNet encoder
down_blocks.{0..3}.attentions.{0..1}.*           # Copied from UNet encoder (where present)
down_blocks.{0..2}.downsamplers.0.*              # Copied from UNet encoder
mid_block.*                                       # Copied from UNet mid block
```

## Algorithm Steps

### ControlNet Forward Pass (SD1.5/SDXL)

```
Input: noisy_latent [B,4,H,W], timestep, text_embeddings, control_image [B,3,H*8,W*8]

1. Process control image through input hint block:
   guided_hint = ControlNetConditioningEmbedding(control_image)
   # Result: [B, model_channels, H, W]

2. Compute timestep embedding:
   t_emb = timestep_embedding(timestep)  # sinusoidal -> MLP

3. Initial convolution on noisy latent:
   sample = conv_in(noisy_latent)  # [B, 320, H, W]

4. Add control hint to sample:
   sample = sample + guided_hint

5. Apply zero conv to initial sample:
   outputs = [zero_convs[0](sample)]

6. For each encoder block (12 blocks total for SD1.5):
   sample = encoder_block(sample, t_emb, text_embeddings)
   outputs.append(zero_convs[i](sample))

7. Process through mid block:
   sample = mid_block(sample, t_emb, text_embeddings)
   mid_output = controlnet_mid_block(sample)  # zero conv

8. Apply conditioning_scale:
   outputs = [out * conditioning_scale for out in outputs]
   mid_output = mid_output * conditioning_scale

9. Return (down_block_res_samples=outputs, mid_block_res_sample=mid_output)
```

### UNet Integration (How residuals are injected)

```
Input: noisy_latent, timestep, text_embeddings,
       down_block_additional_residuals (from ControlNet),
       mid_block_additional_residual (from ControlNet)

1. Encoder pass (frozen): produces skip_connections list
   For each down block:
     skip_connections.append(block_output)

2. Mid block pass:
   h = mid_block(h)
   h = h + mid_block_additional_residual  # ADD ControlNet mid residual

3. Decoder pass:
   For each up block:
     skip = skip_connections.pop()
     skip = skip + down_block_additional_residuals.pop()  # ADD ControlNet residual
     h = torch.cat([h, skip], dim=1)  # concatenate with decoder features
     h = up_block(h)
```

### Multi-ControlNet Combination

```
For each ControlNet_i with control_image_i, scale_i:
    outputs_i = ControlNet_i(sample, timestep, embeddings, control_image_i,
                             conditioning_scale=scale_i)

Combined residuals = element-wise sum of all ControlNet outputs:
    down_combined[j] = sum(outputs_i.down[j] for all i)
    mid_combined = sum(outputs_i.mid for all i)
```

### Flux ControlNet Forward Pass

```
Input: hidden_states [B, seq_len, inner_dim], controlnet_cond, timestep, text_embeddings

1. Embed hidden states: hidden_states = x_embedder(hidden_states)
2. Process control condition through hint block (if present)
3. Add control: hidden_states += controlnet_x_embedder(controlnet_cond)
4. For each joint transformer block (5 or 19):
   hidden_states, encoder_hidden_states = joint_block(hidden_states, encoder_hidden_states)
   block_sample = controlnet_blocks[i](hidden_states)  # zero-initialized linear
   block_samples.append(block_sample)
5. Concatenate streams: hidden_states = cat([encoder_hidden_states, hidden_states])
6. For each single transformer block (0 or 38):
   hidden_states = single_block(hidden_states)
   single_sample = controlnet_single_blocks[i](hidden_states)
   single_block_samples.append(single_sample)
7. Scale all: samples *= conditioning_scale
8. If shallow ControlNet, use align_res_stack_to_original_blocks to distribute
   outputs across all DiT blocks (repeating as needed)
9. Return (controlnet_block_samples, controlnet_single_block_samples)
```

## Reference Implementations

| Implementation | Language | Notes |
|---------------|----------|-------|
| [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | Python/PyTorch | Original implementation, SD1.5 |
| [lllyasviel/ControlNet-v1-1-nightly](https://github.com/lllyasviel/ControlNet-v1-1-nightly) | Python/PyTorch | Updated v1.1 with 14 models |
| [diffusers ControlNetModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/controlnet.py) | Python/PyTorch | HuggingFace SD1.5/SD2 implementation |
| [diffusers FluxControlNetModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/controlnet_flux.py) | Python/PyTorch | Flux ControlNet |
| [diffusers MultiControlNetModel](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnets/multicontrolnet.py) | Python/PyTorch | Multi-ControlNet combiner |
| [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) | Python | A1111 WebUI extension |
| [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) | Python | Preprocessing library for all control types |
| [xinsir6/ControlNetPlus](https://github.com/xinsir6/ControlNetPlus) | Python/PyTorch | Union model implementation |

## Differences Between Implementations

### Original (lllyasviel) vs diffusers

| Aspect | Original | diffusers |
|--------|----------|-----------|
| Integration | `ControlledUnetModel` wraps UNet, ControlNet runs inside | ControlNet is a separate model, outputs passed to UNet |
| Skip connection injection | `h = cat([h, hs.pop() + control.pop()])` inline | Residuals passed as `additional_residuals` parameter |
| Scaling | `self.control_scales = [1.0] * 13` on ControlNet | `conditioning_scale` applied externally |
| Multi-ControlNet | Not natively supported | `MultiControlNetModel` with element-wise sum |
| Weight format | .pth checkpoint | safetensors with diffusers naming |
| SDXL support | Not in original repo | Full support with addition embeddings |

### SD1.5 vs SDXL ControlNet

| Aspect | SD1.5 | SDXL |
|--------|-------|------|
| Injection points | 13 (12 skip + 1 mid) | 10 (9 skip + 1 mid) |
| Block out channels | [320, 640, 1280, 1280] | [320, 640, 1280] |
| Down block types | 3x CrossAttn + 1x Plain | 1x Plain + 2x CrossAttn |
| Cross attention dim | 768 | 2048 |
| Transformer layers/block | [1, 1, 1, 0] | [1, 2, 10] |
| Additional embeddings | None | text_time (size/crop conditioning) |
| Base resolution | 512x512 | 1024x1024 |
| ControlNet params | ~361M | ~1.2B (estimate) |

### UNet ControlNet vs Flux ControlNet

| Aspect | UNet (SD1.5/SDXL) | Flux |
|--------|-------------------|------|
| Base architecture | UNet (encoder-decoder with skips) | DiT (transformer, no skip connections) |
| Injection mechanism | Residuals added to skip connections | Residuals added to transformer block outputs |
| Zero modules | Conv2d 1x1 (zero convolutions) | nn.Linear (zero-initialized linear layers) |
| Output structure | down_block_res_samples + mid_block_res | block_samples + single_block_samples |
| Spatial dimensions | Varies per block (progressive downsampling) | Flat sequence (all same seq_len) |
| Shallow variant | Not typical | Common (5 blocks vs 19 full) |
| Control fusion point | After each encoder block | After each transformer block |
| Typical param overhead | ~100% of encoder | ~0.1% to ~27.5% of full model |

## Open Questions

- [ ] Exact parameter count for SDXL ControlNet (estimated ~1.2B but not confirmed from source)
- [ ] Whether ControlNet-XS (lightweight variant) is worth supporting alongside standard ControlNet
- [ ] SD3 ControlNet architecture specifics (MMDiT-based, similar to Flux but different block structure)

## Implementation Notes

### For HartsyInference C# Implementation

1. **ControlNet is a separate model** that runs in parallel with the main UNet/transformer. Its forward pass receives the same noisy latent, timestep, and text embeddings as the main model, plus the control image.

2. **Weight loading**: ControlNet safetensors contain both the copied encoder weights and the zero convolution weights. The encoder weight keys mirror the UNet encoder keys. The `controlnet_down_blocks.{N}` and `controlnet_mid_block` keys are the zero convolutions.

3. **Memory considerations**:
   - SD1.5 ControlNet adds ~361M params (~1.4 GB in FP32, ~700 MB in FP16)
   - SDXL ControlNet adds ~1.2B params (~4.8 GB in FP32, ~2.4 GB in FP16)
   - Multi-ControlNet multiplies this linearly
   - Union models avoid this by handling all control types in one network

4. **The input hint block is tiny** (a few hundred KB) and always runs on the control image before the main forward pass. It converts [B,3,H*8,W*8] to [B,model_channels,H,W].

5. **Critical implementation detail**: The control residuals are added to the UNet's skip connections, NOT to the decoder block inputs. In the original implementation: `h = cat([h, hs.pop() + control.pop()], dim=1)`. The control is added to the skip connection tensor before concatenation with the decoder tensor.

6. **For Flux**: The shallow ControlNet pattern (5 joint blocks instead of 19) means the `align_res_stack_to_original_blocks` method must be implemented to distribute fewer control outputs across more transformer blocks. This repeats outputs to fill the gaps.

7. **Preprocessing is external**: The C# implementation should accept pre-processed control images. Implementing Canny edge detection, MiDaS depth estimation, or OpenPose in C# is a separate concern from the ControlNet inference itself. Users can preprocess with Python tools or we can provide basic preprocessors (Canny is straightforward, depth/pose require separate neural networks).

8. **The conditioning_scale should be user-configurable** and default to 1.0. For multi-ControlNet, accept a list of scales. The control_guidance_start/end parameters can skip ControlNet evaluation at certain timesteps to save compute.

9. **SDXL ControlNet requires additional conditioning**: The `addition_embed_type = "text_time"` means the ControlNet must receive the same additional embeddings (original_size, crop_coords, target_size) as the SDXL UNet. Forgetting this will produce garbage outputs.

10. **Union model support**: If implementing ControlNet Union, add a `control_mode` integer parameter to the forward pass. The model uses this to select internal behavior for different control types.
