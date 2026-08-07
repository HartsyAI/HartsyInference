# ControlNet — Research Notes

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

ControlNet (Zhang et al., 2023) adds spatial conditioning controls to pretrained text-to-image diffusion models by creating a trainable copy of the encoder blocks and connecting them to the frozen model through "zero convolutions" — 1x1 convolutions with weights and biases initialized to zero. This ensures the ControlNet produces zero output at initialization, preventing any distortion to the base model before training begins. Different ControlNet variants (Canny, Depth, OpenPose, Scribble, etc.) share the same architecture but differ only in their input preprocessing. For SD1.5, there are 13 injection points (12 encoder skip connections + 1 mid block). For SDXL, there are 10 injection points due to the smaller encoder. For Flux, ControlNet operates on transformer blocks rather than UNet blocks, injecting residuals into both the joint (double-stream) and single-stream transformer blocks. Multi-ControlNet combines residuals from multiple ControlNets via element-wise addition with per-ControlNet scaling factors.

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
