# Research Requirements

> Back to [Core Design](CORE_DESIGN.md)

Every area below needs a `docs/Research/` document **before** implementation begins.

## Model Formats & Loading

| Document | Research Topic | Needed Before |
|---|---|---|
| [SAFETENSORS_FORMAT.md](../Research/SAFETENSORS_FORMAT.md) | Binary layout, header JSON, multi-shard, dtype mapping | ModelHandler |
| [GGUF_FORMAT.md](../Research/GGUF_FORMAT.md) | GGUF v3 header, metadata keys, Q4_K_M block layout, dequant math | ModelHandler |
| [QUANTIZATION_DIFFUSION.md](../Research/QUANTIZATION_DIFFUSION.md) | Which components tolerate Q8_0 vs FP16 | ModelHandler, Diffusion |

## GPU / Compute

| Document | Research Topic | Needed Before |
|---|---|---|
| [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) | cuInit, cuModuleLoadData, cuLaunchKernel, cuMemPool, cuBLAS HGEMM; cross-ref dotLLM's `CudaDriverApi.cs` | Cuda |
| [PTX_KERNELS.md](../Research/PTX_KERNELS.md) | PTX ISA compute_80+, shared memory tiling, warp shuffle, FP16 intrinsics, register pressure; image-specific: Conv2D, GroupNorm, spatial attention | Cuda / Ptx |
| [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) | cuDNN P/Invoke, tensor descriptors, algorithm selection, workspace; im2col+cuBLAS HGEMM alternative | Cuda |
| [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md) | Instance/device/queue creation, compute pipelines, descriptor sets, push constants, command buffers, memory allocation, staging; `[LibraryImport("vulkan-1")]` P/Invoke surface | Vulkan |
| [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md) | GLSL compute syntax, SPIR-V compilation, subgroup ops, shared memory barriers, FP16 extensions, workgroup size per vendor (NVIDIA 32, AMD 64, Intel 8–32) | Vulkan / Spirv |
| [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md) | `vkGetPhysicalDeviceMemoryProperties`, heap types, sub-allocation, device-local vs host-visible, staging buffers | Vulkan |
| [SIMD_INTRINSICS_DOTNET.md](../Research/SIMD_INTRINSICS_DOTNET.md) | `System.Runtime.Intrinsics` AVX2/AVX-512, `TensorPrimitives`, dotLLM SIMD dispatch | Cpu |

## CPU Kernel Algorithms

| Document | Research Topic | Needed Before |
|---|---|---|
| [IM2COL_CPU.md](../Research/IM2COL_CPU.md) | im2col + GEMM, cache layout, padding/stride/dilation | Cpu |
| [GROUPNORM_MATH.md](../Research/GROUPNORM_MATH.md) | GroupNorm vs BatchNorm vs LayerNorm, FP16 accumulation, fusion with Conv2D | Cpu |
| [FLASH_ATTENTION.md](../Research/FLASH_ATTENTION.md) | Tiled SDPA O(N), dotLLM's implementation, cross-attention adaptation | Cpu, Cuda |

## Diffusion Model Architectures

| Document | Research Topic | Needed Before |
|---|---|---|
| [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) | UNet blocks, channels, cross-attention, skip connections, timestep embedding | Diffusion (UNet) |
| [SDXL_ARCHITECTURE.md](../Research/SDXL_ARCHITECTURE.md) | Dual CLIP encoder, larger UNet, SDXL conditioning, refiner | Diffusion (SDXL) |
| [FLUX_ARCHITECTURE.md](../Research/FLUX_ARCHITECTURE.md) | MMDiT double/single-stream, flow-matching, RoPE, Flux LoRA format | Diffusion (Flux) |
| [SD3_ARCHITECTURE.md](../Research/SD3_ARCHITECTURE.md) | MMDiT joint attention, 3 text encoders (CLIP-L/G, T5), QK-norm | Diffusion (SD3) |
| [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) | Encoder/decoder, KL latent, scaling factors (0.18215 SD, 0.13025 SDXL), tiled decode blending | Diffusion (VAE) |

## Diffusion Techniques

| Document | Research Topic | Needed Before |
|---|---|---|
| [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) | DDPM, DDIM, Euler, DPM++ 2M/SDE, LCM, flow-matching Euler | Diffusion (Schedulers) |
| [CFG_AND_GUIDANCE.md](../Research/CFG_AND_GUIDANCE.md) | CFG math, negative prompts, guidance scale, PAG | Diffusion (Pipelines) |
| [LORA_FORMAT.md](../Research/LORA_FORMAT.md) | Rank decomposition, alpha scaling, safetensors storage, Flux vs SD naming, multi-LoRA stacking | Diffusion (Adapters) |
| [CONTROLNET.md](../Research/CONTROLNET.md) | Encoder copy + zero conv, residual injection, preprocessing per type | Diffusion (Adapters) |

## Text Encoders & Tokenizers

| Document | Research Topic | Needed Before |
|---|---|---|
| [CLIP_ARCHITECTURE.md](../Research/CLIP_ARCHITECTURE.md) | ViT patch embedding, CLS token, transformer, OpenAI vs LAION weights | Vision, Diffusion |
| [T5_ARCHITECTURE.md](../Research/T5_ARCHITECTURE.md) | T5 encoder-only, relative positional bias, ReLU-gated FFN, SentencePiece | Diffusion, Tokenizers |

## Audio

| Document | Research Topic | Needed Before |
|---|---|---|
| [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) | Conv1D extractor, encoder/decoder transformer, timestamp tokens, GGUF Whisper format | Audio |
| [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) | Whisper preprocessing: FFT 400, hop 160, 80 mel bins, log compression, normalization; must match whisper.cpp | Audio |
| [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) | StyleTTS2-based, phoneme encoder, acoustic model, HiFiGAN, voice embeddings | Audio |
| [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) | HiFiGAN generator, upsampling, residual dilated conv, I/O formats | Audio |

## Vision

| Document | Research Topic | Needed Before |
|---|---|---|
| [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) | YOLOv8/v11 backbone (C2f, SPPF), FPN, detection head, NMS, output tensor format | Vision |

## Server / API

| Document | Research Topic | Needed Before |
|---|---|---|
| [OPENAI_IMAGE_API.md](../Research/OPENAI_IMAGE_API.md) | `/v1/images/generations` schema, URL vs base64, edits/variations | Server |

## Reference Architecture

| Document | Research Topic | Needed Before |
|---|---|---|
| [DOTLLM_ARCHITECTURE.md](../Research/DOTLLM_ARCHITECTURE.md) | Tensor system, CUDA P/Invoke, PTX management, SIMD dispatch, memory management, server architecture | All packages |

## Research Document Template

```markdown
# [Topic] — Research Notes
> Status: Draft | Complete | Last Updated: YYYY-MM-DD | Needed Before: [Component]

## Summary
[1-2 paragraph overview]

## Detailed Findings
[Main content]

## Key Numbers / Constants
[Exact values code needs]

## Reference Implementations
[Links to code studied]

## Open Questions
[Unresolved items]

## Implementation Notes
[Decisions made]
```
