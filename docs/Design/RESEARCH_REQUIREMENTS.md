# Research Requirements

> Back to [Core Design](CORE_DESIGN.md)

---

## Overview

Every area listed below requires a research document (`docs/Research/`) to be written **before** implementation begins. These research docs become the agent instruction context for each build phase.

---

## Model Formats & Loading

| Document | What to Research | Needed Before |
|---|---|---|
| [SAFETENSORS_FORMAT.md](../Research/SAFETENSORS_FORMAT.md) | Exact binary layout, header JSON schema, multi-shard conventions, dtype string mapping | ModelHandler |
| [GGUF_FORMAT.md](../Research/GGUF_FORMAT.md) | GGUF v3 header, all metadata key types, Q4_K_M block layout, dequant math | ModelHandler |
| [QUANTIZATION_DIFFUSION.md](../Research/QUANTIZATION_DIFFUSION.md) | Which parts of diffusion models tolerate Q8_0 vs need FP16, SDXL vs Flux quantization behavior | ModelHandler, Diffusion |

---

## GPU / Compute

| Document | What to Research | Needed Before |
|---|---|---|
| [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) | cuInit, cuModuleLoadData, cuLaunchKernel, async memory allocation, cuMemPool, cuBLAS HGEMM API signatures. Cross-reference dotLLM's `CudaDriverApi.cs` (~34 P/Invoke declarations) | Cuda |
| [PTX_KERNELS.md](../Research/PTX_KERNELS.md) | PTX ISA for compute_80+, shared memory tiling, warp shuffle reductions, FP16 intrinsics, register pressure. Image-specific: Conv2D tiling, GroupNorm shared memory reduction, spatial attention tiling | Cuda / Ptx |
| [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) | cuDNN P/Invoke bindings, tensor descriptor format, algorithm selection, workspace allocation. Also: im2col+cuBLAS HGEMM as alternative to cuDNN | Cuda |
| [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md) | Vulkan instance/device/queue creation, compute pipeline creation, descriptor sets, push constants, command buffer recording, memory allocation and types, staging buffer transfers. P/Invoke surface design for `[LibraryImport("vulkan-1")]`. Compare to dotLLM's CUDA P/Invoke pattern | Vulkan |
| [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md) | GLSL compute shader syntax, SPIR-V compilation via glslangValidator, subgroup operations (arithmetic, shuffle), shared memory barriers, FP16 extensions (`GL_EXT_shader_explicit_arithmetic_types_float16`), workgroup size selection per vendor (NVIDIA 32, AMD 64, Intel 8–32) | Vulkan / Spirv |
| [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md) | `vkGetPhysicalDeviceMemoryProperties`, memory heap types, sub-allocation strategies (avoid per-tensor `vkAllocateMemory`), device-local vs host-visible memory, staging buffer patterns for host↔device transfers | Vulkan |
| [SIMD_INTRINSICS_DOTNET.md](../Research/SIMD_INTRINSICS_DOTNET.md) | `System.Runtime.Intrinsics` AVX2/AVX-512 patterns, `TensorPrimitives` API, dotLLM's SIMD kernel structure and dispatch pattern | Cpu |

---

## CPU Kernel Algorithms

| Document | What to Research | Needed Before |
|---|---|---|
| [IM2COL_CPU.md](../Research/IM2COL_CPU.md) | im2col algorithm, GEMM after im2col, memory layout for cache efficiency, padding/stride/dilation handling | Cpu |
| [GROUPNORM_MATH.md](../Research/GROUPNORM_MATH.md) | GroupNorm vs BatchNorm vs LayerNorm math, FP16 accumulation stability, fusion with Conv2D | Cpu |
| [FLASH_ATTENTION.md](../Research/FLASH_ATTENTION.md) | Tiled SDPA O(N) memory algorithm, dotLLM's implementation, adapting for cross-attention | Cpu, Cuda |

---

## Diffusion Model Architectures

| Document | What to Research | Needed Before |
|---|---|---|
| [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) | Exact UNet block types, channel dimensions, cross-attention placement, skip connections, timestep embedding | Diffusion (UNet) |
| [SDXL_ARCHITECTURE.md](../Research/SDXL_ARCHITECTURE.md) | Dual CLIP encoder, larger UNet, SDXL-specific conditioning (aesthetic score, crop coords), refiner | Diffusion (SDXL) |
| [FLUX_ARCHITECTURE.md](../Research/FLUX_ARCHITECTURE.md) | Double-stream and single-stream MMDiT blocks, flow-matching vs DDPM, RoPE for joint image+text, Flux LoRA format | Diffusion (Flux) |
| [SD3_ARCHITECTURE.md](../Research/SD3_ARCHITECTURE.md) | MMDiT joint attention, three text encoders (CLIP-L, CLIP-G, T5), SD3 QK-norm | Diffusion (SD3) |
| [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) | VAE encoder/decoder structure, KL latent, scaling factors (0.18215 SD, 0.13025 SDXL), tiled decode blending | Diffusion (VAE) |

---

## Diffusion Techniques

| Document | What to Research | Needed Before |
|---|---|---|
| [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) | DDPM derivation, DDIM, Euler discrete, DPM++ 2M, DPM++ SDE, LCM, flow-matching Euler | Diffusion (Schedulers) |
| [CFG_AND_GUIDANCE.md](../Research/CFG_AND_GUIDANCE.md) | Classifier-free guidance math, negative prompt encoding, guidance scale effect, PAG | Diffusion (Pipelines) |
| [LORA_FORMAT.md](../Research/LORA_FORMAT.md) | LoRA rank decomposition, alpha scaling, safetensors storage, Flux vs SD naming, multi-LoRA stacking | Diffusion (Adapters) |
| [CONTROLNET.md](../Research/CONTROLNET.md) | ControlNet architecture (encoder copy + zero convolutions), residual injection points, preprocessing per type | Diffusion (Adapters) |

---

## Text Encoders & Tokenizers

| Document | What to Research | Needed Before |
|---|---|---|
| [CLIP_ARCHITECTURE.md](../Research/CLIP_ARCHITECTURE.md) | ViT patch embedding, CLS token, transformer encoder, OpenAI vs LAION weights differences | Vision, Diffusion |
| [T5_ARCHITECTURE.md](../Research/T5_ARCHITECTURE.md) | T5 encoder-only (no decoder), relative positional bias, FFN with ReLU gating, SentencePiece tokenizer | Diffusion, Tokenizers |

---

## Audio

| Document | What to Research | Needed Before |
|---|---|---|
| [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) | Conv1D feature extractor, encoder transformer, decoder autoregressive loop, timestamp tokens, GGUF Whisper format | Audio |
| [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) | Exact Whisper preprocessing (FFT 400, hop 160, 80 mel bins, log compression, normalization), must match whisper.cpp | Audio |
| [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) | Kokoro model structure (StyleTTS2-based), phoneme encoder, acoustic model, HiFiGAN, voice embeddings | Audio |
| [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) | HiFiGAN generator architecture, upsampling blocks, residual dilated conv, input/output formats | Audio |

---

## Vision

| Document | What to Research | Needed Before |
|---|---|---|
| [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) | YOLOv8/v11 backbone (C2f, SPPF), FPN neck, detection head, NMS algorithm, output tensor format | Vision |

---

## Server / API

| Document | What to Research | Needed Before |
|---|---|---|
| [OPENAI_IMAGE_API.md](../Research/OPENAI_IMAGE_API.md) | OpenAI `/v1/images/generations` request/response schema, URL vs base64 modes, edits/variations | Server |

---

## Reference Architecture

| Document | What to Research | Needed Before |
|---|---|---|
| [DOTLLM_ARCHITECTURE.md](../Research/DOTLLM_ARCHITECTURE.md) | dotLLM project structure, tensor system (ITensor + TensorRef), CUDA P/Invoke patterns, PTX management, SIMD dispatch, memory management, server architecture — patterns SharpInference should follow | All packages (reference architecture) |

---

## Research Document Template

Each research document should follow this structure:

```markdown
# [Topic] — Research Notes

> Status: Draft | Complete
> Last Updated: YYYY-MM-DD
> Needed Before: [Package/Component]

## Summary
[1-2 paragraph overview]

## Detailed Findings
[Main research content]

## Key Numbers / Constants
[Any magic numbers, dimensions, sizes that code needs]

## Reference Implementations
[Links to Python/C++ code studied]

## Open Questions
[Anything unresolved]

## Implementation Notes
[Decisions made based on this research]
```
