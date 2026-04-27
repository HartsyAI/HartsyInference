# Research Requirements

Every area below needs a `docs/Research/` document **before** implementation begins. See `docs/Agents/RESEARCH.md` for research output format.

## Model Formats & Loading

| Document | Needed Before |
|---|---|
| [SAFETENSORS_FORMAT.md](../Research/SAFETENSORS_FORMAT.md) | ModelHandler |
| [GGUF_FORMAT.md](../Research/GGUF_FORMAT.md) | ModelHandler |
| [QUANTIZATION_DIFFUSION.md](../Research/QUANTIZATION_DIFFUSION.md) | ModelHandler, Diffusion |

## GPU / Compute

| Document | Needed Before |
|---|---|
| [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) | Cuda |
| [PTX_KERNELS.md](../Research/PTX_KERNELS.md) | Cuda / Ptx |
| [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) | Cuda |
| [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md) | Vulkan ✅ |
| [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md) | Vulkan / Spirv ✅ |
| [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md) | Vulkan ✅ |
| [SIMD_INTRINSICS_DOTNET.md](../Research/SIMD_INTRINSICS_DOTNET.md) | Cpu |

## CPU Kernel Algorithms

| Document | Needed Before |
|---|---|
| [IM2COL_CPU.md](../Research/IM2COL_CPU.md) | Cpu |
| [GROUPNORM_MATH.md](../Research/GROUPNORM_MATH.md) | Cpu |
| [FLASH_ATTENTION.md](../Research/FLASH_ATTENTION.md) | Cpu, Cuda |

## Diffusion Architectures

| Document | Needed Before |
|---|---|
| [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) | Diffusion (UNet) |
| [SDXL_ARCHITECTURE.md](../Research/SDXL_ARCHITECTURE.md) | Diffusion (SDXL) |
| [FLUX_ARCHITECTURE.md](../Research/FLUX_ARCHITECTURE.md) | Diffusion (Flux) |
| [SD3_ARCHITECTURE.md](../Research/SD3_ARCHITECTURE.md) | Diffusion (SD3) |
| [VAE_ARCHITECTURE.md](../Research/VAE_ARCHITECTURE.md) | Diffusion (VAE) |

## Diffusion Techniques

| Document | Needed Before |
|---|---|
| [DIFFUSION_SCHEDULERS.md](../Research/DIFFUSION_SCHEDULERS.md) | Diffusion (Schedulers) |
| [CFG_AND_GUIDANCE.md](../Research/CFG_AND_GUIDANCE.md) | Diffusion (Pipelines) |
| [LORA_FORMAT.md](../Research/LORA_FORMAT.md) | Diffusion (Adapters) |
| [CONTROLNET.md](../Research/CONTROLNET.md) | Diffusion (Adapters) |

## Text Encoders & Tokenizers

| Document | Needed Before |
|---|---|
| [CLIP_ARCHITECTURE.md](../Research/CLIP_ARCHITECTURE.md) | Vision, Diffusion |
| [CLIP_TOKENIZER.md](../Research/CLIP_TOKENIZER.md) | Tokenizers |
| [T5_ARCHITECTURE.md](../Research/T5_ARCHITECTURE.md) | Diffusion, Tokenizers |
| [T5_TOKENIZER.md](../Research/T5_TOKENIZER.md) | Tokenizers |

## Audio

| Document | Needed Before |
|---|---|
| [WHISPER_ARCHITECTURE.md](../Research/WHISPER_ARCHITECTURE.md) | Audio |
| [MEL_SPECTROGRAM.md](../Research/MEL_SPECTROGRAM.md) | Audio |
| [KOKORO_ARCHITECTURE.md](../Research/KOKORO_ARCHITECTURE.md) | Audio |
| [HIFIGAN_VOCODER.md](../Research/HIFIGAN_VOCODER.md) | Audio |

## Vision / Server / Reference

| Document | Needed Before |
|---|---|
| [YOLO_ARCHITECTURE.md](../Research/YOLO_ARCHITECTURE.md) | Vision |
| [OPENAI_IMAGE_API.md](../Research/OPENAI_IMAGE_API.md) | Server |
| [DOTLLM_ARCHITECTURE.md](../Research/DOTLLM_ARCHITECTURE.md) | All packages |
