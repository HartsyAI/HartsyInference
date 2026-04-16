# Vision & Goals

> Back to [Core Design](CORE_DESIGN.md)

---

## The Core Motivation

Every existing .NET AI inference solution for image and audio is either:

- **A wrapper around a Python service** — SwarmUI delegates to a Python backend process
- **A wrapper around a C++ binary** — StableDiffusion.NET wraps sd.cpp
- **ONNX Runtime** — requires model conversion and loses flexibility
- **NVIDIA-only** — existing CUDA-based solutions ignore AMD and Intel GPU users entirely

There is no pure C# engine that can load a `.safetensors` diffusion model, run it on **any GPU** (NVIDIA via CUDA, AMD/Intel via Vulkan), and stream image generation results through an ASP.NET endpoint — without any external process.

**SharpInference fills that gap.**

dotLLM ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) proved that production AI inference can be done entirely in managed C# with PTX kernels achieving ~98–100% of native CUDA performance. SharpInference takes that proven approach and extends it in two directions: (1) **non-LLM modalities** (diffusion, audio, vision — requiring Conv2D, GroupNorm, spatial attention, upsampling, FFT kernels that don't exist in dotLLM) and (2) **cross-vendor GPU support via Vulkan** — applying the same P/Invoke-to-driver-API philosophy that dotLLM pioneered for CUDA.

---

## The SwarmUI Backend Angle

SwarmUI is a .NET application that delegates inference to an external Python backend process. This means every deployment requires Python, a full Python inference stack, and all its dependencies. The arrangement is fragile, slow to start, and impossible to embed cleanly.

### With SharpInference as a SwarmUI Backend Extension

| Before (Python Backend) | After (SharpInference) |
|---|---|
| Requires Python + pip + torch + diffusers | A NuGet package reference is the only "install" step |
| Separate process, HTTP round-trips | Inference runs inside the same .NET process |
| 10–30 second startup for Python backend | Starts in milliseconds |
| No C# debuggability | Full C# debuggability and observability |
| Complex Docker with Python + .NET | Single base image Docker deployment |

The extension registers SharpInference's pipeline as a SwarmUI backend, exposes the API surface SwarmUI expects, and handles all standard model formats used in the ecosystem today.

---

## Target Users

| User | How They Use SharpInference |
|---|---|
| **SwarmUI deployers** | Drop-in backend replacement, no Python |
| **ASP.NET developers** | Add AI image/audio generation to any web app via NuGet |
| **Desktop app developers** | Embed inference directly in WPF/MAUI/Avalonia apps |
| **dotLLM users** | Complete the AI stack — LLM + vision + audio in one .NET process |
| **HartsyWeb / WizardPortrait** | Native inference for image generation, STT, TTS, face detection |

---

## What SharpInference Is NOT

- **Not an LLM engine** — dotLLM handles that. SharpInference covers everything else.
- **Not an ONNX wrapper** — models load from their native formats (safetensors, GGUF) directly.
- **Not a training framework** — inference only. No backward pass, no gradient computation.
- **Not a thin binding layer** — all compute kernels are written in C# (CPU SIMD), PTX (CUDA), or SPIR-V (Vulkan). No `libonnxruntime`, no `libtorch`, no managed GPU wrappers.
- **Not NVIDIA-only** — Vulkan backend provides AMD and Intel GPU support using the same pure-C# P/Invoke philosophy dotLLM proved for CUDA.

---

## How Image Inference Differs from LLM Inference

dotLLM handles LLM inference (MatMul-heavy, autoregressive token generation). Image/audio/vision inference requires a fundamentally different kernel set:

| dotLLM (LLM) Kernels | SharpInference (Image) Kernels | Why Different |
|---|---|---|
| RMSNorm | GroupNorm | Diffusion uses group normalization (32 groups), not RMS normalization |
| RoPE (1D positional) | RoPE 2D (spatial + text) | Image tokens have 2D spatial positions, not 1D sequence positions |
| Causal attention mask | Bidirectional + cross-attention | Diffusion attention is bidirectional (no causal mask) with text cross-attention |
| SwiGLU FFN | SiLU activation in ResNet blocks | UNet uses SiLU, not SwiGLU |
| Embedding lookup | Conv2D (3×3, 1×1, depthwise) | Image features processed by convolutions, not embedding tables |
| KV-cache for decode | No KV-cache (full recompute) | Each denoising step is a full forward pass — no autoregressive caching |
| Single-token GEMV | Batch GEMM (full spatial maps) | Image inference always processes full spatial feature maps, not single tokens |
| — | Upsample 2D (nearest, bilinear) | Spatial upsampling between resolution stages in UNet/VAE |
| — | FFT / STFT / Mel filterbank | Audio preprocessing for Whisper — no equivalent in LLM inference |
| — | Timestep conditioning | Sinusoidal embedding + FiLM-style modulation — unique to diffusion |

Despite these differences, the **infrastructure is identical**: same P/Invoke pattern, same PTX/SPIR-V management, same memory model, same tensor types, same SIMD dispatch, same error handling. dotLLM solved the hard infrastructure problems — SharpInference focuses on the domain-specific kernels and model architectures.
