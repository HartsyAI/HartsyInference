# Vision & Goals

> Back to [Core Design](CORE_DESIGN.md)

## Core Motivation

Every existing .NET AI inference solution is either:
- **Python wrapper** (SwarmUI delegates to Python backend)
- **C++ wrapper** (StableDiffusion.NET wraps sd.cpp)
- **ONNX Runtime** (requires model conversion)
- **NVIDIA-only** (ignores AMD/Intel)

No pure C# engine loads `.safetensors` diffusion models, runs on **any GPU** (NVIDIA via CUDA, AMD/Intel via Vulkan), and streams results through ASP.NET — without external processes.

**SharpInference fills that gap.**

dotLLM proved production AI inference in managed C# with PTX kernels at ~98–100% native CUDA performance. SharpInference extends this in two directions: (1) **non-LLM modalities** (diffusion, audio, vision — requiring Conv2D, GroupNorm, FFT, etc.) and (2) **cross-vendor GPU support via Vulkan** — same P/Invoke-to-driver-API philosophy.

## SwarmUI Backend Angle

SwarmUI is .NET but delegates inference to an external Python process — fragile, slow, hard to embed.

| Before (Python) | After (SharpInference) |
|---|---|
| Python + pip + torch + diffusers | NuGet reference only |
| Separate process, HTTP round-trips | In-process inference |
| 10–30s startup | Milliseconds |
| No C# debuggability | Full C# debuggability |
| Complex Docker (Python + .NET) | Single base image |

## Target Users

| User | Use Case |
|---|---|
| SwarmUI deployers | Drop-in backend, no Python |
| ASP.NET developers | AI image/audio via NuGet |
| Desktop developers | WPF/MAUI/Avalonia embedding |
| dotLLM users | Complete AI stack (LLM + vision + audio) |
| HartsyWeb / WizardPortrait | Native image gen, STT, TTS, face detection |

## What It Is NOT

- **Not an LLM engine** — dotLLM handles that
- **Not an ONNX wrapper** — loads native formats directly
- **Not a training framework** — inference only
- **Not a thin binding** — all kernels in C# (SIMD), PTX, or SPIR-V
- **Not NVIDIA-only** — Vulkan supports AMD/Intel

## Image vs LLM Inference

dotLLM handles MatMul-heavy autoregressive LLMs. Image/audio needs different kernels:

| LLM (dotLLM) | Image (SharpInference) | Why Different |
|---|---|---|
| RMSNorm | GroupNorm | Diffusion uses 32-group normalization |
| 1D RoPE | 2D RoPE | Image tokens have 2D spatial positions |
| Causal attention | Bidirectional + cross-attention | No autoregressive masking; text conditioning |
| SwiGLU | SiLU | UNet ResNet blocks use SiLU |
| Embedding lookup | Conv2D (3×3, 1×1, depthwise) | Spatial features, not token embeddings |
| KV-cache decode | No KV-cache | Each denoising step is full forward pass |
| Single-token GEMV | Batch GEMM (spatial maps) | Full feature maps, not single tokens |
| — | Upsample 2D | Spatial upsampling in UNet/VAE |
| — | FFT/STFT/Mel | Audio preprocessing for Whisper |
| — | Timestep conditioning | Sinusoidal embedding + FiLM — unique to diffusion |

Infrastructure is identical: P/Invoke, PTX/SPIR-V, memory model, tensor types, SIMD dispatch, error handling. dotLLM solved the hard problems — SharpInference adds domain-specific kernels.
