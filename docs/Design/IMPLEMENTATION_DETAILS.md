# Implementation Details

> Per-component technical approach. For shared patterns (tensor types, P/Invoke, disposal, CUDA launch), see `docs/CODE_STYLE.md` and `docs/Agents/AGENTS.md`.

## Component Overview

| Component | Approach | Key Risk |
|---|---|---|
| Tensor/TensorShape/TensorRef/TensorView/DType | Pure C# sealed class + readonly record struct | None |
| SafeTensors Loader | Pure C# mmap + JSON header | Multi-shard edge cases |
| GGUF Loader | Clean-room impl (dotLLM is GPLv3) | GGUF v3 dequant complexity |
| CPU Kernels | `System.Runtime.Intrinsics` AVX2/AVX-512/NEON | AVX-512 not universal |
| CUDA Backend | PTX via CUDA Driver API P/Invoke | PTX syntax curve |
| Vulkan Backend | SPIR-V via Vulkan API P/Invoke | Vulkan verbosity |
| CLIP/T5 Text Encoder | Pure C# transformer + kernels | Tokenizer must match exactly |
| UNet/DiT | Pure C# using op set | Cross-attention correctness |
| VAE Decoder | Pure C# Conv2D + GroupNorm | Tiled decode seam blending |
| Schedulers | Pure C# math | FP reproducibility |
| LoRA/ControlNet | Delta weights / separate UNet residuals | Flux format differs |
| Whisper STT / Kokoro TTS | Encoder-decoder / HiFiGAN vocoder | Autoregressive + timestamps |
| OpenAI API | ASP.NET Minimal API | — |

---

## Core — Tensor & Backend

**IBackend** — op-dispatch interface. Domain ops beyond memory management:
```csharp
void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);
void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);
void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);
void Fft(Tensor output, Tensor input);
void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);
void MelFilterbank(Tensor output, Tensor input, Tensor filters);
Tensor AllocateOnDevice(DeviceKind device, TensorShape shape, DType dtype);
void CopyTo(Tensor source, Tensor destination);
```

**Adaptive Thread Pool:** Custom `ComputeThreadPool` with `delegate*` dispatch. SpinWait (~100ns) for latency-critical (denoising), EventBased for throughput (loading). Auto-switches.

---

## Model Handler — Safetensors

**Format:** 8-byte LE header size → UTF-8 JSON header (dtype, shape, data_offsets) → tensor data.

**Loading:** `MemoryMappedFile.CreateFromFile` → parse JSON header → build index → return `TensorView`s into mmap. Multi-shard: sequential load, unified index with adjusted offsets.

---

## CPU — SIMD Kernels

| Kernel | Approach |
|---|---|
| Conv2D | im2col → GEMM. 1x1 degenerates to direct GEMM |
| GroupNorm | Split channels into groups; `TensorPrimitives` with SIMD dispatch |
| SDPA | Tiled attention (L2-cache-sized tiles), O(N) memory, ~1KB stack per head |
| FFT | Cooley-Tukey radix-2. Butterfly loop uses AVX2 complex multiply |
| Weight repacking | R4-style interleave for `Vector256<T>` sequential reads at load time |
| SIMD dispatch | `Vector512` → `Vector256` → scalar. Mandatory scalar fallback |

---

## CUDA — PTX Backend

See `docs/CODE_STYLE.md` for P/Invoke conventions and `docs/Agents/AGENTS.md` for CUDA launch pattern.

**Kernel fusion:** Minimize memory bandwidth. Key fusions: GroupNorm+SiLU, Conv2D+bias+activation, fused attention. Quantize activations (small) rather than dequantize weights (large).

**Conv2D:** cuDNN first for correctness; custom PTX later. 1x1 → cuBLAS HGEMM.

**cuBLAS:** `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for FP16-in/FP32-accumulate, auto Tensor Cores on Ampere+. Handle once per context.

---

## Vulkan — SPIR-V Backend

Same P/Invoke-to-driver philosophy as CUDA. `[LibraryImport("vulkan-1")]` (~40 functions).

**Shader management:** `.glsl` → `.spv` via `glslangValidator --target-env vulkan1.2`. Loaded via `vkCreateShaderModule` → `vkCreateComputePipelines` (cached). Dispatched via `vkCmdDispatch`.

**Memory:** Device-local for tensors, host-visible staging for transfers. Sub-allocate from large blocks (Vulkan ~4096 allocation limit).

**Descriptors:** Layouts cached process-lifetime. Pool pre-allocated. Push constants (≤128B) for scalars. Storage buffers for tensors.

**Key differences from CUDA:** No cuBLAS — tiled GEMM via subgroup ops. Subgroup size varies (32/64/8-32). Explicit sync (fences, semaphores, pipeline barriers).

---

## Diffusion — Pipelines

**Pipeline factory** — inspects model metadata → auto-instantiates correct pipeline. All implement `IAsyncEnumerable<GenerationProgress>`.

**UNet (SD1.5):** 4 down, 1 mid, 4 up. ResNetBlock: `GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv+residual`. CrossAttentionBlock: `LayerNorm→self-attn→cross-attn→FFN`. Timestep: sinusoidal→MLP→FiLM addition.

**VAE Tiled Decode:** Split latent into overlapping tiles, decode independently, blend overlaps with linear fade mask.

**LoRA:** Delta `dW = B × A × scale`. Applied in-place or kept additive for multi-LoRA.

---

## Audio — Whisper

**Preprocessing:** 16kHz PCM → 25ms Hann frames (10ms hop) → FFT → mel (80 bins) → log → normalize → `[1, 80, T]`.

**Encoder:** Two Conv1D (stride 1, stride 2) → positional encoding → N transformer blocks.

**Decoder:** Autoregressive transformer with cross-attention to encoder. KV-cache. Token IDs with optional timestamps.

---

## Server — OpenAI API

DotLLM Minimal API: `ServerState` singleton, source-gen JSON, one file per endpoint.

**Endpoints:** `POST /v1/images/generations` (JSON), `POST /v1/images/edits` (multipart), `POST /v1/audio/transcriptions` (multipart), `POST /v1/audio/speech` (JSON → audio stream), `GET /v1/models`, model load/unload/pull.

**Streaming:** SSE for image progress, chunked transfer for TTS audio.
