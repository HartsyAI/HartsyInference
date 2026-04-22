# Implementation Details

> Back to [Core Design](CORE_DESIGN.md) | Reference: [dotLLM Architecture Research](../Research/DOTLLM_ARCHITECTURE.md)

## Component Overview

| Component | Approach | Key Risk |
|---|---|---|
| Tensor/TensorShape/TensorRef/TensorView/DType | Pure C# sealed class + readonly record struct (dotLLM) | None |
| SafeTensors Loader | Pure C# mmap + JSON header | Multi-shard edge cases |
| GGUF Loader | Clean-room impl (dotLLM is GPLv3) | GGUF v3 dequant complexity |
| CPU Kernels | `System.Runtime.Intrinsics` AVX2/AVX-512/NEON (dotLLM SIMD dispatch) | AVX-512 not universal |
| CPU Conv2D | im2col + SIMD GEMM | Padding/stride edge cases |
| CPU Audio (FFT/STFT) | Cooley-Tukey radix-2 | Numerical precision |
| CUDA Backend | PTX via CUDA Driver API P/Invoke (dotLLM pattern) | PTX syntax curve |
| Vulkan Backend | SPIR-V via Vulkan API P/Invoke (extends dotLLM) | Vulkan verbosity, memory mgmt |
| CUDA Conv2D | cuDNN first, custom PTX later | cuDNN version compat |
| Vulkan Conv2D | SPIR-V compute shader (im2col + GEMM via subgroup ops) | Subgroup size varies |
| cuBLAS HGEMM | P/Invoke to cuBLAS (dotLLM pattern) | Version pinning |
| CLIP/T5 Text Encoder | Pure C# transformer + kernels | Tokenizer must match OpenAI exactly |
| UNet (SD1.5/SDXL) / DiT/MMDiT (SD3/Flux) | Pure C# using op set | Cross-attention correctness, novel architecture |
| VAE Decoder | Pure C# Conv2D + GroupNorm | Tiled decode seam blending |
| Schedulers | Pure C# math | FP reproducibility |
| LoRA/ControlNet | Add delta weights / separate UNet with residuals | Flux LoRA format differs |
| Whisper STT / Kokoro TTS | Pure C# encoder-decoder / HiFiGAN vocoder | Autoregressive decode + timestamps, phoneme encoder |
| STFT/Mel | Pure C# FFT + filterbank | Must match Whisper exactly |
| CLIP Image Encoder / YOLO | Pure C# ViT / CNN + NMS | Patch norm exactness, NMS tuning |
| OpenAI API | ASP.NET Minimal API (dotLLM pattern) | — |

---

## Core -- Tensor & Backend

### Multi-Type Tensor System (dotLLM)

| Type | Role | Key Properties |
|---|---|---|
| `Tensor` (sealed class, `IDisposable`) | Owns memory; lifecycle manager | `NativeMemory.AlignedAlloc` (64-byte aligned) or mmap pointer. Thread-safe disposal via `Interlocked.Exchange` → `AlignedFree`. Finalizer safety net. Shape: inline array up to 6D, row-major strides. GPU variants: `CudaTensor` (`cuMemAlloc_v2`), `VulkanTensor` (`vkAllocateMemory` + `vkBindBufferMemory`) |
| `TensorView` (sealed class, `IDisposable`) | Non-owning view | `Dispose()` no-op. Borrowed references, weight slices. Full `TensorShape` |
| `TensorRef` (readonly record struct) | Zero-alloc compute | `TensorRef(nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device)`. Stack-only, no ownership. `[AggressiveInlining]` accessors. Created via `.AsRef()` |
| `TensorMetadata` (readonly record struct) | Lightweight description | `Shape`, `DType`, `DeviceId`, `DataPointer` — no ownership |
| `DType` (readonly record struct) | Data type metadata | `Name`, `SizeInBytes`, `IsQuantized`, `BlockByteSize`, `BlockElementCount`. Pre-defined: `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, etc. `SizeInBytes = 0` for quantized; `ComputeByteCount()` validates block alignment |

**`IBackend`** — op-dispatch interface (deliberate divergence from dotLLM). `CpuBackend` always available; `CudaBackend` / `VulkanBackend` opt-in. Per-tensor `DeviceKind`; cross-device ops auto-insert `Copy`. Each implementation delegates to static kernel methods internally.

```csharp
// Domain ops beyond dotLLM's device management:
void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW);
void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);
void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);
void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);
void Fft(Tensor output, Tensor input);
void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);
void MelFilterbank(Tensor output, Tensor input, Tensor filters);
Tensor AllocateOnDevice(DeviceKind device, TensorShape shape, DType dtype);
void CopyTo(Tensor source, Tensor destination);
```

**Eager execution** — no computation graph. Each op executes immediately. Fusion is manual at kernel level (e.g., Conv2D + GroupNorm fused kernel).

### Adaptive Thread Pool (dotLLM)

Custom `ComputeThreadPool` with function-pointer dispatch (`delegate*<...>`) for zero-alloc work distribution:
- **SpinWait mode** — latency-critical paths (denoising steps), ~100ns wake latency
- **EventBased mode** — throughput paths (loading, preprocessing), blocks on `ManualResetEventSlim`
- Auto-switch: `_threadPool?.SetDispatchMode(isLatencyCritical ? DispatchMode.SpinWait : DispatchMode.EventBased);`

### ModelConfig Pattern (dotLLM)

Class record with `required` properties — reference semantics avoid copies of large config objects. `required` ensures mandatory fields set at construction.

```csharp
public record ModelConfig
{
    public required string Architecture { get; init; }
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
}
```

### Pipeline Options Pattern (dotLLM)

Three-tier `TextToImageOptions`:
- **Tier 1:** Flat properties (`Steps`, `CfgScale`, `Width`, `Height`, `Seed`)
- **Tier 2:** Explicit composition (`IScheduler`, `LoraSpec`s)
- **Tier 3:** Custom injection (`IPipelineCallback`s)

---

## Model Handler -- Safetensors

**Format:** 8-byte LE `uint64` (header byte count) → UTF-8 JSON header (`dtype`, `shape`, `data_offsets`) → tensor data blob.

**Loading:** `MemoryMappedFile.CreateFromFile` → `MemoryMappedViewAccessor` → `JsonSerializer.Deserialize` header → build index → return `TensorView`s pointing into mmap. Multi-shard: read sequentially, build unified index with adjusted offsets.

**ModelLoader (dotLLM pattern):**
```csharp
public static class ModelLoader
{
    public static (IModel Model, ModelConfig Config) LoadFromSafeTensors(
        string path, IBackend backend, ModelConfig? configOverride = null);
}
```

---

## CPU -- SIMD Kernels

| Kernel | Approach |
|---|---|
| **Conv2D** | im2col transformation → GEMM. 1×1 convolutions degenerate to direct GEMM (no im2col) |
| **GroupNorm** | Split channels into groups; `TensorPrimitives.Sum` / `Norm` with SIMD dispatch |
| **SDPA** | Tiled attention (L2-cache-sized tiles), O(N) memory, ~1KB stack per head. Diffusion: spatial features (HxW) cross-attending to text embeddings |
| **FFT** | Cooley-Tukey radix-2 (400/512 sample windows). Butterfly loop uses AVX2 complex multiply |
| **Weight repacking** | R4 pattern interleaves 4 rows of quantization blocks for `Vector256<T>` sequential reads. Load-time cost pays back per-inference |
| **SIMD dispatch** | Tiered: `Vector512` → `Vector256` → scalar fallback. Cross-platform vectors preferred. **Mandatory scalar fallback** for every kernel |

---

## CUDA -- PTX Backend (dotLLM)

Library name `"cuda"` (not `"nvcuda"`), resolved at runtime by `CudaLibraryResolver`.

**P/Invoke conventions:**
- `[LibraryImport]` (source-generated) over `[DllImport]` — zero-alloc, trimmer-friendly
- Return type `int` (not enum) — `.ThrowOnError()` → `cuGetErrorName`/`cuGetErrorString` → `CudaException`
- `[SuppressGCTransition]` on short calls (< 1us)
- `CudaLibraryResolver` via `NativeLibrary.SetDllImportResolver()`

**PTX loading (NOT embedded):**
```csharp
public sealed class CudaKernels : IDisposable
{
    private readonly nint _conv2dModule, _conv2dF16_3x3Func, _conv2dF16_1x1Func;
    // ... all 18 kernel families as nint FIELDS (not dictionary-cached)

    public CudaKernels(string ptxDir)
    {
        _conv2dModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "conv2d.ptx"));
        _conv2dF16_3x3Func = _conv2dModule.GetFunction("conv2d_f16_3x3");
        // ...
    }
}
```

**Kernel launch (zero-alloc `stackalloc`):**
```csharp
public void LaunchGroupNormSiLU(nint output, nint input, nint weight, nint bias,
    int channels, int groups, float eps, nint stream)
{
    nint outputArg = output, inputArg = input, weightArg = weight, biasArg = bias;
    int channelsArg = channels, groupsArg = groups;
    float epsArg = eps;
    void** args = stackalloc void*[] {&outputArg, &inputArg, &weightArg,
        &biasArg, &channelsArg, &groupsArg, &epsArg};
    CudaDriverApi.cuLaunchKernel(_groupNormSiluFusedFunc,
        (uint)groups, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
}
```

BlockSize typically 256 (matching dotLLM). Grid varies per kernel.

**Kernel fusion:** Minimize memory bandwidth (primary bottleneck). Key fusions: **GroupNorm+SiLU** (saves one tensor R/W), **Conv2D+bias+activation** (eliminates intermediate), **fused attention** (Q×Kᵀ, scale, softmax, ×V in one pass). Quantize activations (small) rather than dequantize weights (large) where applicable.

**Conv2D strategy:** cuDNN P/Invoke first for correctness; custom PTX added in later performance pass. 1×1 convolutions degenerate to cuBLAS HGEMM (no im2col).

**cuBLAS:** `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for FP16-in/FP32-accumulate, auto Tensor Cores on Ampere+. Handle created once per CUDA context. All large GEMM: UNet projections, cross-attention QKV, encoder layers, im2col Conv2D.

---

## Vulkan -- SPIR-V Backend (extends dotLLM)

Same pure-C# philosophy: P/Invoke to driver API, pre-compiled shader binaries, zero native wrappers. `[LibraryImport("vulkan-1")]` (~40 functions), `int` returns (not `VkResult` enum), same `.ThrowOnError()`. `VulkanLibraryResolver` via `NativeLibrary.SetDllImportResolver()`.

**SPIR-V shader management (mirrors PTX):**
1. Build-time: `.glsl` → `.spv` via `glslangValidator --target-env vulkan1.2`
2. Ship: `.spv` as content files (same as PTX)
3. Runtime: `vkCreateShaderModule` → `vkCreateComputePipelines` (cached in `Dictionary<string, nint>`)
4. Dispatch: `vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch(groupX, groupY, groupZ)`

**Vulkan memory management:**
- Device-local memory (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) for tensor storage
- Host-visible staging buffers for host↔device transfers (`memcpy` → staging → `vkCmdCopyBuffer` → device)
- Sub-allocate from large `vkAllocateMemory` blocks (Vulkan limits total allocations to ~4096)

**Descriptor management:**
- Set layouts created once per kernel signature (process-lifetime cached)
- Descriptor pool pre-allocated at startup
- Push constants (up to 128 bytes) for scalar params (stride, padding, epsilon)
- Storage buffers for all tensor data

**Key differences from CUDA:**
- No cuBLAS equivalent — tiled GEMM via subgroup operations + shared memory
- Subgroup size varies by vendor: NVIDIA 32, AMD 64, Intel 8–32
- Explicit sync: fences, semaphores, pipeline barriers (vs CUDA implicit stream ordering)
- No `[SuppressGCTransition]` benefit — command buffer recording adds overhead
- Feature detection via `vkGetPhysicalDeviceFeatures2` (subgroup ops, 16-bit storage, push descriptors)

---

## Diffusion -- Pipelines

**Pipeline factory** — inspects model metadata (safetensors config / GGUF metadata) → auto-instantiates correct pipeline. All implement `IAsyncEnumerable<GenerationProgress>` for streaming.

**UNet (SD1.5):** 4 down-blocks, 1 middle, 4 up-blocks.
- **ResNetBlock:** `GroupNorm → SiLU → Conv2D → GroupNorm → SiLU → Conv2D + residual`
- **CrossAttentionBlock:** `LayerNorm → SDPA(self) → LayerNorm → CrossAttn(text) → LayerNorm → FFN`
- **Timestep conditioning:** sinusoidal embedding → MLP → FiLM-style addition to ResNetBlock hidden state

**VAE Tiled Decode:** Split latent into overlapping tiles, decode independently, blend overlaps with linear fade mask to eliminate seams.

**LoRA:** Low-rank matrices (A, B) produce delta `dW = B × A × scale`. Added in-place to base weights or kept as additive correction for multi-LoRA. Flux LoRA uses different rank decomposition — separate handling.

---

## Audio -- Whisper

**Preprocessing:** Raw PCM (16kHz) → 25ms Hann-windowed frames (10ms hop) → FFT → magnitude spectrogram → mel filterbank (80 bins) → log compression → mean-subtract normalization → `[1, 80, T]` tensor.

**Encoder:** Two `Conv1D` layers (stride 1, stride 2) reduce temporal dim by 2 → sinusoidal positional encoding → N transformer blocks (MHA + FFN with GELU). No cross-attention.

**Decoder:** Autoregressive transformer with cross-attention to encoder outputs. Generates token IDs (language/task tokens, optional timestamps). KV-cache pattern (encoder-decoder, not decoder-only).

---

## Server -- OpenAI-Compatible API

DotLLM Minimal API architecture: `ServerState` singleton, source-generated JSON (`[JsonSerializable]`), one file per endpoint.

**Image API:**
- `POST /v1/images/generations` — JSON: `prompt`, `model`, `size`, `n`, `response_format`, `seed`, generation params. Returns base64 PNG or URL.
- `POST /v1/images/edits` — multipart: `image`, `mask`, `prompt` for img2img/inpainting.

**Audio API:**
- `POST /v1/audio/transcriptions` — multipart audio + model. Returns transcript JSON.
- `POST /v1/audio/speech` — `input` text + `voice` + `model`. Returns audio stream.

**Streaming:**
- Image: SSE `{"step": N, "total": M, "preview": "<base64>"}` per step; final `{"status": "complete", "image": "<base64>"}`
- Audio TTS: chunked transfer encoding for streaming PCM

**Model Management:** `GET /v1/models` (list), `POST /v1/models/load` (trigger load), `DELETE /v1/models/{id}` (unload), `POST /v1/models/pull` (HF download).

**DI and JSON (dotLLM pattern):**
```csharp
public static class SharpInferenceServiceExtensions
{
    public static IServiceCollection AddSharpInference(
        this IServiceCollection services, ServerState state)
    {
        services.AddSingleton(state);
        services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain
                .Insert(0, SharpInferenceJsonContext.Default));
        return services;
    }
}

[JsonSerializable(typeof(ImageGenerationRequest))]
[JsonSerializable(typeof(ImageGenerationResponse))]
[JsonSerializable(typeof(AudioTranscriptionResponse))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(ErrorResponse))]
internal partial class SharpInferenceJsonContext : JsonSerializerContext { }
```
