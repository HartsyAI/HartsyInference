# SharpInference — Core Design Overview

**Version 2.0 | April 2026**

SharpInference is a **native C#/.NET 10 AI inference engine** for non-LLM modalities — image generation, speech-to-text, text-to-speech, vision, object detection, and video. It works alongside **dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) as a complete, self-contained AI platform with zero Python dependencies, zero C++ wrappers, and no external processes.

## Design Pillars

| Pillar | Description |
|---|---|
| **Pure C#** | CUDA via PTX + Driver API P/Invoke; Vulkan via SPIR-V + Vulkan API P/Invoke |
| **Zero-allocation hot paths** | `NativeMemory.AlignedAlloc`; mmap weights; `TensorRef` readonly record struct on kernels; `Span<T>` on hot paths |
| **Modular NuGet** | Pull only what you need |
| **Multi-GPU backend** | CUDA (NVIDIA via PTX + cuBLAS) + Vulkan (AMD/Intel/NVIDIA via SPIR-V + compute shaders) |
| **OpenAI-compatible API** | Drop-in replacement for OpenAI image/audio endpoints |
| **dotLLM alignment** | Same tensor memory, P/Invoke, PTX management, SIMD dispatch, adaptive thread pool — extended to non-LLM modalities with deliberate divergence (`IBackend` op-dispatch) |
| **Production-grade** | Error handling, streaming, memory budgeting, VRAM monitoring, model hot-swap |

## Design Documents Index

| Document | Description |
|---|---|
| [Vision & Goals](VISION_AND_GOALS.md) | Core motivation, SwarmUI backend angle |
| [Features](FEATURES.md) | Full features list |
| [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) | Phase 1-3 model support plan |
| [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependency graph |
| [File Structure](FILE_STRUCTURE.md) | Project layout |
| [Implementation Details](IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [Build Order & Phases](BUILD_ORDER.md) | Implementation sequencing |
| [Validation Strategy](VALIDATION_STRATEGY.md) | Reference implementations and validation methods |
| [Research Requirements](RESEARCH_REQUIREMENTS.md) | Research documents needed before implementation |

## Architecture at a Glance

```
+--------------------------------------------------------------+
|                     SharpInference.Server                     |
|            (OpenAI-compatible REST API + SSE)                 |
+------------+--------------+----------------------------------+
|  Diffusion |    Audio     |           Vision                 |
|  SD/SDXL   |  Whisper STT |  CLIP Embeddings                 |
|  Flux/SD3  |  Kokoro TTS  |  YOLO Detection                  |
|  LoRA/CN   |  Voice Conv  |  SAM Segmentation                |
+------------+--------------+----------------------------------+
|                  SharpInference.Core                          |
|    Tensor + TensorRef . IBackend . Schedulers . Pipelines    |
+--------------+---------------------+------------------------+
| CPU Backend  |    CUDA Backend      |   Vulkan Backend       |
| AVX2/512/NEON| PTX Kernels + cuBLAS | SPIR-V + VkCompute     |
+--------------+---------------------+------------------------+
|                     Model Handler                             |
|        Safetensors . GGUF . HuggingFace . Registry            |
+--------------------------------------------------------------+
```

**Three backends follow the same pattern:** model code programs against `IBackend` only. CPU dispatches to SIMD kernels; CUDA to PTX + cuBLAS; Vulkan to SPIR-V compute shaders. Backend selected at runtime based on hardware detection.

## Key Architectural Decisions

- **Eager execution** — no computation graph. Each op executes immediately. Fusion is manual at kernel level (dotLLM pattern).
- **Multi-type tensor system (dotLLM)** — `Tensor` (sealed, `IDisposable`) owns memory; `TensorView` (sealed, non-owning, `Dispose()` no-op); `TensorRef` (readonly record struct, zero-alloc for kernel internals). Compute path never allocates, never boxes, never touches GC.
- **Unmanaged memory** — `NativeMemory.AlignedAlloc(byteCount, 64)` with 64-byte alignment, or memory-mapped files. Thread-safe disposal via `Interlocked.Exchange` before `NativeMemory.AlignedFree`. Finalizer safety net. (dotLLM `UnmanagedTensor` pattern.)
- **PTX from disk (dotLLM)** — `.cu` → `.ptx` via `nvcc -ptx`; shipped as content files. Loaded at runtime via `CudaModule.LoadFromFile()`, JIT-compiled by driver. Function handles stored as `nint` fields (not dictionary-cached), resolved once in constructor. (dotLLM `CudaKernels` pattern.)
- **SPIR-V for Vulkan** — `.glsl` → `.spv` via `glslangValidator`; shipped as content files. Loaded via `vkCreateShaderModule`, compute pipelines via `vkCreateComputePipelines`, dispatched via `vkCmdDispatch`. All Vulkan access through `[LibraryImport("vulkan-1")]` P/Invoke.
- **IBackend op-dispatch (deliberate divergence)** — dotLLM uses `IBackend` only for device memory management; kernel ops are direct static calls. SharpInference uses `IBackend` for op-dispatch because we have 3 backends × many model types — separate implementations per backend would be unmaintainable. Each `IBackend` implementation delegates to static kernel methods internally, keeping zero-alloc compute patterns.
- **Pipeline factory** — model metadata drives automatic pipeline selection (dotLLM `ModelLoader` pattern).
- **Three-tier options (dotLLM)** — flat properties (simple), explicit step composition (advanced), custom processor injection (full control).

---

## Relationship to dotLLM

SharpInference works alongside **dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) — a ground-up C# LLM inference engine by Konrad Kokosa. Together they form a complete, pure C#/.NET AI platform. dotLLM handles LLM text generation; SharpInference covers everything else.

> **Reference:** See `docs/Research/DOTLLM_ARCHITECTURE.md` for source-verified analysis.

### How dotLLM Inspires SharpInference

| Pattern | dotLLM | SharpInference |
|---|---|---|
| **CUDA access** | `"cuda"` lib name, ~40 `[LibraryImport]` P/Invoke, `int` returns, `CudaLibraryResolver` via `NativeLibrary.SetDllImportResolver()`. PTX loaded from disk directory. Function handles as `nint` fields, resolved in `CudaKernels` constructor | Same approach. `.ThrowOnError()` calls `cuGetErrorName`/`cuGetErrorString` |
| **Vulkan** | Not implemented (CUDA-only) | Extends dotLLM's P/Invoke-to-driver-API philosophy: `[LibraryImport("vulkan-1")]` (~40 functions), pure C# |
| **cuBLAS** | `CublasApi.cs` (~6 functions), `cublasGemmEx` with `CUBLAS_COMPUTE_32F`, auto Tensor Cores on Ampere+. Handle once per context | Same binding strategy. `cublasGemmEx` for FP16/FP32 GEMM in convolution, attention, projections |
| **Tensor types** | `UnmanagedTensor`, `CudaTensor` (with `_ownsMemory`), `TensorView` (non-owning, `Dispose()` no-op), `TensorRef` (readonly record struct, flat `Dim0`/`Dim1`), `TensorMetadata` (readonly record struct) | `Tensor` (sealed), `TensorView` (non-owning), `TensorRef` (kernel internals), `TensorMetadata` |
| **DType** | `readonly record struct` with `Name`, `SizeInBytes`, `IsQuantized`, `BlockByteSize`, `BlockElementCount`. `ComputeByteCount()` with `Debug.Assert` block alignment. `SizeInBytes = 0` for quantized | Same pattern with `Name` for diagnostics |
| **Tensor memory** | `NativeMemory.AlignedAlloc(byteCount, 64)`. Thread-safe disposal via `Interlocked.Exchange`. Finalizer safety net. `AllocateBytes()` for quantized. `ArrayPool<T>` for short-lived managed buffers only | Same 64-byte aligned allocations, same disposal, same safety net. `ArrayPool<T>` for non-tensor temporaries only |
| **Model loading** | `MemoryMappedFile.CreateFromFile()` with OS demand-paging. `ModelLoader` static helper returning `(IModel, GgufFile, ModelConfig)`. `ModelConfig` class record with `required` properties and `init` setters | Same mmap strategy for SafeTensors and GGUF. Static `ModelLoader`. `ModelConfig` as class record with `required` properties |
| **SIMD kernels** | `TensorPrimitives` standard ops, `System.Runtime.Intrinsics` hot loops. Cross-platform vectors preferred. Scalar fallbacks mandatory. R4 weight repacking at load time | Same intrinsics strategy, tiered dispatch, R4-style repacking |
| **IBackend role** | Device memory management only (`AllocateOnDevice`, `CopyBetweenDevices`, `AllReduce`, `Send`, `Receive`). Kernel ops direct static method calls | **Deliberate divergence:** op-dispatch interface. Each implementation delegates to static kernel methods internally |
| **Kernel dispatch** | Direct static calls. `TransformerModel.Forward()` → `MatMul.GemvQ8_0(...)` directly. Separate CPU and CUDA model classes | Routes through `IBackend` (one virtual dispatch) → static kernel methods. One pipeline per model, any backend |
| **Attention** | Flash Attention 2/3 (GPU), CPU tiled attention (L2-cache-sized tiles). Intentionally not behind unified interface | CPU tiled attention adapted for spatial cross-attention. Same Flash Attention PTX for CUDA. Vulkan equivalent via SPIR-V |
| **Quantization** | Dequant for Q4_0/Q4_1/Q5_0/Q5_K/Q6_K/Q8_0/Q4_K. On-the-fly activation quantization (quantize small activations). Custom Q8_0 x Q4_K GEMV | Same dequant support. Activation quantization strategy. VAE always FP16; UNet/DiT supports Q8_0 |
| **Kernel fusion** | RMSNorm+Quantize, SwiGLU+L1-tiling, on-the-fly activation quantization. Memory bandwidth is THE bottleneck | GroupNorm+SiLU, Conv2D+bias+activation, fused attention. Bandwidth-first optimization |
| **Adaptive thread pool** | `ComputeThreadPool` with function-pointer dispatch (`delegate*<...>`). SpinWait (~100ns) vs EventBased (`ManualResetEventSlim`). Auto-switches based on `seqLen == 1` | Same pool with function-pointer dispatch. SpinWait during denoising steps, EventBased during loading/preprocessing |
| **Options pattern** | `InferenceOptions` class record: flat properties (auto-build), explicit `ISamplerStep` list (composable), custom `ILogitProcessor` list (full control) | Same three-tier pattern for pipeline options |
| **Streaming** | `IAsyncEnumerable<GenerationToken>` with readonly record struct, `[EnumeratorCancellation] CancellationToken` | `IAsyncEnumerable<GenerationProgress>` with readonly record struct |
| **Server** | ASP.NET Minimal API. `ServerState` singleton created before DI. Source-generated JSON via `[JsonSerializable]`. One file per endpoint. SSE via `Results.Stream()` | Same Minimal API architecture. `ServerState` singleton. Source-generated JSON |
| **Error handling** | `int` CUDA returns. `.ThrowOnError()` → `cuGetErrorName`/`cuGetErrorString` → `CudaException(int, string)`. `Environment.FailFast` for unrecoverable compute thread errors | Same `int` returns, same `.ThrowOnError()`, `Environment.FailFast` for corrupted workers |
| **Build system** | `.slnx` solution format. Central package management via `Directory.Packages.props`. Minimal `.csproj` files | Same: `.slnx`, `Directory.Build.props`, `Directory.Packages.props` |
| **Code standards** | `[MethodImpl(AggressiveInlining)]`, `[SkipLocalsInit]`, `[SuppressGCTransition]`, `Span<T>`, nullable reference types, file-scoped namespaces, `readonly record struct`, `sealed`, no `#region`, XML doc comments | Identical conventions |

### Why IBackend Op-Dispatch (Deliberate Divergence)

In dotLLM, `IBackend` is purely device memory management; model code calls kernels directly (e.g., `TransformerModel.Forward()` → `MatMul.GemvQ8_0(...)`). This works for LLMs with ~1 model architecture × 2 backends.

SharpInference diverges because the problem space is different:

| Factor | dotLLM | SharpInference |
|---|---|---|
| Backends | 2 (CPU, CUDA) | 3 (CPU, CUDA, Vulkan) |
| Model types | ~1 (transformer decoder) | Many (UNet, DiT, VAE, Whisper, Kokoro, YOLO, CLIP, etc.) |
| Duplicated code | 2 copies per model | 3 copies × many models = unmaintainable |

With `IBackend` op-dispatch, each pipeline is written once and runs on any backend. Virtual dispatch overhead (~2ns for `IBackend.Conv2D()`) is negligible vs kernel runtime (milliseconds). Each `IBackend` delegates to static kernel methods internally, keeping zero-alloc compute patterns.

### dotLLM's CUDA Kernel Coverage

dotLLM ships 24 PTX kernels from `native/ptx/`. Kernel domains: rmsnorm, rope, attention, softmax, swiglu, embedding, dequantization (Q8_0/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K), quantized GEMV, bias_add, type conversion, KV-cache quantization, fused_add_rmsnorm, per_head_rmsnorm.

### SharpInference's Kernel Coverage

~18 kernel families, each with FP16/FP32 variants, both PTX (CUDA) and SPIR-V (Vulkan):

| Kernel | Domain |
|---|---|
| `conv2d_f16_3x3` / `1x1` | UNet ResNet, projections |
| `group_norm_f16` / `group_norm_silu_fused` | UNet/VAE normalization |
| `layer_norm_f16` | Attention, text encoders |
| `sdpa_f16` | Diffusion attention (tiled O(N) memory) |
| `upsample2d_nearest` / `bilinear` | Spatial upsampling |
| `elementwise_f16` / `silu_f16` / `gelu_f16` | Activations, residuals |
| `dequant_q8` / `dequant_q4k` | On-the-fly dequantization |
| `fft_radix2` / `mel_filterbank` | Audio preprocessing |
| `rope_2d` | Flux/SD3 DiT position encoding |
| `timestep_embed` | UNet conditioning |
| `conv2d_bias_silu_fused` | Fused ResNet block (bandwidth opt) |

### How Vulkan Follows dotLLM Pattern

| dotLLM CUDA | SharpInference Vulkan |
|---|---|
| `[LibraryImport("cuda")]` (~40 functions) | `[LibraryImport("vulkan-1")]` (~40 functions) |
| `.cu` → `.ptx` via `nvcc -ptx` | `.glsl` → `.spv` via `glslangValidator` |
| `CudaModule.LoadFromFile()` → driver JIT SASS | `vkCreateShaderModule` → driver compiles to target ISA |
| `cuModuleGetFunction` → `nint` field | `vkCreateComputePipelines` → cached pipeline handle |
| `cuLaunchKernel(func, grid, block, args)` | `vkCmdDispatch(commandBuffer, groupCountX/Y/Z)` |
| `cuMemAlloc_v2` / `cuMemFree_v2` | `vkAllocateMemory` + `vkBindBufferMemory` |
| `cuMemcpyHtoD` / `cuMemcpyDtoH` | Staging buffer + `vkCmdCopyBuffer` |
| `cuStreamCreate` | `VkQueue` + `VkCommandBuffer` |
| `CudaLibraryResolver` | `VulkanLibraryResolver`: `vulkan-1.dll` (Win), `libvulkan.so.1` (Linux) |
| `nint` fields for handles | `Dictionary<string, nint>` pipeline cache |
| `stackalloc void*[]` kernel args | Descriptor sets + push constants |

**Vulkan specifics:**
- Explicit memory type selection via `vkGetPhysicalDeviceMemoryProperties`: `DEVICE_LOCAL` for tensors, `HOST_VISIBLE | HOST_COHERENT` for staging
- GLSL `layout(set=N, binding=M)` mapped to descriptor sets
- Push constants (up to 128 bytes) for scalar params (stride, padding, epsilon)
- `VkFence` / `VkSemaphore` for GPU sync
- Subgroup operations (`subgroupAdd`, `subgroupShuffle`) replace CUDA warp shuffles

### Integration Points

- **Prompt enhancement** — dotLLM generates/refines prompts for SharpInference diffusion pipelines
- **Multimodal** — SharpInference vision output (CLIP embeddings, detections) consumed by dotLLM
- **Shared CUDA context** — both P/Invoke same CUDA Driver API; share context + device
- **Unified model registry** — single local model store serves both projects
- **Common API surface** — dotLLM `/v1/chat/completions`, SharpInference `/v1/images/*` + `/v1/audio/*`; compose into single server

### Design Philosophy Alignment

Both share: **production AI inference belongs in managed code**. dotLLM proved pure C# with PTX achieves 66-88% of llama.cpp decode throughput. SharpInference extends this to diffusion/audio/vision workloads.

Key shared principles:
- **No external processes** — in-process only
- **No native shared libraries** — CUDA via Driver API P/Invoke; Vulkan via Vulkan API P/Invoke
- **Predictable memory** — 64-byte aligned unmanaged allocations, `ArrayPool<T>` for temporaries, never GC for large buffers
- **Zero-GC hot paths** — no managed heap allocations during inference; metadata as readonly record structs
- **Validate against references** — numerical correctness vs Python implementations
- **Process-lifetime caching** — PTX modules, SPIR-V pipelines, handles, cuBLAS, descriptor layouts created once
- **Don't abstract prematurely** — optimize per-backend where performance gain justifies complexity

### Licensing Note

dotLLM is GPLv3. SharpInference uses clean-room implementations where functionality overlaps to maintain licensing independence. Architectural patterns are not subject to copyright — inspiration is in *how* inference should be structured, not copied code.
