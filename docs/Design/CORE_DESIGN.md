# SharpInference — Core Design Overview

**Version 2.0 | April 2026**

---

## What Is SharpInference?

SharpInference is a **native C#/.NET 10 AI inference engine** for all non-LLM inference modalities — image generation (diffusion), speech-to-text, text-to-speech, voice conversion, image understanding, object detection, and video generation. It works closely alongside **dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) to form a complete, self-contained AI platform with zero Python dependencies, zero C++ wrappers, and no external processes. SharpInference draws significant architectural inspiration from dotLLM — adopting its approach to CUDA access via PTX and the Driver API, unmanaged tensor memory with 64-byte aligned allocations, SIMD kernel dispatch, and adaptive thread pool — and extends these proven patterns to non-LLM inference workloads.

---

## Design Pillars

| Pillar | What It Means |
|---|---|
| **Pure C#** | No native shared libraries required; CUDA accessed via PTX through the CUDA Driver API P/Invoke; Vulkan accessed via SPIR-V through the Vulkan API P/Invoke — same pure-C# philosophy for both GPU backends |
| **Zero-allocation hot paths** | All tensor data in `NativeMemory.AlignedAlloc`; model weights memory-mapped; `TensorRef` readonly record struct on all kernel signatures; `Span<T>` on hot paths |
| **Modular NuGet packages** | Pull in only what you need; no bloated single-assembly distribution |
| **Multi-GPU backend** | CUDA (NVIDIA via PTX + cuBLAS) and Vulkan (AMD/Intel/NVIDIA via SPIR-V + compute shaders) — both following dotLLM's P/Invoke-to-driver-API pattern |
| **OpenAI-compatible API** | Drop-in replacement for OpenAI image and audio endpoints |
| **dotLLM alignment** | Directly follows dotLLM's proven inference patterns; same tensor memory model, same P/Invoke conventions, same PTX management, same SIMD dispatch, same adaptive thread pool — extended to non-LLM modalities with one deliberate divergence (IBackend as op-dispatch, explained below) |
| **Production-grade** | Proper error handling, streaming progress, memory budgeting, VRAM monitoring, model hot-swap |

---

## Design Documents Index

Each section of the design is broken out into its own document for focused reference:

| Document | Description |
|---|---|
| [Vision & Goals](VISION_AND_GOALS.md) | Core motivation, the SwarmUI backend angle, and why this project exists |
| [Features](FEATURES.md) | Complete features list across infrastructure, image, audio, vision, and server |
| [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) | Phase 1-3 model support plan with formats and priorities |
| [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependency graph, and minimum install examples |
| [File Structure](FILE_STRUCTURE.md) | Full project file and folder layout |
| [Implementation Details](IMPLEMENTATION_DETAILS.md) | Per-component technical approach, architecture decisions, and key algorithms |
| [Build Order & Phases](BUILD_ORDER.md) | Implementation sequencing and phase dependencies |
| [Validation Strategy](VALIDATION_STRATEGY.md) | Reference implementations and validation methods for every component |
| [Research Requirements](RESEARCH_REQUIREMENTS.md) | Research documents needed before implementation, organized by component |

---

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

**The three backends follow the same pattern:** model code programs against `IBackend` and never references any backend directly. CPU dispatches to SIMD kernels. CUDA dispatches to PTX kernels + cuBLAS. Vulkan dispatches to SPIR-V compute shaders. The backend is selected at runtime based on hardware detection.

---

## Key Architectural Decisions

- **Eager execution** — no computation graph. Each op executes immediately. Simpler to implement and debug; sufficient for inference (no gradients needed). Fusion is done manually at the kernel level. This is how dotLLM works.
- **Multi-type tensor system (from dotLLM)** — `Tensor` (sealed class, `IDisposable`) owns memory and manages lifecycle. `TensorView` (sealed class) is a non-owning view where `Dispose()` is a no-op. `TensorRef` (readonly record struct) is a zero-alloc view used internally in kernel implementations. This separation means the compute path never allocates, never boxes, and never touches the GC.
- **Unmanaged memory** — all tensor storage via `NativeMemory.AlignedAlloc(byteCount, 64)` with 64-byte alignment, or memory-mapped files. Zero GC pressure on the inference hot path. Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree`. Finalizer safety net for forgotten `Dispose()` calls. All of this follows dotLLM's `UnmanagedTensor` pattern.
- **PTX from disk directory (from dotLLM)** — CUDA `.cu` source files are compiled to `.ptx` via `nvcc -ptx` and shipped as content files in a directory alongside the .NET assemblies. At runtime, PTX is loaded from the directory via `CudaModule.LoadFromFile()` and JIT-compiled by the GPU driver to native SASS. Function handles stored as `nint` fields (not dictionary-cached), resolved once in the constructor. This is exactly how dotLLM's `CudaKernels` class works.
- **SPIR-V for Vulkan (extending dotLLM's approach)** — the same philosophy applied to Vulkan: `.glsl` compute shaders are compiled to `.spv` via `glslangValidator` and shipped as content files. At runtime, SPIR-V is loaded via `vkCreateShaderModule`, compute pipelines created via `vkCreateComputePipelines`, and dispatched via `vkCmdDispatch`. All Vulkan API access is through `[LibraryImport("vulkan-1")]` P/Invoke — the same pure-C# pattern dotLLM proved with CUDA.
- **IBackend as op-dispatch (deliberate divergence from dotLLM)** — In dotLLM, `IBackend` is purely device memory management and kernel ops are called directly as static methods. SharpInference uses `IBackend` as an op-dispatch interface because we have 3 backends x many model types — separate model implementations per backend would be unmaintainable. Each `IBackend` implementation immediately delegates to static kernel methods internally. See the full rationale in the dotLLM section below.
- **Pipeline factory** — model metadata (architecture key) drives automatic pipeline selection. Users load a model file and get the right pipeline without manual configuration. Follows dotLLM's `ModelLoader` static helper pattern.
- **Three-tier options (from dotLLM)** — Pipeline options follow dotLLM's `InferenceOptions` pattern: flat properties for simple use, explicit step composition for advanced use, custom processor injection for full control.

---

## Relationship to dotLLM

SharpInference is designed to work **closely alongside dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) and draws heavy inspiration from its architecture and inference approach. dotLLM is a ground-up LLM inference engine written entirely in C# by Konrad Kokosa — a 20+ year .NET veteran and author of *Pro .NET Memory Management* — supporting Llama, Mistral, Phi, Qwen, and DeepSeek transformer architectures on .NET 10. Together, the two projects form a complete, pure C#/.NET AI platform — dotLLM handles LLM text generation while SharpInference covers every other modality.

> **Reference:** See `docs/Research/DOTLLM_ARCHITECTURE.md` for the complete source-verified analysis of dotLLM's codebase. Every pattern described below has been validated against the actual source code.

### How dotLLM Inspires SharpInference

dotLLM pioneered many of the patterns that SharpInference adopts and extends for non-LLM inference. The following table has been verified against dotLLM's actual source code:

| Pattern | How dotLLM Does It | How SharpInference Adopts It |
|---|---|---|
| **CUDA access** | `private const string LibName = "cuda"` with ~40 `[LibraryImport(LibName)]` P/Invoke declarations. Return types are `int` (not an enum). `CudaLibraryResolver` maps `"cuda"` to `nvcuda.dll` (Windows) / `libcuda.so` (Linux) at runtime via `NativeLibrary.SetDllImportResolver()`. PTX loaded from a **directory on disk** via `CudaModule.LoadFromFile()`, function handles stored as **`nint` fields**, resolved once in the `CudaKernels` constructor | **Same approach:** `"cuda"` library name with `CudaLibraryResolver`. PTX loaded from content file directory. Function handles as `nint` fields. `int` return types with `.ThrowOnError()` that calls `cuGetErrorName`/`cuGetErrorString` |
| **Vulkan (SharpInference extension)** | dotLLM does not implement Vulkan — CUDA-only | SharpInference extends dotLLM's P/Invoke-to-driver-API philosophy to Vulkan: `VulkanApi.cs` with `[LibraryImport("vulkan-1")]` P/Invoke declarations (~40 functions). Same pure-C# pattern — no native wrappers, no managed Vulkan libraries |
| **cuBLAS** | Separate `CublasApi.cs` P/Invoke surface (~6 functions). `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for FP16-in/FP32-accumulate, automatic Tensor Core usage on Ampere+. Handle created once per CUDA context | **Same cuBLAS binding strategy.** `cublasGemmEx` for FP16/FP32 GEMM in convolution (im2col + GEMM), attention (QKT and AV), and text encoder projections |
| **Tensor types** | **Three concrete types:** `UnmanagedTensor` (owns CPU memory), `CudaTensor` (owns GPU memory, has `_ownsMemory` flag), `TensorView` (non-owning, `Dispose()` is no-op). All implement `ITensor` interface. **Plus** `TensorRef` (readonly record struct with flat `Dim0`/`Dim1` fields — no `TensorShape`) for zero-alloc hot paths. **Plus** `TensorMetadata` (readonly record struct) | **Same multi-type approach:** `Tensor` (sealed class), `TensorView` (non-owning), `TensorRef` (readonly record struct) for kernel internals, `TensorMetadata` for descriptions |
| **DType** | `readonly record struct DType(string Name, int SizeInBytes, bool IsQuantized, int BlockByteSize = 0, int BlockElementCount = 1)` with `ComputeByteCount()` using `Debug.Assert` for block alignment. `SizeInBytes = 0` for quantized types | **Same pattern** with `Name` field for diagnostics. `ComputeByteCount()` validates block alignment |
| **Tensor memory** | `NativeMemory.AlignedAlloc(byteCount, 64)` with 64-byte alignment. Thread-safe disposal via `Interlocked.Exchange`. Finalizer safety net. `CudaTensor` has `AllocateBytes()` for quantized types where per-element size is 0. `ArrayPool<T>` for short-lived managed buffers only | **Same 64-byte aligned allocations**, same `Interlocked.Exchange` disposal, same finalizer safety net. `ArrayPool<T>` for non-tensor temporaries only |
| **Model loading** | GGUF via `MemoryMappedFile.CreateFromFile()` with OS demand-paging. `ModelLoader` is a **static helper class** returning `(IModel, GgufFile, ModelConfig)`. `ModelConfig` is a **class record** with `required` properties and `init` setters | **Same mmap strategy** for SafeTensors and GGUF. Static `ModelLoader` pattern. `ModelConfig` as class record with `required` properties |
| **SIMD kernels** | `TensorPrimitives` for standard ops, `System.Runtime.Intrinsics` for hot loops. Cross-platform vectors preferred. Scalar fallbacks mandatory. R4 weight repacking at load time | **Same intrinsics strategy.** Same tiered dispatch. Same R4-style repacking for quantized weights |
| **IBackend role** | **Device memory management only** — `AllocateOnDevice`, `CopyBetweenDevices`, `AllReduce`, `Send`, `Receive`. Kernel ops called directly as static methods from model code | **Deliberate divergence:** op-dispatch interface (see rationale below). Each implementation immediately delegates to static kernel methods |
| **Kernel dispatch** | Direct static method calls. `TransformerModel.Forward()` calls `MatMul.GemvQ8_0(...)` directly. Separate CPU and CUDA model classes | SharpInference routes through `IBackend` (one virtual dispatch) which delegates to static kernel methods. One pipeline implementation per model, works on any backend |
| **Attention** | Flash Attention 2/3 (GPU), CPU tiled attention (L2-cache-sized tiles). CPU and GPU implementations intentionally not behind a unified interface | CPU tiled attention adapted for spatial cross-attention. Same Flash Attention PTX for CUDA. Vulkan equivalent via SPIR-V |
| **Quantization** | Dequant for Q4_0, Q4_1, Q5_0, Q5_K, Q6_K, Q8_0, Q4_K. On-the-fly activation quantization (quantize small activations, not large weights). Custom Q8_0 x Q4_K GEMV kernels | Same dequant support. Activation quantization strategy. VAE always FP16, UNet/DiT supports Q8_0 |
| **Kernel fusion** | RMSNorm+Quantize, SwiGLU+L1-tiling, on-the-fly activation quantization. Memory bandwidth is THE bottleneck | GroupNorm+SiLU, Conv2D+bias+activation, fused attention. Bandwidth-first optimization |
| **Adaptive thread pool** | `ComputeThreadPool` with **function-pointer dispatch** (`delegate*<nint, int, int, void>`) for zero-alloc work distribution. SpinWait (~100ns wake) vs EventBased (`ManualResetEventSlim`). Auto-switches based on `seqLen == 1` | Same `ComputeThreadPool` with function-pointer dispatch. SpinWait during denoising steps, EventBased during loading/preprocessing |
| **Options pattern** | `InferenceOptions` class record with **three-tier API**: flat properties (auto-build), explicit `ISamplerStep` list (composable), custom `ILogitProcessor` list (full control) | Same three-tier pattern for pipeline options |
| **Streaming** | `IAsyncEnumerable<GenerationToken>` with readonly record struct, `[EnumeratorCancellation] CancellationToken` | Same: `IAsyncEnumerable<GenerationProgress>` with readonly record struct |
| **Server** | ASP.NET Minimal API. `ServerState` singleton created before DI. Source-generated JSON via `[JsonSerializable]`. One file per endpoint. SSE via `Results.Stream()` | Same Minimal API architecture. `ServerState` singleton. Source-generated JSON |
| **Error handling** | `int` return from CUDA calls. `.ThrowOnError()` looks up error via `cuGetErrorName`/`cuGetErrorString`, throws `CudaException(int, string)`. `Environment.FailFast` for unrecoverable compute thread errors | Same `int` returns, same `.ThrowOnError()` with diagnostic lookup. `Environment.FailFast` for corrupted workers |
| **Build system** | `.slnx` solution format. Central package management via `Directory.Packages.props`. Minimal `.csproj` files | Same: `.slnx`, `Directory.Build.props`, `Directory.Packages.props` |
| **Code standards** | `[MethodImpl(AggressiveInlining)]`, `[SkipLocalsInit]`, `[SuppressGCTransition]`, `Span<T>`, nullable reference types, file-scoped namespaces, `readonly record struct`, `sealed`, no `#region`, XML doc comments | **Identical conventions throughout** |

### Why SharpInference Uses IBackend for Op-Dispatch (Deliberate Divergence)

In dotLLM, `IBackend` is purely a device memory management interface. Model code calls kernel functions directly — `TransformerModel` calls CPU kernels, `CudaTransformerModel` calls CUDA kernels. This works well for LLMs because there is essentially one model architecture pattern (transformer decoder) with two backend variants.

SharpInference deliberately diverges on this point because our problem space is different:

| Factor | dotLLM | SharpInference |
|---|---|---|
| Backends | 2 (CPU, CUDA) | 3 (CPU, CUDA, Vulkan) |
| Model types | ~1 pattern (transformer decoder) | Many (UNet, DiT, VAE, Whisper, Kokoro, YOLO, CLIP, etc.) |
| Duplicated code if direct dispatch | 2 copies per model | 3 copies per model x many models = unmaintainable |

With `IBackend` as op-dispatch, we write each pipeline once and it runs on any backend. The virtual dispatch overhead is negligible — a single `interface` call to `IBackend.Conv2D()` adds ~2ns, while the Conv2D kernel itself takes milliseconds. Each `IBackend` implementation immediately delegates to static kernel methods internally, keeping the same zero-alloc patterns in the actual compute.

### dotLLM's CUDA Kernel Coverage

For reference, dotLLM ships 24 PTX kernels loaded from a `native/ptx/` directory. Each module has multiple function entry points (e.g., `rmsnorm.ptx` contains `rmsnorm_f16`; `embedding.ptx` contains `embedding_lookup_f32`, `embedding_lookup_f16`, `embedding_lookup_q8_0`). Kernel domains: rmsnorm, rope, attention, softmax, swiglu, embedding, dequantization (Q8_0/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K), quantized GEMV, bias_add, type conversion, KV-cache quantization, fused_add_rmsnorm, and per_head_rmsnorm.

### SharpInference's Kernel Coverage (Image/Audio/Vision Domain)

SharpInference's GPU kernels cover a different domain than dotLLM but follow identical PTX/SPIR-V authoring, loading, and caching patterns. Each kernel exists as both a PTX file (CUDA) and a SPIR-V file (Vulkan):

| Kernel | Domain | Why Needed for Image Inference |
|---|---|---|
| `conv2d_f16_3x3` | Diffusion UNet, VAE | Core building block — every ResNet block uses 3x3 convolutions |
| `conv2d_f16_1x1` | Diffusion UNet, projections | Pointwise convolutions for channel mixing, skip connections |
| `group_norm_f16` | Diffusion UNet, VAE | Every ResNet block uses GroupNorm (typically 32 groups) |
| `group_norm_silu_fused` | Diffusion UNet | **Fused GroupNorm+SiLU** — eliminates one full tensor read/write per ResNet block |
| `layer_norm_f16` | Attention, text encoders | Pre-attention normalization in cross-attention blocks |
| `sdpa_f16` | Diffusion attention | Spatial self-attention and text cross-attention. Tiled for O(N) memory |
| `upsample2d_nearest` | UNet decoder, VAE decoder | Spatial upsampling between resolution stages |
| `upsample2d_bilinear` | VAE decoder | Higher-quality upsampling for final image output |
| `elementwise_f16` | Everywhere | Add, mul, scale, residual connections |
| `silu_f16` | UNet ResNet blocks | SiLU activation (x * sigmoid(x)) |
| `gelu_f16` | Text encoders, DiT FFN | GELU activation in transformer feed-forward blocks |
| `dequant_q8` | Model loading | On-the-fly dequantization of Q8_0 weights to FP16 |
| `dequant_q4k` | Model loading | On-the-fly dequantization of Q4_K weights to FP16 |
| `fft_radix2` | Audio (Whisper) | Cooley-Tukey FFT for mel spectrogram computation |
| `mel_filterbank` | Audio (Whisper) | Apply mel filter matrix to FFT magnitude spectrogram |
| `rope_2d` | Flux/SD3 DiT | 2D rotary position embedding for spatial+text tokens |
| `timestep_embed` | UNet conditioning | Sinusoidal timestep embedding + MLP projection |
| `conv2d_bias_silu_fused` | UNet ResNet blocks | **Fused Conv2D+bias+SiLU** — key bandwidth optimization |

**Total: ~18 kernel families** (each with FP16 and FP32 variants, and both PTX and SPIR-V implementations).

### How Vulkan Follows the dotLLM Pattern

dotLLM proved that CUDA can be accessed entirely from C# via P/Invoke to the Driver API + pre-compiled PTX. SharpInference applies the same philosophy to Vulkan:

| dotLLM CUDA Pattern | SharpInference Vulkan Equivalent |
|---|---|
| `[LibraryImport("cuda")]` P/Invoke (~40 functions) | `[LibraryImport("vulkan-1")]` P/Invoke (~40 functions) |
| `.cu` -> `.ptx` via `nvcc -ptx` at build time | `.glsl` compute -> `.spv` via `glslangValidator` at build time |
| `CudaModule.LoadFromFile()` loads PTX from disk, driver JIT-compiles to SASS | `vkCreateShaderModule` loads SPIR-V from disk (driver compiles to target ISA) |
| `cuModuleGetFunction` -> `nint` field | `vkCreateComputePipelines` -> cached pipeline handle |
| `cuLaunchKernel(func, grid, block, args)` | `vkCmdDispatch(commandBuffer, groupCountX/Y/Z)` |
| `cuMemAlloc_v2` / `cuMemFree_v2` for device memory | `vkAllocateMemory` + `vkBindBufferMemory` for device memory |
| `cuMemcpyHtoD` / `cuMemcpyDtoH` for host<->device | Staging buffer + `vkCmdCopyBuffer` for host<->device |
| `cuStreamCreate` for async execution | `VkQueue` + `VkCommandBuffer` for async execution |
| `CudaLibraryResolver` for cross-platform lib names | `VulkanLibraryResolver`: `vulkan-1.dll` (Windows), `libvulkan.so.1` (Linux) |
| `nint` fields for function handles | `Dictionary<string, nint>` for compute pipeline cache |
| `stackalloc void*[]` for kernel arguments | Descriptor sets + push constants for shader arguments |

**Key Vulkan specifics:**
- Vulkan requires explicit memory type selection via `vkGetPhysicalDeviceMemoryProperties` — choose `DEVICE_LOCAL` for tensors, `HOST_VISIBLE | HOST_COHERENT` for staging
- Compute shaders use GLSL `layout(set=N, binding=M)` for buffer bindings — mapped to descriptor sets
- Push constants (up to 128 bytes) used for scalar parameters (stride, padding, epsilon) — equivalent to dotLLM's `stackalloc void*[]` kernel args
- `VkFence` and `VkSemaphore` for GPU synchronization — equivalent to CUDA stream synchronization
- Subgroup operations (`subgroupAdd`, `subgroupShuffle`) replace CUDA warp shuffles for reductions

### Integration Points

SharpInference and dotLLM are designed to interoperate directly in the same process:

- **Prompt enhancement** — dotLLM generates or refines text prompts that feed into SharpInference diffusion pipelines
- **Multimodal pipelines** — vision output from SharpInference (CLIP embeddings, object detection) can be consumed by dotLLM for visual understanding tasks
- **Shared CUDA context** — both libraries P/Invoke the same CUDA Driver API and can share a single CUDA context and device
- **Unified model registry** — model discovery and caching patterns align so that a single local model store serves both projects
- **Common API surface** — dotLLM serves `/v1/chat/completions`, SharpInference serves `/v1/images/generations` and `/v1/audio/*` — compose naturally into a single server

### Design Philosophy Alignment

Both projects share a core conviction: **production AI inference belongs in managed code**. Where the broader ecosystem wraps Python or C++ libraries behind thin interop layers (ONNX Runtime, llama.cpp bindings), dotLLM proved that a pure C# implementation with PTX kernels can achieve 66-88% of llama.cpp decode throughput (with 2-5x slower prefill due to RyuJIT register pressure) while offering superior integration, deployment simplicity, and developer experience. SharpInference extends this approach to diffusion, audio, and vision workloads where the compute profile (large matrix operations, Conv2D) is more amenable to cuBLAS-level performance parity.

Key shared principles:
- **No external processes** — everything runs in-process, no sidecar Python services
- **No native shared libraries** — CUDA accessed exclusively through the Driver API P/Invoke surface; Vulkan through the Vulkan API P/Invoke surface
- **Predictable memory** — 64-byte aligned unmanaged allocations with explicit lifetime, `ArrayPool<T>` for temporaries, never relying on GC for large buffers
- **Zero-GC hot paths** — no managed heap allocations during inference; tensor data entirely in unmanaged memory, metadata as readonly record structs
- **Validate against references** — both projects validate numerical correctness against established Python implementations
- **Process-lifetime caching** — PTX modules, SPIR-V pipelines, function handles, cuBLAS handles, Vulkan descriptor set layouts — all created once and cached forever
- **Don't abstract prematurely** — dotLLM's CPU and CUDA attention implementations are intentionally not behind a unified interface. Apply this lesson: optimize per-backend where the performance gain justifies the complexity

### Licensing Note

dotLLM is licensed under GPLv3. Where functionality overlaps (e.g., GGUF loading, certain kernel implementations), SharpInference uses clean-room implementations to maintain licensing independence. Architectural patterns and design approaches are not subject to copyright — the inspiration is in *how* inference should be structured, not copied code.
