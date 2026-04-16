# SharpInference — Core Design Overview

**Version 1.0 | April 2026**

---

## What Is SharpInference?

SharpInference is a **native C#/.NET 10 AI inference engine** for all non-LLM inference modalities — image generation (diffusion), speech-to-text, text-to-speech, voice conversion, image understanding, object detection, and video generation. It works closely alongside **dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) to form a complete, self-contained AI platform with zero Python dependencies, zero C++ wrappers, and no external processes. SharpInference draws significant architectural inspiration from dotLLM — adopting its approach to CUDA access via PTX and the Driver API, unmanaged tensor memory with 64-byte aligned allocations, `IBackend` abstraction, and SIMD kernel dispatch — and extends these proven patterns to non-LLM inference workloads.

---

## Design Pillars

| Pillar | What It Means |
|---|---|
| **Pure C#** | No native shared libraries required; CUDA accessed via PTX through the CUDA Driver API P/Invoke; Vulkan accessed via SPIR-V through the Vulkan API P/Invoke — same pure-C# philosophy for both GPU backends |
| **Zero-allocation hot paths** | All tensor data in `NativeMemory.AlignedAlloc`; model weights memory-mapped; `TensorRef` readonly record struct on all kernel signatures; `Span<T>` on hot paths |
| **Modular NuGet packages** | Pull in only what you need; no bloated single-assembly distribution |
| **Multi-GPU backend** | CUDA (NVIDIA via PTX + cuBLAS) and Vulkan (AMD/Intel/NVIDIA via SPIR-V + compute shaders) — both following dotLLM's P/Invoke-to-driver-API pattern |
| **OpenAI-compatible API** | Drop-in replacement for OpenAI image and audio endpoints |
| **dotLLM alignment** | Directly follows dotLLM's proven inference architecture; same dual tensor types, same P/Invoke patterns, same PTX management, same SIMD dispatch, same memory model — extended to non-LLM modalities |
| **Production-grade** | Proper error handling, streaming progress, memory budgeting, VRAM monitoring, model hot-swap |

---

## Design Documents Index

Each section of the design is broken out into its own document for focused reference:

| Document | Description |
|---|---|
| [Vision & Goals](VISION_AND_GOALS.md) | Core motivation, the SwarmUI backend angle, and why this project exists |
| [Features](FEATURES.md) | Complete features list across infrastructure, image, audio, vision, and server |
| [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) | Phase 1–3 model support plan with formats and priorities |
| [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependency graph, and minimum install examples |
| [File Structure](FILE_STRUCTURE.md) | Full project file and folder layout |
| [Implementation Details](IMPLEMENTATION_DETAILS.md) | Per-component technical approach, architecture decisions, and key algorithms |
| [Build Order & Phases](BUILD_ORDER.md) | Implementation sequencing and phase dependencies |
| [Validation Strategy](VALIDATION_STRATEGY.md) | Reference implementations and validation methods for every component |
| [Research Requirements](RESEARCH_REQUIREMENTS.md) | Research documents needed before implementation, organized by component |

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                     SharpInference.Server                     │
│            (OpenAI-compatible REST API + SSE)                 │
├────────────┬──────────────┬──────────────────────────────────┤
│  Diffusion │    Audio     │           Vision                 │
│  SD/SDXL   │  Whisper STT │  CLIP Embeddings                 │
│  Flux/SD3  │  Kokoro TTS  │  YOLO Detection                  │
│  LoRA/CN   │  Voice Conv  │  SAM Segmentation                │
├────────────┴──────────────┴──────────────────────────────────┤
│                  SharpInference.Core                          │
│    Tensor + TensorRef · IBackend · Schedulers · Pipelines    │
├──────────────┬──────────────────────┬────────────────────────┤
│ CPU Backend  │    CUDA Backend      │   Vulkan Backend       │
│ AVX2/512/NEON│ PTX Kernels + cuBLAS │ SPIR-V + VkCompute     │
├──────────────┴──────────────────────┴────────────────────────┤
│                     Model Handler                             │
│        Safetensors · GGUF · HuggingFace · Registry            │
└──────────────────────────────────────────────────────────────┘
```

**The three backends follow the same pattern:** model code programs against `IBackend` and never references any backend directly. CPU dispatches to SIMD kernels. CUDA dispatches to PTX kernels + cuBLAS. Vulkan dispatches to SPIR-V compute shaders. The backend is selected at runtime based on hardware detection.

---

## Key Architectural Decisions

- **Eager execution** — no computation graph. Each op executes immediately. Simpler to implement and debug; sufficient for inference (no gradients needed). Fusion is done manually at the kernel level. This is how dotLLM works.
- **Dual tensor types (from dotLLM)** — `Tensor` (sealed class, `IDisposable`) owns memory and manages lifecycle. `TensorRef` (readonly record struct) is a zero-alloc view used in all kernel signatures and hot paths. This separation means the compute path never allocates, never boxes, and never touches the GC.
- **Unmanaged memory** — all tensor storage via `NativeMemory.AlignedAlloc(byteCount, 64)` with 64-byte alignment, or memory-mapped files. Zero GC pressure on the inference hot path. Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree`. Finalizer safety net for forgotten `Dispose()` calls. All of this follows dotLLM's `UnmanagedTensor` pattern exactly.
- **PTX over CUDA C (from dotLLM)** — CUDA `.cu` source files are compiled to `.ptx` via `nvcc -ptx` and shipped as content files alongside the .NET assemblies. At runtime, PTX is loaded via `cuModuleLoadData` and JIT-compiled by the GPU driver to native SASS — no native shared libraries to ship. This pattern was proven by dotLLM's `CudaModule.cs` and is adopted directly here.
- **SPIR-V for Vulkan (extending dotLLM's approach)** — the same philosophy applied to Vulkan: `.glsl` compute shaders are compiled to `.spv` via `glslangValidator` and shipped as content files. At runtime, SPIR-V is loaded via `vkCreateShaderModule`, compute pipelines created via `vkCreateComputePipelines`, and dispatched via `vkCmdDispatch`. All Vulkan API access is through `[LibraryImport("vulkan-1")]` P/Invoke — the same pure-C# pattern dotLLM proved with CUDA.
- **IBackend abstraction** — all model code programs against `IBackend`. CPU, CUDA, and Vulkan are swappable without changing model logic. Cross-device ops insert automatic copies.
- **Pipeline factory** — model metadata (architecture key) drives automatic pipeline selection. Users load a model file and get the right pipeline without manual configuration.

---

## Relationship to dotLLM

SharpInference is designed to work **closely alongside dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) and draws heavy inspiration from its architecture and inference approach. dotLLM is a ground-up LLM inference engine written entirely in C# by Konrad Kokosa — explicitly not a wrapper around llama.cpp or Python — supporting Llama, Mistral, Phi, Qwen, and DeepSeek transformer architectures on .NET 10. Together, the two projects form a complete, pure C#/.NET AI platform — dotLLM handles LLM text generation while SharpInference covers every other modality.

### How dotLLM Inspires SharpInference

dotLLM pioneered many of the patterns that SharpInference adopts and extends for non-LLM inference:

| Pattern | How dotLLM Does It | How SharpInference Adopts It |
|---|---|---|
| **CUDA access** | `.cu` files compiled to `.ptx` via `nvcc -ptx`, shipped as content files. `CudaDriverApi.cs` contains ~34 `[LibraryImport("nvcuda")]` P/Invoke declarations. PTX loaded via `cuModuleLoadData`, function handles cached via `cuModuleGetFunction`, kernels launched via `cuLaunchKernel` with args marshaled on the stack via `stackalloc` | **Identical approach:** pre-compiled PTX content files, `cuModuleLoadData` loading, JIT-compiled by the GPU driver to native SASS, function handles cached in `Dictionary<string, nint>` for process lifetime. Same `CudaLibraryResolver` registered via `NativeLibrary.SetDllImportResolver()` for cross-platform resolution |
| **Vulkan (SharpInference extension)** | dotLLM does not implement Vulkan — CUDA-only | SharpInference extends dotLLM's P/Invoke-to-driver-API philosophy to Vulkan: `VulkanApi.cs` with `[LibraryImport("vulkan-1")]` P/Invoke declarations (~40 functions). `.glsl` compute shaders compiled to `.spv` via `glslangValidator`, shipped as content files. SPIR-V loaded via `vkCreateShaderModule`, compute pipelines built via `vkCreateComputePipelines`, dispatched via `vkCmdDispatch`. Same pattern: no native wrappers, no managed Vulkan libraries — pure P/Invoke to the Vulkan loader |
| **cuBLAS** | Separate `CublasApi.cs` P/Invoke surface (~6 functions) for FP16 GEMM with automatic Tensor Core usage. `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for FP16-in/FP32-accumulate | **Same cuBLAS binding strategy.** `cublasGemmEx` for FP16/FP32 GEMM in convolution (im2col + GEMM), attention (QK^T and AV), and text encoder projections. cuBLAS handle created once per CUDA context, cached for process lifetime |
| **Dual tensor types** | `ITensor` (interface, `IDisposable`) for lifecycle. `TensorRef` (readonly record struct: `nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device`) for zero-alloc compute in all kernel signatures | **Identical pattern:** `Tensor` (sealed class, `IDisposable`) for lifecycle. `TensorRef` (readonly record struct) for all kernel signatures. Kernels never see `Tensor` — only `TensorRef`. This means the hot path never allocates, never boxes, never touches the GC |
| **Tensor memory** | `UnmanagedTensor` uses `NativeMemory.AlignedAlloc(byteCount, 64)` with 64-byte alignment. Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree`. Finalizer safety net. `CudaTensor` wraps device pointers from `cuMemAlloc_v2`. `ArrayPool<T>` for temporary buffers | **Same 64-byte aligned unmanaged allocations**, same `Interlocked.Exchange` disposal pattern, same finalizer safety net. `CudaTensor` wraps `cuMemAlloc_v2` device pointers. `VulkanTensor` wraps `vkAllocateMemory` + `vkBindBufferMemory` device pointers. `ArrayPool<T>` for non-tensor temporaries only |
| **Model loading** | GGUF via `MemoryMappedFile.CreateFromFile()` with OS demand-paging — multi-GB models load in milliseconds. Tensor descriptors provide name, shape, quantization type, and byte offset | **Same mmap loading strategy** for both SafeTensors and GGUF. Clean-room GGUF implementation to maintain licensing independence. For GPU: weights paged in on-demand and copied to VRAM via `cuMemcpyHtoD` (CUDA) or staging buffer + `vkCmdCopyBuffer` (Vulkan) |
| **SIMD kernels** | `System.Numerics.Tensors.TensorPrimitives` for standard ops, `System.Runtime.Intrinsics` (`Vector128<T>`/`Vector256<T>`/`Vector512<T>`) for hot loops. Cross-platform vectors preferred over platform-specific; scalar fallbacks mandatory for every kernel | **Same intrinsics strategy:** `TensorPrimitives` where applicable, `Vector256<T>` for inner loops (Conv2D, GroupNorm, FFT), scalar fallbacks for all paths. Same tiered dispatch: `Vector512.IsHardwareAccelerated` → `Vector256.IsHardwareAccelerated` → scalar. Same `[MethodImpl(AggressiveInlining)]` on all SIMD helper methods |
| **Attention** | Flash Attention 2 (SM80+ Ampere), Flash Attention 3 (SM90+ Hopper), CPU tiled attention (L2-cache-sized tiles), naive fallback. CPU and GPU have separate optimized implementations intentionally not behind a unified interface | CPU tiled attention adapted for spatial cross-attention in diffusion (H×W tokens cross-attending to text embeddings). Same Flash Attention PTX for CUDA. Vulkan equivalent via SPIR-V compute shader with shared memory tiling |
| **Backend abstraction** | `IBackend` in `DotLLM.Core` with `DotLLM.Cpu` and `DotLLM.Cuda` as separate NuGet packages | `IBackend` in `SharpInference.Core` with `SharpInference.Cpu`, `SharpInference.Cuda`, and `SharpInference.Vulkan` as separate NuGet packages. Same package-per-backend design |
| **Quantization** | Dequant kernels for Q4_0, Q5_0, Q5_K, Q6_K, Q8_0, Q4_K. GPU does on-the-fly dequantization into a reusable scratch buffer before cuBLAS ops. Custom quantized GEMV kernels for decode | Same dequant format support (Q4_0, Q8_0, Q4_K_M). On-the-fly dequant into scratch buffers before cuBLAS/Vulkan GEMM. For image inference: VAE weights always FP16 (never quantized), UNet/DiT backbone supports Q8_0 |
| **Kernel fusion** | Fuses operations to minimize memory bandwidth: RMSNorm+Quantize, SwiGLU+L1-tiling, on-the-fly activation quantization | Same fusion philosophy: GroupNorm+SiLU in UNet blocks, Conv2D+bias+activation, fused attention score computation. Memory bandwidth is the bottleneck — every kernel targets bandwidth reduction |
| **Adaptive thread pool** | `ComputeThreadPool` with SpinWait (latency-critical single-token decode) and EventBased (throughput-oriented prefill) modes, switching automatically | Same `ComputeThreadPool` pattern: SpinWait during denoising steps (latency between steps matters for streaming), EventBased during model loading and preprocessing |
| **Package structure** | Layered NuGet packages: `DotLLM.Core` → `DotLLM.Cpu`/`DotLLM.Cuda` → `DotLLM.Models` → `DotLLM.Engine` → `DotLLM.Server` | Mirror structure: `SharpInference.Core` → `SharpInference.Cpu`/`SharpInference.Cuda`/`SharpInference.Vulkan` → `SharpInference.Diffusion`/`Audio`/`Vision` → `SharpInference.Server` |
| **Streaming** | `IAsyncEnumerable<GenerationToken>` where `GenerationToken` is a readonly record struct — zero allocation per yield. `CancellationToken` for abort | Same pattern: `IAsyncEnumerable<GenerationProgress>` where `GenerationProgress` is a readonly record struct. Step count, optional preview image, timing info |
| **Error handling** | `CuResult` enum with `.ThrowOnError()` extension. `Environment.FailFast` for unrecoverable compute thread errors. Shape validation at operation boundaries | Same patterns: `.ThrowOnError()` on every CUDA/Vulkan call. `Environment.FailFast` for corrupted worker threads. Custom exceptions: `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException` |
| **Code standards** | `[MethodImpl(AggressiveInlining)]` on hot paths, `[SkipLocalsInit]`, `[SuppressGCTransition]` for short CUDA calls, `Span<T>` in signatures, nullable reference types, file-scoped namespaces, `readonly record struct` for value types, `sealed` on all non-inheritable classes | **Identical conventions throughout** |

### dotLLM's CUDA Kernel Coverage

For reference, dotLLM ships 24 PTX kernels covering: rmsnorm, rope, attention, softmax, swiglu, embedding, dequantization (Q8_0/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K), quantized GEMV, bias_add, type conversion, KV-cache quantization, fused_add_rmsnorm, and per_head_rmsnorm — each with FP16 and FP32 variants.

### SharpInference's Kernel Coverage (Image/Audio/Vision Domain)

SharpInference's GPU kernels cover a different domain than dotLLM but follow identical PTX/SPIR-V authoring, loading, and caching patterns. Each kernel exists as both a PTX file (CUDA) and a SPIR-V file (Vulkan):

| Kernel | Domain | Why Needed for Image Inference |
|---|---|---|
| `conv2d_f16_3x3` | Diffusion UNet, VAE | Core building block — every ResNet block uses 3×3 convolutions |
| `conv2d_f16_1x1` | Diffusion UNet, projections | Pointwise convolutions for channel mixing, skip connections |
| `group_norm_f16` | Diffusion UNet, VAE | Every ResNet block uses GroupNorm (typically 32 groups) |
| `group_norm_silu_fused` | Diffusion UNet | **Fused GroupNorm+SiLU** — eliminates one full tensor read/write per ResNet block |
| `layer_norm_f16` | Attention, text encoders | Pre-attention normalization in cross-attention blocks |
| `sdpa_f16` | Diffusion attention | Spatial self-attention and text cross-attention. Tiled for O(N) memory |
| `upsample2d_nearest` | UNet decoder, VAE decoder | Spatial upsampling between resolution stages |
| `upsample2d_bilinear` | VAE decoder | Higher-quality upsampling for final image output |
| `elementwise_f16` | Everywhere | Add, mul, scale, residual connections |
| `silu_f16` | UNet ResNet blocks | SiLU activation (x × sigmoid(x)) |
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
| `[LibraryImport("nvcuda")]` P/Invoke (~34 functions) | `[LibraryImport("vulkan-1")]` P/Invoke (~40 functions) |
| `.cu` → `.ptx` via `nvcc -ptx` at build time | `.glsl` compute → `.spv` via `glslangValidator` at build time |
| `cuModuleLoadData` loads PTX, driver JIT-compiles to SASS | `vkCreateShaderModule` loads SPIR-V (already compiled to target ISA by driver) |
| `cuModuleGetFunction` retrieves kernel handle | `vkCreateComputePipelines` creates compute pipeline |
| `cuLaunchKernel(func, grid, block, args)` | `vkCmdDispatch(commandBuffer, groupCountX/Y/Z)` |
| `cuMemAlloc_v2` / `cuMemFree_v2` for device memory | `vkAllocateMemory` + `vkBindBufferMemory` for device memory |
| `cuMemcpyHtoD` / `cuMemcpyDtoH` for host↔device | Staging buffer + `vkCmdCopyBuffer` for host↔device |
| `cuStreamCreate` for async execution | `VkQueue` + `VkCommandBuffer` for async execution |
| `CudaLibraryResolver` for cross-platform lib names | `VulkanLibraryResolver`: `vulkan-1.dll` (Windows), `libvulkan.so.1` (Linux) |
| `Dictionary<string, nint>` function handle cache | `Dictionary<string, VkPipeline>` compute pipeline cache |
| `stackalloc void*[]` for kernel arguments | Descriptor sets + push constants for shader arguments |

**Key Vulkan specifics:**
- Vulkan requires explicit memory type selection via `vkGetPhysicalDeviceMemoryProperties` — choose `DEVICE_LOCAL` for tensors, `HOST_VISIBLE | HOST_COHERENT` for staging
- Compute shaders use GLSL `layout(set=N, binding=M)` for buffer bindings — mapped to descriptor sets
- Push constants (up to 128 bytes) used for scalar parameters (stride, padding, epsilon) — equivalent to dotLLM's `stackalloc void*[]` kernel args
- `VkFence` and `VkSemaphore` for GPU synchronization — equivalent to CUDA stream synchronization
- Subgroup operations (`subgroupAdd`, `subgroupShuffle`) replace CUDA warp shuffles for reductions

### Integration Points

SharpInference and dotLLM are designed to interoperate directly in the same process:

- **Prompt enhancement** — dotLLM generates or refines text prompts that feed into SharpInference diffusion pipelines (e.g., LLM-powered prompt expansion for image generation)
- **Multimodal pipelines** — vision output from SharpInference (CLIP embeddings, object detection results) can be consumed by dotLLM for visual understanding tasks
- **Shared CUDA context** — both libraries P/Invoke the same CUDA Driver API and can share a single CUDA context and device, avoiding redundant GPU initialization and enabling efficient VRAM coordination
- **Unified model registry** — model discovery and caching patterns align so that a single local model store serves both projects
- **Common API surface** — dotLLM serves `/v1/chat/completions`, SharpInference serves `/v1/images/generations` and `/v1/audio/*` — the OpenAI-compatible endpoints compose naturally into a single server

### Design Philosophy Alignment

Both projects share a core conviction: **production AI inference belongs in managed code**. Where the broader ecosystem wraps Python or C++ libraries behind thin interop layers (ONNX Runtime, llama.cpp bindings), dotLLM proved that a pure C# implementation with PTX kernels can achieve ~98–100% of native CUDA performance while offering superior integration, deployment simplicity, and developer experience. SharpInference extends this proof to diffusion, audio, and vision workloads.

Key shared principles:
- **No external processes** — everything runs in-process, no sidecar Python services
- **No native shared libraries** — CUDA is accessed exclusively through the Driver API P/Invoke surface; Vulkan is accessed exclusively through the Vulkan API P/Invoke surface. dotLLM evaluated and rejected ILGPU (lacks Tensor Core access, no bfloat16), ManagedCuda (GPLv3 license conflict), ComputeSharp (Windows-only DirectX), and Silk.NET (no CUDA support). SharpInference additionally evaluated and rejected Vortice.Vulkan (unnecessary abstraction layer) and Evergine (limited compute shader support)
- **Predictable memory** — 64-byte aligned unmanaged allocations with explicit lifetime, `ArrayPool<T>` for temporaries, never relying on GC for large buffers
- **Zero-GC hot paths** — no managed heap allocations during inference; tensor data entirely in unmanaged memory, metadata as readonly record structs (`TensorRef`, `DType`, `TensorShape`)
- **Validate against references** — both projects validate numerical correctness against established Python implementations
- **Process-lifetime caching** — PTX modules, SPIR-V pipelines, function handles, cuBLAS handles, Vulkan descriptor set layouts — all created once and cached forever. No repeated initialization in hot paths

### Licensing Note

dotLLM is licensed under GPLv3. Where functionality overlaps (e.g., GGUF loading, certain kernel implementations), SharpInference uses clean-room implementations to maintain licensing independence. Architectural patterns and design approaches are not subject to copyright — the inspiration is in *how* inference should be structured, not copied code.
