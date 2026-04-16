# Implementation Details

> Back to [Core Design](CORE_DESIGN.md)
> Reference: [dotLLM Architecture Research](../Research/DOTLLM_ARCHITECTURE.md) for source-verified patterns

---

## Component Overview

| Component | Approach | From Scratch? | Key Risk |
|---|---|---|---|
| Tensor / TensorShape / TensorRef | Pure C# sealed class + readonly record struct (dotLLM pattern) | Yes -- but simple | None |
| TensorView | Non-owning view, `Dispose()` is no-op (dotLLM `TensorView` pattern) | Yes | None |
| DType | Readonly record struct with Name, block-level quantization metadata (dotLLM pattern) | Yes | None |
| SafeTensors Loader | Pure C# mmap + JSON header | Yes -- simple format | Multi-shard edge cases |
| GGUF Loader | Clean-room impl (dotLLM is GPLv3) | Yes | GGUF v3 dequant complexity |
| CPU Kernels (matmul, norm) | `System.Runtime.Intrinsics` AVX2/AVX-512/NEON (dotLLM SIMD dispatch) | Yes -- follow dotLLM patterns | AVX-512 not universal |
| CPU Conv2D | im2col + SIMD GEMM | Yes | Padding / stride edge cases |
| CPU Audio (FFT, STFT) | Cooley-Tukey radix-2 FFT in C# | Yes | Numerical precision vs reference |
| CUDA Backend | PTX via CUDA Driver API P/Invoke (dotLLM `CudaDriverApi.cs` pattern) | Yes -- follow dotLLM pattern | PTX syntax learning curve |
| Vulkan Backend | SPIR-V via Vulkan API P/Invoke (extending dotLLM's approach to Vulkan) | Yes -- new territory | Vulkan verbosity, memory management |
| CUDA Conv2D kernel | cuDNN first, custom PTX later | Hybrid | cuDNN version compatibility |
| Vulkan Conv2D kernel | SPIR-V compute shader (im2col + GEMM via subgroup ops) | Yes | Subgroup size varies by vendor |
| cuBLAS HGEMM | P/Invoke to cuBLAS (dotLLM `CublasApi.cs` pattern) | Just bindings | Version pinning |
| CLIP Text Encoder | Pure C# transformer + our kernels | Yes | Tokenizer must match OpenAI exactly |
| T5 Text Encoder | Pure C# encoder-only transformer | Yes | Large model (11GB), needs Q8_0 |
| UNet (SD1.5/SDXL) | Pure C# using our op set | Yes -- significant | Cross-attention conditioning correctness |
| DiT / MMDiT (SD3/Flux) | Pure C# using our op set | Yes | Novel architecture |
| VAE Decoder | Pure C# Conv2D + GroupNorm | Yes | Tiled decode seam blending |
| Schedulers | Pure C# math | Yes -- well-specified | Floating point reproducibility |
| LoRA loading | Add delta weights to model params | Yes -- simple | Flux LoRA format differs |
| ControlNet | Separate UNet injecting residuals | Yes -- needs UNet first | Residual injection architecture |
| Whisper STT | Pure C# encoder-decoder transformer | Yes | Autoregressive decode + timestamps |
| Kokoro TTS | Pure C# + HiFiGAN vocoder | Yes | Phoneme encoder complexity |
| STFT / Mel | Pure C# FFT + filterbank | Yes | Must match Whisper exactly |
| CLIP Image Encoder | Pure C# ViT + our kernels | Yes | Patch normalization must be exact |
| YOLO Detection | Pure C# CNN + NMS | Yes | NMS post-processing tuning |
| OpenAI API | ASP.NET Minimal API (dotLLM Minimal API pattern) | Straightforward | N/A |

---

## Core -- Tensor & Backend

### Multi-Type Tensor System (from dotLLM)

Following dotLLM's proven architecture, SharpInference splits tensor concerns into multiple distinct types:

**`Tensor` (sealed class, implements `IDisposable`)** -- Lifecycle and allocation:
- Owns the underlying memory (`NativeMemory.AlignedAlloc` with 64-byte alignment, or pointer into mmap)
- Tracks shape, strides, dtype, device
- Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree`
- Finalizer safety net for forgotten `Dispose()` calls
- Shape stored as fixed-size inline array of up to 6 dimensions, strides precomputed row-major
- Used for model weights, intermediate buffers, and any tensor with explicit lifetime
- GPU variants: `CudaTensor` (wraps `cuMemAlloc_v2`, has `_ownsMemory` flag and `AllocateBytes()` for quantized types), `VulkanTensor` (wraps `vkAllocateMemory` + `vkBindBufferMemory`)

**`TensorView` (sealed class, implements `IDisposable`)** -- Non-owning view (from dotLLM):
- `Dispose()` is a **no-op** -- does not free underlying memory
- Used for borrowed tensor references, weight slices, KV-cache views
- Has full `TensorShape` unlike `TensorRef`
- Allows passing non-owning references through `IDisposable`-expecting APIs safely

**`TensorRef` (readonly record struct)** -- Zero-alloc compute:
- `readonly record struct TensorRef(nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device)`
- Passed by value on the stack -- zero heap allocation
- No ownership, no disposal -- just a view into existing memory
- Used internally in kernel implementations for zero-alloc hot paths
- `[MethodImpl(AggressiveInlining)]` on all accessors
- Created from any `Tensor` via `.AsRef()`

**`TensorMetadata` (readonly record struct)** -- Lightweight description:
- `readonly record struct TensorMetadata(TensorShape Shape, DType DType, int DeviceId, nint DataPointer)`
- Pure value type for passing tensor descriptions without ownership semantics

**`DType` (readonly record struct)** -- Data type metadata:
- `readonly record struct DType(string Name, int SizeInBytes, bool IsQuantized, int BlockByteSize = 0, int BlockElementCount = 1)`
- Pre-defined instances: `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `I8`, `U8`, etc.
- `SizeInBytes = 0` for quantized types -- use `ComputeByteCount(elementCount)` for byte calculations
- `ComputeByteCount()` includes `Debug.Assert(!IsQuantized || elementCount % BlockElementCount == 0)` for alignment validation
- `Name` field enables diagnostic logging and error messages

**IBackend** is the op-dispatch interface all model code programs against (deliberate divergence from dotLLM where `IBackend` is purely device memory management). `CpuBackend` is always available. `CudaBackend` and `VulkanBackend` are opt-in. Backend selection is per-tensor (tensors carry their `DeviceKind`) and is resolved at operation time. Cross-device operations automatically insert a `Copy` call. Each `IBackend` implementation immediately delegates to static kernel methods internally -- the public `IBackend` methods accept `Tensor` for lifecycle safety and the kernel functions use raw pointers/spans internally.

**IBackend operations for image inference (beyond dotLLM's LLM-focused ops):**
```csharp
// SharpInference's IBackend is an op-dispatch interface (deliberate divergence from dotLLM).
// In dotLLM, IBackend has only: AllocateOnDevice, CopyBetweenDevices, AllReduce, Send, Receive.
// We add domain ops so pipeline code can be backend-agnostic:
void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias,
    int strideH, int strideW, int padH, int padW);
void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias,
    int groups, float eps);
void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);
void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW);
void Fft(Tensor output, Tensor input);
void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);
void MelFilterbank(Tensor output, Tensor input, Tensor filters);

// Plus dotLLM's device management methods:
Tensor AllocateOnDevice(DeviceKind device, TensorShape shape, DType dtype);
void CopyTo(Tensor source, Tensor destination);
```

**No computation graph** (unlike GGML). SharpInference uses eager execution throughout -- each op executes immediately when called. This is simpler to implement and debug, and sufficient for inference (no gradient computation). Operator fusion is done manually at the kernel level (e.g., Conv2D + GroupNorm fused kernel).

### Adaptive Thread Pool (from dotLLM)

A custom `ComputeThreadPool` (not `ThreadPool` or `Task.Run`) with two dispatch modes and **function-pointer dispatch** (`delegate*<nint, int, int, void>`) for zero-alloc work distribution:
- **SpinWait mode** -- for latency-critical paths (denoising steps during streaming): near-zero wake latency (~100ns vs ~15us for OS thread wake)
- **EventBased mode** -- for throughput-oriented paths (model loading, batch preprocessing): blocks on `ManualResetEventSlim`, no wasted CPU
- The pool switches modes automatically based on operation type

```csharp
// dotLLM's exact pattern for mode switching:
_threadPool?.SetDispatchMode(isLatencyCritical ? DispatchMode.SpinWait : DispatchMode.EventBased);
```

### ModelConfig Pattern (from dotLLM)

Model configuration uses a **class record** (not struct) with `required` properties:

```csharp
public record ModelConfig
{
    public required string Architecture { get; init; }
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
    // ... additional fields with init-only setters and defaults
}
```

Class record is appropriate because `ModelConfig` is created once and shared -- reference semantics avoid unnecessary copies of large config objects. `required` ensures mandatory fields are always set at construction.

### Pipeline Options Pattern (from dotLLM's InferenceOptions)

Three-tier customization following dotLLM's proven pattern:

```csharp
public record TextToImageOptions
{
    // Tier 1: Flat properties (auto-build pipeline from these)
    public int Steps { get; init; } = 20;
    public float CfgScale { get; init; } = 7.5f;
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;
    public int? Seed { get; init; }

    // Tier 2: Explicit composition (advanced users)
    public IScheduler? Scheduler { get; init; }
    public IReadOnlyList<LoraSpec>? Loras { get; init; }

    // Tier 3: Custom injection (full control)
    public IReadOnlyList<IPipelineCallback>? Callbacks { get; init; }
}
```

---

## Model Handler -- Safetensors

The safetensors format:
1. 8-byte little-endian `uint64` -> header byte count
2. That many bytes of UTF-8 JSON mapping tensor name -> `{ "dtype", "shape", "data_offsets": [start, end] }`
3. The tensor data blob

**Loading strategy:** `MemoryMappedFile.CreateFromFile` -> `MemoryMappedViewAccessor` for the entire file -> `JsonSerializer.Deserialize` the header -> build index dictionary -> return `TensorView` objects pointing into the mmap.

Multi-shard loading reads shards sequentially and builds a unified index with adjusted offsets.

**ModelLoader static helper (from dotLLM):**
```csharp
public static class ModelLoader
{
    public static (IModel Model, ModelConfig Config) LoadFromSafeTensors(
        string path, IBackend backend, ModelConfig? configOverride = null)
    {
        // Load safetensors, extract config, build model, return tuple
    }
}
```

---

## CPU -- SIMD Kernels

### Conv2D
Implemented via **im2col transformation**: rearrange input tensor into a matrix where each column is a flattened receptive field, then GEMM against the weight matrix. Converts Conv2D into standard matrix multiply with full SIMD benefit. For 1x1 convolutions (used heavily in UNet), im2col is a no-op -- degenerates to direct GEMM.

### GroupNorm
Split channels into groups, compute mean and variance per group, normalize. Inner loops use `TensorPrimitives.Sum` and `TensorPrimitives.Norm` with SIMD dispatch.

### SDPA (Scaled Dot-Product Attention)
Inspired by dotLLM's CPU tiled attention (L2-cache-sized tiles, `IAttentionStrategy` kernel selection) -- O(N) memory, ~1KB stack per head. Difference: diffusion attention operates on spatial features (HxW tokens) cross-attending to text embeddings.

### FFT
Cooley-Tukey radix-2 for Whisper's mel spectrogram. Only need power-of-2 sized real FFTs of audio windows (typically 400 or 512 samples). Innermost butterfly loop uses AVX2 complex multiply.

### Weight Repacking at Load Time (from dotLLM)
For quantized weights, repack at model load time for optimal SIMD access. dotLLM's R4 pattern interleaves 4 rows of quantization blocks so sequential memory reads fill `Vector256<T>` registers without gather operations. Apply the same concept for quantized vision/audio model weights -- the millisecond cost at load time pays back microseconds on every inference.

### SIMD Dispatch Convention (from dotLLM)
All CPU kernels follow a tiered dispatch: `Vector512.IsHardwareAccelerated` -> `Vector256.IsHardwareAccelerated` -> scalar fallback. Cross-platform vector types (`Vector256<float>`) preferred over platform-specific intrinsics. Scalar fallback is **mandatory** for every SIMD kernel -- no exceptions.

---

## CUDA -- PTX Backend (from dotLLM)

Follow dotLLM's proven pattern exactly. The library name is `"cuda"` (not `"nvcuda"`) -- resolved at runtime by `CudaLibraryResolver`.

**P/Invoke declarations (matching dotLLM's actual pattern):**
```csharp
private const string LibName = "cuda";

[LibraryImport(LibName)]
internal static partial int cuInit(uint flags);

[LibraryImport(LibName)]
internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

[LibraryImport(LibName)]
internal static partial int cuModuleGetFunction(out nint function, nint module,
    [MarshalAs(UnmanagedType.LPStr)] string name);

[LibraryImport(LibName)]
internal static partial int cuLaunchKernel(
    nint function,
    uint gridDimX, uint gridDimY, uint gridDimZ,
    uint blockDimX, uint blockDimY, uint blockDimZ,
    uint sharedMemBytes, nint stream,
    nint kernelParams, nint extra);
```

**P/Invoke conventions (from dotLLM):**
- `[LibraryImport]` (source-generated) over `[DllImport]` -- zero-alloc marshaling, trimmer-friendly
- Return type is `int` (not an enum) -- `.ThrowOnError()` extension looks up error via `cuGetErrorName`/`cuGetErrorString` and throws `CudaException(int errorCode, string message)`
- `[SuppressGCTransition]` on short CUDA calls (< 1us) to avoid GC cooperation overhead
- `CudaLibraryResolver` registered via `NativeLibrary.SetDllImportResolver()` for cross-platform resolution (`nvcuda.dll` on Windows, `libcuda.so` on Linux)

**PTX loading from directory (from dotLLM -- NOT embedded resources):**
```csharp
// dotLLM loads ALL PTX modules from a directory in the CudaKernels constructor.
// Function handles stored as nint FIELDS, not dictionary-cached.
public sealed class CudaKernels : IDisposable
{
    private readonly nint _conv2dModule;
    private readonly nint _conv2dF16_3x3Func;
    private readonly nint _conv2dF16_1x1Func;
    private readonly nint _groupNormModule;
    private readonly nint _groupNormF16Func;
    private readonly nint _groupNormSiluFusedFunc;
    // ... all 18 kernel families

    public CudaKernels(string ptxDir)
    {
        _conv2dModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "conv2d.ptx"));
        _conv2dF16_3x3Func = _conv2dModule.GetFunction("conv2d_f16_3x3");
        _conv2dF16_1x1Func = _conv2dModule.GetFunction("conv2d_f16_1x1");
        _groupNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "group_norm.ptx"));
        _groupNormF16Func = _groupNormModule.GetFunction("group_norm_f16");
        _groupNormSiluFusedFunc = _groupNormModule.GetFunction("group_norm_silu_fused_f16");
        // ...
    }
}
```

**Kernel argument marshaling (zero-alloc via stackalloc, from dotLLM):**
```csharp
public void LaunchGroupNormSiLU(nint output, nint input, nint weight, nint bias,
    int channels, int groups, float eps, nint stream)
{
    // Local variables for stable addresses (dotLLM pattern)
    nint outputArg = output, inputArg = input, weightArg = weight, biasArg = bias;
    int channelsArg = channels, groupsArg = groups;
    float epsArg = eps;

    void** args = stackalloc void*[] {&outputArg, &inputArg, &weightArg,
        &biasArg, &channelsArg, &groupsArg, &epsArg};
    CudaDriverApi.cuLaunchKernel(_groupNormSiluFusedFunc,
        (uint)groups, 1, 1, 256, 1, 1,
        0, stream, (nint)args, 0).ThrowOnError();
}
```

Note: `BlockSize` is typically 256 (matching dotLLM). Grid dimensions vary per kernel.

**CudaException pattern (from dotLLM):**
```csharp
public sealed class CudaException : Exception
{
    public int ErrorCode { get; }
    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}") { ErrorCode = errorCode; }
}
```

**Kernel fusion strategy (from dotLLM):**
- Fuse operations to minimize memory bandwidth (the primary bottleneck in inference)
- Key image inference fusions: **GroupNorm+SiLU** in UNet ResNet blocks (saves one full tensor read/write), **Conv2D+bias+activation** (eliminate intermediate tensor), **fused attention score** (QxKT, scale, softmax, xV in one pass)
- Follow dotLLM's principle: quantize activations (small) rather than dequantize weights (large) where applicable

### Conv2D Strategy for Image Inference
Initial: cuDNN via `[LibraryImport("cudnn")]` P/Invoke for correct results with good performance. Custom PTX Conv2D kernel added in later performance pass once correctness is validated. For 1x1 convolutions (used heavily in UNet projections), im2col is a no-op -- degenerates to direct cuBLAS HGEMM.

### cuBLAS (from dotLLM)
```csharp
[LibraryImport("cublas64_12")]
internal static partial int cublasCreate_v2(out nint handle);

[LibraryImport("cublas64_12")]
internal static partial int cublasGemmEx(
    nint handle, int transa, int transb,
    int m, int n, int k, nint alpha,
    nint A, int Atype, int lda,
    nint B, int Btype, int ldb, nint beta,
    nint C, int Ctype, int ldc,
    int computeType, int algo);
```
HGEMM handles all large matrix multiplies: UNet projections, cross-attention QKV, T5/CLIP encoder layers, im2col Conv2D. FP16 path with `CUBLAS_COMPUTE_32F` gives ~2x throughput over FP32 and automatically uses Tensor Cores on Ampere+. cuBLAS handle created once per CUDA context, reused for all GEMM calls.

---

## Vulkan -- SPIR-V Backend (extending dotLLM's approach)

SharpInference extends dotLLM's pure-C# GPU philosophy to Vulkan, enabling AMD, Intel, and NVIDIA GPU support without CUDA. The pattern is identical in spirit: P/Invoke to the driver API, pre-compiled shader binaries, zero native wrappers.

**P/Invoke declarations (`VulkanApi.cs`, ~40 functions):**
```csharp
[LibraryImport("vulkan-1")]
internal static partial int vkCreateInstance(in VkInstanceCreateInfo createInfo,
    nint allocator, out nint instance);

[LibraryImport("vulkan-1")]
internal static partial int vkCreateShaderModule(nint device,
    in VkShaderModuleCreateInfo createInfo, nint allocator, out nint shaderModule);

[LibraryImport("vulkan-1")]
internal static partial int vkCreateComputePipelines(nint device, nint pipelineCache,
    uint createInfoCount, VkComputePipelineCreateInfo* createInfos, nint allocator, nint* pipelines);

[LibraryImport("vulkan-1")]
internal static partial void vkCmdDispatch(nint commandBuffer,
    uint groupCountX, uint groupCountY, uint groupCountZ);
```

Note: Vulkan return types are also `int` (matching the dotLLM CUDA pattern), not a `VkResult` enum. Same `.ThrowOnError()` extension pattern.

**Cross-platform library resolution:**
```csharp
public sealed class VulkanLibraryResolver
{
    // Resolves "vulkan-1" to:
    //   Windows: vulkan-1.dll (system PATH, typically GPU driver)
    //   Linux:   libvulkan.so.1 (LD_LIBRARY_PATH or /usr/lib)
}
```
Registered via `NativeLibrary.SetDllImportResolver()` at startup -- same mechanism as dotLLM's `CudaLibraryResolver`.

**SPIR-V shader management (mirrors PTX pattern):**
1. **Build-time:** `.glsl` compute shaders compiled to `.spv` via `glslangValidator --target-env vulkan1.2 -S comp -o kernel.spv kernel.comp.glsl`
2. **Ship:** `.spv` files as content files in a directory (same as PTX)
3. **Runtime load:** `vkCreateShaderModule` loads SPIR-V binary
4. **Pipeline creation:** `vkCreateComputePipelines` wraps shader + pipeline layout (cached in `Dictionary<string, nint>`)
5. **Dispatch:** `vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch(groupX, groupY, groupZ)`

**Vulkan compute shader example (GroupNorm, equivalent to PTX kernel):**
```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer OutputBuf { float16_t output[]; };
layout(set = 0, binding = 1) buffer InputBuf  { float16_t input[];  };
layout(set = 0, binding = 2) buffer WeightBuf { float16_t weight[]; };
layout(set = 0, binding = 3) buffer BiasBuf   { float16_t bias[];   };
layout(push_constant) uniform Params { int channels; int groups; float eps; };

shared float sharedMean;
shared float sharedVar;

void main() {
    // GroupNorm: per-group mean/variance via subgroup reduction
    uint gid = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    int groupSize = channels / groups;
    // ... subgroup reduction for mean, variance, normalize, affine transform
}
```

**Vulkan memory management:**
- `vkGetPhysicalDeviceMemoryProperties` to enumerate memory heaps and types
- Device-local memory (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) for tensor storage -- equivalent to CUDA `cuMemAlloc_v2`
- Host-visible staging buffers (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`) for host<->device transfers
- Transfer via staging: `memcpy` to staging -> `vkCmdCopyBuffer` staging -> device -- equivalent to CUDA `cuMemcpyHtoD`
- Vulkan Memory Allocator pattern: sub-allocate from large `vkAllocateMemory` blocks to avoid per-tensor allocation overhead (Vulkan limits total allocations to ~4096)

**Vulkan descriptor management:**
- Descriptor set layouts created once per kernel signature (cached for process lifetime)
- Descriptor pool pre-allocated at startup with enough sets for concurrent operations
- Push constants (up to 128 bytes) for scalar kernel parameters (stride, padding, epsilon, scale)
- Storage buffers (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`) for all tensor data

**Key differences from CUDA backend:**
- No equivalent to cuBLAS -- matrix multiply implemented as SPIR-V compute shaders using subgroup operations and shared memory tiling
- Subgroup size varies by vendor (32 on NVIDIA, 64 on AMD, 8-32 on Intel) -- kernels must handle variable subgroup widths
- Vulkan requires explicit synchronization (fences, semaphores, pipeline barriers) vs CUDA's implicit stream ordering
- No equivalent to `[SuppressGCTransition]` benefit -- Vulkan calls are inherently more expensive due to command buffer recording model
- Vulkan compute feature detection via `vkGetPhysicalDeviceFeatures2` (subgroup operations, 16-bit storage, push descriptors)

---

## Diffusion -- Pipelines

**Pipeline factory** inspects model metadata (architecture key from safetensors config or GGUF metadata) and instantiates the correct pipeline. All pipelines implement `IAsyncEnumerable<GenerationProgress>` for streaming progress.

### UNet (SD1.5)
4 down-blocks, 1 middle block, 4 up-blocks. Each contains:
- **ResNetBlock:** `GroupNorm -> SiLU -> Conv2D -> GroupNorm -> SiLU -> Conv2D + residual`
- **CrossAttentionBlock:** `LayerNorm -> SDPA(self) -> LayerNorm -> CrossAttn(text) -> LayerNorm -> FFN`

**Timestep conditioning:** sinusoidal embedding -> MLP -> added to each ResNetBlock's hidden state (FiLM-style).

### VAE Tiled Decode
Split latent into overlapping tiles (e.g., 64x64 latent = 512x512 pixels each), decode independently, blend overlapping regions using linear fade mask to eliminate seam artifacts.

### LoRA
LoRA weights are a pair of low-rank matrices (A, B) producing delta `dW = B x A x scale`. Added directly to base model weights in-place, or kept as additive correction at forward pass time for multi-LoRA. Flux LoRA uses a different rank decomposition -- requires separate handling.

---

## Audio -- Whisper

### Preprocessing Pipeline
Raw PCM (16kHz) -> 25ms Hann-windowed frames with 10ms hop -> FFT -> magnitude spectrogram -> mel filterbank (80 bins) -> log compression -> mean-subtract normalization -> `[1, 80, T]` tensor.

### Encoder
Two `Conv1D` layers (stride 1 and stride 2) reduce temporal dimension by 2, then sinusoidal positional encoding, then N transformer blocks (MHA + FFN with GELU). No cross-attention in encoder.

### Decoder
Standard autoregressive transformer with cross-attention to encoder outputs. Generates token IDs including special language/task tokens and optionally timestamp tokens. Decode loop uses KV-cache pattern (Whisper is encoder-decoder, not decoder-only).

---

## Server -- OpenAI-Compatible API

Following dotLLM's Minimal API architecture: `ServerState` singleton, source-generated JSON via `[JsonSerializable]` contexts, one file per endpoint.

### Image API
- `POST /v1/images/generations` -- JSON body with `prompt`, `model`, `size`, `n`, `response_format`, `seed`, generation params (`steps`, `cfg_scale`, `sampler`). Returns base64 PNG or URL.
- `POST /v1/images/edits` -- multipart form with `image`, `mask`, `prompt` for img2img and inpainting.

### Audio API
- `POST /v1/audio/transcriptions` -- multipart audio + model. Returns transcript JSON.
- `POST /v1/audio/speech` -- `input` text + `voice` + `model`. Returns audio stream.

### Streaming
Image generation uses SSE (`text/event-stream`): `{"step": N, "total": M, "preview": "<base64>"}` per step, final `{"status": "complete", "image": "<base64>"}`. Audio TTS uses chunked transfer encoding for streaming PCM.

### Model Management
- `GET /v1/models` -- list loaded models
- `POST /v1/models/load` -- trigger model load
- `DELETE /v1/models/{id}` -- unload from VRAM
- `POST /v1/models/pull` -- trigger HuggingFace download

### DI and JSON Patterns (from dotLLM)

```csharp
// Service registration (from dotLLM's ServiceCollectionExtensions pattern)
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

// Source-generated JSON -- no reflection (from dotLLM)
[JsonSerializable(typeof(ImageGenerationRequest))]
[JsonSerializable(typeof(ImageGenerationResponse))]
[JsonSerializable(typeof(AudioTranscriptionResponse))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(ErrorResponse))]
internal partial class SharpInferenceJsonContext : JsonSerializerContext { }
```
