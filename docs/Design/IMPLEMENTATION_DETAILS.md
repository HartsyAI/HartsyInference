# Implementation Details

> Back to [Core Design](CORE_DESIGN.md)

---

## Component Overview

| Component | Approach | From Scratch? | Key Risk |
|---|---|---|---|
| Tensor / TensorShape / TensorRef | Pure C# sealed class + readonly record struct (dotLLM pattern) | Yes — but simple | None |
| DType | Readonly record struct with block-level quantization metadata (dotLLM pattern) | Yes | None |
| SafeTensors Loader | Pure C# mmap + JSON header | Yes — simple format | Multi-shard edge cases |
| GGUF Loader | Clean-room impl (dotLLM is GPLv3) | Yes | GGUF v3 dequant complexity |
| CPU Kernels (matmul, norm) | `System.Runtime.Intrinsics` AVX2/AVX-512/NEON (dotLLM SIMD dispatch) | Yes — follow dotLLM patterns | AVX-512 not universal |
| CPU Conv2D | im2col + SIMD GEMM | Yes | Padding / stride edge cases |
| CPU Audio (FFT, STFT) | Cooley-Tukey radix-2 FFT in C# | Yes | Numerical precision vs reference |
| CUDA Backend | PTX via CUDA Driver API P/Invoke (dotLLM `CudaDriverApi.cs` pattern) | Yes — follow dotLLM pattern | PTX syntax learning curve |
| Vulkan Backend | SPIR-V via Vulkan API P/Invoke (extending dotLLM's approach to Vulkan) | Yes — new territory | Vulkan verbosity, memory management |
| CUDA Conv2D kernel | cuDNN first, custom PTX later | Hybrid | cuDNN version compatibility |
| Vulkan Conv2D kernel | SPIR-V compute shader (im2col + GEMM via subgroup ops) | Yes | Subgroup size varies by vendor |
| cuBLAS HGEMM | P/Invoke to cuBLAS (dotLLM `CublasApi.cs` pattern) | Just bindings | Version pinning |
| CLIP Text Encoder | Pure C# transformer + our kernels | Yes | Tokenizer must match OpenAI exactly |
| T5 Text Encoder | Pure C# encoder-only transformer | Yes | Large model (11GB), needs Q8_0 |
| UNet (SD1.5/SDXL) | Pure C# using our op set | Yes — significant | Cross-attention conditioning correctness |
| DiT / MMDiT (SD3/Flux) | Pure C# using our op set | Yes | Novel architecture |
| VAE Decoder | Pure C# Conv2D + GroupNorm | Yes | Tiled decode seam blending |
| Schedulers | Pure C# math | Yes — well-specified | Floating point reproducibility |
| LoRA loading | Add delta weights to model params | Yes — simple | Flux LoRA format differs |
| ControlNet | Separate UNet injecting residuals | Yes — needs UNet first | Residual injection architecture |
| Whisper STT | Pure C# encoder-decoder transformer | Yes | Autoregressive decode + timestamps |
| Kokoro TTS | Pure C# + HiFiGAN vocoder | Yes | Phoneme encoder complexity |
| STFT / Mel | Pure C# FFT + filterbank | Yes | Must match Whisper exactly |
| CLIP Image Encoder | Pure C# ViT + our kernels | Yes | Patch normalization must be exact |
| YOLO Detection | Pure C# CNN + NMS | Yes | NMS post-processing tuning |
| OpenAI API | Standard ASP.NET minimal API (dotLLM Minimal API pattern) | Straightforward | N/A |

---

## Core — Tensor & Backend

### Dual Tensor Type Pattern (from dotLLM)

Following dotLLM's proven architecture, SharpInference splits tensor concerns into two distinct types:

**`Tensor` (sealed class, implements `IDisposable`)** — Lifecycle and allocation:
- Owns the underlying memory (`NativeMemory.AlignedAlloc` with 64-byte alignment, or pointer into mmap)
- Tracks shape, strides, dtype, device
- Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree`
- Finalizer safety net for forgotten `Dispose()` calls
- Shape stored as fixed-size inline array of up to 6 dimensions, strides precomputed row-major
- Used for model weights, intermediate buffers, and any tensor with explicit lifetime

**`TensorRef` (readonly record struct)** — Zero-alloc compute:
- `readonly record struct TensorRef(nint DataPointer, TensorShape Shape, DType DType, DeviceKind Device)`
- Passed by value on the stack — zero heap allocation
- No ownership, no disposal — just a view into existing memory
- Used in all kernel signatures and hot-path compute
- `[MethodImpl(AggressiveInlining)]` on all accessors
- Created from any `Tensor` via `.AsRef()`

**`DType` (readonly record struct)** — Data type metadata:
- `readonly record struct DType(int SizeInBytes, bool IsQuantized, int BlockByteSize, int BlockElementCount)`
- Pre-defined instances: `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, etc.
- `SizeInBytes = 0` for quantized types — use `(elementCount / BlockElementCount) * BlockByteSize` for byte calculations

**IBackend** is the abstraction all model code programs against (same as dotLLM's `IBackend`). `CpuBackend` is always available. `CudaBackend` and `VulkanBackend` are opt-in. Backend selection is per-tensor (tensors carry their `DeviceKind`) and is resolved at operation time. Cross-device operations automatically insert a `Copy` call. All kernel-level methods accept `TensorRef` parameters (not `Tensor`) — the public `IBackend` methods accept `Tensor` for lifecycle safety and call `.AsRef()` internally before dispatching to kernel functions.

**IBackend operations for image inference (beyond dotLLM's LLM-focused ops):**
```csharp
// dotLLM's IBackend has: MatMul, RmsNorm, Softmax, RoPE, SwiGLU, Embedding
// SharpInference's IBackend adds these for image/audio/vision:
void Conv2D(TensorRef output, TensorRef input, TensorRef weight, TensorRef? bias,
    int strideH, int strideW, int padH, int padW);
void GroupNorm(TensorRef output, TensorRef input, TensorRef weight, TensorRef bias,
    int groups, float eps);
void UpsampleNearest2D(TensorRef output, TensorRef input, int scaleH, int scaleW);
void UpsampleBilinear2D(TensorRef output, TensorRef input, int scaleH, int scaleW);
void Fft(TensorRef output, TensorRef input);
void Stft(TensorRef output, TensorRef input, int fftSize, int hopLength, TensorRef window);
void MelFilterbank(TensorRef output, TensorRef input, TensorRef filters);
```

**No computation graph** (unlike GGML). SharpInference uses eager execution throughout — each op executes immediately when called. This is simpler to implement and debug, and sufficient for inference (no gradient computation). Operator fusion is done manually at the kernel level (e.g., Conv2D + GroupNorm fused kernel).

### Adaptive Thread Pool (from dotLLM)

A custom `ComputeThreadPool` (not `ThreadPool` or `Task.Run`) with two dispatch modes:
- **SpinWait mode** — for latency-critical paths (denoising steps during streaming): near-zero wake latency
- **EventBased mode** — for throughput-oriented paths (model loading, batch preprocessing): no wasted CPU
- The pool switches modes automatically based on operation type

---

## Model Handler — Safetensors

The safetensors format:
1. 8-byte little-endian `uint64` → header byte count
2. That many bytes of UTF-8 JSON mapping tensor name → `{ "dtype", "shape", "data_offsets": [start, end] }`
3. The tensor data blob

**Loading strategy:** `MemoryMappedFile.CreateFromFile` → `MemoryMappedViewAccessor` for the entire file → `JsonSerializer.Deserialize` the header → build index dictionary → return `TensorView` objects pointing into the mmap.

Multi-shard loading reads shards sequentially and builds a unified index with adjusted offsets.

---

## CPU — SIMD Kernels

### Conv2D
Implemented via **im2col transformation**: rearrange input tensor into a matrix where each column is a flattened receptive field, then GEMM against the weight matrix. Converts Conv2D into standard matrix multiply with full SIMD benefit. For 1×1 convolutions (used heavily in UNet), im2col is a no-op — degenerates to direct GEMM.

### GroupNorm
Split channels into groups, compute mean and variance per group, normalize. Inner loops use `TensorPrimitives.Sum` and `TensorPrimitives.Norm` with SIMD dispatch.

### SDPA (Scaled Dot-Product Attention)
Inspired by dotLLM's CPU tiled attention (L2-cache-sized tiles, `IAttentionStrategy` kernel selection) — O(N) memory, ~1KB stack per head. Difference: diffusion attention operates on spatial features (H×W tokens) cross-attending to text embeddings.

### FFT
Cooley-Tukey radix-2 for Whisper's mel spectrogram. Only need power-of-2 sized real FFTs of audio windows (typically 400 or 512 samples). Innermost butterfly loop uses AVX2 complex multiply.

### Weight Repacking at Load Time (from dotLLM)
For quantized weights, repack at model load time for optimal SIMD access. dotLLM's R4 pattern interleaves 4 rows of quantization blocks so sequential memory reads fill `Vector256<T>` registers without gather operations. Apply the same concept for quantized vision/audio model weights — the millisecond cost at load time pays back microseconds on every inference.

### SIMD Dispatch Convention (from dotLLM)
All CPU kernels follow a tiered dispatch: `Vector512.IsHardwareAccelerated` → `Vector256.IsHardwareAccelerated` → scalar fallback. Cross-platform vector types (`Vector256<float>`) preferred over platform-specific intrinsics. Scalar fallback is **mandatory** for every SIMD kernel — no exceptions.

---

## CUDA — PTX Backend (from dotLLM)

Follow dotLLM's proven pattern (`CudaDriverApi.cs` / `CudaModule.cs`) exactly: no native shared library, no DllImport of `libcuda.so`. Instead, `[LibraryImport("nvcuda")]` P/Invoke to the CUDA Driver API (~34 declarations in dotLLM's implementation). CUDA `.cu` source files are compiled to `.ptx` via `nvcc -ptx -arch=compute_80` and shipped as content files alongside .NET assemblies. At runtime, PTX is loaded via `cuModuleLoadData` (JIT-compiled by the GPU driver to native SASS for the actual GPU), function handles retrieved via `cuModuleGetFunction`, and cached in `Dictionary<string, nint>` for process lifetime. Kernel arguments are marshaled to local variables and pointer arrays built on the stack via `stackalloc void*[]`, then launched via `cuLaunchKernel`.

**P/Invoke declarations (from dotLLM):**
```csharp
[LibraryImport("nvcuda")]
internal static partial CuResult cuInit(uint flags);

[LibraryImport("nvcuda")]
internal static partial CuResult cuModuleLoadData(out nint module, nint image);

[LibraryImport("nvcuda")]
internal static partial CuResult cuModuleGetFunction(out nint function, nint module,
    [MarshalAs(UnmanagedType.LPStr)] string name);

[LibraryImport("nvcuda")]
internal static partial CuResult cuLaunchKernel(
    nint function,
    uint gridDimX, uint gridDimY, uint gridDimZ,
    uint blockDimX, uint blockDimY, uint blockDimZ,
    uint sharedMemBytes, nint stream,
    nint kernelParams, nint extra);
```

**P/Invoke conventions (from dotLLM):**
- `[LibraryImport]` (source-generated) over `[DllImport]` — zero-alloc marshaling, trimmer-friendly
- `CuResult` enum with `.ThrowOnError()` extension — every call checked, no silent failures
- `[SuppressGCTransition]` on short CUDA calls (< 1µs) to avoid GC cooperation overhead
- `CudaLibraryResolver` registered via `NativeLibrary.SetDllImportResolver()` for cross-platform resolution (`nvcuda.dll` on Windows, `libcuda.so.1` on Linux)

**Kernel argument marshaling (from dotLLM, zero-alloc via stackalloc):**
```csharp
void LaunchGroupNormSiLU(nint output, nint input, nint weight, nint bias, int channels, int groups, float eps)
{
    void** args = stackalloc void*[7];
    args[0] = &output; args[1] = &input; args[2] = &weight;
    args[3] = &bias; args[4] = &channels; args[5] = &groups; args[6] = &eps;
    CudaDriverApi.cuLaunchKernel(_groupNormSiluFunc, gridDim, 1, 1, blockDim, 1, 1,
        sharedMem, stream, (nint)args, nint.Zero).ThrowOnError();
}
```

**Kernel fusion strategy (from dotLLM):**
- Fuse operations to minimize memory bandwidth (the primary bottleneck in inference)
- Key image inference fusions: **GroupNorm+SiLU** in UNet ResNet blocks (saves one full tensor read/write), **Conv2D+bias+activation** (eliminate intermediate tensor), **fused attention score** (Q×K^T, scale, softmax, ×V in one pass)
- Follow dotLLM's principle: quantize activations (small) rather than dequantize weights (large) where applicable

### Conv2D Strategy for Image Inference
Initial: cuDNN via `[LibraryImport("cudnn")]` P/Invoke for correct results with good performance. Custom PTX Conv2D kernel added in later performance pass once correctness is validated. For 1×1 convolutions (used heavily in UNet projections), im2col is a no-op — degenerates to direct cuBLAS HGEMM.

### cuBLAS (from dotLLM)
```csharp
[LibraryImport("cublas64_12")]
internal static partial CublasStatus cublasGemmEx(
    nint handle, CublasOperation transa, CublasOperation transb,
    int m, int n, int k, nint alpha,
    nint A, CudaDataType Atype, int lda,
    nint B, CudaDataType Btype, int ldb, nint beta,
    nint C, CudaDataType Ctype, int ldc,
    CublasComputeType computeType, CublasGemmAlgo algo);
```
HGEMM handles all large matrix multiplies: UNet projections (linear layers in ResNet and attention blocks), cross-attention QKV projections, T5 encoder layers, CLIP encoder layers, im2col Conv2D. FP16 path with `CUBLAS_COMPUTE_32F` gives ~2x throughput over FP32 and automatically uses Tensor Cores on Ampere+. cuBLAS handle created once per CUDA context, reused for all GEMM calls.

---

## Vulkan — SPIR-V Backend (extending dotLLM's approach)

SharpInference extends dotLLM's pure-C# GPU philosophy to Vulkan, enabling AMD, Intel, and NVIDIA GPU support without CUDA. The pattern is identical in spirit: P/Invoke to the driver API, pre-compiled shader binaries, zero native wrappers.

**P/Invoke declarations (`VulkanApi.cs`, ~40 functions):**
```csharp
[LibraryImport("vulkan-1")]
internal static partial VkResult vkCreateInstance(in VkInstanceCreateInfo createInfo,
    nint allocator, out nint instance);

[LibraryImport("vulkan-1")]
internal static partial VkResult vkCreateShaderModule(nint device,
    in VkShaderModuleCreateInfo createInfo, nint allocator, out nint shaderModule);

[LibraryImport("vulkan-1")]
internal static partial VkResult vkCreateComputePipelines(nint device, nint pipelineCache,
    uint createInfoCount, VkComputePipelineCreateInfo* createInfos, nint allocator, nint* pipelines);

[LibraryImport("vulkan-1")]
internal static partial void vkCmdDispatch(nint commandBuffer,
    uint groupCountX, uint groupCountY, uint groupCountZ);
```

**Cross-platform library resolution:**
```csharp
public sealed class VulkanLibraryResolver
{
    // Resolves "vulkan-1" to:
    //   Windows: vulkan-1.dll (system PATH, typically GPU driver)
    //   Linux:   libvulkan.so.1 (LD_LIBRARY_PATH or /usr/lib)
}
```
Registered via `NativeLibrary.SetDllImportResolver()` at startup — same mechanism as dotLLM's `CudaLibraryResolver`.

**SPIR-V shader management (mirrors PTX pattern):**
1. **Build-time:** `.glsl` compute shaders compiled to `.spv` via `glslangValidator --target-env vulkan1.2 -S comp -o kernel.spv kernel.comp.glsl`
2. **Ship:** `.spv` files as embedded resources or content files (same as PTX)
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
- Device-local memory (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) for tensor storage — equivalent to CUDA `cuMemAlloc_v2`
- Host-visible staging buffers (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`) for host↔device transfers
- Transfer via staging: `memcpy` to staging → `vkCmdCopyBuffer` staging → device — equivalent to CUDA `cuMemcpyHtoD`
- Vulkan Memory Allocator pattern: sub-allocate from large `vkAllocateMemory` blocks to avoid per-tensor allocation overhead (Vulkan limits total allocations to ~4096)

**Vulkan descriptor management:**
- Descriptor set layouts created once per kernel signature (cached for process lifetime)
- Descriptor pool pre-allocated at startup with enough sets for concurrent operations
- Push constants (up to 128 bytes) for scalar kernel parameters (stride, padding, epsilon, scale)
- Storage buffers (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`) for all tensor data

**Key differences from CUDA backend:**
- No equivalent to cuBLAS — matrix multiply implemented as SPIR-V compute shaders using subgroup operations and shared memory tiling
- Subgroup size varies by vendor (32 on NVIDIA, 64 on AMD, 8–32 on Intel) — kernels must handle variable subgroup widths
- Vulkan requires explicit synchronization (fences, semaphores, pipeline barriers) vs CUDA's implicit stream ordering
- No equivalent to `[SuppressGCTransition]` benefit — Vulkan calls are inherently more expensive due to command buffer recording model
- Vulkan compute feature detection via `vkGetPhysicalDeviceFeatures2` (subgroup operations, 16-bit storage, push descriptors)

---

## Diffusion — Pipelines

**Pipeline factory** inspects model metadata (architecture key from safetensors config or GGUF metadata) and instantiates the correct pipeline. All pipelines implement `IAsyncEnumerable<GenerationProgress>` for streaming progress.

### UNet (SD1.5)
4 down-blocks, 1 middle block, 4 up-blocks. Each contains:
- **ResNetBlock:** `GroupNorm → SiLU → Conv2D → GroupNorm → SiLU → Conv2D + residual`
- **CrossAttentionBlock:** `LayerNorm → SDPA(self) → LayerNorm → CrossAttn(text) → LayerNorm → FFN`

**Timestep conditioning:** sinusoidal embedding → MLP → added to each ResNetBlock's hidden state (FiLM-style).

### VAE Tiled Decode
Split latent into overlapping tiles (e.g., 64×64 latent = 512×512 pixels each), decode independently, blend overlapping regions using linear fade mask to eliminate seam artifacts.

### LoRA
LoRA weights are a pair of low-rank matrices (A, B) producing delta `ΔW = B × A × scale`. Added directly to base model weights in-place, or kept as additive correction at forward pass time for multi-LoRA. Flux LoRA uses a different rank decomposition — requires separate handling.

---

## Audio — Whisper

### Preprocessing Pipeline
Raw PCM (16kHz) → 25ms Hann-windowed frames with 10ms hop → FFT → magnitude spectrogram → mel filterbank (80 bins) → log compression → mean-subtract normalization → `[1, 80, T]` tensor.

### Encoder
Two `Conv1D` layers (stride 1 and stride 2) reduce temporal dimension by 2, then sinusoidal positional encoding, then N transformer blocks (MHA + FFN with GELU). No cross-attention in encoder.

### Decoder
Standard autoregressive transformer with cross-attention to encoder outputs. Generates token IDs including special language/task tokens and optionally timestamp tokens. Decode loop uses KV-cache pattern (Whisper is encoder-decoder, not decoder-only).

---

## Server — OpenAI-Compatible API

### Image API
- `POST /v1/images/generations` — JSON body with `prompt`, `model`, `size`, `n`, `response_format`, `seed`, generation params (`steps`, `cfg_scale`, `sampler`). Returns base64 PNG or URL.
- `POST /v1/images/edits` — multipart form with `image`, `mask`, `prompt` for img2img and inpainting.

### Audio API
- `POST /v1/audio/transcriptions` — multipart audio + model. Returns transcript JSON.
- `POST /v1/audio/speech` — `input` text + `voice` + `model`. Returns audio stream.

### Streaming
Image generation uses SSE (`text/event-stream`): `{"step": N, "total": M, "preview": "<base64>"}` per step, final `{"status": "complete", "image": "<base64>"}`. Audio TTS uses chunked transfer encoding for streaming PCM.

### Model Management
- `GET /v1/models` — list loaded models
- `POST /v1/models/load` — trigger model load
- `DELETE /v1/models/{id}` — unload from VRAM
- `POST /v1/models/pull` — trigger HuggingFace download
