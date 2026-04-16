# dotLLM Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: All SharpInference packages (reference architecture)

## Summary

dotLLM ([github.com/kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) is a ground-up, pure C#/.NET 10 LLM inference engine created by Konrad Kokosa. It supports Llama, Mistral, Phi, Qwen, and DeepSeek architectures with both CPU (AVX2/AVX-512) and CUDA (PTX + cuBLAS) backends. dotLLM proves that production AI inference can be done entirely in managed code — no Python, no C++ wrappers, no ONNX Runtime — while achieving ~98–100% of native CUDA performance. SharpInference is designed to follow dotLLM's proven patterns and extend them to non-LLM inference modalities (diffusion, audio, vision).

This document catalogs dotLLM's architecture, implementation patterns, and design decisions so that SharpInference can adopt the same approaches consistently. Where dotLLM has solved a problem (e.g., PTX loading, tensor memory, SIMD dispatch), SharpInference should follow the same solution rather than inventing a new one.

Sources: [dotLLM GitHub repository](https://github.com/kkokosa/dotLLM), [Konrad Kokosa's .NET blog](https://prodotnetmemory.com/), [dotLLM README](https://github.com/kkokosa/dotLLM/blob/main/README.md)

## Detailed Findings

### Project Structure and NuGet Layering

dotLLM uses a layered NuGet package architecture with strict dependency direction — higher layers depend on lower layers, never the reverse:

```
DotLLM.Server          ← ASP.NET Minimal API (OpenAI-compatible endpoints)
    ↓
DotLLM.Engine          ← Orchestration, pipeline management, model loading
    ↓
DotLLM.Models          ← Architecture-specific model implementations (Llama, Phi, etc.)
    ↓
DotLLM.Cpu / DotLLM.Cuda  ← Backend implementations (IBackend)
    ↓
DotLLM.Core            ← Tensor, DType, IBackend interface, base abstractions
    ↓
DotLLM.Tokenizers      ← BPE/SentencePiece tokenization
DotLLM.HuggingFace     ← Model download and config parsing
DotLLM.Diagnostics     ← Logging and tracing
DotLLM.Telemetry       ← Performance counters
```

**Key design rule:** Each package has a single responsibility. Model code never references CPU or CUDA directly — it programs against `IBackend`. The backend packages are selected at runtime.

Source: [dotLLM solution structure](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Mirror this exact layering: `SharpInference.Core` → `SharpInference.Cpu`/`SharpInference.Cuda` → `SharpInference.Diffusion`/`Audio`/`Vision` → `SharpInference.Server`.

### Dual Tensor Types: ITensor + TensorRef

dotLLM splits tensor concerns into two distinct types — this is one of its most important architectural patterns:

**`ITensor` (interface)** — Lifecycle and allocation:
- Implements `IDisposable` for deterministic cleanup
- Owns the underlying memory (unmanaged or device)
- Tracks shape, strides, dtype, device
- Used for model weights, KV-cache, and any tensor that outlives a single operation
- Concrete implementations: `UnmanagedTensor` (CPU), `CudaTensor` (GPU)

**`TensorRef` (readonly record struct)** — Zero-alloc compute:
```csharp
public readonly record struct TensorRef(
    nint DataPointer,
    TensorShape Shape,
    DType DType,
    DeviceKind Device);
```
- Passed by value on the stack — zero heap allocation
- No ownership, no disposal — just a view into existing memory
- Used in all kernel signatures and hot-path compute
- `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on all accessors
- Can be created from any `ITensor` via `.AsRef()` or similar

**Why this matters:** Kernel functions accept `TensorRef` parameters, which means the hot path never allocates, never boxes, and never touches the GC. The `ITensor` interface handles the "business logic" of tensor lifecycle (allocation, disposal, device transfers) while `TensorRef` handles the "math" side.

Source: [dotLLM tensor implementation](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Implement the same dual-type pattern. `Tensor` class (or `ITensor`) for lifecycle, `TensorRef` readonly record struct for all kernel signatures and compute paths.

### DType as Readonly Record Struct

dotLLM represents data types as a `readonly record struct` with rich metadata:

```csharp
public readonly record struct DType(
    int SizeInBytes,
    bool IsQuantized,
    int BlockByteSize,
    int BlockElementCount)
{
    public static readonly DType F32 = new(4, false, 4, 1);
    public static readonly DType F16 = new(2, false, 2, 1);
    public static readonly DType BF16 = new(2, false, 2, 1);
    public static readonly DType Q4_0 = new(0, true, 18, 32);  // 18 bytes per 32 elements
    public static readonly DType Q4_1 = new(0, true, 20, 32);
    public static readonly DType Q8_0 = new(0, true, 34, 32);  // 34 bytes per 32 elements
    public static readonly DType Q4_K = new(0, true, 144, 256);
    public static readonly DType Q5_K = new(0, true, 176, 256);
    public static readonly DType Q6_K = new(0, true, 210, 256);
}
```

**Why `SizeInBytes = 0` for quantized types:** Quantized formats pack multiple elements into blocks — there is no meaningful "per-element" byte size. Byte calculation must use `BlockByteSize` and `BlockElementCount` instead: `totalBytes = (elementCount / BlockElementCount) * BlockByteSize`.

Source: [dotLLM DType definitions](https://github.com/kkokosa/dotLLM), [GGML quantization formats](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

**SharpInference adoption:** Use the same `DType` readonly record struct pattern with identical field semantics. Add additional dtypes as needed for vision/audio (e.g., INT8, UINT8 for image pixels).

### Unmanaged Memory Management

All tensor data lives in unmanaged memory — never on the managed heap:

**Allocation:**
```csharp
nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
```
- 64-byte alignment ensures optimal cache line usage and SIMD vector alignment
- `NativeMemory.AlignedAlloc` maps to `_aligned_malloc` (Windows) or `aligned_alloc` (Linux)
- Returns `void*`, cast to `nint` for storage

**Thread-safe disposal:**
```csharp
public void Dispose()
{
    nint ptr = Interlocked.Exchange(ref _pointer, 0);
    if (ptr != 0)
        NativeMemory.AlignedFree((void*)ptr);
    GC.SuppressFinalize(this);
}

~UnmanagedTensor()
{
    nint ptr = Interlocked.Exchange(ref _pointer, 0);
    if (ptr != 0)
        NativeMemory.AlignedFree((void*)ptr);
}
```

**Why `Interlocked.Exchange`:** Prevents double-free in concurrent scenarios. The exchange is atomic — only one thread will see a non-zero pointer and proceed to free it. The finalizer acts as a safety net for forgotten `Dispose()` calls.

**Temporary buffers:** `ArrayPool<T>.Shared` for short-lived managed buffers (metadata parsing, string building) — never for tensor data.

Source: [dotLLM UnmanagedTensor](https://github.com/kkokosa/dotLLM), [NativeMemory docs](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativememory)

**SharpInference adoption:** Same 64-byte aligned allocation, same `Interlocked.Exchange` disposal pattern, same finalizer safety net. Use `ArrayPool<T>` only for non-tensor temporaries.

### CUDA Integration via P/Invoke

dotLLM accesses CUDA entirely through the Driver API — no Runtime API, no managed CUDA wrappers:

**P/Invoke declarations (~34 functions in `CudaDriverApi.cs`):**
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

**Key decisions:**
- `[LibraryImport]` (source-generated) over `[DllImport]` — zero-alloc marshaling, trimmer-friendly
- `CuResult` enum with `.ThrowOnError()` extension method — every call is checked
- `[SuppressGCTransition]` on short CUDA calls (< 1µs) to avoid GC cooperation overhead

**Cross-platform library resolution:**
```csharp
public sealed class CudaLibraryResolver
{
    // Resolves "nvcuda" to:
    //   Windows: nvcuda.dll (system PATH)
    //   Linux:   libcuda.so.1 (LD_LIBRARY_PATH or /usr/lib)
    // Resolves "cublas64_12" to:
    //   Windows: cublas64_12.dll
    //   Linux:   libcublas.so.12
}
```
Registered via `NativeLibrary.SetDllImportResolver()` at startup.

Source: [dotLLM CudaDriverApi.cs](https://github.com/kkokosa/dotLLM), [CUDA Driver API docs](https://docs.nvidia.com/cuda/cuda-driver-api/)

**SharpInference adoption:** Same P/Invoke surface, same `CudaLibraryResolver`, same error-checking pattern. Add additional Driver API functions as needed for SharpInference-specific operations.

### PTX Kernel Management

PTX files are pre-compiled from `.cu` sources and shipped as embedded resources or content files:

**Compilation (build-time):**
```bash
nvcc -ptx -arch=compute_80 -o kernel.ptx kernel.cu
```
- `-arch=compute_80` for SM80 (Ampere) baseline
- PTX is forward-compatible — the GPU driver JIT-compiles to native SASS for the actual GPU

**Loading (runtime):**
```csharp
// 1. Read PTX text from embedded resource or content file
byte[] ptxBytes = File.ReadAllBytes("kernels/rmsnorm.ptx");

// 2. Load as CUDA module
fixed (byte* ptxPtr = ptxBytes)
{
    CudaDriverApi.cuModuleLoadData(out nint module, (nint)ptxPtr).ThrowOnError();
}

// 3. Get function handle (cached for process lifetime)
CudaDriverApi.cuModuleGetFunction(out nint function, module, "rmsnorm_f16").ThrowOnError();
```

**Kernel argument marshaling (zero-alloc via stackalloc):**
```csharp
void LaunchRmsNorm(nint output, nint input, nint weights, int n, float eps)
{
    void** args = stackalloc void*[5];
    args[0] = &output;
    args[1] = &input;
    args[2] = &weights;
    args[3] = &n;
    args[4] = &eps;

    CudaDriverApi.cuLaunchKernel(
        _rmsnormFunction,
        gridDim, 1, 1,
        blockDim, 1, 1,
        sharedMem, stream,
        (nint)args, nint.Zero).ThrowOnError();
}
```

**Function handle caching:**
```csharp
private readonly Dictionary<string, nint> _functionCache = new();

public nint GetFunction(string name)
{
    if (!_functionCache.TryGetValue(name, out nint func))
    {
        CudaDriverApi.cuModuleGetFunction(out func, _module, name).ThrowOnError();
        _functionCache[name] = func;
    }
    return func;
}
```

**dotLLM ships 24 PTX kernels:** rmsnorm, rope, attention, softmax, swiglu, embedding, dequantization (Q8_0/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K), quantized GEMV, bias_add, type conversion, KV-cache quantization, fused_add_rmsnorm, per_head_rmsnorm — each with FP16 and FP32 variants.

Source: [dotLLM CUDA kernels](https://github.com/kkokosa/dotLLM), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

**SharpInference adoption:** Same PTX loading, caching, and launch patterns. SharpInference kernels cover a different domain (Conv2D, GroupNorm, SDPA for spatial attention, upsampling, FFT/STFT) but use identical infrastructure.

### cuBLAS Integration

Separate P/Invoke surface for cuBLAS matrix operations:

```csharp
[LibraryImport("cublas64_12")]
internal static partial CublasStatus cublasCreate_v2(out nint handle);

[LibraryImport("cublas64_12")]
internal static partial CublasStatus cublasSetStream_v2(nint handle, nint stream);

[LibraryImport("cublas64_12")]
internal static partial CublasStatus cublasGemmEx(
    nint handle,
    CublasOperation transa, CublasOperation transb,
    int m, int n, int k,
    nint alpha,
    nint A, CudaDataType Atype, int lda,
    nint B, CudaDataType Btype, int ldb,
    nint beta,
    nint C, CudaDataType Ctype, int ldc,
    CublasComputeType computeType,
    CublasGemmAlgo algo);
```

- `cublasGemmEx` automatically uses Tensor Cores when available (FP16 inputs, FP32 accumulate)
- cuBLAS handle created once per CUDA context, reused for all GEMM calls
- ~6 cuBLAS P/Invoke functions total

Source: [dotLLM CublasApi.cs](https://github.com/kkokosa/dotLLM), [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/)

**SharpInference adoption:** Same cuBLAS binding strategy for FP16/FP32 GEMM in convolution (im2col + GEMM) and attention operations.

### Memory-Mapped Model Loading (GGUF)

GGUF files are loaded via memory-mapped I/O — multi-GB models appear to load in milliseconds:

```csharp
var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
byte* basePtr = null;
accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
```

**How it works:**
- `MemoryMappedFile.CreateFromFile()` creates a virtual address mapping — no data is read from disk yet
- The OS loads pages on-demand as they are accessed (demand paging)
- Tensor descriptors parsed from the GGUF header provide name, shape, quantization type, and byte offset
- Tensor data accessed as raw pointers: `basePtr + tensorOffset`
- For GPU inference, data is copied to VRAM via `cuMemcpyHtoD` — only the weights actually needed are paged in

**GGUF header parsing:** Clean-room implementation (not copied from llama.cpp due to licensing). Parses the GGUF magic number, version, tensor count, metadata key-value pairs, and tensor descriptors.

Source: [dotLLM GGUF loader](https://github.com/kkokosa/dotLLM), [GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

**SharpInference adoption:** Same mmap loading for both SafeTensors and GGUF. Clean-room GGUF implementation to maintain licensing independence from both dotLLM (GPLv3) and llama.cpp.

### IBackend Abstraction

All model code programs against `IBackend` — never calls CPU or CUDA kernels directly:

```csharp
public interface IBackend
{
    void MatMul(TensorRef output, TensorRef a, TensorRef b);
    void RmsNorm(TensorRef output, TensorRef input, TensorRef weights, float eps);
    void Softmax(TensorRef output, TensorRef input);
    void RoPE(TensorRef output, TensorRef input, int position, float theta);
    void SwiGLU(TensorRef output, TensorRef gate, TensorRef up);
    // ... other ops
}
```

**Key behaviors:**
- CPU backend dispatches to SIMD kernels (AVX2/AVX-512/NEON with scalar fallback)
- CUDA backend dispatches to PTX kernels or cuBLAS
- Cross-device ops insert automatic copies (CPU→GPU via `cuMemcpyHtoD`, GPU→CPU via `cuMemcpyDtoH`)
- Backend selected at startup based on hardware detection; model code is backend-agnostic

Source: [dotLLM IBackend](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same `IBackend` interface pattern. SharpInference adds additional ops (Conv2D, GroupNorm, Upsample, SDPA, FFT) but follows the same abstraction boundary.

### SIMD Kernel Dispatch

CPU kernels use a tiered SIMD dispatch strategy:

```
AVX-512 (Vector512<T>) → AVX2 (Vector256<T>) → Scalar fallback
```

**Dispatch pattern:**
```csharp
if (Vector512.IsHardwareAccelerated)
    RmsNormAvx512(output, input, weights, eps);
else if (Vector256.IsHardwareAccelerated)
    RmsNormAvx256(output, input, weights, eps);
else
    RmsNormScalar(output, input, weights, eps);
```

**Key conventions:**
- `System.Runtime.Intrinsics` for hot inner loops (`Vector128<T>`, `Vector256<T>`, `Vector512<T>`)
- `System.Numerics.Tensors.TensorPrimitives` for standard element-wise ops (add, multiply, etc.)
- Cross-platform vector types preferred over platform-specific (e.g., `Vector256<float>` not `Avx2.xxx`)
- Scalar fallback is **mandatory** — every SIMD kernel must have a scalar equivalent
- `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on all SIMD helper methods

Source: [dotLLM CPU kernels](https://github.com/kkokosa/dotLLM), [.NET SIMD docs](https://learn.microsoft.com/en-us/dotnet/standard/simd)

**SharpInference adoption:** Same dispatch hierarchy. SharpInference's CPU kernels (Conv2D, GroupNorm, FFT, mel spectrogram) follow identical patterns.

### R4 Weight Repacking for SIMD

dotLLM repacks quantized weight matrices at model load time for optimal SIMD access:

**Problem:** GGUF stores quantization blocks sequentially by row. When computing a matrix-vector product, the SIMD kernel needs to process multiple rows simultaneously to fill vector registers efficiently.

**Solution — R4 repacking:** Interleave 4 rows of quantization blocks so that sequential memory reads fill `Vector256<T>` registers without gather operations:

```
Original layout (row-major blocks):
Row 0: [block0][block1][block2]...
Row 1: [block0][block1][block2]...
Row 2: [block0][block1][block2]...
Row 3: [block0][block1][block2]...

R4 repacked layout:
[Row0.block0][Row1.block0][Row2.block0][Row3.block0]
[Row0.block1][Row1.block1][Row2.block1][Row3.block1]
...
```

This ensures each cache line read provides data for 4 output elements, maximizing SIMD utilization.

Source: [dotLLM weight repacking](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Apply same repacking concept for quantized vision/audio model weights where applicable.

### Fused Kernels

dotLLM fuses operations to minimize memory bandwidth:

**RMSNorm + Quantize fusion:**
- Instead of: RMSNorm → write to memory → read from memory → Quantize
- Fused: RMSNorm → quantize in-register → write quantized output
- Saves one full tensor read+write (significant for large hidden dimensions)

**SwiGLU fusion with L1-cache tiling:**
- Processes tiles that fit in L1 cache (~32KB per core)
- Gate and up projections processed together, element-wise SwiGLU applied in-register
- Avoids cache thrashing on large intermediate tensors

**On-the-fly activation quantization:**
- Instead of dequantizing weight matrix (expensive, large), quantize the input activation to Q8_0 (cheap, small)
- Q8_0 × Q4_K GEMV kernel operates directly on quantized formats
- Dramatically reduces memory bandwidth for single-token decode

Source: [dotLLM fused kernels](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Apply fusion philosophy throughout. Key candidates: GroupNorm+SiLU fusion in UNet blocks, Conv2D+bias+activation fusion, attention score computation.

### ComputeThreadPool with Adaptive Dispatch

dotLLM uses a custom thread pool (not `ThreadPool` or `Task.Run`) with two dispatch modes:

**SpinWait mode** — for single-token decode (latency-critical):
- Worker threads spin-wait on a shared flag
- Near-zero wake latency (~100ns vs ~15µs for OS thread wake)
- Burns CPU cycles but minimizes inter-token latency

**EventBased mode** — for prefill / batch processing (throughput-oriented):
- Worker threads block on `ManualResetEventSlim`
- No wasted CPU during longer operations
- Higher wake latency acceptable because operations are larger

**Adaptive switching:** The pool detects whether the current operation is latency-sensitive (single token) or throughput-sensitive (prefill/batch) and switches modes accordingly.

Source: [dotLLM ComputeThreadPool](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Implement similar adaptive thread pool. For diffusion: SpinWait during denoising steps (latency between steps matters for streaming), EventBased during model loading and preprocessing.

### Error Handling Patterns

**CUDA error checking:**
```csharp
public static void ThrowOnError(this CuResult result)
{
    if (result != CuResult.Success)
        throw new CudaException(result);
}
```
Every CUDA call is checked — no silent failures.

**Worker thread crashes:**
```csharp
catch (Exception ex)
{
    Environment.FailFast($"Worker thread crashed: {ex.Message}", ex);
}
```
`Environment.FailFast` is used for unrecoverable errors in compute worker threads — a corrupted worker is worse than a crash because it produces silent wrong results.

**Tensor shape validation:**
- Shape checks at operation boundaries with clear error messages
- Fail fast before any computation begins

Source: [dotLLM error handling](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same patterns. Custom exceptions (`SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException`). `Environment.FailFast` for unrecoverable compute thread errors.

### Streaming via IAsyncEnumerable

dotLLM streams tokens using `IAsyncEnumerable<T>`:

```csharp
public async IAsyncEnumerable<GenerationToken> GenerateAsync(
    string prompt,
    GenerationOptions options,
    [EnumeratorCancellation] CancellationToken ct = default)
{
    // ... setup ...
    while (!done)
    {
        int tokenId = DecodeNextToken();
        yield return new GenerationToken(tokenId, _tokenizer.Decode(tokenId));
    }
}
```

- `GenerationToken` is a `readonly record struct` — zero allocation per yield
- Supports `CancellationToken` for graceful abort
- ASP.NET server maps this to Server-Sent Events (SSE) for OpenAI-compatible streaming

Source: [dotLLM streaming](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same pattern for diffusion progress streaming: `IAsyncEnumerable<GenerationProgress>` where `GenerationProgress` is a readonly record struct carrying step number, preview image (optional), and timing info.

### ASP.NET Server Architecture

dotLLM's server uses ASP.NET Minimal APIs:

**Key patterns:**
- One file per endpoint (e.g., `ChatCompletionEndpoint.cs`, `ModelsEndpoint.cs`)
- `ServerState` singleton holds loaded models, backend reference, and configuration
- Source-generated JSON serialization (`[JsonSerializable]` contexts) — no reflection
- OpenAI-compatible request/response types

```csharp
app.MapPost("/v1/chat/completions", async (ChatCompletionRequest request, ServerState state) =>
{
    // ... validation, model selection ...
    return Results.Stream(async stream =>
    {
        await foreach (var token in engine.GenerateAsync(prompt, options))
        {
            await stream.WriteAsync(FormatSSE(token));
        }
    }, "text/event-stream");
});
```

Source: [dotLLM server](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same Minimal API architecture. SharpInference serves `/v1/images/generations`, `/v1/audio/transcriptions`, `/v1/audio/speech`, and `/v1/audio/translations`. Shared `ServerState` pattern, source-generated JSON, per-endpoint files.

### Testing Strategy

**SIMD vs Scalar verification:**
```csharp
[Fact]
public void RmsNorm_SimdMatchesScalar()
{
    var input = CreateRandomTensor(shape);
    var resultSimd = RmsNormSimd(input, weights, eps);
    var resultScalar = RmsNormScalar(input, weights, eps);
    AssertTensorsEqual(resultSimd, resultScalar, tolerance: 1e-5f);
}
```
Every SIMD kernel is tested against its scalar fallback — if they disagree, the SIMD implementation has a bug.

**Known-value tests:** Fixed inputs with pre-computed expected outputs (validated against Python reference).

**Integration tests with lazy model download:**
```csharp
[Fact]
[Trait("Category", "Integration")]
public async Task Llama_GeneratesCoherentText()
{
    var model = await TestModels.GetOrDownload("llama-3.2-1B-Q4_0.gguf");
    // ... inference test ...
}
```
Models are downloaded once and cached. Integration tests are tagged so they can be excluded from CI.

Source: [dotLLM tests](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same three-tier testing: SIMD-vs-scalar unit tests, known-value tests against Python outputs, integration tests with lazy model download.

### Performance Attributes and Code Style

dotLLM uses specific C# attributes consistently:

| Attribute | When Used |
|-----------|-----------|
| `[MethodImpl(AggressiveInlining)]` | Small hot-path methods (< ~20 IL bytes), tensor accessors, SIMD helpers |
| `[SkipLocalsInit]` | Methods with large `stackalloc` or performance-critical paths |
| `[SuppressGCTransition]` | Short CUDA P/Invoke calls (< 1µs) to avoid GC cooperation overhead |
| `readonly record struct` | Value types (DType, TensorShape, TensorRef, GenerationToken) |
| `file-scoped namespaces` | All files |
| `sealed` | All classes that aren't designed for inheritance |
| `readonly` | All fields that don't change after construction |

**Naming and formatting:**
- Standard C# naming (PascalCase types/methods, camelCase locals, _camelCase fields)
- Nullable reference types enabled project-wide
- XML doc comments on all public APIs
- No `#region` blocks

Source: [dotLLM code conventions](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Follow all of these conventions identically.

### Configuration and Model Discovery

**Model registry pattern:**
- Models discovered from a configurable local directory (default: `~/.cache/sharpinference/models/`)
- HuggingFace download support for automatic model fetching
- Architecture key in model metadata drives automatic pipeline selection
- Model hot-swap supported without process restart

**Configuration:**
- `appsettings.json` for server configuration
- Command-line argument overrides
- Environment variable support for containerized deployment

Source: [dotLLM configuration](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same model discovery patterns. Shared model cache directory when both dotLLM and SharpInference are used together.

## Key Patterns Summary

| Pattern | Description | Why It Matters |
|---------|-------------|----------------|
| Dual tensor types | `ITensor` for lifecycle + `TensorRef` for compute | Zero-alloc hot paths |
| 64-byte aligned alloc | `NativeMemory.AlignedAlloc(bytes, 64)` | Cache line + SIMD alignment |
| `Interlocked.Exchange` dispose | Atomic pointer swap before free | Thread-safe, no double-free |
| `[LibraryImport]` P/Invoke | Source-generated marshaling | Zero-alloc, trimmer-friendly |
| `stackalloc void*[]` kernel args | Kernel params on stack | No heap allocation for launches |
| Function handle cache | `Dictionary<string, nint>` | One-time lookup per kernel |
| `.ThrowOnError()` extension | Every CUDA call checked | No silent failures |
| R4 weight repacking | Interleave rows at load time | Sequential SIMD reads |
| Fused kernels | Combine ops to reduce memory traffic | Bandwidth-bound optimization |
| Adaptive thread pool | SpinWait vs EventBased | Latency vs throughput tradeoff |
| `IAsyncEnumerable` streaming | `yield return` readonly record structs | Zero-alloc streaming |
| SIMD-vs-scalar tests | Every SIMD kernel tested against scalar | Correctness verification |
| `Environment.FailFast` | Unrecoverable compute thread errors | No silent corruption |
| Source-generated JSON | `[JsonSerializable]` contexts | No reflection in server |

## Architectural Lessons for SharpInference

1. **Don't abstract prematurely.** dotLLM's CPU and CUDA attention implementations are intentionally *not* behind a unified interface — the optimization strategies are too different. Follow this principle: `IBackend` abstracts the operation, but backend implementations can be radically different internally.

2. **Memory bandwidth is the bottleneck.** Most inference kernels are memory-bandwidth-bound, not compute-bound. Fused kernels, activation quantization, and cache-tiled loops all target bandwidth reduction. Apply this lens to every SharpInference kernel.

3. **Quantization on the small side.** dotLLM quantizes activations (small) rather than dequantizing weights (large) for GEMV. Look for analogous opportunities in SharpInference — always move the smaller tensor.

4. **Load-time preprocessing pays off.** R4 repacking costs milliseconds at load time but saves microseconds on every inference. SharpInference should similarly preprocess weights at load time for optimal runtime access patterns.

5. **Process-lifetime caching.** PTX modules, function handles, cuBLAS handles — created once and cached forever. Avoid repeated initialization in hot paths.

6. **Test the SIMD, not just the math.** The SIMD-vs-scalar test pattern catches register width bugs, lane ordering issues, and remainder handling errors that unit tests with known values might miss.

## References

1. [dotLLM GitHub Repository](https://github.com/kkokosa/dotLLM) — Full source code, GPLv3 licensed
2. [Konrad Kokosa — Pro .NET Memory Management](https://prodotnetmemory.com/) — Author's background in .NET memory optimization
3. [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/) — P/Invoke target API surface
4. [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) — Matrix multiplication API
5. [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX instruction set
6. [.NET NativeMemory API](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativememory) — Unmanaged allocation API
7. [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Model file format
8. [System.Runtime.Intrinsics](https://learn.microsoft.com/en-us/dotnet/standard/simd) — .NET SIMD programming guide
9. [ASP.NET Minimal APIs](https://learn.microsoft.com/en-us/aspnet/core/fundamentals/minimal-apis) — Server framework
10. [System.Numerics.Tensors.TensorPrimitives](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives) — Standard tensor operations
