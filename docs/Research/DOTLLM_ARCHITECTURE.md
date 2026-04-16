# dotLLM Architecture — Research Notes

> Status: Complete (updated with source code verification)
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
public readonly record struct TensorRef
{
    public int Dim0 { get; }      // e.g., sequence length
    public int Dim1 { get; }      // e.g., KV stride
    public DType DType { get; }
    public int DeviceId { get; }  // -1 = CPU, 0..N = GPU
    public nint DataPointer { get; }
    public long ElementCount { get; }
    public long ByteCount => DType.ComputeByteCount(ElementCount);
}
```
- Flat dimension fields (Dim0, Dim1) — no `TensorShape`, no interface dispatch
- Passed by value on the stack — zero heap allocation
- No ownership, no disposal — just a view into existing memory
- Used on the inference hot path for KV-cache updates
- Two constructors: 1D (Dim0 only) and 2D (Dim0, Dim1)

**`TensorView` (sealed class, implements `ITensor`)** — Non-owning view:
- Implements `ITensor` but `Dispose()` is a no-op
- Used for KV-cache slices and borrowed tensor references
- Has full `TensorShape` unlike `TensorRef`

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

**P/Invoke declarations (~40 functions in `CudaDriverApi.cs`):**
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

**Key decisions:**
- `[LibraryImport]` (source-generated) over `[DllImport]` — zero-alloc marshaling, trimmer-friendly
- Return type is `int` (not an enum) — `.ThrowOnError()` extension checks for non-zero
- `[SuppressGCTransition]` on short CUDA calls (e.g., `cuMemFree_v2`) to avoid GC cooperation overhead
- Library name is `"cuda"` — resolved at runtime by `CudaLibraryResolver`

**Cross-platform library resolution:**
```csharp
public sealed class CudaLibraryResolver
{
    // Resolves "cuda" to:
    //   Windows: nvcuda.dll (system PATH)
    //   Linux:   libcuda.so.1 (LD_LIBRARY_PATH or /usr/lib)
}
```
Registered via `NativeLibrary.SetDllImportResolver()` at startup. Called from `CudaBackend` constructor.

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

**Loading (runtime) — from PTX directory, NOT embedded resources:**
```csharp
// CudaModule wraps cuModuleLoadData + cuModuleGetFunction
var module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rmsnorm.ptx"));
nint func = module.GetFunction("rmsnorm_f16");
```

All 24+ modules are loaded in the `CudaKernels` constructor. Function handles stored as `nint` fields (not dictionary cached).

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

**dotLLM ships 24 PTX kernels (each as a separate `.ptx` file with FP16 and FP32 variants):** rmsnorm, rope, attention, softmax, swiglu, embedding, dequantization (Q8_0/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K), quantized GEMV, bias_add, type conversion, KV-cache quantization, fused_add_rmsnorm, per_head_rmsnorm — each with FP16 and FP32 variants.

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

`IBackend` is a **device memory management** interface — NOT an op dispatch interface. Model code calls kernel functions directly:

```csharp
public interface IBackend : IDisposable
{
    int DeviceCount { get; }
    ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype);
    void CopyBetweenDevices(ITensor source, ITensor destination);
    void AllReduce(ReadOnlySpan<ITensor> tensors);
    void Send(ITensor tensor, int targetDevice);
    ITensor Receive(int sourceDevice, TensorShape shape, DType dtype);
}
```

**Key behaviors:**
- `CpuBackend` allocates via `UnmanagedTensor.Allocate()`, throws `NotSupportedException` for multi-device ops
- `CudaBackend` allocates via `CudaTensor.Allocate()`, handles host-device copies via `cuMemcpyHtoD_v2`/`cuMemcpyDtoH_v2`
- Kernel ops (MatMul, RmsNorm, RoPE, etc.) are called directly as static methods — NOT through `IBackend`
- `TransformerModel.Forward()` calls `MatMul.GemvQ8_0(...)`, `RmsNorm.Execute(...)` etc. directly
- `CudaTransformerModel` calls `CudaKernels.LaunchRmsNorm(...)`, `CudaGemm.Hgemm(...)` directly

Source: [dotLLM IBackend](https://github.com/kkokosa/dotLLM)

**SharpInference adoption:** Same `IBackend` as device/memory manager. Kernel dispatch happens in model-specific code or a separate layer, not through `IBackend`. This avoids a bloated interface that must be implemented for every backend.

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

## Addendum: Source Code Verification (2026-04-16)

The following details are derived from reading the actual dotLLM source code at commit e81ee20 and the author's blog post at kokosa.dev. These correct, clarify, and expand on the patterns described above.

### Exact IBackend Interface (Corrected)

The `IBackend` interface in dotLLM is NOT an op-dispatch interface. It is a **device memory management** interface. Model code calls kernels directly — not through `IBackend`:

```csharp
// DotLLM.Core/Backends/IBackend.cs — ACTUAL interface
public interface IBackend : IDisposable
{
    int DeviceCount { get; }
    ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype);
    void CopyBetweenDevices(ITensor source, ITensor destination);
    void AllReduce(ReadOnlySpan<ITensor> tensors);
    void Send(ITensor tensor, int targetDevice);
    ITensor Receive(int sourceDevice, TensorShape shape, DType dtype);
}
```

There are no `MatMul`, `RmsNorm`, `Softmax`, etc. methods on `IBackend`. Those are called directly as static methods in the kernel classes (`MatMul.GemvQ8_0(...)`, `RmsNorm.Execute(...)`, etc.) from `TransformerModel.Forward()`.

**SharpInference implication:** Our `IBackend` should similarly be about memory and device management, NOT op dispatch. Kernel dispatch happens in model-specific code or in a separate op dispatcher.

### Exact ITensor Interface

```csharp
public interface ITensor : IDisposable
{
    TensorShape Shape { get; }
    DType DType { get; }
    int DeviceId { get; }      // -1 = CPU, 0..N = GPU index
    nint DataPointer { get; }
    TensorMetadata Metadata { get; }
    long ElementCount { get; }
    long ByteCount { get; }
}
```

Note: No strides. No `.AsRef()` method. No `AsSpan()`. Tensor data is accessed by casting `DataPointer` to the appropriate pointer type in unsafe code.

### Three Tensor Types (Not Two)

dotLLM actually has **three** tensor representations:

1. **`UnmanagedTensor`** (class, implements `ITensor`) — Owns CPU memory via `NativeMemory.AlignedAlloc(bytes, 64)`. Thread-safe disposal via `Interlocked.Exchange`. Has finalizer safety net.

2. **`CudaTensor`** (class, implements `ITensor`) — Owns GPU memory via `cuMemAlloc_v2`. Has `_ownsMemory` flag for non-owning wraps. Also has `AllocateBytes()` for quantized types where per-element size is 0.

3. **`TensorView`** (class, implements `ITensor`) — Non-owning view over existing memory. `Dispose()` is a no-op. Used for KV-cache slices and other borrowed references.

4. **`TensorRef`** (readonly record struct) — Lightweight value type with flat dimension fields (Dim0, Dim1). No `TensorShape`. No interface dispatch. Used exclusively on the inference hot path for zero-alloc KV-cache updates.

5. **`TensorMetadata`** (readonly record struct) — `(TensorShape, DType, DeviceId, DataPointer)`. Pure value type, no ownership.

### Exact DType Definition

```csharp
public readonly record struct DType(string Name, int SizeInBytes, bool IsQuantized,
    int BlockByteSize = 0, int BlockElementCount = 1)
```

Note the `Name` field (string) and the `ComputeByteCount` method with `Debug.Assert` for block alignment:

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
public long ComputeByteCount(long elementCount)
{
    Debug.Assert(!IsQuantized || elementCount % BlockElementCount == 0);
    return IsQuantized ? elementCount / BlockElementCount * BlockByteSize : elementCount * SizeInBytes;
}
```

Supported types: Float32, Float16, BFloat16, Int8, UInt8, Int32, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K.

### CUDA Integration: "cuda" Not "nvcuda"

The actual library name used in P/Invoke is `"cuda"` (not `"nvcuda"`):

```csharp
private const string LibName = "cuda";

[LibraryImport(LibName)]
internal static partial int cuInit(uint flags);
```

A `CudaLibraryResolver` registered at startup maps `"cuda"` to `libcuda.so` (Linux) / `nvcuda.dll` (Windows). Return types are `int` (not a `CuResult` enum) with a `.ThrowOnError()` extension.

### PTX Loading: From Directory, Not Embedded Resources

dotLLM loads PTX from a **directory on disk**, NOT embedded resources:

```csharp
public CudaKernels(string ptxDir)
{
    _rmsnormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rmsnorm.ptx"));
    _ropeModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "rope.ptx"));
    // ... 24 modules total
}
```

The `native/ptx/` directory contains pre-compiled `.ptx` files. The `.cu` source files live in `native/kernels/`. Each PTX module has multiple function entry points (e.g., `rmsnorm.ptx` has `rmsnorm_f16`; `embedding.ptx` has `embedding_lookup_f32`, `embedding_lookup_f16`, `embedding_lookup_q8_0`).

Function handles are stored as `nint` fields — NOT in a dictionary cache. They are resolved once in the constructor.

### CudaKernels Launch Pattern (Exact)

Kernel arguments are marshaled via `stackalloc void*[]` with local variables for each argument (to get stable addresses):

```csharp
public void LaunchRmsNorm(nint input, nint weight, nint output,
                           int hiddenSize, float eps, int rows, nint stream)
{
    nint inputArg = input, weightArg = weight, outputArg = output;
    int nArg = hiddenSize;
    float epsArg = eps;

    void** args = stackalloc void*[] {&inputArg, &weightArg, &outputArg, &nArg, &epsArg};
    CudaDriverApi.cuLaunchKernel(_rmsnormFunc,
            (uint)rows, 1, 1, BlockSize, 1, 1,
            0, stream, (nint)args, 0).ThrowOnError();
}
```

Note: `BlockSize` is a constant 256. Grid dimensions vary per kernel.

### CudaException Pattern

```csharp
public sealed class CudaException : Exception
{
    public int ErrorCode { get; }
    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}") { ErrorCode = errorCode; }
}
```

The `.ThrowOnError()` extension method is on `CudaErrorHelper` and looks up error name/string via `cuGetErrorName`/`cuGetErrorString`.

### TransformerModel.Forward() — The Full CPU Forward Pass

The `TransformerModel` class is `unsafe` and operates entirely with raw pointers. It pre-allocates all scratch buffers at model load time via `TransformerForwardState`:

```csharp
float* hidden = (float*)_state.HiddenState;
float* residual = (float*)_state.Residual;
float* normOut = (float*)_state.NormOutput;
float* q = (float*)_state.Q;
// ... all pre-allocated, reused across forward passes
```

The forward pass steps:
1. Embedding lookup (supports F32, F16, Q8_0, or generic dequant fallback)
2. For each layer: RMSNorm -> Q/K/V projections -> optional bias -> optional QK-norm -> RoPE -> Attention (cached or uncached) -> O projection -> residual add -> FFN RMSNorm -> Gate/Up projections -> SwiGLU -> Down projection -> residual add
3. Final RMSNorm
4. LM Head (logits)

**Key optimization: Decode vs Prefill paths.** Single-token decode uses fused ops (`FusedQkvDecode`, `FusedGateUpDecode`) that dispatch all projections in one thread pool call. Prefill uses unfused per-token loops.

**Adaptive dispatch mode switching:**
```csharp
_threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);
```

### Sampling Pipeline (Exact API)

```csharp
public interface ISamplerStep
{
    void Apply(Span<float> logits, SamplerContext context);
}

public interface ILogitProcessor
{
    void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context);
}
```

`SamplerPipeline` orchestrates: processors first (repetition penalty), then steps in order (temperature -> topK -> topP -> minP), then categorical sample. Greedy mode short-circuits to `TensorPrimitives.IndexOfMax(logits)`.

The pipeline can be built from `InferenceOptions` (auto-build) or composed explicitly from step instances.

### IKvCache with TensorRef Overloads

The `IKvCache` interface has **dual overloads** — one taking `ITensor` and one taking `TensorRef`:

```csharp
void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex);
void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex);

ITensor GetKeys(int layerIndex);
TensorRef GetKeysRef(int layerIndex);  // Zero-allocation hot path
```

The `Rollback(int length)` method supports speculative decoding — discards entries beyond the given position while retaining allocated memory.

### ModelConfig as Record (Not Record Struct)

```csharp
public record ModelConfig  // class record, NOT struct
{
    public required Architecture Architecture { get; init; }
    public required int VocabSize { get; init; }
    public required int HiddenSize { get; init; }
    // ... many more fields with init-only setters
}
```

Uses `required` keyword for mandatory fields. Optional fields have defaults. This is a reference type (class record), not a value type — appropriate since it's created once and shared.

### InferenceOptions Pattern

```csharp
public record InferenceOptions
{
    // Simple flat properties (auto-build sampling pipeline)
    public float Temperature { get; init; } = 0.7f;
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int MaxTokens { get; init; } = 2048;
    public int? Seed { get; init; }

    // Advanced: explicit pipeline composition
    public IReadOnlyList<ISamplerStep>? SamplerSteps { get; init; }
    public IReadOnlyList<ILogitProcessor>? LogitProcessors { get; init; }
    public IReadOnlyList<IStopCondition>? StopConditions { get; init; }

    // Constraints
    public ResponseFormat? ResponseFormat { get; init; }

    // Diagnostics
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; } = 5;

    // Threading
    public ThreadingConfig? Threading { get; init; }
}
```

Three tiers of customization: flat properties (easy), explicit sampler steps (composable), and custom processors/conditions (full control).

### ModelLoader — Static Helper Pattern

```csharp
public static class ModelLoader
{
    public static (IModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, ThreadingConfig? threading = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config, threading ?? ThreadingConfig.SingleThreaded);
        return (model, gguf, config);
    }
}
```

Returns a tuple of (model, gguf file handle, config). The GGUF file handle must be kept alive for the model's lifetime (mmap).

### Server DI Pattern

```csharp
public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddDotLLM(this IServiceCollection services, ServerState state)
    {
        services.AddSingleton(state);
        services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, ServerJsonContext.Default));
        return services;
    }
}
```

`ServerState` is a singleton created before the DI container is built, then registered. Source-generated `ServerJsonContext` is injected into the JSON serializer chain.

### IInferenceHook — Zero-Cost Diagnostics

```csharp
public interface IInferenceHook
{
    HookPoint HookPoint { get; }
    HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context);
}
```

Zero cost when no hooks registered — callers null-check before invoking. `HookResult` determines whether to continue with original activation or replace it.

### ComputeThreadPool — Function Pointer Dispatch

```csharp
void Dispatch(nint context, delegate*<nint, int, int, void> fn)
```

Uses unmanaged function pointers (not delegates) for zero-alloc dispatch. Caller thread executes as thread 0. Workers coordinate via `Interlocked` operations on a generation counter. `CountdownEvent` for completion synchronization. Per-worker scratch buffers via `GetWorkerScratch(int threadIdx, int minBytes)` returning 64-byte-aligned memory.

### Project Build Configuration

**Directory.Build.props (root):**
- Target: `net10.0`
- `AllowUnsafeBlocks: true`
- `EnforceCodeStyleInBuild: true`
- `Nullable: enable`
- `ImplicitUsings: enable`
- Author: Konrad Kokosa, License: GPL-3.0-only
- Deterministic builds on CI

**Directory.Packages.props** (Central Package Management):
- `System.Numerics.Tensors` 9.0.3
- `xunit` 2.9.3 / `xunit.runner.visualstudio` 3.0.2
- `BenchmarkDotNet` 0.14.0
- `Spectre.Console` 0.49.1 (CLI)
- `MinVer` 6.0.0 (versioning)

**Individual .csproj files are minimal** — `DotLLM.Core.csproj` contains only a `<Description>` element. All framework/build settings inherited from `Directory.Build.props`.

### Solution File

Uses `.slnx` format (new XML-based solution format in .NET 10).

### Test Organization

- `DotLLM.Tests.Unit` — Pure unit tests organized by package (Cpu/Kernels, Engine/Samplers, Models/Gguf, Tensors, Tokenizers)
- `DotLLM.Tests.Integration` — Tests requiring real model files, organized with shared fixtures (SmallModelFixture, Q4KModelFixture, etc.) and a `TestModelDownloader` for lazy download
- Tests use xunit with `[Trait("Category", "Integration")]` for model-dependent tests
- Skippable tests via `Xunit.SkippableFact` for hardware-dependent tests (CUDA)

### GGUF Reader — Pure Functional Parser

```csharp
public static class GgufReader
{
    public static GgufHeader ReadHeader(BinaryReader reader) { ... }
    public static Dictionary<string, GgufMetadataValue> ReadMetadata(BinaryReader reader, GgufHeader header) { ... }
    public static List<GgufTensorDescriptor> ReadTensorInfos(BinaryReader reader, GgufHeader header) { ... }
}
```

Static pure functions: bytes in, structs out. Handles GGUF v2 (uint32 counts) and v3 (uint64 counts). Returns strongly-typed arrays for common metadata element types.

### Benchmark Organization

- `DotLLM.Benchmarks` project using BenchmarkDotNet
- Custom columns for decode tok/s and prefill tok/s
- Benchmarks: DequantizeBenchmarks, KernelBenchmarks, InferenceBenchmarks, SchemaConstraintBenchmarks, TopKSamplerBenchmarks, etc.
- Results stored in `benchmarks/results/`

### Performance Numbers (From Blog)

CPU decode throughput (AMD Ryzen 9 5950X, 16 threads):
- SmolLM-135M Q4_K_M: 279.1 tok/s (83% of llama.cpp)
- SmolLM-135M Q8_0: 197.7 tok/s (77% of llama.cpp)
- Llama 3.2 1B Q4_K_M: 32.4 tok/s (66% of llama.cpp)
- Llama 3.2 3B Q8_0: 9.9 tok/s (88% of llama.cpp)

Prefill: 2-5x slower than llama.cpp due to RyuJIT register pressure (16 YMM registers for AVX2 vs unlimited in hand-written assembly).

### Development Phase Status

Phase 6 complete. Phase 7 in progress (diagnostics, LoRA, observability). The project has been through 7 quality "waves" of systematic review.

## References

1. [dotLLM GitHub Repository](https://github.com/kkokosa/dotLLM) — Full source code, GPLv3 licensed
2. [Konrad Kokosa's Blog: Introducing dotLLM](https://kokosa.dev/blog/2026/dotllm/) — Architecture blog post with benchmarks
3. [dotLLM Website](https://dotllm.dev/) — Project homepage
4. [Konrad Kokosa — Pro .NET Memory Management](https://prodotnetmemory.com/) — Author's background in .NET memory optimization
5. [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/) — P/Invoke target API surface
6. [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) — Matrix multiplication API
7. [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX instruction set
8. [.NET NativeMemory API](https://learn.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.nativememory) — Unmanaged allocation API
9. [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Model file format
10. [System.Runtime.Intrinsics](https://learn.microsoft.com/en-us/dotnet/standard/simd) — .NET SIMD programming guide
11. [ASP.NET Minimal APIs](https://learn.microsoft.com/en-us/aspnet/core/fundamentals/minimal-apis) — Server framework
12. [System.Numerics.Tensors.TensorPrimitives](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives) — Standard tensor operations
