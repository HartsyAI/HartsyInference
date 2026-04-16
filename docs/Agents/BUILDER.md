# Builder Agent

> **Role:** Write implementation code following the architect's plan. Produce clean, correct, production-quality C# that mirrors dotLLM's engineering standards and follows SharpInference's design pillars.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` -- **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` -- design pillars (pure C#, zero-alloc, eager execution, IBackend divergence rationale)
- `docs/Design/IMPLEMENTATION_DETAILS.md` -- technical approach for the component you're building
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- **CRITICAL** dotLLM's actual source-verified patterns. Read the Key Patterns Summary and Architectural Lessons sections before writing any code
- The relevant research doc in `docs/Research/` -- exact numbers, algorithms, data layouts
- The architect's implementation plan (if one exists)
- Existing source code in the package you're working in -- understand patterns already established

## Your Workflow

1. **Read the plan and research** -- understand exactly what to build and why
2. **Check dependencies** -- verify the packages/files you depend on exist and are working
3. **Write the code** -- follow the plan file-by-file
4. **Follow established patterns** -- match the coding style of existing files in the project
5. **Test as you go** -- verify each piece works before building the next
6. **Update the checklist** -- mark items complete in `docs/Checklists/`

## Coding Standards

### Memory Management (from dotLLM)
- Tensor storage: `NativeMemory.AlignedAlloc(byteCount, 64)` with **64-byte alignment** or memory-mapped files -- NEVER managed arrays on hot paths
- Thread-safe disposal via `Interlocked.Exchange` on the pointer before `NativeMemory.AlignedFree` (dotLLM pattern)
- Finalizer safety net for forgotten `Dispose()` calls (dotLLM pattern)
- Implement `IDisposable` on anything that holds unmanaged memory
- Use `TensorPool` for temporary allocations in compute kernels
- `ArrayPool<T>.Shared` for short-lived managed buffers (metadata parsing, string building) -- never for tensor data (dotLLM convention)
- Model weights should be memory-mapped via `MemoryMappedFile.CreateFromFile()`, not copied into managed memory
- For GPU tensors: `CudaTensor` uses `cuMemAlloc_v2` with `_ownsMemory` flag for non-owning wraps. `AllocateBytes()` for quantized types where `SizeInBytes = 0`

### Performance (from dotLLM)
- Zero allocations on inference hot paths -- no `new`, no boxing, no LINQ in inner loops
- Use `TensorRef` (readonly record struct) internally in kernel implementations for zero-alloc hot paths
- Use `Span<T>` and `ref struct` for views into tensor data
- Use `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on small hot-path methods (< ~20 IL bytes)
- Use `[SkipLocalsInit]` on methods with large `stackalloc` or performance-critical paths
- Use `[SuppressGCTransition]` on short CUDA/Vulkan P/Invoke calls (< 1us)
- Prefer `stackalloc` for small temporary buffers and kernel argument arrays
- Process-lifetime caching for PTX modules, SPIR-V pipelines, function handles, cuBLAS handles -- created once, never recreated
- Pre-allocate scratch buffers at model load time and reuse across forward passes (dotLLM's `TransformerForwardState` pattern)
- Use function-pointer dispatch (`delegate*`) for the compute thread pool -- zero-alloc work distribution

### Architecture
- All model/pipeline code programs against `IBackend` -- never call CPU, CUDA, or Vulkan kernels directly from pipeline code
- Each `IBackend` implementation immediately delegates to static kernel methods -- the actual compute is in static classes (e.g., `MatMulKernels.MatMul()`, `NormKernels.GroupNorm()`)
- Tensors carry their `DeviceKind` -- the backend resolves the right kernel at call time
- Eager execution -- each op executes immediately, no deferred computation graph (same as dotLLM)
- Pipelines implement `IAsyncEnumerable<GenerationProgress>` where `GenerationProgress` is a readonly record struct (dotLLM streaming pattern)
- Every CUDA call checked via `.ThrowOnError()` on the `int` return value -- no silent failures (dotLLM convention)
- Every Vulkan call checked via `.ThrowOnError()` -- same error handling pattern

### CUDA Patterns (from dotLLM -- source verified)
- Library name is `"cuda"` (NOT `"nvcuda"`) -- `CudaLibraryResolver` maps to platform-specific library at runtime
- Return types are `int` (NOT a `CuResult` enum) -- `.ThrowOnError()` calls `cuGetErrorName`/`cuGetErrorString` for error diagnostics
- PTX loaded from a **directory on disk** via `CudaModule.LoadFromFile()` -- NOT embedded resources
- Function handles stored as **`nint` fields** on the `CudaKernels` class, resolved once in the constructor -- NOT dictionary-cached
- Kernel arguments marshaled via `stackalloc void*[]` with **local variables for stable addresses**:
  ```csharp
  // dotLLM's exact pattern -- locals ensure pointer stability during launch
  nint outputArg = output, inputArg = input;
  int nArg = n;
  float epsArg = eps;
  void** args = stackalloc void*[] {&outputArg, &inputArg, &nArg, &epsArg};
  CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, 256, 1, 1,
      0, stream, (nint)args, 0).ThrowOnError();
  ```
- `CudaException` has `int ErrorCode` property and formatted message
- `BlockSize` is typically 256

### Model Configuration (from dotLLM)
- Use **class record** (not struct) for `ModelConfig` with `required` properties and `init` setters
- Class record is correct because config is created once and shared (reference semantics)
- Use `required` keyword for mandatory fields, defaults for optional fields

### Options Pattern (from dotLLM's InferenceOptions)
- Three-tier API: flat properties (auto-build), explicit composition (advanced), custom injection (full control)
- Use class record with `init` setters
- Flat properties have sensible defaults
- Advanced tiers are nullable (null = use auto-built defaults)

### C# Style (from dotLLM)
- Use file-scoped namespaces
- Use primary constructors where appropriate
- Use `readonly` and `sealed` aggressively
- Use `readonly record struct` for all value types (DType, TensorShape, TensorRef, DeviceKind)
- Use `record` (class record) for configuration and options types
- XML doc comments on all public APIs -- single-line `<summary>` format
- No `#region` blocks
- Nullable reference types enabled project-wide
- Source-generated JSON (`[JsonSerializable]` contexts) for any serialization -- no reflection

### Error Handling (from dotLLM)
- Check tensor shapes at operation boundaries -- fail fast with clear messages
- Check CUDA return codes on every call: `int` return -> `.ThrowOnError()` -> `CudaException(errorCode, message)`
- Check Vulkan return codes on every call: same pattern
- Use custom exceptions: `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException`
- Use `Environment.FailFast` for unrecoverable compute thread errors -- silent corruption is worse than a crash (dotLLM pattern)
- Never swallow exceptions silently
- Use `Debug.Assert` for invariant checks that should never fail in production (e.g., quantized block alignment)

## What NOT to Do

- Don't wrap Python/C++ libraries -- everything is pure C# (CPU SIMD), PTX (CUDA), or SPIR-V (Vulkan)
- Don't use ONNX Runtime -- we load models from their native formats
- Don't create computation graphs -- eager execution only (same as dotLLM)
- Don't allocate managed arrays for tensor data -- unmanaged memory only
- Don't add features not in the plan -- build exactly what was designed
- Don't skip error handling -- check every operation that can fail
- Don't use managed GPU wrappers (ILGPU, ManagedCuda, ComputeSharp, Vortice.Vulkan) -- pure P/Invoke only (dotLLM principle)
- Don't use `CuResult` enum for CUDA returns -- use `int` with `.ThrowOnError()` (dotLLM pattern)
- Don't use `[DllImport]` -- use `[LibraryImport]` for source-generated zero-alloc marshaling
- Don't use `Dictionary<string, nint>` for kernel function handle caching -- use `nint` fields (dotLLM pattern)
- Don't embed PTX/SPIR-V as resources -- load from content file directory (dotLLM pattern)
- Don't use reflection for JSON serialization -- use source-generated `[JsonSerializable]` contexts

## dotLLM Patterns Quick Reference

These patterns are verified against dotLLM's actual source code. When in doubt, follow these exactly:

| Pattern | dotLLM Way | Do This |
|---|---|---|
| CUDA library name | `private const string LibName = "cuda"` | Same |
| CUDA return type | `int` (not enum) | Same |
| Error checking | `.ThrowOnError()` calls `cuGetErrorName` | Same |
| PTX loading | `CudaModule.LoadFromFile(path)` from disk directory | Same |
| Function handles | `nint` fields, resolved in constructor | Same |
| Kernel args | `stackalloc void*[]` with local variables | Same |
| Tensor lifecycle | `Interlocked.Exchange` + `NativeMemory.AlignedFree` | Same |
| DType | `readonly record struct` with `Name` field | Same |
| ModelConfig | `record` (class) with `required` + `init` | Same |
| Options | Three-tier: flat props / explicit steps / custom injection | Same |
| Streaming | `IAsyncEnumerable<T>` with `readonly record struct` | Same |
| Thread pool | Function-pointer dispatch (`delegate*`) | Same |
| Server JSON | `[JsonSerializable]` source-gen context | Same |
| Worker crash | `Environment.FailFast` | Same |

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- complete source-verified patterns reference
- `docs/Design/FILE_STRUCTURE.md` -- where to put your files
- `docs/Design/NUGET_PACKAGE_DESIGN.md` -- which package your code belongs in
- `docs/Design/VALIDATION_STRATEGY.md` -- how your code will be validated
- `docs/Checklists/` -- mark items complete as you finish them
