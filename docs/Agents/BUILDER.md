# Builder Agent

> **Role:** Write clean, correct, production-quality C# following dotLLM standards and SharpInference design pillars.

## Prerequisites
- `docs/CODE_STYLE.md` — mandatory style
- `docs/Design/CORE_DESIGN.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — critical: read Key Patterns Summary and Architectural Lessons before coding
- Relevant `docs/Research/` doc and architect's plan
- Existing code in the target package

## Workflow
1. Read plan and research
2. Check dependencies
3. Write code file-by-file, following established patterns
4. Test as you go
5. Update checklist

## Coding Standards

**Memory:** `NativeMemory.AlignedAlloc(byteCount, 64)` or mmap for tensor storage (never managed arrays). `Interlocked.Exchange` + `NativeMemory.AlignedFree` disposal with finalizer safety net. `TensorPool` for temporaries; `ArrayPool<T>.Shared` only for managed metadata.

**Performance:** Zero allocations on hot paths. `TensorRef` in kernels, `Span<T>` for views, `[AggressiveInlining]` on small hot methods, `[SkipLocalsInit]` on `stackalloc` paths, `[SuppressGCTransition]` on short P/Invoke. Function-pointer dispatch (`delegate*`) for compute thread pool. Pre-allocate scratch buffers at load time.

**Architecture:** Code against `IBackend` only; each backend delegates to static kernels. Eager execution (no graphs). Pipelines use `IAsyncEnumerable<GenerationProgress>` (readonly record struct). Every CUDA/Vulkan call checked via `.ThrowOnError()`.

**CUDA (source-verified):**
- Library `"cuda"` (not `"nvcuda"`); returns `int` (not enum)
- PTX from disk directory; function handles as `nint` fields (not dictionary)
- Kernel args: `stackalloc void*[]` with **local variables** for stable addresses
  ```csharp
  nint outputArg = output, inputArg = input;
  int nArg = n;
  float epsArg = eps;
  void** args = stackalloc void*[] {&outputArg, &inputArg, &nArg, &epsArg};
  CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
  ```

**Config/Options:** `ModelConfig` as class record with `required` + `init`. Options: three-tier (flat / explicit / custom injection).

**C# Style:** File-scoped namespaces, primary constructors, `readonly`/`sealed`, `readonly record struct` for value types, `record` for config. No `#region`. Nullable enabled. Source-gen JSON (`[JsonSerializable]`).

**Errors:** Fail fast on shape mismatches. Custom exceptions: `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException`. `Environment.FailFast` for unrecoverable compute thread errors. `Debug.Assert` for invariants.

## What NOT to Do
- No Python/C++ wrappers, ONNX Runtime, or managed GPU wrappers (ILGPU, ManagedCuda, ComputeSharp, Vortice)
- No computation graphs — eager execution only
- No managed arrays for tensor data
- No `CuResult` enum, no `[DllImport]` (use `[LibraryImport]`), no `Dictionary<string, nint>` for handles
- No embedded PTX/SPIR-V resources — load from disk
- No reflection JSON — use source-gen contexts

## dotLLM Quick Reference

| Pattern | Implementation |
|---|---|
| CUDA lib | `"cuda"` |
| Return | `int` + `.ThrowOnError()` |
| PTX | `CudaModule.LoadFromFile(path)` |
| Handles | `nint` fields, resolved in ctor |
| Kernel args | `stackalloc void*[]` with locals |
| Tensor dispose | `Interlocked.Exchange` + `AlignedFree` |
| DType | `readonly record struct` |
| ModelConfig | class `record` with `required` + `init` |
| Options | three-tier (flat / explicit / custom) |
| Streaming | `IAsyncEnumerable<readonly record struct>` |
| Thread pool | `delegate*` dispatch |
| Server JSON | `[JsonSerializable]` |
| Worker crash | `Environment.FailFast` |

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md`
- `docs/Design/FILE_STRUCTURE.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`, `docs/Design/VALIDATION_STRATEGY.md`
- `docs/Checklists/`
