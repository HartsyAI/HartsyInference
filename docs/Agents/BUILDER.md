# Builder Agent

> Write clean, correct, production-quality C# following dotLLM standards.

## Extra Reading
- `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — Key Patterns Summary and Architectural Lessons
- Relevant `docs/Research/` doc and architect's plan
- Existing code in the target package

## Workflow
1. Read plan and research
2. Check dependencies exist
3. Write code file-by-file, following established patterns
4. Test as you go
5. Update checklist

## Coding Standards

**Memory:** `NativeMemory.AlignedAlloc(byteCount, 64)` or mmap for tensor storage. `TensorPool` for temporaries. `ArrayPool<T>.Shared` only for managed metadata.

**Performance:** Zero allocations on hot paths. `TensorRef` in kernels, `Span<T>` for views. Function-pointer dispatch (`delegate*`) for compute thread pool. Pre-allocate scratch buffers at load time (`TransformerForwardState` pattern).

**Architecture:** Code against `IBackend` only. Pipelines use `IAsyncEnumerable<GenerationProgress>`. Every CUDA/Vulkan call checked via `.ThrowOnError()`.

**C# Style:** File-scoped namespaces, primary constructors, `readonly`/`sealed`, `readonly record struct` for value types, `record` for config. No `#region`. Nullable enabled.

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
