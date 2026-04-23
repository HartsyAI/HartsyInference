# Agent Core — Shared Context & Routing

> **Every agent conversation starts here.** Read this file first, then load the specialized agent file for your task.

## Before Any Task

1. Read `docs/CODE_STYLE.md` — mandatory, no exceptions
2. Read `docs/Design/CORE_DESIGN.md` — architecture overview
3. Check `docs/Checklists/` — find the active phase (earliest with unchecked items)

## Task Routing

Pick the specialized agent file that matches your task. Read it before starting work.

| Task | Agent File |
|---|---|
| Research a topic | `RESEARCH.md` |
| Plan an implementation | `ARCHITECT.md` |
| Write implementation code | `BUILDER.md` |
| Write SIMD/PTX/SPIR-V kernels | `KERNEL.md` |
| Write or run tests | `TESTER.md` |
| Review code | `REVIEWER.md` |
| Debug a failure | `DEBUG.md` |
| Refactor or optimize | `REFACTOR.md` |
| Build server/API endpoints | `API.md` |
| Wire cross-package integration | `INTEGRATION.md` |
| Convert model formats | `CONVERT.md` |
| Run benchmarks | `BENCHMARK.md` |
| Update docs/README | `DOCS.md` |
| Update checklists | `CHECKLIST.md` |
| Package for NuGet | `DEPLOY.md` |

If your task spans multiple agents (e.g., build + test), load both files.

## Shared Design Rules

These apply to ALL agents. Specialized files only add task-specific rules.

**Pure C# only** — no native shared libraries, no Python, no C++ wrappers, no ONNX Runtime, no managed GPU wrappers (ILGPU, ManagedCuda, ComputeSharp, Vortice).

**Eager execution** — no computation graphs; ops execute immediately.

**Zero GC on hot paths** — no managed allocations during inference. Use `NativeMemory.AlignedAlloc(byteCount, 64)`, `TensorPool` for temporaries, `ArrayPool<T>.Shared` only for managed metadata.

**IBackend abstraction** — model code never calls CPU/CUDA/Vulkan directly. Each backend delegates to static kernels.

**Package boundaries** — respect `docs/Design/NUGET_PACKAGE_DESIGN.md`. Don't leak CUDA/Vulkan into CPU packages.

## dotLLM Patterns (Single Source of Truth)

All code follows patterns verified from the dotLLM codebase. See `docs/CODE_STYLE.md` for full P/Invoke and disposal patterns.

### Tensor Type System

| Type | Owns Memory | Dispose | Use For |
|---|---|---|---|
| `Tensor` | Yes | `Interlocked.Exchange` + `AlignedFree` | Weights, intermediates |
| `TensorView` | No | No-op | Borrowed refs, mmap slices |
| `TensorRef` | No | N/A (value type) | Kernel hot paths |

**Rules:** Creator disposes `Tensor`. `TensorView` never outlives backing memory. `TensorRef` is stack-only.

### CUDA Launch Pattern
```csharp
nint outputArg = output, inputArg = input;
int nArg = n;
float epsArg = eps;
void** args = stackalloc void*[] {&outputArg, &inputArg, &nArg, &epsArg};
CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
```
Key: `stackalloc void*[]` with **local variables** for stable addresses. Never pass field refs directly.

### PTX Loading
PTX from disk via `CudaModule.LoadFromFile(path)`. Function handles as `nint` fields (not dictionary). Target `sm_80` minimum.

### Config & Options
- `ModelConfig`: class `record` with `required` + `init`
- Options: three-tier (flat props / explicit composition / custom injection)
- JSON: source-generated `[JsonSerializable]` contexts — no reflection

### Streaming
`IAsyncEnumerable<GenerationProgress>` where `GenerationProgress` is `readonly record struct`.

### Error Handling
- Shape mismatches: fail fast with `SharpInferenceException`
- CUDA/Vulkan: `.ThrowOnError()` on every call
- Compute threads: `Environment.FailFast` for unrecoverable errors
- Custom exceptions: `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException`

### Performance Attributes
| Attribute | When |
|---|---|
| `[AggressiveInlining]` | Small hot methods, tensor accessors, SIMD helpers |
| `[SkipLocalsInit]` | Large `stackalloc` paths |
| `[SuppressGCTransition]` | Short CUDA P/Invoke (< 1us) |

### What NOT to Do
- No Python/C++ wrappers, ONNX Runtime, managed GPU wrappers
- No computation graphs — eager only
- No managed arrays for tensor data
- No `CuResult` enum, no `[DllImport]` (use `[LibraryImport]`)
- No `Dictionary<string, nint>` for CUDA handles
- No embedded PTX/SPIR-V resources — load from disk
- No reflection JSON — use source-gen contexts
