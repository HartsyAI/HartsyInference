# Audit Agent

> Review / audit code for correctness, safety, and quality — and diagnose failures. Assumes you've read
> `AGENTS.md` + `docs/CODE_STYLE.md`. Known bug classes live in `docs/Checklists/TROUBLESHOOTING.md` — check
> a finding against it before writing it up.

## Review checklist (by category)

- **Architecture** — right package? math routes through `IBackend`? tensors carry `DeviceKind`, cross-device
  via `CopyTo()`? eager execution, minimal public API?
- **Memory safety** — every `AlignedAlloc` has a `Free`? `IDisposable` on unmanaged holders? `TensorView`
  never escapes to the heap? no use-after-free? no managed arrays on hot paths?
- **Performance** — zero allocations in inner loops? `Span<T>` not `T[]`? SIMD tail handled? no needless
  CPU↔GPU copies? `[AggressiveInlining]` on small hot methods?
- **Correctness** — math matches the reference? shapes validated at entry? edge cases? every CUDA/Vulkan
  call `.ThrowOnError()`?
- **Security** — no path traversal / command injection in model loading? public inputs (sizes/shapes/paths)
  validated? no hardcoded secrets?

## Severity ladder

**Critical** (leak / use-after-free / vuln / corruption → must fix) · **High** (wrong math, missing error
handling, perf regression → should fix) · **Medium** (suboptimal pattern, missing docs) · **Low** (nit).

## The highest-signal things to catch

```csharp
// ❌ Critical: direct DataPointer read on a model weight — crashes after GPU preload + CPU disposal
float* w = (float*)weight.DataPointer;
// ✅ route through the backend so the GPU weight cache is honored
backend.Linear(output, input, weight, bias);
```

```csharp
// ❌ High: hand-rolled element loop where a backend op exists (and this one is on a GPU hot path)
for (int i = 0; i < n; i++) dst[i] = src[off + i];
// ✅ backend.SliceRows / backend.Transpose2D / the shared Utilities/* helper
//    (a CPU-side F32 batch-slice like CfgHelper.SliceBatchElement is fine; the same shape on a GPU
//     forward pass is a residency bug — flag it)
```

```csharp
// ❌ Critical: swallowed / unchecked
try { Load(path); } catch { }
CudaDriverApi.cuMemAlloc(out p, n);
// ✅ log + rethrow at boundaries; check every native return
try { Load(path); } catch (Exception ex) { Logs.Error($"load {path}", ex); throw; }
CudaDriverApi.cuMemAlloc(out p, n).ThrowOnError();
```

## Diagnosing a failure

Reproduce → isolate the component → compare against the Python/C++ reference → apply the **minimal** fix →
verify no regressions → add a regression test.

- Fix the root cause, not the symptom. **Don't refactor while debugging**, and **never weaken a tolerance**
  to make a test pass.
- `INF` (not NaN) points at an F32→F16 overflow, not a logic bug; a step-dependent NaN is a threshold
  crossing. A garbled-but-finite image means the math ran on mis-shaped/mis-split data — check attention
  head split, GEGLU last-dim split, and patchify order first (all in TROUBLESHOOTING.md).
