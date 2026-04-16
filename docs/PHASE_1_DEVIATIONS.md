# Phase 1 — Deviations from Design Plan

This document records intentional deviations from `CORE_DESIGN.md` and `IMPLEMENTATION_DETAILS.md` discovered during Phase 1 implementation. Each deviation includes rationale and a recommendation for whether to align or keep the deviation.

---

## 1. DType: Enum vs Readonly Record Struct

**Design:** `DType` is a `readonly record struct` with fields `SizeInBytes`, `IsQuantized`, `BlockByteSize`, `BlockElementCount`.

**Actual:** `DType` is a `byte` enum with static extension methods (`ElementSize()`, `BlockSize()`, `BlockByteSize()`, `IsQuantized()`).

**Rationale:** An enum is simpler, fits in a single byte, enables exhaustive `switch` checking by the compiler, and avoids accidental construction of invalid dtypes. The extension methods provide the same metadata without carrying fields in every tensor.

**Recommendation:** Keep the enum. The design's record struct offers no practical advantage over enum + extensions, and the enum prevents invalid states by construction.

---

## 2. TensorRef Not Implemented — TensorView Used Instead

**Design:** Dual tensor types — `Tensor` (sealed class, IDisposable) for lifecycle, `TensorRef` (readonly record struct) for zero-alloc compute in all kernel signatures. `IBackend` methods accept `TensorRef` parameters internally.

**Actual:** `Tensor` (sealed class, IDisposable) exists as designed. Instead of `TensorRef`, there is `TensorView` — a `readonly unsafe ref struct` (not a record struct). `IBackend` methods accept `Tensor` directly, not `TensorRef`/`TensorView`.

**Rationale:** `TensorView` as a `ref struct` cannot escape to the heap, which is a stronger safety guarantee than a `readonly record struct` that could accidentally be stored in a field and outlive its source data. However, `ref struct` cannot be stored in generic collections or used as interface method parameters, which is why `IBackend` uses `Tensor` directly.

**Recommendation:** Consider adding `TensorRef` as a `readonly record struct` alongside `TensorView` for Phase 2. `TensorRef` would be used in kernel signatures (matching the design), while `TensorView` would remain for slicing/sub-views. For Phase 1, the current approach is functional and safe.

---

## 3. IBackend Accepts Tensor, Not TensorRef

**Design:** IBackend methods accept `TensorRef` parameters (zero-alloc views), with public methods accepting `Tensor` and calling `.AsRef()` internally.

**Actual:** IBackend methods accept `Tensor` directly. No `.AsRef()` conversion exists.

**Rationale:** Without `TensorRef` (see deviation #2), there is no zero-alloc struct to convert to. Passing `Tensor` (a reference type) is still zero-alloc at the call site — it just passes a pointer-sized reference. The design's concern about GC pressure does not apply since `Tensor` is already a reference on the stack.

**Recommendation:** Revisit in Phase 2 if `TensorRef` is added. For Phase 1, the overhead is negligible — no boxing, no allocation.

---

## 4. ComputeThreadPool Simplified

**Design:** Custom `ComputeThreadPool` with two modes — SpinWait (latency-critical denoising steps) and EventBased (throughput-oriented model loading), switching automatically.

**Actual:** `ComputeThreadPool` is a thin wrapper over `Parallel.For` with a fixed `MaxDegreeOfParallelism`. No SpinWait or EventBased mode switching. Single-iteration optimization (falls back to sequential for count <= 1).

**Rationale:** Phase 1 only has CPU kernels and no denoising loop yet. The dual-mode design is an optimization for the inference hot path in later phases (Diffusion, Audio). Implementing it now would be premature — there is no workload to benchmark or validate the mode switching against.

**Recommendation:** Implement dual-mode thread pool in Phase 2 or Phase 3 when diffusion pipeline streaming is built and can be profiled. The current `Parallel.For` wrapper is sufficient for Phase 1 kernel testing.

---

## 5. NumaAffinity Simplified

**Design:** NUMA-aware core pinning with P-core detection for hybrid architectures (Alder Lake, etc.).

**Actual:** `NumaAffinity.cs` exists (45 lines) but only provides basic core count detection and affinity hints. Full P-core/E-core detection and NUMA topology mapping are not implemented.

**Rationale:** .NET 10 does not expose NUMA topology or hybrid core types through managed APIs. Implementing this requires platform-specific P/Invoke (Windows `GetLogicalProcessorInformationEx`, Linux `/sys/devices/system/cpu/`). This is an optimization that should be profiled on actual hybrid hardware before investing in.

**Recommendation:** Defer to Phase 3 (optimization phase). Profile on Intel 12th/13th/14th Gen and AMD Ryzen 7000+ to determine if NUMA/P-core affinity makes a measurable difference for inference workloads.

---

## 6. No Fused Kernels Yet

**Design:** Fused kernels for bandwidth reduction — GroupNorm+SiLU, Conv2D+bias+activation.

**Actual:** All kernels are separate, unfused operations. Conv2D, GroupNorm, SiLU, bias addition are all independent IBackend calls.

**Rationale:** Fused kernels are an optimization for the GPU pipeline (reducing memory bandwidth) and the diffusion inference loop. Phase 1 establishes correctness of individual kernels. Fusion should be done after individual kernels are validated against reference implementations.

**Recommendation:** Implement fused CPU kernels in Phase 3 (optimization). Implement fused GPU kernels in Phase 2 alongside the CUDA backend. Always keep unfused versions as reference for correctness validation.

---

## Summary

| # | Deviation | Severity | Action |
|---|---|---|---|
| 1 | DType is enum, not record struct | Low | Keep — better design |
| 2 | TensorView ref struct instead of TensorRef record struct | Medium | Add TensorRef in Phase 2 |
| 3 | IBackend accepts Tensor, not TensorRef | Medium | Revisit with TensorRef in Phase 2 |
| 4 | ComputeThreadPool lacks dual-mode | Low | Implement when diffusion loop exists |
| 5 | NumaAffinity simplified | Low | Defer to Phase 3 optimization |
| 6 | No fused kernels | Low | Implement in Phase 2/3 |

None of these deviations impact Phase 1 correctness or functionality. All are either intentional simplifications (deferring optimization to later phases) or improvements over the original design (enum DType, ref struct TensorView).
