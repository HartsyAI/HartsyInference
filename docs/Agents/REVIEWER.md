# Reviewer Agent

> Review code for correctness, memory safety, performance, security, and design pillar adherence.

## Extra Reading
- `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Agents/BUILDER.md`, `docs/Agents/KERNEL.md`

## Workflow
1. Read code — understand what it does vs what it should do
2. Check against design, correctness, memory safety, performance, security
3. Report findings by severity with suggested fixes

## Review Checklist

**Architecture:** Correct package? Uses `IBackend`? Tensors carry `DeviceKind`? Cross-device via `CopyTo()`? Eager execution? Minimal public API?

**Memory Safety:** Every `AlignedAlloc` has a `Free`? `IDisposable` on unmanaged resources? `TensorView` never escapes to heap? No use-after-free? No managed arrays on hot paths?

**Performance:** Zero allocations in inner loops? `Span<T>` used? SIMD handles tail? Cache-friendly access? No unnecessary CPU↔GPU copies? `AggressiveInlining` on hot methods?

**Correctness:** Math matches reference? Shapes validated? Edge cases handled? CUDA/Vulkan return codes checked?

**Security:** No path traversal in model loading? No command injection? Input validated at API endpoints? No hardcoded secrets?

**Code Quality:** File-scoped namespaces? XML docs on public APIs? No `#region`? `readonly`/`sealed`? No dead code?

## Severity

| Level | Meaning | Action |
|---|---|---|
| Critical | Memory leak, use-after-free, security vuln, data corruption | Must fix before merge |
| High | Incorrect math, missing error handling, perf regression | Should fix before merge |
| Medium | Missing docs, suboptimal pattern, style | Fix or acknowledge |
| Low | Naming nit, formatting, minor improvement | Optional |
