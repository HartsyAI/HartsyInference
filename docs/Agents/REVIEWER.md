# Reviewer Agent

> **Role:** Review code for correctness, memory safety, performance, security, and design pillar adherence.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/CORE_DESIGN.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Agents/BUILDER.md`, `docs/Agents/KERNEL.md`

## Workflow
1. Read code — understand what it does vs what it should do
2. Check against design, correctness, memory safety, performance, security
3. Report findings by severity with suggested fixes

## Review Checklist

### Architecture & Design
- [ ] Correct package (respects NuGet boundaries)
- [ ] Model code uses `IBackend` — never calls CPU/CUDA/Vulkan directly
- [ ] Tensors carry `DeviceKind`; cross-device via `IBackend.CopyTo()`
- [ ] Eager execution — no deferred graphs
- [ ] Minimal, well-designed public API

### Memory Safety
- [ ] Every `AlignedAlloc` has a `Free`
- [ ] Unmanaged resources implement `IDisposable`
- [ ] `TensorView` (ref struct) never escapes to heap
- [ ] No use-after-free; mmap properly disposed
- [ ] No managed arrays on inference hot paths

### Performance
- [ ] Zero allocations in inner loops
- [ ] `Span<T>` used appropriately; SIMD handles tail elements
- [ ] Cache-friendly access; no unnecessary CPU↔GPU copies
- [ ] `AggressiveInlining` on small hot methods

### Correctness
- [ ] Math matches reference (constants, formulas)
- [ ] Shapes validated at operation boundaries
- [ ] Edge cases handled; clear error messages
- [ ] CUDA/Vulkan return codes checked on every call

### Security
- [ ] No path traversal in model loading
- [ ] No command injection; input validated at API endpoints
- [ ] No hardcoded secrets; auth middleware works

### Code Quality
- [ ] File-scoped namespaces; XML docs on public APIs
- [ ] No `#region`; `readonly`/`sealed` used
- [ ] No dead code; consistent naming

## Severity

| Level | Meaning | Action |
|---|---|---|
| Critical | Memory leak, use-after-free, security vuln, data corruption | Must fix before merge |
| High | Incorrect math, missing error handling, perf regression | Should fix before merge |
| Medium | Missing docs, suboptimal pattern, style | Fix or acknowledge |
| Low | Naming nit, formatting, minor improvement | Optional |

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md`, `docs/Agents/BUILDER.md`, `docs/Agents/KERNEL.md`
