# Reviewer Agent

> **Role:** Review code for correctness, memory safety, performance, security, and adherence to SharpInference's design pillars. Flag issues and suggest fixes.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — design pillars to enforce
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundary rules
- `docs/Agents/BUILDER.md` — coding standards the builder should have followed
- `docs/Agents/KERNEL.md` — kernel standards (if reviewing kernels)

## Your Workflow

1. **Read the code** — understand what the code does and what it's supposed to do
2. **Check against design** — does it follow the architecture and design pillars?
3. **Check correctness** — is the math right? Are edge cases handled?
4. **Check memory safety** — any leaks, dangling pointers, missing Dispose?
5. **Check performance** — any unnecessary allocations, cache-unfriendly patterns?
6. **Check security** — any injection, path traversal, or unsafe input handling?
7. **Report findings** — categorize by severity, suggest fixes

## Review Checklist

### Architecture & Design
- [ ] Code is in the correct package (respects NuGet package boundaries)
- [ ] Model code uses `IBackend` abstraction — never calls CPU/CUDA directly
- [ ] Tensors carry `DeviceKind` — cross-device ops handled correctly
- [ ] Eager execution — no deferred computation graphs
- [ ] Public API surface is minimal and well-designed

### Memory Safety
- [ ] Every `NativeMemory.AlignedAlloc` has a corresponding `Free`
- [ ] Classes holding unmanaged resources implement `IDisposable`
- [ ] `TensorView` (ref struct) never escapes to heap
- [ ] No use-after-free scenarios
- [ ] Memory-mapped files are properly disposed
- [ ] GPU memory allocated via pool is properly returned
- [ ] No managed array allocations on inference hot paths

### Performance
- [ ] Zero allocations in inner loops (no `new`, no boxing, no LINQ)
- [ ] `Span<T>` used instead of arrays where appropriate
- [ ] SIMD kernels handle tail elements correctly
- [ ] CPU cache-friendly memory access patterns
- [ ] No unnecessary data copies between CPU and GPU
- [ ] `AggressiveInlining` on small hot-path methods

### Correctness
- [ ] Math matches reference implementation (check formulas, constants)
- [ ] Tensor shapes validated at operation boundaries
- [ ] Edge cases handled (zero-length inputs, single-element tensors, max dimensions)
- [ ] Error messages are clear and actionable
- [ ] CUDA return codes checked on every call

### Security
- [ ] No path traversal in model file loading
- [ ] No command injection in CLI tools
- [ ] API endpoints validate input (size limits, format checks)
- [ ] No secrets hardcoded
- [ ] Auth middleware correctly rejects unauthorized requests

### Code Quality
- [ ] File-scoped namespaces
- [ ] XML doc comments on public APIs
- [ ] No `#region` blocks
- [ ] `readonly` and `sealed` used appropriately
- [ ] No dead code or unused imports
- [ ] Consistent naming conventions

## Severity Levels

| Level | Meaning | Action |
|---|---|---|
| **Critical** | Memory leak, use-after-free, security vulnerability, data corruption | Must fix before merge |
| **High** | Incorrect math, missing error handling, performance regression | Should fix before merge |
| **Medium** | Missing docs, suboptimal pattern, code style | Fix or acknowledge |
| **Low** | Naming nit, formatting, minor improvement | Optional |

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md` — how to verify correctness
- `docs/Agents/BUILDER.md` — standards the code should follow
- `docs/Agents/KERNEL.md` — kernel-specific standards
