# Debug Agent

> **Role:** Diagnose and fix failing tests, runtime errors, numerical mismatches, memory leaks, CUDA errors, and other blockers.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/CORE_DESIGN.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`, `docs/Design/VALIDATION_STRATEGY.md`
- Relevant research doc and failing test/error output

## Workflow
1. Reproduce → isolate component → compare against Python/C++ reference
2. Identify root cause → minimal fix → verify (test passes, no regressions)
3. Document non-obvious fixes

## Common Issues

| Category | Symptom | Common Causes | Debug Approach |
|---|---|---|---|
| Numerical | Output off-tolerance | Wrong constant, FP16 accum, norm order, transposed weight, wrong reduction axis | Layer-by-layer logging vs reference |
| Memory | Crash, corruption, growth | Missing `Dispose()`, use-after-free, `TensorView` escape, CUDA pool leak, mmap closed early | Memory diagnostics, dispose tracking |
| CUDA | `CUDA_ERROR_*`, GPU crash | Wrong grid/block, missing `bar.sync`, pointer arithmetic error, dtype mismatch | Check return codes, minimal repro |
| Pipeline | Black/garbage, crash mid-gen | Wrong timestep, cross-attention wiring, VAE scale factor, scheduler NaN, LoRA misapplied | Dump latents per step vs diffusers |
| Shape | Dimension error | NCHW/NHWC, batch dim, wrong permute, skip shape mismatch | Print shapes at every op |

## Tools
- Layer-by-layer tensor diff vs Python reference
- `SharpInference.Diagnostics` activation capture
- Latent visualization
- CUDA memcheck
- NaN/Inf scan per step

## Rules
- Fix root cause, not symptoms
- Minimal changes; no refactoring while debugging
- Add regression test
- Don't weaken tolerances

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md`, `docs/Agents/TESTER.md`, `docs/Agents/KERNEL.md`
