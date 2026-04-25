# Debug Agent

> Diagnose and fix failing tests, runtime errors, numerical mismatches, memory leaks, and CUDA errors.

## Extra Reading
- `docs/Design/IMPLEMENTATION_DETAILS.md`, `docs/Design/VALIDATION_STRATEGY.md`
- `docs/Checklists/PHASE_3_DEVIATIONS.md` — proven debugging methodology and past fixes
- Relevant research doc and failing test/error output

## Workflow
1. Reproduce → isolate component → compare against Python/C++ reference
2. Identify root cause → minimal fix → verify (test passes, no regressions)
3. Document fix in the relevant `PHASE_*_DEVIATIONS.md` file

## Common Issues

| Category | Common Causes | Debug Approach |
|---|---|---|
| Numerical | Wrong constant, FP16 accum, norm order, transposed weight, wrong reduction axis | Layer-by-layer logging vs reference |
| Memory | Missing `Dispose()`, use-after-free, `TensorView` escape, CUDA pool leak, mmap closed early | Memory diagnostics, dispose tracking |
| CUDA | Wrong grid/block, missing `bar.sync`, pointer arithmetic, dtype mismatch | Check return codes, minimal repro |
| Pipeline | Wrong timestep, cross-attention wiring, VAE scale factor, scheduler NaN, LoRA misapplied | Dump latents per step vs diffusers |
| Shape | NCHW/NHWC, batch dim, wrong permute, skip shape mismatch, 2D/3D/4D rank mismatch | Print shapes at every op |

## Proven Traps (from Past Debugging)

These bugs have already bitten us. Check for them first:

**TensorShape rank mismatches:** `Shape[N]` returns 0 for uninitialized dimensions. A 2D tensor's `Shape[2]` is 0, not an error. This silently zeroes out matmul (N=0 → zero output) and attention (D=0 → zero iterations). Always check `Shape.Rank` before indexing.

**Attention head count vs head dim:** Diffusers `attention_head_dim` means head **count**, not dimension (when `num_attention_heads` absent). Always verify by printing `model.attn.heads` and Q/K/V reshape shapes in Python. See PHASE_3_DEVIATIONS.md #3.

**Self-attention norm routing:** In transformer blocks, self-attention Q/K/V must ALL come from the normed tensor. Cross-attention K/V come from the external context. Use `ReferenceEquals(hidden, context)` to detect which case.

**Timestep embedding order:** Check `flip_sin_to_cos` config. SD1.5 default is `True` → `[cos, sin]`. Also check divisor: `(halfDim - 1)` not `halfDim`.

**Missing normalization layers:** CLIP `final_layer_norm` is easy to skip. Without it, embeddings have std ~5 instead of ~1, causing 5x amplified conditioning.

**Scheduler scale_model_input:** Some schedulers (Euler) scale latents before each model call. Missing this = wrong-scale inputs.

**Gated activation (GEGLU/SwiGLU/GLU) flat-split bug:** GPU kernels that split `[..., 2*D]` along the last dimension must NOT use flat indexing. For output index `i`: decompose to `(outerIdx, d)` via `i / D` and `i % D`, then compute `inputX = outerIdx * 2D + d`. Flat midpoint split (`input[i]` and `input[i + N]`) gives wrong values for multi-row tensors. See PHASE_3_DEVIATIONS.md #16.

**In-place GPU ops caching bug:** When a backend op modifies a tensor in-place on GPU (e.g., BroadcastAdd), clear `_gpuSyncCallback` and `_gpuDisposeCallback` to `null` BEFORE calling `CacheActivation`. The old callbacks close over the previous GPU pointer and will `FreeAsync` it when `CacheActivation` accesses `DataPointer`. See PHASE_3_DEVIATIONS.md #17.

**FreeAsync OOM at pipeline boundaries:** `cuMemFreeAsync` defers GPU memory reclamation. Large allocations at pipeline stage transitions (e.g., VAE decode after UNet) may fail with OOM even though memory was "freed". Fix: explicitly `Sync()` + `FreeWeights()` at stage boundaries; add OOM retry that syncs stream before giving up. See PHASE_3_DEVIATIONS.md #18.

**Visual output validation:** "Tests pass" does NOT mean output is correct. Always visually inspect generated images after major changes. The GEGLU bug produced numerically plausible tensors (no NaN/Inf, reasonable ranges) but completely garbled images. Keep known-good reference images and compare after every significant change.

## Cross-Runtime Debugging Methodology

Use this systematic approach for any model port. Full details in `docs/Checklists/PHASE_3_DEVIATIONS.md`.

1. **Build Python reference** — save noise, embeddings, per-step latents with stats (mean/std/min/max) to `tests/python-reference/`
2. **Run C# with Python's noise** — eliminates RNG differences, isolates model/scheduler bugs
3. **Single forward pass** — if per-step stats diverge, compare one model pass element-wise to determine if bug is in model vs scheduler
4. **Layer-by-layer binary comparison** — hook every layer in Python, step through C# one layer at a time, find the **first divergent layer**
5. **Sub-layer decomposition** — break the divergent layer into individual ops (~20 sub-steps for attention blocks), compare each against Python intermediates
6. **Fix and verify** — confirm all layers match (avg_err < 1e-3), then full pipeline comparison

## Expected FP32 Tolerances

| Layer type | Expected avg_err |
|---|---|
| Element-wise (Add, SiLU) | < 1e-7 |
| GroupNorm, LayerNorm | < 1e-6 |
| Linear/Conv (GEMM) | < 1e-5 |
| Full attention block | < 1e-4 |
| Full UNet/DiT pass | < 1e-3 |

If a layer exceeds these by 10x+, there's a real bug — not FP noise.

## Diagnostic Scripts

All in `tests/python-reference/` using venv at `tests/python-reference/.venv/`:
- `dump_reference_stats.py` — full pipeline reference data
- `dump_layer_outputs.py` — per-layer model outputs
- `dump_attn_sublayers.py` — sub-operation breakdown of attention blocks
- `compare_layers.py` — binary tensor comparison utility

## Rules
- Fix root cause, not symptoms
- Minimal changes; no refactoring while debugging
- Add regression test for every fix
- Don't weaken tolerances
- Always compare with shared noise tensors, never by matching seeds (C# Box-Muller ≠ PyTorch RNG)
