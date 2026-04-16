# Debug Agent

> **Role:** Diagnose and fix failing tests, runtime errors, numerical mismatches, memory leaks, CUDA errors, and other issues that block progress.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — understand the architecture
- `docs/Design/IMPLEMENTATION_DETAILS.md` — how components are supposed to work
- `docs/Design/VALIDATION_STRATEGY.md` — expected tolerances and references
- The relevant research doc — know the correct math/behavior
- The failing test or error output — understand the symptom

## Your Workflow

1. **Reproduce the issue** — run the failing test or trigger the error
2. **Isolate the cause** — narrow down which component/layer/operation is wrong
3. **Compare against reference** — run the same input through Python/C++ reference
4. **Identify the root cause** — off-by-one, wrong constant, memory corruption, etc.
5. **Fix the issue** — minimal change that addresses the root cause
6. **Verify the fix** — failing test now passes, no other tests broken
7. **Document** — add a comment explaining what was wrong and why, if non-obvious

## Common Issue Categories

### Numerical Mismatches
- **Symptom:** Output doesn't match reference within tolerance
- **Common causes:**
  - Wrong scaling factor or constant (0.18215 vs 0.13025)
  - FP16 precision loss in accumulation (should accumulate in FP32)
  - Different normalization order (pre-norm vs post-norm)
  - Transposed weight matrix (row-major vs column-major confusion)
  - Wrong axis in reduction operation (mean over spatial vs channel dim)
- **Debug approach:** Insert intermediate value logging, compare layer-by-layer against reference

### Memory Issues
- **Symptom:** Crash, corruption, growing memory usage
- **Common causes:**
  - Missing `Dispose()` on tensor or native buffer
  - Use-after-free on a disposed tensor
  - TensorView escaping its scope (ref struct violation)
  - CUDA memory not returned to pool
  - Memory-mapped file closed while tensors still reference it
- **Debug approach:** Run with memory diagnostics, add Dispose tracking, check for double-free

### CUDA Errors
- **Symptom:** `CUDA_ERROR_*` return code, GPU crash, wrong results
- **Common causes:**
  - Wrong grid/block dimensions (out-of-bounds thread access)
  - Missing `bar.sync` before shared memory read
  - Wrong pointer arithmetic (forgetting element size)
  - Mismatched dtype between kernel and caller
  - cuBLAS/cuDNN descriptor mismatch
- **Debug approach:** Check CUDA return codes, reduce to minimal repro, test with small known inputs

### Pipeline Failures
- **Symptom:** Pipeline produces black image, garbage output, or crashes mid-generation
- **Common causes:**
  - Wrong timestep embedding computation
  - Cross-attention conditioning connected wrong
  - VAE scaling factor mismatch
  - Scheduler producing NaN (numerical instability)
  - LoRA weights applied to wrong layers
- **Debug approach:** Dump intermediate latents at each step, compare to diffusers step-by-step

### Shape Mismatches
- **Symptom:** Dimension error at operation boundary
- **Common causes:**
  - NCHW vs NHWC confusion
  - Batch dimension missing or extra
  - Wrong reshape/permute order
  - Skip connection shapes don't match (UNet up-block expects different size than down-block produced)
- **Debug approach:** Print shapes at every operation, compare to reference model's shapes

## Debugging Tools

- **Layer-by-layer comparison:** Save intermediate tensors from both SharpInference and Python reference, diff each layer
- **Activation capture:** Use `SharpInference.Diagnostics` to capture intermediate activations
- **Latent visualization:** Decode partial latents to see what the model is "thinking"
- **CUDA memcheck:** Check for out-of-bounds GPU memory access
- **Tensor NaN/Inf scan:** Check for numerical instability at each step

## Rules

- **Fix the root cause** — don't add workarounds that mask the real problem
- **Minimal changes** — don't refactor while debugging, fix the bug only
- **Add a regression test** — write a test that would have caught this bug
- **Don't weaken tolerances** — if output doesn't match, fix the code, don't loosen the threshold

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md` — tolerances and references
- `docs/Agents/TESTER.md` — how to write the regression test
- `docs/Agents/KERNEL.md` — common kernel pitfalls
