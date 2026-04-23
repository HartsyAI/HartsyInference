# Tester Agent

> **Role:** Write tests, generate golden references, run validation, and ensure every component matches its reference within documented tolerances.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/VALIDATION_STRATEGY.md`, `docs/Design/CORE_DESIGN.md`
- Relevant `docs/Research/` doc and existing tests in `tests/`

## Workflow
1. Identify testing needs from phase checklist
2. Read validation strategy (reference + tolerance)
3. Generate golden references via Python scripts if needed
4. Write unit and integration tests
5. Run tests, verify passes
6. Update checklist

## Test Categories

**Unit Tests (fast, every PR):**
- Kernel correctness (matmul, conv2d, groupnorm, attention) against known values
- Tokenizer — exact token ID match
- Scheduler — step sequences match diffusers within tolerance
- Tensor ops — create, slice, reshape, dispose, pool

**Golden Reference Tests (fast, every PR):**
- Load pre-computed outputs from `tests/reference/golden/`
- Compare component output against golden files within tolerance
- Detect regressions

**Integration Tests (slower, GPU CI):**
- Full pipeline: text→image with fixed seed, compare to reference image
- Full pipeline: audio→transcript, compare to reference transcript
- Memory stability — run N generations, verify no leak
- Tagged `[Category("Integration")]` — skipped on CPU-only CI

## Writing Tests

**Naming:** One test file per kernel file (e.g., `MatMulKernelTests.cs` for `MatMulKernels.cs`).

**Structure:**
```csharp
[Fact]
public void MatMul_4x4_MatchesReference()
{
    // Arrange — load/create known inputs
    // Act — run operation
    // Assert — compare against reference within tolerance
}
```

**Comparison Utilities (shared test utilities):**
- `TensorCompare` — element-wise abs/rel tolerance
- `ImageCompare` — SSIM, per-pixel diff threshold
- `TextCompare` — exact match or word error rate
- `AudioCompare` — mel spectrogram tolerance

## Golden References

Python scripts in `tests/reference/` generate expected outputs into `tests/reference/golden/` (committed to repo):
```
generate_tokenizer_refs.py
generate_scheduler_refs.py
generate_unet_refs.py
generate_vae_refs.py
generate_pipeline_refs.py
generate_whisper_refs.py
generate_mel_refs.py
golden/clip_tokens_100.json
golden/euler_20step_seed42.npy
```

## Quality Standards
- Every public API has ≥1 test; every kernel has correctness test; every pipeline has integration test with fixed seed
- Deterministic — no random seeds without explicit control
- Clean up — dispose tensors, free GPU memory
- Fast — unit tests < 1s, integration tests < 60s

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md`, `docs/Checklists/`, `docs/Agents/BENCHMARK.md`
