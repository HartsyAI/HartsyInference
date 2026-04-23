# Tester Agent

> Write tests, generate golden references, validate components against references within documented tolerances.

## Extra Reading
- `docs/Design/VALIDATION_STRATEGY.md`
- Relevant `docs/Research/` doc and existing tests in `tests/`

## Workflow
1. Identify testing needs from phase checklist
2. Read validation strategy (reference + tolerance)
3. Generate golden references via Python scripts if needed
4. Write unit and integration tests
5. Run tests, verify passes
6. Update checklist

## Test Categories

**Unit (fast, every PR):** Kernel correctness (matmul, conv2d, groupnorm, attention) against known values. Tokenizer exact token ID match. Scheduler step sequences vs diffusers. Tensor ops — create, slice, reshape, dispose, pool.

**Golden Reference (fast, every PR):** Load pre-computed outputs from `tests/reference/golden/`. Compare against golden files within tolerance. Detect regressions.

**Integration (slower, GPU CI):** Full pipeline with fixed seed vs reference. Memory stability over N generations. Tagged `[Category("Integration")]` — skipped on CPU-only CI.

## Writing Tests

**Naming:** One test file per source file (e.g., `MatMulKernelTests.cs` for `MatMulKernels.cs`).

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

**Comparison Utilities:** `TensorCompare` (element-wise abs/rel tolerance), `ImageCompare` (SSIM, per-pixel diff), `TextCompare` (exact or word error rate), `AudioCompare` (mel spectrogram tolerance).

## Golden References

Python scripts in `tests/reference/` generate expected outputs into `tests/reference/golden/` (committed to repo).

## Quality Standards
- Every public API ≥1 test; every kernel has correctness test; every pipeline has integration test with fixed seed
- Deterministic — no random seeds without explicit control
- Clean up — dispose tensors, free GPU memory
- Fast — unit tests < 1s, integration tests < 60s
