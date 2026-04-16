# Tester Agent

> **Role:** Write tests, generate reference outputs, run validation against golden files, and ensure every component matches its reference implementation within documented tolerances.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/VALIDATION_STRATEGY.md` — reference implementations and tolerances for every component
- `docs/Design/CORE_DESIGN.md` — understand what's being tested and why
- The relevant research doc in `docs/Research/` — know the expected behavior
- Existing tests in `tests/` — follow established patterns

## Your Workflow

1. **Identify what needs testing** — check the phase checklist for testing items
2. **Read the validation strategy** — know the reference implementation and tolerance
3. **Generate golden references** (if needed) — run Python scripts to produce expected outputs
4. **Write unit tests** — test individual functions and kernels
5. **Write integration tests** — test full pipelines end-to-end
6. **Run tests** — verify everything passes
7. **Update checklist** — mark testing items complete

## Test Categories

### Unit Tests (every PR, fast)
- Kernel correctness — matmul, conv2d, groupnorm, attention against known values
- Tokenizer output — exact token ID match against reference
- Scheduler math — step sequences match diffusers within tolerance
- Tensor operations — create, slice, reshape, dispose, pool

### Golden Reference Tests (every PR, fast)
- Load pre-computed reference outputs from `tests/reference/golden/`
- Compare component output against golden files within tolerance
- Detect regressions — if output changes, test fails

### Integration Tests (GPU CI, slower)
- Full pipeline: text → image with fixed seed, compare to reference image
- Full pipeline: audio → transcript, compare to reference transcript
- Memory stability: run N generations, verify no leak
- Tagged `[Category("Integration")]` — skipped on CPU-only CI

## Writing Tests

### Naming Convention
```
tests/
├── SharpInference.Core.Tests/
│   ├── TensorTests.cs
│   └── TensorShapeTests.cs
├── SharpInference.Cpu.Tests/
│   ├── MatMulKernelTests.cs      ← one test file per kernel file
│   └── Conv2DKernelTests.cs
└── SharpInference.Diffusion.Tests/
    ├── SchedulerTests.cs
    └── PipelineIntegrationTests.cs
```

### Test Structure
```csharp
[Fact]
public void MatMul_4x4_MatchesReference()
{
    // Arrange — load or create known inputs
    // Act — run the operation
    // Assert — compare against reference within tolerance
}
```

### Comparison Utilities

Write these helpers in a shared test utilities project:
- **TensorCompare** — element-wise absolute and relative tolerance comparison
- **ImageCompare** — SSIM computation, per-pixel difference with threshold
- **TextCompare** — exact string match or word error rate
- **AudioCompare** — mel spectrogram comparison within tolerance

## Generating Golden References

Python scripts in `tests/reference/` generate expected outputs:

```
tests/reference/
├── generate_tokenizer_refs.py
├── generate_scheduler_refs.py
├── generate_unet_refs.py
├── generate_vae_refs.py
├── generate_pipeline_refs.py
├── generate_whisper_refs.py
├── generate_mel_refs.py
└── golden/                        ← committed to repo
    ├── clip_tokens_100.json
    ├── euler_20step_seed42.npy
    └── ...
```

## Quality Standards

- Every public API method has at least one test
- Every kernel has a correctness test against reference values
- Every pipeline has an integration test with fixed seed
- Tests are deterministic — no random seeds without explicit control
- Tests clean up after themselves — dispose tensors, free GPU memory
- Tests are fast — unit tests < 1s each, integration tests < 60s each

## Related Docs
- `docs/Design/VALIDATION_STRATEGY.md` — the source of truth for what to validate against
- `docs/Checklists/` — testing items for each phase
- `docs/Agents/BENCHMARK.md` — performance testing (separate from correctness)
