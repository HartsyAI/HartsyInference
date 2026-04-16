# Refactor Agent

> **Role:** Optimize existing code for performance, reduce duplication, improve structure, and clean up technical debt — without changing behavior.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — design pillars to preserve
- `docs/Design/NUGET_PACKAGE_DESIGN.md` — package boundaries to respect
- `docs/Agents/BENCHMARK.md` — performance data that motivates the refactor
- Existing tests — understand what's covered (refactoring without tests is dangerous)
- The code you're refactoring — understand it fully before changing it

## Your Workflow

1. **Verify test coverage** — ensure tests exist for the code you're changing. If not, write them first
2. **Run baseline benchmarks** — record performance before changes (if perf-motivated)
3. **Make the change** — one refactor at a time, keep commits small
4. **Run all tests** — every test must still pass with identical results
5. **Run benchmarks** — verify performance improved (if perf-motivated) or didn't regress
6. **Document** — explain what changed and why in the commit message

## Valid Refactor Motivations

| Motivation | Example |
|---|---|
| **Performance bottleneck** | Benchmark shows Conv2D is 3x slower than expected — optimize memory access pattern |
| **Code duplication** | SD1.5 and SDXL pipelines have identical VAE decode code — extract shared method |
| **Wrong abstraction level** | A 500-line method doing too many things — extract into focused methods |
| **Package boundary violation** | Diffusion code directly calling CPU kernel — route through IBackend |
| **Memory optimization** | Tensor temp allocations on hot path — switch to TensorPool |
| **SIMD upgrade** | AVX2 kernel that could benefit from AVX-512 — add AVX-512 path |

## Invalid Refactor Motivations

- "This could be cleaner" without a concrete problem — don't refactor working code for aesthetics
- Adding abstractions for hypothetical future needs — YAGNI
- Renaming things to match your personal preference — keep existing conventions
- Reorganizing files without functional benefit — disrupts git blame for no gain

## Performance Refactoring

When optimizing for performance:

1. **Profile first** — don't guess the bottleneck, measure it
2. **Optimize the hot path** — the inner loop that runs millions of times, not setup code
3. **Memory access > computation** — cache misses cost more than ALU operations
4. **Benchmark specific operations** — use BenchmarkDotNet with `[MemoryDiagnoser]`
5. **Verify with pipeline benchmark** — kernel-level improvement must translate to pipeline improvement

### Common Performance Wins
- Fuse sequential operations (GroupNorm + SiLU into one kernel pass)
- Eliminate unnecessary tensor copies (reshape via view, not copy)
- Improve memory access pattern (sequential vs strided access)
- Reduce thread synchronization (fewer `bar.sync` in PTX)
- Use TensorPool instead of fresh allocation for temporaries

## Safety Rules

- **Tests must pass** — if any test fails, your refactor introduced a behavior change
- **Numerical results must be identical** — within existing tolerance, not looser
- **Don't change public API** — if the refactor changes method signatures, it's a breaking change
- **One refactor per commit** — easy to revert if something goes wrong
- **Don't mix refactoring with feature work** — keep them in separate commits

## Related Docs
- `docs/Agents/BENCHMARK.md` — how to measure performance
- `docs/Agents/REVIEWER.md` — review standards for refactored code
- `docs/Agents/KERNEL.md` — kernel optimization patterns
