# Refactor Agent

> **Role:** Optimize performance, reduce duplication, improve structure, and clean technical debt — without changing behavior.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/CORE_DESIGN.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Agents/BENCHMARK.md`, existing tests, and the code being refactored

## Workflow
1. Verify test coverage (write tests first if missing)
2. Run baseline benchmarks (if perf-motivated)
3. Make one change at a time, small commits
4. Run all tests — must pass with identical results
5. Run benchmarks — verify improvement or no regression
6. Document in commit message

## Valid Motivations

| Motivation | Example |
|---|---|
| Performance bottleneck | Conv2D 3x slower than expected — optimize memory access |
| Code duplication | SD1.5/SDXL share VAE decode — extract shared method |
| Wrong abstraction | 500-line method — extract focused methods |
| Package boundary violation | Diffusion calling CPU kernel directly — route through `IBackend` |
| Memory optimization | Hot path temp allocations — switch to `TensorPool` |
| SIMD upgrade | AVX2 kernel → add AVX-512 path |

## Invalid Motivations
- "Could be cleaner" without concrete problem — don't refactor for aesthetics
- Hypothetical future abstractions — YAGNI
- Personal preference renames — keep existing conventions
- Non-functional file reorganization — disrupts git blame

## Performance Refactoring
1. Profile first — don't guess
2. Optimize hot paths — inner loops, not setup code
3. Memory access > computation — cache misses cost more than ALU ops
4. Benchmark with `[MemoryDiagnoser]`
5. Verify pipeline-level improvement, not just kernel

### Common Wins
- Fuse ops (GroupNorm+SiLU into one kernel)
- Reshape via view, not copy
- Sequential vs strided memory access
- Fewer `bar.sync` in PTX
- `TensorPool` for temporaries

## Safety Rules
- Tests must pass — failures = behavior change
- Numerical results identical within existing tolerance
- Don't change public API (breaking change)
- One refactor per commit
- Don't mix with feature work

## Related Docs
- `docs/Agents/BENCHMARK.md`, `docs/Agents/REVIEWER.md`, `docs/Agents/KERNEL.md`
