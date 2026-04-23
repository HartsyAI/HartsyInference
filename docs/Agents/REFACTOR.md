# Refactor Agent

> Optimize performance, reduce duplication, improve structure — without changing behavior.

## Extra Reading
- `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Agents/BENCHMARK.md`, existing tests, and the code being refactored

## Workflow
1. Verify test coverage (write tests first if missing)
2. Run baseline benchmarks (if perf-motivated)
3. One change at a time, small commits
4. All tests must pass with identical results
5. Benchmarks verify improvement or no regression

## Valid Motivations
- Performance bottleneck (profiled, not guessed)
- Code duplication (shared logic extracted)
- Wrong abstraction (500-line method → focused methods)
- Package boundary violation (route through `IBackend`)
- Memory optimization (hot path temps → `TensorPool`)
- SIMD upgrade (AVX2 → add AVX-512 path)

## Invalid Motivations
- "Could be cleaner" without concrete problem
- Hypothetical future abstractions (YAGNI)
- Personal preference renames
- Non-functional file reorganization

## Performance Refactoring
1. Profile first — don't guess
2. Optimize hot paths — inner loops, not setup code
3. Memory access > computation — cache misses cost more than ALU ops
4. Benchmark with `[MemoryDiagnoser]`
5. Verify pipeline-level improvement, not just kernel

**Common Wins:** Fuse ops (GroupNorm+SiLU), reshape via view not copy, sequential vs strided memory access, fewer `bar.sync` in PTX, `TensorPool` for temporaries.

## Safety Rules
- Tests must pass — failures = behavior change
- Numerical results identical within existing tolerance
- Don't change public API
- One refactor per commit; don't mix with feature work
