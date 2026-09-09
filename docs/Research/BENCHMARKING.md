# Cross-runtime benchmarking

Use [PROFILING_METHODOLOGY.md](PROFILING_METHODOLOGY.md) and the
[benchmark harness](../../benchmarks/README.md). Results belong in [scoreboards](../../benchmarks/scoreboards/).

Compare identical checkpoint/quant, geometry, scheduler, steps, conditioning, and hardware. Numerical parity
must share saved noise/embeddings; equal seeds across different RNG implementations do not give equal inputs.
Measure cold startup, warm generation/step latency, and peak memory separately. Sample enough to expose
variance and GPU contention; a one-second nvidia-smi poll can miss short memory peaks.

Drive current CLI/Engine/Swarm paths rather than deleted generation-test names. Inspect actual output as well
as metrics. A speedup only matters when the same quality gate passes. Investigate regressions against a
matched recorded baseline; do not infer them from old estimates or multiply speculative kernel speedups.
