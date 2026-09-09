# CUDA performance research map

The Phase B plan is historical (git history); the harness it proposed already exists.
[ROADMAP.md](../Checklists/ROADMAP.md#2-gpu-kernel-performance) owns open work and
[scoreboards](../../benchmarks/scoreboards/) own measurements.

- [PROFILING_METHODOLOGY.md](PROFILING_METHODOLOGY.md): matched inputs, hardware/software fingerprints,
  cold/warm separation, multiple trials, GPU vs host timing, significance and quality gates.
- [DEEP_KERNEL_OPTIMIZATION.md](DEEP_KERNEL_OPTIMIZATION.md): tensor-core tiling, fusion, graphs, convolution.
- [STEP_ACCELERATION.md](STEP_ACCELERATION.md): distillation, caching, and guidance techniques.
- [QUANTIZATION_LOW_PRECISION_INFERENCE.md](QUANTIZATION_LOW_PRECISION_INFERENCE.md): precision methods.
- [MEMORY_SCHEDULING_SERVING.md](MEMORY_SCHEDULING_SERVING.md): allocation, streaming, cache/serving choices.

Historical H6 (referenced by MemoryAllocFreeBenchmarks): measure allocation/reuse overhead before adding
an arena. This is a hypothesis to test, not a fixed speedup or an outstanding implementation order.
Dependency pins live in benchmarks/python-baseline/requirements.txt; they describe that baseline environment,
not the minimum or latest dependencies for every reference model.
