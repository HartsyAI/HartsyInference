# Kernels and performance

Read relevant backend research and known failures on demand. Profile before choosing an optimization; validate against a scalar/reference implementation with documented dtype- and operation-specific tolerances. Use FP32 accumulation for reduced-precision arithmetic unless the validated algorithm requires otherwise.

## CUDA

- Follow [kernel source/build policy](../../src/HartsyInference.Cuda/Kernels/README.md). CUDA sources produce checked-in PTX; legacy handwritten PTX exists. Load artifacts from disk. Match PTX ISA to the deployment driver; do not impose a universal .version 9.0 header.
- Launch arguments point to stable locals; cache function handles in nint fields. Check native status results.
- Use wide indexing where tensor products can overflow 32 bits. Test multi-row gated activation splits along the last dimension.
- Respect stream ordering, deferred frees, graph buffer lifetime and in-place callback ownership in [core](AGENTS.md). Never bypass GPU residency by directly reading disposed host weights.

## SIMD and Vulkan

- Dispatch supported SIMD widths with scalar/tail coverage. Benchmark wider vectors rather than assuming speedup.
- Vulkan features and subgroup sizes require queried support. Validate full input/output dtype combinations, partial tiles, shared-memory barriers and buffer visibility; no single tiled GEMM replaces every layout/precision path.
- GLSL is source; SPIR-V is a build artifact. Rebuild with src/HartsyInference.Vulkan/Shaders/build.sh and commit generated changes together; register new variants in the build lists.
- Run VulkanShaderDriftTests with a compiler that supports the shader set. An unavailable/inadequate compiler produces inconclusive coverage, not a verified rebuild. See [SPIR-V reference](../Research/SPIRV_COMPUTE_SHADERS.md).

## Measurement

Measure end-to-end throughput/latency and memory as well as kernels; include host glue and transfers. Pin GPU identity (CUDA_DEVICE_ORDER=PCI_BUS_ID when using ordinals), warmup, settings, driver/toolchain and quality gate. Archive results through [benchmark guidance](../../benchmarks/README.md). GPU traits do not automatically filter dotnet test; use explicit filters/resource gates from [style](../CODE_STYLE.md).
