# Kernels and performance

Use this file for GPU math, SIMD, PTX, SPIR-V and performance work. Read the relevant backend research and known failures
when you need them. Profile before you choose an optimization. Validate each kernel against a scalar or reference
implementation, using tolerances documented for that dtype and operation. Accumulate in FP32 for reduced-precision
arithmetic unless the validated algorithm requires otherwise.

## CUDA

- Follow the [kernel source and build policy](../../src/HartsyInference.Cuda/Kernels/README.md). CUDA sources produce
  checked-in PTX, and some legacy handwritten PTX remains. Load artifacts from disk. Match the PTX ISA to the deployment
  driver, because there is no universal `.version 9.0` header.
- Point launch arguments at stable locals, and cache function handles in `nint` fields. Check every native status result.
- Use wide indexing wherever a tensor product can overflow 32 bits. Test multi-row gated activation splits along the
  last dimension.
- Respect stream ordering, deferred frees, graph buffer lifetimes and in-place callback ownership, as described in
  [engine patterns](ENGINE_PATTERNS.md). Never read disposed host weights directly, because that bypasses GPU residency.

## SIMD and Vulkan

- Dispatch each supported SIMD width with scalar and tail coverage. Benchmark wider vectors instead of assuming they are faster.
- Query Vulkan feature and subgroup-size support before you rely on it. Validate every input and output dtype combination,
  partial tiles, shared-memory barriers and buffer visibility. One tiled GEMM does not cover every layout and precision path.
- GLSL is the source, and SPIR-V is a build artifact. Rebuild with `src/HartsyInference.Vulkan/Shaders/build.sh`, commit
  the generated changes together with the source, and register new variants in the build lists.
- Run `VulkanShaderDriftTests` with a compiler that supports the shader set. A missing or inadequate compiler gives
  inconclusive coverage, which is not a verified rebuild. See the [SPIR-V reference](../Research/SPIRV_COMPUTE_SHADERS.md).

## Measurement

Measure end-to-end throughput, latency and memory as well as the kernels, because host glue and transfers count. Pin the GPU
identity (set `CUDA_DEVICE_ORDER=PCI_BUS_ID` when you use ordinals), and record warmup, settings, driver and toolchain, and
the quality gate. Archive results following the [benchmark guidance](../../benchmarks/README.md). GPU traits do not
filter `dotnet test` automatically, so pass explicit filters and resource gates from [code style](../CODE_STYLE.md).
