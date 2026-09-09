# SPIR-V compute shader reference

Sources: [Shaders](../../src/HartsyInference.Vulkan/Shaders/); shipped artifacts:
[Spirv](../../src/HartsyInference.Vulkan/Spirv/). The current build script and kernel registry define variants.
Old example scripts, kernel catalogs, claimed vendor speeds, and unimplemented phase plans were removed.

## Build and dispatch contract

Run bash src/HartsyInference.Vulkan/Shaders/build.sh after shader edits and commit rebuilt artifacts.
Register new variants in its arrays; merely adding a source file does not build it. Use a compiler that
supports the entire shader set (the distro glslang frontend previously lacked integer-dot-product support).
Validate emitted SPIR-V and run the drift check described in [KERNEL.md](../Agents/KERNEL.md).

The runtime loads SPIR-V and builds compute pipelines with specialization constants. Every cache key must
include the actual dtype, shape/device specialization, and descriptor/push-constant contract. Storage layout
and alignment must match C# exactly; do not assume a scalar array and packed vector have identical stride.

## Parallel algorithms

- Query supported subgroup operations and sizes. Reductions must handle the complete workgroup, including
  multiple subgroups and partial tiles; a warp-sized algorithm is not automatically portable.
- Every invocation required by a barrier must reach it. Replace divergent early returns with guarded
  loads/stores around unconditional barrier participation.
- Accumulate normalization/softmax/GEMM in F32 where required even with F16 storage.
- GEGLU/SwiGLU split the last logical dimension, not the flat tensor midpoint. Test multiple rows.
- Use bounds-checked loads/stores for partial GEMM tiles. Krea2's joint text+image length is generally not
  tile-aligned; padding via separate buffers adds lifetime/barrier risks.
- Use 64-bit indexing or reject overflowing geometry before launch. im2col products can exceed 32-bit
  limits even with ordinary image dimensions.
- Tune shared-memory tiles, bank layout, workgroup size, and occupancy against actual hardware limits.
  No fixed NVIDIA bank/warp assumption is a cross-vendor rule.
- Cooperative matrices require queried supported types, dimensions, and scopes. Coopmat1/coopmat2 and
  scalar paths already exist; extension support alone is not a throughput guarantee.
- Attention must preserve causal/additive masks, GQA grouping, softmax stability, and layout. Unsupported
  combinations need an explicit valid fallback rather than silently losing a feature.

## Measurement and validation

Compare saved inputs against CPU/CUDA/upstream results with operation-specific tolerances. Cover aligned
and partial dimensions, F16/F32 outputs, bias, multiple batches, and chained GPU consumption; a host read
can conceal a stale GPU binding. llvmpipe exercises portability, not device throughput.

Use VulkanGpuTimer timestamps for device time; measure end-to-end separately. Host dispatch recording time
cannot explain GPU execution collected at a later Sync. Backend performance and hardware coverage live in
[scoreboards](../../benchmarks/scoreboards/VULKAN.md) and [ROADMAP](../Checklists/ROADMAP.md).

## References

- [SPIR-V 1.6 Specification](https://registry.khronos.org/SPIR-V/specs/1.6/SPIRV.html)
- [GLSL 4.60 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [GL_KHR_shader_subgroup](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt)
- [Vulkan GLSL extensions](https://github.com/KhronosGroup/GLSL/tree/main/extensions/khr)
- [glslang](https://github.com/KhronosGroup/glslang)
- [SPIRV-Tools (`spirv-opt`/`spirv-val`/`spirv-dis`)](https://github.com/KhronosGroup/SPIRV-Tools)
- [llama.cpp Vulkan backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-vulkan)
- [VkFFT source](https://github.com/DTolm/VkFFT)
- [Khronos Vulkan Guide — Compute](https://docs.vulkan.org/guide/latest/computeshader.html)
- [AMD GPU Performance Guide](https://gpuopen.com/learn/concurrent-execution-asynchronous-queues/)
- [NVIDIA Vulkan Tips](https://developer.nvidia.com/blog/vulkan-tips/)
- [shaderc](https://github.com/google/shaderc)
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)
- [GL_EXT_shader_explicit_arithmetic_types_float16](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)
- [Khronos Vulkan Guide — Subgroups](https://docs.vulkan.org/guide/latest/subgroups.html)
- [VK_KHR_cooperative_matrix proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html)
- [Vulkan-Samples — compute samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api)
- [NCNN Vulkan backend](https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan)
- [AMD wave32 vs wave64 guide](https://gpuopen.com/learn/wave_intrinsics_unleashed/)
- [Intel Vulkan compute samples](https://github.com/intel/compute-runtime)
