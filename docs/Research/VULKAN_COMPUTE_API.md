# Vulkan compute API reference

Current bindings, feature queries, and dispatch live in [HartsyInference.Vulkan](../../src/HartsyInference.Vulkan/).
Vulkan is the cross-vendor backend design; actual AMD/Intel model verification remains open.
Portability through an API is not proof that every device or MoltenVK supports the engine's required features.

## ABI and feature contracts

- Bind the loader and resolve extension entry points through instance/device proc-address APIs.
  A supported extension entry point need not be exported directly by the loader.
- Match Khronos headers exactly: dispatchable handles are pointer-sized; non-dispatchable handles have
  a 64-bit ABI. Structure layout, alignment, sType, and pNext lifetime must be correct.
- Most creation/feature structures carry sType/pNext; not every Vulkan structure does.
  Keep enum values in the bindings, not a second documentation table.
- VkResult nonnegative values can mean partial/not-ready/timeout, not unconditional completion.
  Handle the result according to the called API, including two-call enumeration and VK_INCOMPLETE.
- Query and enable required features individually. API version alone does not guarantee optional FP16,
  subgroup, cooperative-matrix, or memory features. Check marshalling/layout before blaming a driver query.
- Query workgroup/shared-memory/storage-buffer/push-constant limits and subgroup operations before dispatch.
  The portable push-constant budget is 128 bytes; use the actual capability record for larger designs.
- Select a compute-capable queue family. Cross-family transfer requires ownership handling;
  a dedicated compute queue is a candidate to measure, not an automatic speedup.

## Memory, descriptors, and commands

- Tensor buffers need storage plus applicable transfer usage flags. Respect memoryTypeBits and alignment.
- Suballocate; one VkDeviceMemory per temporary hits allocation-count and latency limits.
  See [VULKAN_MEMORY_MANAGEMENT.md](VULKAN_MEMORY_MANAGEMENT.md).
- HOST_VISIBLE permits mapping; HOST_COHERENT removes explicit flush/invalidate requirements, not GPU/CPU
  synchronization. Noncoherent ranges must respect nonCoherentAtomSize.
- Descriptor bindings, command buffers, allocations, and pipelines must outlive GPU use. A submitted
  command buffer cannot be reset while in flight. Do not reuse pool sets still referenced by replay.
- Synchronization2 stage/access scopes must describe the actual read/write dependency. Queue order alone
  does not replace memory dependencies; host reads need completion plus appropriate visibility.
- Push descriptors bake bindings into recorded commands. Step replay therefore retains referenced buffers
  until reset; retained weight casts and temporaries can exceed an otherwise viable eager VRAM budget.
- Pipeline specialization constants select stable shape/device variants. Cache by the complete signature,
  including dtypes and descriptor/push-constant layout; invalidate incompatible persistent caches.

## Shaders and validation

GLSL is compiled to SPIR-V at build time; inference loads artifacts from disk. Shader/source rebuild rules
are in [KERNEL.md](../Agents/KERNEL.md). Do not assume subgroup size is 32 or assume all workgroups can
participate in a barrier after divergent early returns.

Use validation layers during development where available. Measure GPU time with timestamp queries
(VulkanGpuTimer), separately from host recording/submit/wait time. Nsight Compute is not a Vulkan profiler.
Use llvmpipe for portable/small-subgroup checks and real hardware for throughput claims.
Current backend-specific failures are in [Troubleshooting](../Checklists/TROUBLESHOOTING.md),
with measurements in the [Vulkan scoreboard](../../benchmarks/scoreboards/VULKAN.md).

## References

- [Vulkan 1.3 Specification](https://docs.vulkan.org/spec/latest/index.html)
- [Vulkan API Registry](https://registry.khronos.org/vulkan/)
- [Khronos Vulkan Guide — Compute](https://docs.vulkan.org/guide/latest/computeshader.html)
- [VK_EXT_subgroup_size_control proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_subgroup_size_control.html)
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)
- [Vulkan-Samples (compute_nbody, compute_op)](https://github.com/KhronosGroup/Vulkan-Samples)
- [Sascha Willems Vulkan Samples](https://github.com/SaschaWillems/Vulkan)
- [VkFFT (real Vulkan compute lib in C)](https://github.com/DTolm/VkFFT)
- [Khronos Vulkan Spec §3 Instance](https://docs.vulkan.org/spec/latest/chapters/initialization.html)
- [Khronos Vulkan Guide — Subgroups](https://docs.vulkan.org/guide/latest/subgroups.html)
- [llama.cpp Vulkan backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-vulkan)
- [Granite Vulkan Compute Examples](https://github.com/Themaister/Granite/tree/master/granite/compute)
- [Mesa RADV source](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/amd/vulkan)
- [Intel ANV source](https://gitlab.freedesktop.org/mesa/mesa/-/tree/main/src/intel/vulkan)
- [VK_KHR_cooperative_matrix proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html)
- [VK_KHR_synchronization2 proposal](https://www.khronos.org/blog/vulkan-sdk-1.2.182-released-with-new-extensions-for-vulkan-synchronization-and-pipeline-management)
- [Vulkan-Headers (vulkan_core.h)](https://github.com/KhronosGroup/Vulkan-Headers/blob/main/include/vulkan/vulkan_core.h)
- [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
