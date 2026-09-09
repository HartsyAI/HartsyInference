# Vulkan memory and lifetime constraints

Implementation: [Vulkan package](../../src/HartsyInference.Vulkan/). This note records design constraints;
allocation sizes, data structures, feature flags, and tests are defined by code, not copied here.

## Memory placement

Choose a memory type permitted by memoryTypeBits and the operation's required flags. Prefer device-local
storage for weights/activations, host-visible staging for uploads, and cached host-visible memory for readback.
HOST_COHERENT removes flush/invalidate calls, not synchronization with GPU use. Align noncoherent ranges
to nonCoherentAtomSize. ReBAR/UMA placement is hardware-specific and must be measured.

Suballocate aligned ranges from larger blocks. Distinguish live bytes, reserved bytes, reusable empty blocks,
and driver budget. Pooling large dedicated allocations avoids repeated allocation/free cost, but retention
can reduce headroom. Reclaim based on completed timeline values, not merely host-side disposal.

## Ownership and synchronization

- Tensor identity keys GPU caches. Check cached weights before touching CPU storage, which may already be disposed.
- Activation disposal schedules release after its last GPU use; it must not free a still-recorded buffer.
  Replacing a cached output requires explicit ownership of the old binding.
- Host readback waits for completion and visibility. Host scalar writes must not race a previously recorded
  dispatch that still expects the old value.
- Descriptor pools/sets and command buffers cannot be reset while in flight. Push descriptors remove the
  pool-set lifetime hazard but do not remove buffer lifetime requirements.
- A recorded graph captures addresses. Keep every referenced allocation alive until graph reset unless
  a verified reuse plan proves disjoint lifetimes. Weight-cast retention may make capture infeasible even
  when eager execution fits; recover to eager without stale bindings or leaked allocations.
- Cross-queue copies need synchronization and, where applicable, queue-family ownership transfer.
- At component transitions, synchronize and release the intended weights/activations; broad eviction can
  accidentally invalidate shared components or graph state.

## OOM and diagnostics

On allocation failure, account for pending frees and reclaimable pool storage before retrying. Do not classify
transient capacity failure as permanent kernel incompatibility. Report requested size, live/reserved usage,
cache/cast/graph retention, and device budget separately. High nvidia-smi usage alone is not a leak.

Historical Krea2 capture OOM was a one-time retention peak, not accumulating activations. A separate empty-block
retention finding affected VRAM headroom; direct timing retracted the claim that it explained the throughput gap.
See [Troubleshooting](../Checklists/TROUBLESHOOTING.md) and [Vulkan measurements](../../benchmarks/scoreboards/VULKAN.md).

## Verification

Exercise allocation reuse, mixed sizes, alignment, deferred frees, repeated outputs, CPU-disposed weights,
noncoherent visibility, descriptor turnover, graph abort/replay/reset, and repeated model switching.
A synthetic allocator test does not prove a full model's residency behavior. Distinguish skipped GPU tests
from executed checks; no dedicated Vulkan CI lane is currently configured.

## References

- [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- [Vulkan 1.3 Spec — Resources](https://docs.vulkan.org/spec/latest/chapters/resources.html)
- [VMA documentation site](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/)
- [Vulkan Memory Types Tutorial](https://docs.vulkan.org/guide/latest/memory_allocation.html)
- [VK_EXT_memory_budget proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_memory_budget.html)
- [Adam Sawicki — Vulkan Memory Heaps Cheatsheet](https://gpuopen.com/learn/vulkan-device-memory/)
- [llama.cpp Vulkan allocator](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- [Vulkan Spec § Memory Allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html)
- [VK_KHR_push_descriptor](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_push_descriptor.html)
- [VK_KHR_timeline_semaphore](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_timeline_semaphore.html)
- [VK_KHR_synchronization2](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html)
- [NCNN Vulkan allocator](https://github.com/Tencent/ncnn/blob/master/src/allocator.cpp)
- [Granite Vulkan allocator](https://github.com/Themaister/Granite/blob/master/granite/vulkan/memory_allocator.cpp)
