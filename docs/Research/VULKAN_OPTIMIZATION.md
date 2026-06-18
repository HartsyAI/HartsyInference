# Vulkan Compute Optimization — Research Notes

Optimization techniques for the pure-C#/.NET `HartsyInference.Vulkan` backend (raw P/Invoke via `LibraryImport`, hand-written GLSL compiled to SPIR-V, no managed wrappers). Primary GPU: NVIDIA RTX 3060 (GA106, Ampere CC 8.6, subgroup/warp = 32, Vulkan 1.4 driver). This complements the existing [`VULKAN_COMPUTE_API.md`](VULKAN_COMPUTE_API.md), [`VULKAN_MEMORY_MANAGEMENT.md`](VULKAN_MEMORY_MANAGEMENT.md), and [`SPIRV_COMPUTE_SHADERS.md`](SPIRV_COMPUTE_SHADERS.md); it is the optimization survey, and a sibling to the CUDA [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) / [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md).

**Implementation status (2026-06-17).** The two headline recommendations below, "subgroup reductions in the norm/softmax kernels" (§3.2) and "coopmat1 as the default matmul with tiled fallback" (§4.1), were already implemented in the backend before this survey; the research agents inferred shared-memory trees and an unenabled coopmat path without reading the shaders. The real work was making those existing paths **cross-vendor correct**: the coopmat sTypes were swapped and the path was not shape-gated, and the cross-subgroup reduction dropped partials on small-subgroup devices. Both are now fixed and validated on the RTX 3060 (subgroup 32) and llvmpipe (subgroup 8) — see deviations #18-19 in [`../Checklists/PHASE_3_5_DEVIATIONS.md`](../Checklists/PHASE_3_5_DEVIATIONS.md). **Shared-memory bank padding (§3.3) is also now applied** to the `matmul_tiled` fallback path (`Asub` padded to stride `BK+1` to remove the 4-way column-load conflict at `BK=32`); correctness validated on the 3060 (coopmat-disabled tiled path) and llvmpipe. **The INT8 dot-product GEMM (§4.3) is implemented and built out**: `matmul_int8` + `VulkanBackend.MatMulInt8` use `dotPacked4x8` (the cross-vendor DP4a/IMMA path), now **shared-memory tiled** (register microtile, bank-padded), with **per-row scales** and an `Int8Quantizer.RowwiseSymmetric` helper for converting FP weights/activations. Validated bit-exact on the 3060 (single/multi-tile/partial-edge) and end-to-end at ~0.5% relative error from quantized FP inputs; gracefully absent on llvmpipe — see deviation #20. Remaining: per-shape tile selection and wiring the quantizer into model loading. The other items below (coopmat2 fused dequant, async transfer streaming) are genuine future work.

Single most useful fact: the 3060 on a Vulkan 1.4 driver supports almost everything below as core (<= 1.3) or as a long-shipping NVIDIA extension, so most high-value items need no capability gating. Feature support below was confirmed against a live device report (vulkan.gpuinfo.org report 40937, driver 580.88). The card reports both `cooperativeMatrix` (KHR coopmat1) and the coopmat2 flexible-dimensions/tensor-addressing/reduction flags, plus `shaderFloat16`, `shaderInt8`, and `shaderIntegerDotProduct` with `integerDotProduct4x8BitPackedSignedAccelerated = true`.

---

## 1. Compute pipeline and dispatch

### 1.1 Specialization constants (bake local_size and loop bounds)

A spec constant is supplied at `vkCreateComputePipelines` time before the driver's final compile, so it folds as a true compile-time constant: loops unroll, dead branches drop, workgroup size is baked. Strictly more powerful than a push constant (which the compiler must treat as an unknown runtime value). GLSL: `layout(constant_id = 0) const uint TILE = 16;` and `layout(local_size_x_id = 2, local_size_y_id = 3) in;`. Gotcha: a literal `local_size_x = N` overrides `LocalSizeId`, so do not hardcode a size on an axis you specialize. Plain spec constants are core 1.0; `local_size_id` is core 1.3 (maintenance4). P/Invoke: `VkSpecializationMapEntry { uint constantID; uint offset; nuint size; }` + `VkSpecializationInfo`, attached via `VkPipelineShaderStageCreateInfo.pSpecializationInfo`. Build one pipeline per variant, cache by (tile, unroll, dtype). The engine already uses spec constants for `local_size` (see `SpecConstant` in `VulkanKernels.cs`).

### 1.2 Push descriptors + descriptor update templates

`vkCmdPushDescriptorSetKHR` writes descriptors straight into the command buffer, removing the pool allocate + `vkUpdateDescriptorSets` + `vkCmdBindDescriptorSets` sequence. Layout needs `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`. Promoted to Vulkan 1.4 core. The engine already prefers this (`VulkanDescriptorManager.PushSet`, gated on `PushDescriptorActive`). Descriptor update templates (`vkUpdateDescriptorSetWithTemplate`, core 1.1) capture a fixed update shape once and gather from one `void*`, avoiding per-call `VkWriteDescriptorSet[]` marshaling for the allocated-set fallback path.

### 1.3 Buffer device address (bindless via push constant)

`VK_KHR_buffer_device_address` (core 1.2, `bufferDeviceAddress = true` on the 3060) turns a `VkBuffer` into a 64-bit GPU address (`vkGetBufferDeviceAddress`, a `ulong`). Put it in a push constant and deref in GLSL via `GL_EXT_buffer_reference`, with zero descriptors for that tensor. The clean bindless primitive for an engine with many weight tensors: allocate with `SHADER_DEVICE_ADDRESS_BIT` and `MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`, pass addresses by push constant. Avoid `VK_EXT_descriptor_buffer` (NVIDIA implements it with an indirection penalty and it is now deprecated for `VK_EXT_descriptor_heap`).

### 1.4 Serialized pipeline cache on disk

`vkGetPipelineCacheData` to disk, seed `VkPipelineCache` next launch so `vkCreateComputePipelines` reuses baked compilation. Matters because spec-constant variants (1.1) multiply the pipeline count. The cache is not portable across machine / driver version / hardware, so wrap it with a header storing vendorID, deviceID, `driverVersion`, ABI, `pipelineCacheUUID`, and a hash; validate every field; if `vkCreatePipelineCache` still fails on validated data, catch and recreate empty; write atomically (temp + rename). The engine has `VulkanPipelineCache`; confirm it serializes to disk with this guard.

### 1.5 Skip list

Pipeline derivatives (`DERIVATIVE_BIT`) are accepted but unaccelerated on NVIDIA. Secondary command buffers help only multithreaded recording, irrelevant to a small static compute graph. `VK_EXT_descriptor_buffer` is deprecated/penalized. `pipeline_executable_properties` is a useful dev-only introspection tool (register counts, ISA disasm) but must stay off in production (the capture bit can change compilation).

---

## 2. Barriers and synchronization

### 2.1 The correct minimal compute-to-compute barrier

When dispatch B reads what dispatch A wrote: `srcStage = COMPUTE_SHADER`, `srcAccess = SHADER_STORAGE_WRITE`, `dstStage = COMPUTE_SHADER`, `dstAccess = SHADER_STORAGE_READ`. The sync2 dedicated storage bits are tighter than the generic `SHADER_READ`/`SHADER_WRITE` (they exclude uniform/sampled). The engine's `RecordGlobalComputeBarrier` uses exactly this pair (note: the `ShaderStorage*` constants were wrong by one bit until fixed, see [PHASE_3_5_DEVIATIONS.md](../Checklists/PHASE_3_5_DEVIATIONS.md)). Over-sync mistakes to avoid: `ALL_COMMANDS` on either mask, `BOTTOM_OF_PIPE`/`TOP_OF_PIPE` for memory visibility (they do none), the debug full barrier (`MEMORY_READ|MEMORY_WRITE` + `ALL_COMMANDS`), and read-to-read barriers (cost, do nothing).

### 2.2 vkCmdPipelineBarrier2 + one global barrier

`VK_KHR_synchronization2` (core 1.3) carries stage and access together in `VkMemoryBarrier2`, removing the old footgun of matching function stage masks to barrier access masks by hand. For SSBO-only compute chains, prefer one global `VkMemoryBarrier2` over many `VkBufferMemoryBarrier2`: no real GPU cares for the SSBO case, and one global barrier means no per-buffer struct array to pin each step. Reserve buffer/image barriers for queue-ownership transfers and image layout transitions (VAE storage images). The masks are `ulong` (8 bytes) in sync2; a wrong width shifts every later field.

### 2.3 Do not barrier between independent dispatches

A barrier creates a false dependency. Independent branches (separate attention heads, channels, preprocessing) writing disjoint regions get zero barriers, joined by one barrier before the consumer. The current backend emits a global barrier after every dispatch (`VulkanBackend.cs` Dispatch path), which is correct but serializes independent work such as the SDPA per-head loop. Scoping the barrier to the actually-written buffer, or dropping it between disjoint-region dispatches, would let Ampere overlap them. This is a perf opportunity, not a correctness fix.

### 2.4 Transfer-to-compute barriers

The compute-to-compute global barrier does not cover a `vkCmdCopyBuffer` (transfer stage) feeding a dispatch. Every host-to-device upload that feeds a kernel needs `srcStage = TRANSFER, srcAccess = TRANSFER_WRITE, dstStage = COMPUTE_SHADER, dstAccess = SHADER_STORAGE_READ`. The engine does this in `RecordCopyAndBarrier`; the invariant to maintain is that no upload path enqueues a copy without that barrier.

---

## 3. Workgroup, occupancy, subgroups

### 3.1 local_size for the 3060

Warp = 32, so a non-multiple-of-32 workgroup wastes the tail warp. The 3060 has 48 max resident warps/SM (not 64 like A100), so for full occupancy the warp count must divide 48: 64, 128, 192, 256, 384, 512 threads. Defaults: 128 or 256 for norm/softmax/elementwise, 256 for tiled matmul. Drive `local_size` from a spec constant so one SPIR-V sweeps sizes. Note the engine hardcodes `local_size_x = 256` for norms (`VulkanBackend.cs`), which is a good 3060 value but worth spec-constant-izing for portability. Vulkan exposes no occupancy-calculator API, so compute resident-blocks/SM yourself from `VkPhysicalDeviceLimits` read once at startup.

### 3.2 Subgroup reductions (the highest-value low-effort win)

Subgroup ops exchange and reduce across the 32 lanes through registers, with no shared-memory round trip and no `barrier()`. Replacing the shared-memory tree reductions in layernorm / groupnorm / rmsnorm / softmax collapses a log2(N)-barrier tree into one barrier and removes most atomics. Core Vulkan 1.1; the 3060 exposes the full set (BASIC, VOTE, ARITHMETIC, BALLOT, SHUFFLE, CLUSTERED, QUAD). Pattern for a 256-thread group:
```glsl
#extension GL_KHR_shader_subgroup_arithmetic : require
shared float partial[8];                 // 256/32 subgroups
float warpSum = subgroupAdd(v);          // 32-wide, no barrier
if (subgroupElect()) partial[gl_SubgroupID] = warpSum;
barrier();                               // single barrier
if (gl_SubgroupID == 0) {
    float s = (gl_SubgroupInvocationID < gl_NumSubgroups) ? partial[gl_SubgroupInvocationID] : 0.0;
    float total = subgroupAdd(s);
}
```
For softmax: `subgroupMax` then `subgroupAdd(exp(x - max))`. Pair with `VK_EXT_subgroup_size_control` (core 1.3): set `requiredSubgroupSize = 32` and `REQUIRE_FULL_SUBGROUPS_BIT` so there is no partial tail subgroup. This is also a prerequisite for coopmat1 correctness (partial subgroups give wrong coopmat results; llama.cpp added exactly this in its coopmat path).

Note the existing norm shaders size `shared float warp_sum[32]` for the cross-subgroup combine, which is safe on the 3060 (256/32 = 8 subgroups) but overflows if a device ever forces subgroup size < 8. Size the array from a spec constant or assert `gl_NumSubgroups <= 32`.

### 3.3 Shared-memory bank conflicts

32 banks x 4 bytes (word address mod 32). Column access of a `shared float[N][N]` tile with N a multiple of 32 is a 32-way conflict, fully serialized. Fix with one padding column: `shared float tile[32][33]` shifts each row by a bank. Costs 128 bytes/tile, negligible against the 100 KB budget. XOR swizzle (`col ^ row`) is the zero-waste alternative. Apply to `matmul_tiled` and `transpose` shared tiles.

---

## 4. Cooperative matrix (tensor cores)

### 4.1 coopmat1 (VK_KHR_cooperative_matrix)

GLSL: `#extension GL_KHR_cooperative_matrix : require`, type `coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>`, with `coopMatLoad` / `coopMatMulAdd(A, B, C)` / `coopMatStore`. Maps directly to NVIDIA tensor cores with no application changes. The matrix lives only in registers (spread across the 32-lane subgroup), not in shared/buffer storage. Must enumerate supported shapes at startup with `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and pick a combo (do not hardcode); the Ampere FP16 set centers on 16x16x16 with FP16 or FP32 accumulate, and INT8 is `sint8 x sint8 -> sint32`. Alignment rule: tile base offset and stride must be at least 16-byte aligned (a multiple of 8 elements for FP16). Enable `cooperativeMatrixRobustBufferAccess` (free on the 3060) so out-of-bounds tile loads are safe, or pad buffers to tile multiples.

Accumulate precision: default to FP32 accumulate (`CType = float32`) for diffusion linear/conv-as-matmul where K is large (hidden dims 1024 to 4096); this is what ggml-vulkan defaults to. Expose FP16 accumulate (about 2x throughput) as an opt-in per-op flag for short reductions / attention scores. The engine already has a `matmul_coopmat` shader; make it the default matmul with the tiled FP16 shader as a runtime fallback (mirror llama.cpp's `GGML_VK_DISABLE_COOPMAT` escape hatch).

### 4.2 coopmat2 (VK_NV_cooperative_matrix2, the high ceiling)

All coopmat2 flags are present on the 3060. Adds over coopmat1: workgroup scope (larger tiles, e.g. 128x256, auto-staged through shared memory); tensor addressing (`tensorLayoutNV`, `coopMatLoadTensorNV`) that handles multidim slicing/padding/clamp and removes manual index math; a decode/dequant callback on load (dequantize int4/block-quant weights during the tensor load, no separate dequant pass); `coopMatReduceNV` (fuse softmax row-max/sum into the matmul); `coopMatPerElementNV` (fuse bias/activation epilogue); and accumulator-to-A/B conversions (fuse back-to-back matmuls, i.e. attention QK then times V without a global-memory round trip). It is NVIDIA-only but the single-target 3060 has every flag. Plan it as the high-ceiling target behind the same fallback chain.

### 4.3 INT8 quantized GEMM (the IMMA equivalent)

`VK_KHR_shader_integer_dot_product` (core 1.3) with `dotPacked4x8AccSatEXT` is the DP4a/IMMA-style fused multiply-add-accumulate, hardware-accelerated on the 3060 (`integerDotProduct4x8BitPackedSignedAccelerated = true`). ggml-vulkan's `mul_mmq.comp` keeps weights as int8 quant blocks with per-block scales, runs `dotPacked4x8` directly on the int8 values without dequantizing, then applies the per-block FP scale to the integer accumulator at the end. 2 to 4x memory savings plus native int8 throughput; needs the quant-format plumbing. Eventually move it into the coopmat2 decode callback to also get tensor cores.

---

## 5. Memory and transfer

### 5.1 Memory type selection (the 3060 table)

Query, never hardcode. The reference 3060 layout (ReBAR off): Heap 0 = 12 GB `DEVICE_LOCAL`; Heap 1 = system RAM with `HOST_VISIBLE|HOST_COHERENT` and `HOST_VISIBLE|HOST_COHERENT|HOST_CACHED` types; Heap 2 = ~214 MB `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT` (the small pinned-BAR pool). `nonCoherentAtomSize = 64`. With ReBAR on, Heap 0 gains a `DEVICE_LOCAL|HOST_VISIBLE` type covering all 12 GB and Heap 2 shrinks; detect ReBAR by scanning for a `DEVICE_LOCAL|HOST_VISIBLE` type whose heap is multi-GB. Rules: weights/activations in pure `DEVICE_LOCAL`; upload staging in `HOST_VISIBLE|HOST_COHERENT`; readback in `HOST_VISIBLE|HOST_CACHED` (reading from write-combined upload or BAR memory is orders of magnitude slower); small per-step pushes via the BAR type. On this discrete GPU, never CPU-map device-local; populate via staging + `vkCmdCopyBuffer`. The `FindMemoryType` test is `(props & required) == required`, never `props == required`.

Non-coherent flush/invalidate (a no-op on the 3060 since all host-visible types are coherent here, but required for portability): `VkMappedMemoryRange.offset` must be a multiple of `nonCoherentAtomSize` and `size` a multiple of it or extend to the allocation end. Aligning the offset down requires extending the end to still cover the data (the engine had a range-math bug here, since fixed: `start = AlignDown(off, atom); end = AlignUp(off + size, atom)`).

### 5.2 Suballocation (correctness on Windows, not just perf)

`maxMemoryAllocationCount` caps live `VkDeviceMemory` objects; spec floor is 4096. Linux NVIDIA reports effectively unlimited, but Windows WDDM caps at 4096 regardless, so one-allocation-per-buffer hits `VK_ERROR_OUT_OF_DEVICE_MEMORY` on count with gigabytes free. Suballocate: a few large blocks (VMA default block 256 MiB), bind resources at offsets via `vkBindBufferMemory(..., offset)` rounded to each resource's `VkMemoryRequirements.alignment`. `VK_KHR_dedicated_allocation` (core 1.1) for big tensors that prefer/require it. `VK_EXT_memory_budget` (function is core 1.1): chain `VkPhysicalDeviceMemoryBudgetPropertiesEXT` into `vkGetPhysicalDeviceMemoryProperties2`, gate every allocation on `budget - usage`, re-queried per frame; essential on a 12 GB card sharing VRAM with the desktop.

### 5.3 Staging, persistent mapping, transfer queue

Host-visible staging (`TRANSFER_SRC`, persistently mapped, mapped once) to device-local (`TRANSFER_DST`) via `vkCmdCopyBuffer`; device VRAM bandwidth is 16 to 32x PCIe, so anything read repeatedly must be resident. A dedicated transfer-only queue family (detect by `(flags & TRANSFER) && !(flags & (GRAPHICS|COMPUTE))`) overlaps host-device copies with compute. Cross-queue gotcha: different families need `VK_SHARING_MODE_CONCURRENT` (simplest, slight cost) or explicit release/acquire ownership-transfer barriers, a frequently-missed bug that leaves buffer contents undefined.

### 5.4 Timeline semaphores and weight streaming

`VK_KHR_timeline_semaphore` (core 1.2): one monotonic 64-bit counter for device-device and device-host sync, no reset, allows wait-before-signal (clean producer/consumer between transfer and compute queues). They order execution only, not memory coherency, so still pair with barriers. For models bigger than 12 GB, replicate the CUDA pinned-double-buffer prefetch: persistently-mapped staging (= pinned host memory) + transfer queue (= copy engine) + timeline semaphore ("compute of block N waits until upload of block N signaled value N"). Prefetch depth 1; keep as much resident as fits and stream only the overflow; the win exists only when per-block compute >= per-block PCIe upload.

---

## Ranked table (impact vs effort, all confirmed on RTX 3060)

| # | Technique | Impact | Effort | Vulkan |
|---|---|---|---|---|
| 1 | Subgroup reductions in norm/softmax + subgroup_size_control | High | Low | 1.1 / 1.3 |
| 2 | coopmat1 FP16 matmul (FP32 accum), tiled fallback | Very high | Medium | KHR (core 1.4) |
| 3 | One global sync2 barrier; no barrier between independent dispatches | High | Low | 1.3 |
| 4 | Shared-mem bank padding `[N][N+1]` in matmul/transpose | High | Low | HW |
| 5 | Spec-constant local_size + on-disk pipeline cache | High | Medium | 1.3 / 1.0 |
| 6 | Suballocator over large blocks + memory_budget gating | Critical (Windows correctness) | Medium | 1.0 / EXT |
| 7 | Staging + persistent map + HOST_CACHED readback | High | Low | 1.0 |
| 8 | buffer_device_address bindless weights | Medium | Med | 1.2 |
| 9 | INT8 quant GEMM via dotPacked4x8AccSatEXT | High (VRAM) | Med-High | 1.3 |
| 10 | Transfer queue + timeline-semaphore weight streaming | High (if > VRAM) | High | 1.2 |
| 11 | coopmat2 (workgroup tiles, fused decode + reductions + attention) | Very high (ceiling) | High | NV |

## Top recommendations for HartsyInference.Vulkan

1. **Subgroup reductions first** in every norm/softmax kernel (`subgroupAdd`/`subgroupMax`, one final barrier), wiring `subgroup_size_control` with `REQUIRE_FULL_SUBGROUPS_BIT` + `requiredSubgroupSize = 32` as shared infra (coopmat also needs it). Lowest effort, immediate win.
2. **Make coopmat1 the default matmul** (FP32 accumulate), enumerate shapes at startup, honor the 16-byte tile alignment, enable robustness, keep the tiled shader as a runtime fallback.
3. **Audit barriers**: keep the corrected global sync2 barrier, but scope it to the written buffer (or drop it) between independent dispatches such as the SDPA per-head loop.
4. **Add `+1` padding** to the `matmul_tiled` and `transpose` shared tiles.
5. **Suballocator + `memory_budget`** before anything else memory-side: it is a Windows correctness requirement, not just an optimization.
6. Plan **coopmat2** as the high-ceiling target (fused dequant + large tiles + `coopMatReduceNV` softmax + fused attention), behind the same fallback chain; it is NVIDIA-only but the single-target 3060 has every flag.
7. For models that exceed 12 GB, add the **transfer-queue + timeline-semaphore** block-swap prefetch (depth 1), getting the cross-queue ownership transfer right.

Everything in the top list is Vulkan core <= 1.3 except coopmat1 (core 1.4), buffer_device_address (1.2), and the transfer queue, so it is available unconditionally on the 3060.

## Primary sources

- Khronos Vulkan spec + registry (sync2, subgroups, cooperative matrix, memory model, sparse) — https://docs.vulkan.org , https://registry.khronos.org/vulkan/
- GLSL extension specs (KHR_cooperative_matrix, KHR_shader_subgroph, EXT_integer_dot_product, EXT_buffer_reference) — https://github.com/KhronosGroup/GLSL/tree/master/extensions
- NVIDIA: Machine Learning Acceleration in Vulkan with Cooperative Matrices — https://developer.nvidia.com/blog/machine-learning-acceleration-vulkan-cooperative-matrices/
- NVIDIA Vulkan Dos and Don'ts — https://developer.nvidia.com/blog/vulkan-dos-donts/
- ggml-vulkan shaders (`mul_mm.comp` coopmat, `mul_mmq.comp` int8 dot) — https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan
- Khronos Synchronization Examples — https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples
- GPUOpen: Vulkan Barriers Explained, Vulkan Device Memory — https://gpuopen.com/learn/
- Vulkan Memory Allocator docs (suballocation, budget, mapping) — https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/
- asawicki: Vulkan memory types on PC — https://asawicki.info/news_1740
- zeux.io: robust pipeline cache serialization — https://zeux.io/2019/07/17/serializing-pipeline-cache/
- RTX 3060 capability report — vulkan.gpuinfo.org report id 40937
