# Phase 3.5 — Vulkan Backend (Cross-Vendor GPU)

> **Goal:** Run SD1.5 on AMD / Intel / NVIDIA via Vulkan compute shaders. Match CUDA reference within 1e-3 per kernel and SSIM > 0.99 end-to-end.
>
> **Packages:** `SharpInference.Vulkan` (built), `SharpInference.Core` (minor — capabilities + DeviceKind already in place), `SharpInference.Diffusion` (untouched — `IBackend` abstraction handles routing).
>
> **Status:** Backend functional and perf-tuned. Compiles + 33/33 smoke tests pass on Linux + NVIDIA. End-to-end Flux Schnell FP8 4-step generation: 178 s → 129.5 s after Phase C tuning (1.38× faster). SD1.5 / SDXL integration tests + SSIM helper added (skip-when-checkpoint-missing). Remaining work: AMD / Intel cross-vendor verification (hardware-blocked), further perf tuning (currently ~6.5× CUDA wall-clock — target ≤ 1.6× — per-dispatch overhead is the dominant bottleneck), SSIM gates need CUDA reference images to run. See [PHASE_3_5_DEVIATIONS.md](PHASE_3_5_DEVIATIONS.md) for bugs hit and resolved. Use this checklist to track remaining progression.

---

## 1. Research

- [x] [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md) — P/Invoke surface (~55 functions), instance/device/queue, buffers, descriptors, push constants, command buffers, sync2, timeline semaphores, fences, vendor compatibility matrix
- [x] [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md) — GLSL → SPIR-V toolchain, compute shader anatomy, subgroup primitives, FP16, kernel catalog, **tiled GEMM (cuBLAS replacement)**, per-kernel designs, validation tolerances, vendor tuning
- [x] [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md) — slab allocator, GPU weight cache port, lazy-sync activation cache port, OOM retry, descriptor management, timeline-semaphore stream model
- [ ] [CUDA_PERFORMANCE.md](../Research/CUDA_PERFORMANCE.md) — re-read for the activation-cache patterns that we are porting; not new research, but required reading
- [ ] llama.cpp `ggml-vulkan.cpp` — read once for production-reference idioms (tiled GEMM, dequant, K-cache, push-descriptor usage). Our impl is clean-room but the patterns map closely.

## 2. Planning

- [x] **Vulkan SDK installed locally** — `glslangValidator`, `spirv-tools` (`apt install vulkan-tools glslang-tools spirv-tools` on Linux; LunarG Vulkan SDK for richest validation layers)
- [ ] **Test hardware confirmed** — at minimum Linux + NVIDIA (for A/B against CUDA backend) + Linux + AMD RDNA2 or RDNA3 (the cross-vendor proof). Ideal: also Linux + Intel Arc (variable subgroup size validates `requiredSubgroupSize` path).
- [x] **`SharpInference.Vulkan.csproj`** committed empty with project references and build script wiring (no code yet)
- [x] **`native/vulkan/` directory** committed with empty `shaders/` and `build.sh` (no .glsl yet)
- [ ] **CI matrix updated** — at minimum a Linux runner with `mesa-vulkan-drivers` for unit tests via Mesa LLVMpipe (software Vulkan) so allocator / API tests run without a physical GPU
- [x] **`PHASE_3_5_DEVIATIONS.md`** scaffolded and populated — 9 entries covering FP8 GEMM dtype mismatch, mid-op auto-flush UAF, NVIDIA feature-query bug, slab-size OOM, transient-buffer leak, deferred-free off-by-one, GLSL `erf` workaround, matmul `writeonly` constraint, glslangValidator `-O` flag

## 3. Implementation — `SharpInference.Vulkan`

### 3a. Bring-up (instance → device → trivial dispatch)

The smallest possible end-to-end. **Goal:** allocate a buffer, run an `add` shader, read the result back.

- [x] `VulkanLibraryResolver.cs` — `NativeLibrary.SetDllImportResolver`: `vulkan-1` → `libvulkan.so.1` / `vulkan-1.dll` / `libvulkan.1.dylib` (mirror `CudaLibraryResolver.cs`)
- [x] `VulkanApi.cs` — `[LibraryImport("vulkan-1")]` for the ~55 functions in [VULKAN_COMPUTE_API.md § P/Invoke Function List](../Research/VULKAN_COMPUTE_API.md#pinvoke-function-list-phase-35-minimum-surface). Flat static class. No marshalling.
- [x] `VulkanException.cs` — wraps `VkResult`; `.ThrowOnError(string op)` extension method
- [x] `VulkanStructs.cs` — `VkApplicationInfo`, `VkInstanceCreateInfo`, `VkDeviceCreateInfo`, `VkDeviceQueueCreateInfo`, `VkPhysicalDeviceProperties2`, `VkPhysicalDeviceFeatures2`, `VkPhysicalDeviceMemoryProperties`, `VkQueueFamilyProperties`, `VkMemoryAllocateInfo`, `VkBufferCreateInfo`, `VkMemoryRequirements`, `VkMappedMemoryRange`, `VkSubgroupProperties`, `VkSubgroupSizeControlProperties`, `VkPhysicalDeviceVulkan11/12/13Features`, `VkSpecializationInfo`, `VkSpecializationMapEntry`, `VkPipelineShaderStageCreateInfo`, `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`, `VkComputePipelineCreateInfo`, `VkPipelineLayoutCreateInfo`, `VkDescriptorSetLayoutBinding`, `VkDescriptorSetLayoutCreateInfo`, `VkDescriptorPoolCreateInfo`, `VkDescriptorPoolSize`, `VkDescriptorSetAllocateInfo`, `VkWriteDescriptorSet`, `VkDescriptorBufferInfo`, `VkPushConstantRange`, `VkCommandPoolCreateInfo`, `VkCommandBufferAllocateInfo`, `VkCommandBufferBeginInfo`, `VkBufferCopy`, `VkBufferMemoryBarrier2`, `VkMemoryBarrier2`, `VkDependencyInfo`, `VkSubmitInfo2`, `VkCommandBufferSubmitInfo`, `VkSemaphoreSubmitInfo`, `VkSemaphoreCreateInfo`, `VkSemaphoreTypeCreateInfo`, `VkSemaphoreSignalInfo`, `VkSemaphoreWaitInfo`, `VkFenceCreateInfo`, `VkPipelineCacheCreateInfo`. All `[StructLayout(LayoutKind.Sequential)]`.
- [x] `VulkanEnums.cs` — `VkResult`, `VkStructureType`, `VkDescriptorType`, `VkBufferUsageFlags`, `VkMemoryPropertyFlags`, `VkPipelineStageFlags2`, `VkAccessFlags2`, `VkShaderStageFlags`, `VkCommandPoolCreateFlags`, `VkCommandBufferUsageFlags`, `VkSubgroupFeatureFlags`, `VkPipelineShaderStageCreateFlags`, `VkSemaphoreType`. Match exact spec values.
- [x] `VulkanInstance.cs` — `Create(bool enableValidation)`. Builds `VkApplicationInfo` (api 1.3) + `VkInstanceCreateInfo`. Returns instance handle + selected layer list. Reads `SHARPINFERENCE_VK_VALIDATION` env var.
- [x] `VulkanDevice.cs` — physical device enumeration, scoring (deviceType discrete > integrated > CPU; FP16 + subgroupSizeControl + ARITHMETIC + SHUFFLE features required; VRAM size as tiebreaker). Logs full `BackendCapabilities`. Builds `pNext` chain (`VkPhysicalDeviceFeatures2 → Vulkan11Features → Vulkan12Features → Vulkan13Features`). Calls `vkCreateDevice` with required extensions: `VK_KHR_synchronization2` (for 1.2 fallback), `VK_EXT_memory_budget`, optional `VK_KHR_push_descriptor`, optional `VK_KHR_cooperative_matrix`. Picks compute queue family (compute-only > compute+graphics).
- [x] `VulkanCapabilities` (in `Core` package, mirror `BackendCapabilities` extension fields) — `SubgroupSize`, `MinSubgroupSize`, `MaxSubgroupSize`, `HasReBAR`, `HasPushDescriptor`, `HasMemoryBudget`, `HasCoopMatrix`, `MaxComputeSharedMem`, `VendorId`. Surface to upper layers.
- [x] **Smoke test:** `tests/SharpInference.Vulkan.Tests/VulkanDeviceInfoTest.cs` + `VulkanFeatureProbe.cs` — enumerate physical devices, log device names + capabilities, isolate per-struct feature query (caught the NVIDIA feature-query bug — see [PHASE_3_5_DEVIATIONS.md #3](PHASE_3_5_DEVIATIONS.md))

### 3b. Memory allocator + buffer wrapper

- [x] `VulkanMemoryBlock.cs` — internal: one `VkDeviceMemory`, sorted free-list, persistent mapped pointer for host-visible blocks
- [x] `VulkanMemoryAllocator.cs` — public surface. `Allocate(size, alignment, memoryTypeBits, required, preferred)`, `Free(VulkanAllocation)`. Two slabs: 256 MB large pool (weights/big activations), 16 MB small pool. Memory-type selection algorithm from [VULKAN_MEMORY_MANAGEMENT.md § Memory Type Selection Algorithm](../Research/VULKAN_MEMORY_MANAGEMENT.md#memory-type-selection-algorithm). Handles bigger-than-block allocations via dedicated `vkAllocateMemory`. OOM-retry path drains deferred-free list before re-attempting.
- [x] `VulkanBuffer.cs` — `VkBuffer` + `VulkanAllocation`. `Dispose()` returns slab region. `AsSpan<T>` for host-visible mapping.
- [x] `VulkanGpuTransferHelper.cs` — `Dictionary<Tensor, VulkanBuffer>` weight cache (reference equality). `CopyToDevice(Tensor)`, `PreloadWeight`, `PreloadWeights(IEnumerable<Tensor>)` with batched staging (one staging buf per ~64 MB batch). `FreeWeights` mirrors CUDA's `FreePreloadedWeights`. ReBAR fast path when `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` type exists.
- [x] `VulkanDeferredFreeList.cs` — `Dictionary<ulong /*timelineValue*/, List<VulkanAllocation>>`. `FreeAsync(alloc, currentValue)`. `Reclaim(completedValue)` returns regions to allocator.
- [ ] **Tests:** allocator round-trip, alignment, OOM-retry (synthetic), ReBAR-fast-path detection, batched preload measurement (SDXL preload should drop from minutes to ≤ 30 s) — *covered transitively by Flux end-to-end run; dedicated unit tests still pending. OOM-retry path was added in response to [PHASE_3_5_DEVIATIONS.md #4](PHASE_3_5_DEVIATIONS.md) but not yet covered by a synthetic test.*

### 3c. Command buffer & sync layer

- [x] `VulkanCommandPool.cs` — one pool per backend; `Acquire()` (allocate in batches of 8), `RecycleAllInUse()` (reset, not free)
- [x] `VulkanCommandStream.cs` — single timeline semaphore + monotonic counter. `RecordOp(Action<nint>)`, `SubmitAndAdvance() → ulong`, `WaitTimeline(ulong)`, `CurrentValue()`. Replaces CUDA's `CudaStream` semantically; one logical "stream".
- [x] `VulkanBarriers.cs` — emitter helpers: `EmitComputeToCompute(cb, buffer, offset, size)`, `EmitTransferToCompute(...)`, `EmitComputeToHost(...)`, `EmitComputeToTransfer(...)`. Always per-buffer scope.
- [ ] **Tests:** record empty CB → submit → timeline counter advances; double-submit → second waits on first. *Covered transitively by every smoke test (all dispatch through `_stream`); dedicated test still pending.*

### 3d. Pipeline & descriptor management

- [x] `SpirVShaderLoader.cs` — `LoadFromFile(string path)` returns `VkShaderModule` handle. Reads `.spv` from `Spirv/` content directory. Validates `codeSize % 4 == 0`.
- [x] `VulkanPipelineCache.cs` — read/write cache blob to `~/.cache/sharpinference/vulkan/<deviceUUID>.pipeline_cache`. `Persist()` on backend dispose.
- [x] `VulkanPipelineLayoutFactory.cs` — pre-builds the ~10–12 distinct `VkDescriptorSetLayout` shapes (`L_2SSBO`, `L_3SSBO`, `L_4SSBO`, `L_5SSBO`, `L_3SSBO_QKV`); pre-builds matching `VkPipelineLayout` per (descriptor layout × push-constant range).
- [x] `VulkanDescriptorManager.cs` — push-descriptor path when `VK_KHR_push_descriptor` supported (no pool); pool ring fallback (two pools, alternate, reset between phases). Pre-allocate `MAX_SETS_PER_POOL = 4096`.
- [x] `VulkanKernels.cs` — kernel registry. `Dictionary<KernelKey, VulkanKernel>` with `KernelKey = (Name, SpecConstHash)`. `Get(KernelKey)` returns or builds the `VkPipeline`. Builds via `VkComputePipelineCreateInfo` with `VkSpecializationInfo` and `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` chained.
- [ ] **Tests:** load `add.spv`, build pipeline twice with different spec consts → two cached entries; persist + reload pipeline cache. *Pipeline build covered transitively by every smoke test; cache persist+reload still untested.*

### 3e. Backend implementation (`VulkanBackend : IBackend`)

- [x] `VulkanBackend.cs` — top-level. Fields: `_instance`, `_device`, `_queue`, `_alloc`, `_xferHelper`, `_cmdStream`, `_kernels`, `_descMgr`. Lifecycle: ctor builds all of the above; `Dispose()` tears down in reverse order with `vkDeviceWaitIdle` first.
- [x] All `IBackend` ops dispatch through one helper: `Dispatch(kernelName, specConsts, descriptorSet, pushConstants, groupX, groupY, groupZ)`. Exact mirror of `CudaBackend.Launch*` methods. Lazy-sync activation cache via `_xfer` weight cache + transient-buffer tracking. Multi-dispatch ops are wrapped in `using OpScope _ = EnterOp();` to suppress mid-op auto-flush (see [PHASE_3_5_DEVIATIONS.md #2](PHASE_3_5_DEVIATIONS.md)).
- [x] `MatMul`, `Linear`, `BatchedMatMul` — dispatch `matmul_tiled_*.spv`. Variant selected by transposes + dtype + bias + activation. Falls back to non-fused path when activation kernel not present.
- [x] `Conv2D` — im2col + tiled GEMM + col2bias_add (or fused into GEMM). 64-bit indexing for resolutions ≥ 1024.
- [x] `GroupNorm`, `GroupNormSilu`, `LayerNorm`, `RmsNorm` — direct GLSL ports of the PTX kernels; cross-warp reductions via subgroup arithmetic + `shared` memory.
- [x] `ScaledDotProductAttention` — naive 3-pass (Q×Kᵀ → softmax → ×V) for SD1.5 / SDXL. FlashAttention-style is Phase 4+ optimization.
- [x] `Gelu`, `Silu`, `Add`, `Mul`, `Scale`, `Clamp` — single `elementwise.spv` with op selected via spec const.
- [x] `Transpose2D`, `Permute0213`, `GeGlu` (last-dim split — see PHASE_3_DEVIATIONS #16), `BroadcastAdd` (channel-aware indexing).
- [x] `Concat`, `Split` — accept the CPU-fallback initially (CUDA does too), GPU shaders later.
- [x] `UpsampleNearest2D`, `UpsampleBilinear2D` — straightforward.
- [x] `Fft`, `Stft`, `MelFilterbank` — `NotSupportedException` for now; audio doesn't need GPU in Phase 3.5.
- [x] `CastToF16`, `CastToF32`, `CastF8E4M3ToF16`, `CastF16ToF8E4M3` — dedicated cast shaders.
- [x] `Sync()` — `vkWaitSemaphores(timeline, currentValue)` then drain deferred free list.
- [x] `FreeWeights(IEnumerable<Tensor>)` — drop from cache, deferred-free regions.
- [x] **Capability flags** — `BackendCapabilities { SupportsF16 = features12.shaderFloat16; SupportsBF16 = false; SupportsConv2D = true; SupportsSdpa = true; ... }`.

### 3f. SPIR-V kernels

Build script: `native/vulkan/build.sh`. CSProj target invokes it before build. `.spv` files copied to `Spirv/` content directory; `Content/CopyToOutputDirectory=PreserveNewest`.

Each kernel ships **FP32 + FP16 variants** (same .glsl, different `-DUSE_FP16` define). Validate every shader against CPU reference within tolerance from [SPIRV_COMPUTE_SHADERS.md § Validation Tolerances](../Research/SPIRV_COMPUTE_SHADERS.md#validation-tolerances).

- [x] `elementwise.comp.glsl` — Add, Mul, Scale, SiLU, GELU, GELU-tanh, Sigmoid, Clamp (op via spec const). FP32 + FP16. Smoke-test: `add` matches CPU; `silu` matches CPU.
- [x] `transpose.comp.glsl` — 32×33 padded shared tile. FP32 + FP16. Test `[B, D1, D2] → [B, D2, D1]`.
- [x] `permute_0213.comp.glsl` — `[B, S, H, D] → [B, H, S, D]`. FP32 + FP16.
- [x] `geglu.comp.glsl` — last-dim split (decompose `outerIdx = i / D; d = i % D`, then `inputX = outerIdx * 2*D + d`, `inputGate = inputX + D`). FP32 + FP16. **Multi-row test** `[2, 2, 2*D]` mandatory — flat-midpoint bug regression check.
- [x] `broadcast_add.comp.glsl` — channel-aware: `hidden[B, C, ...spatial] += bias[B, C]`. FP32 + FP16.
- [x] `groupnorm.comp.glsl` — per-(batch, group) workgroup; subgroup reduce + cross-warp shared-mem reduce; FP32 accumulator. FP32 + FP16.
- [x] `groupnorm_silu.comp.glsl` — fused variant. Eliminates intermediate write. **Major UNet speedup.**
- [x] `layernorm.comp.glsl` — per-token. FP32 + FP16.
- [x] `rmsnorm.comp.glsl` — RMS variant (no mean subtraction). FP32 + FP16.
- [x] `softmax.comp.glsl` — three-pass with shared-memory broadcast of max + sum. FP32 accumulator. FP32 + FP16.
- [x] `im2col.comp.glsl` — 64-bit indexing via `GL_EXT_shader_explicit_arithmetic_types_int64`. FP32 + FP16. **Multi-resolution test:** 64×64, 256×256, 1024×1024.
- [x] `col2bias_add.comp.glsl` — reshape + bias. FP32 + FP16.
- [x] `upsample_nearest2d.comp.glsl`, `upsample_bilinear2d.comp.glsl` — straightforward.
- [x] `cast_f32_f16.comp.glsl`, `cast_f16_f32.comp.glsl`, `cast_f8e4m3_f16.comp.glsl`, `cast_f16_f8e4m3.comp.glsl`.
- [x] **`matmul_tiled.comp.glsl`** — the centerpiece. Spec consts: `BM`, `BN`, `BK`, `TM`, `TN`, `USE_FP16`, `TRANSPOSE_A`, `TRANSPOSE_B`, `HAS_BIAS`, `ACTIVATION`. Default tile (128, 128, 16, 8, 8). Build all variants: `nn`, `nt`, `tn` × `f32`, `f16` × bias-on/off × silu/gelu-fused/none. Total ~16 .spv files. Validates within 1e-3 of CPU GEMM. **Performance gate: ≥ 60% of cuBLAS HGEMM on the same NVIDIA hardware.**
- [ ] **Vendor tile-size tuning** — at first launch, run all `(BM, BN)` candidate combos for each typical SDXL/Flux GEMM shape; persist winner to disk. Auto-tuner is Phase 4 polish; v1 ships hard-coded vendor table.

### 3g. Optional / Phase-4 carryover

- [ ] `sdpa_flash.comp.glsl` — FlashAttention-2 style (online softmax, Br=64 / Bc=64 tiles). Phase 4 optimization; v1 uses naive 3-pass. **Phase C2 profile data shows SDPA is only 4.4% of host time on Flux Schnell** — FlashAttention is now low-priority versus the per-dispatch-overhead lever (push descriptors / kernel fusion).
- [x] `matmul_coopmat.comp.glsl` — `VK_KHR_cooperative_matrix` variant. **Implemented in Phase C2 step 2.** FP16 input, FP32-or-FP16 output (gated by `OUTPUT_F32` spec const — see [PHASE_3_5_DEVIATIONS.md #11](PHASE_3_5_DEVIATIONS.md) for why both are required), transpose-aware via `gl_CooperativeMatrixLayout{Row,Column}Major`. Bias add via follow-up `BroadcastAdd(N, 1)` dispatch. Linear time on Flux 43.8 s → 37.0 s (15%); wall-clock 138 s → 129.5 s (6%). Less than projected — see § 8 perf measurements.
- [ ] `dequant_q4_k.comp.glsl`, `dequant_q8_0.comp.glsl` — GGUF Q4/Q8 dequant on GPU. Phase 4.
- [ ] Multi-queue / async-compute — Phase 7 if needed.

## 4. Implementation — Diffusion (no changes expected)

The whole point of `IBackend` is that the diffusion package needs **zero changes**. Validate this:

- [ ] `Sd15Pipeline` runs against `VulkanBackend` with **no source changes** — flip via `Backend = new VulkanBackend(deviceOrdinal: 0)` in the integration test
- [ ] `SdxlPipeline` likewise — Phase 3.5 acceptance gate is SD1.5 only, but SDXL should also produce reasonable output (may have perf gaps)
- [x] `FluxPipeline` — Flux Schnell FP8 4-step 512×512 generation produces real photographic content matching prompt on RTX 3060 ([FluxVulkanGenerationTest.cs](../../tests/SharpInference.Diffusion.Tests/FluxVulkanGenerationTest.cs)). Two correctness bugs surfaced and fixed during this run — see [PHASE_3_5_DEVIATIONS.md #1, #2](PHASE_3_5_DEVIATIONS.md). Wall-clock ~178 s vs CUDA ~20 s — perf tuning still required.
- [x] No `weight.DataPointer` regressions — `IBackend` abstraction held cleanly; diffusion package required zero source changes

## 5. Testing & Validation

### Unit tests (`SharpInference.Vulkan.Tests`)

- [ ] `InstanceBringUpTests` — physical device enumeration; capability discovery
- [ ] `MemoryAllocatorTests` — alloc/free round-trip, alignment, large allocation, OOM retry, ReBAR detection
- [ ] `WeightCacheTests` — reference equality hits; cache survives CPU `Tensor.Dispose()`
- [ ] `StagingUploadTests` — coherent vs non-coherent flush correctness; ReBAR fast path bypass
- [ ] `DescriptorManagerTests` — pool flip on overflow; push-descriptor path when supported
- [ ] `CommandStreamTests` — timeline semaphore advances; wait blocks until value reached
- [ ] `KernelLoadTests` — every shipped `.spv` loads + builds a pipeline successfully on every supported vendor

### Per-kernel correctness (`VulkanKernelTests`)

For every kernel: dispatch on Vulkan, dispatch on CPU, compare element-wise within tolerance from [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md#validation-tolerances).

- [x] `Add_Vs_Cpu_F32` (`Backend_Add_Matches_Cpu`) — F16 variant still pending
- [ ] `Mul_Vs_Cpu_F32` / `_F16`
- [ ] `Scale_Vs_Cpu_F32` / `_F16`
- [x] `Silu_Vs_Cpu` (`Backend_Silu_Matches_Reference`)
- [ ] `Gelu_Vs_Cpu`
- [ ] `Transpose2D_Vs_Cpu`
- [ ] `Permute0213_Vs_Cpu`
- [x] `GeGlu_Vs_Cpu` (`Backend_GeGlu_Matches_Reference`) — multi-row coverage needs verification
- [ ] `BroadcastAdd_Vs_Cpu`
- [ ] `GroupNorm_Vs_Cpu` — 32-group, FP16 accumulate-FP32
- [ ] `GroupNormSilu_Vs_Cpu` — fused
- [x] `LayerNorm_Vs_Cpu` (`Backend_LayerNorm_Matches_Reference`)
- [x] `RmsNorm_Vs_Cpu` (`Backend_RmsNorm_Matches_Reference`) — *added beyond original plan*
- [ ] `Softmax_Vs_Cpu` — long-row stability test (4096 elements)
- [ ] `Im2Col_Vs_Cpu` — 64×64, 256×256, **1024×1024** (regression for 64-bit indexing)
- [ ] `Col2BiasAdd_Vs_Cpu`
- [ ] `Upsample_Vs_Cpu`
- [x] `Cast_F8E4M3_F16_Vs_Cpu` (`Backend_FP8_Cast_Matches_Cpu_AllBytes`) — F32↔F16 cast tests still pending
- [x] `MatMul_Vs_Cpu` — partial: `Backend_MatMul_Matches_Cpu_Reference`, `Backend_MatMul_F16_Roundtrip`, `Backend_MatMul_LargeFp16_Matches_Cpu`. Full transpose-combo matrix still pending.
- [x] `Linear_Vs_Cpu` — `Backend_Linear_FluxShape_F32_Matches_Cpu`. Bias-off + F16 variants still pending.
- [x] `Conv2D_Vs_Cpu` — 1×1, 3×3, stride 1 + 2, pad 0 + 1
- [x] `Sdpa_Vs_Cpu` — `Backend_SDPA_MultiHead_Matches_Cpu_Reference` + `Backend_SDPA_FluxShape_AllHeads_Match_Cpu` (H=24, S=64, D=128 — caught the auto-flush UAF, see [PHASE_3_5_DEVIATIONS.md #2](PHASE_3_5_DEVIATIONS.md))

### Cross-backend consistency (`VulkanVsCudaTests`) — runs only on dual-GPU box

- [ ] `MatMul_Vs_Cuda` — within 1e-3
- [ ] `GroupNorm_Vs_Cuda` — within 1e-3
- [x] `Conv2D_Vs_Cuda` — within 1e-3
- [ ] `Sdpa_Vs_Cuda` — within 1e-3
- [ ] **End-to-end:** `Sd15_512x512_Vulkan_Vs_Cuda` — same seed, same prompt → SSIM > 0.99 (visually indistinguishable)
- [ ] **End-to-end:** `Sdxl_1024x1024_Vulkan_Vs_Cuda` — same seed, same prompt → SSIM > 0.95 (some FP16 path differences expected)

### Performance benchmarks (`SharpInference.Benchmarks`)

- [ ] `MatMulBench` — Vulkan vs CUDA on same NVIDIA HW; target ≥ 60% of cuBLAS HGEMM
- [ ] `Sd15_Pipeline_Bench` — RTX 3060: target ≤ 8 s (CUDA: ~5 s). RX 7900 XTX: target ≤ 10 s (no CUDA reference; absolute target).
- [ ] `Sdxl_1024_Bench` — RTX 3060: target ≤ 180 s (CUDA: ~110 s).

### Memory leak validation

- [x] 100-iteration MatMul cycle on Vulkan → device-memory delta ≤ 16 MB epsilon ([VulkanLeakTests.cs](../../tests/SharpInference.Vulkan.Tests/VulkanLeakTests.cs)). Surfaced two real fixes en route — see [PHASE_3_5_DEVIATIONS.md](PHASE_3_5_DEVIATIONS.md) (callbacks neutralized in `FreeAllCached`; transient-buffer Dispose in helper Dispose).
- [ ] Full 100-step generation cycle on a real model (SD1.5 / Flux) — needs checkpoint, deferred until SSIM gate.
- [ ] Validation layer reports zero leaks on shutdown.

## 8. Performance Measurements (RTX 3060 Linux + NVIDIA driver)

Captured via `SHARPINFERENCE_VK_PROFILE=1`. Generation: Flux Schnell FP8, 4 steps, 512×512, prompt "A photograph of an astronaut riding a horse", seed=42.

| Stage | Wall-clock | Profiled host time | Linear total | Linear avg/call | Notes |
|---|---|---|---|---|---|
| Pre-Phase-C baseline | 178 s | n/a | n/a | n/a | Sync `WaitIdleHost` per op-boundary |
| C1 (async timeline-semaphore submit) | 160 s | 47.6 s | 44.9 s | 20.6 ms | `DrainAndFlush` no longer host-waits |
| C2.1 (matmul tiles 32×32→128×128) | 140 s | 49.3 s | 46.4 s | 21.3 ms | 16× fewer workgroups; GPU compute drops |
| C2.2 (coopmat F16-only guard) | 138 s | 46.4 s | 43.8 s | 20.1 ms | **Path silently bypassed** — see deviation #11 |
| **C2.2 (coopmat F32-output added)** | **129.5 s** | **39.5 s** | **37.0 s** | **17.0 ms** | Real coopmat hit on Flux Linears |
| C2.3a (push descriptors, default on) | 139.6 s | 46.7 s | 43.8 s | 20.1 ms | **Regression on NVIDIA** — see deviation #12; default reverted to opt-in via env |

**Cumulative: 178 s → 129.5 s, 1.38× faster, 27% reduction.**

CUDA reference is ~20 s on the same hardware. Vulkan is **~6.5× off-target** for the Phase 3.5 acceptance gate (≤ 1.6× CUDA). Diagnosis from the profile:

- Linear is 93.6% of profiled host time (37.0 s of 39.5 s).
- 11,319 dispatches across 2,180 Linear calls = ~5 dispatches per call (input cast + weight cast + matmul + bias add + occasionally another cast).
- 17 ms/call vs ~1 ms theoretical compute → **94% of per-call time is per-dispatch overhead** (descriptor binding, command buffer recording, vkQueueSubmit2). Bigger tiles and tensor-core compute reduce the 6% compute portion only.

Remaining levers (Phase 4 carryover):

- **Q/K/V projection fusion** — concat the three QKV weight matrices at weight-load time so the
  three sequential `Linear` calls become one matmul producing `[batch, seqLen, 3*hidden]`,
  followed by tensor-view-based slicing into Q, K, V. Cuts ~14% of total Linear dispatches
  (Q/K/V Linears are ~21% of the total; saves 2/3 of those). Estimated ~5 s wall-clock saving.
  **Deferred to Phase 4** — requires: (a) tensor view/slice API on `Tensor` (currently only
  `Reshape` exists), (b) per-block weight-loading code in `FluxDoubleStreamBlock`,
  `FluxSingleStreamBlock`, `Flux2DoubleBlock`, `Flux2SingleBlock`, and SDXL attention blocks.
  Estimated 2–3 days of careful work.
- **Pre-cast FP8 weights once at load** — currently FP8→F16 cast runs ~2 of the 5 dispatches
  per Linear. Trades VRAM (Flux Schnell ~12 GB FP8 → ~24 GB F16) for fewer dispatches; not
  feasible at full Flux size on a 12 GB card, but viable for selective caching of hot weights.
  Phase 4 work item.
- **Cooperative-matrix bias fusion** — currently coopmat path runs `BroadcastAdd` as a
  follow-up dispatch when bias is present. Adding `HAS_BIAS` to `matmul_coopmat.comp.glsl`
  would save ~50% of those follow-ups (~1,000 dispatches/generation). Estimated ~1 s wall-clock
  saving. Not worth more shader-side complexity right now.
- **Push descriptors on AMD/Intel** — measured as a *regression* on NVIDIA (deviation #12) but
  may be a win on other vendors per published guides. Code path is implemented but
  default-off; opt-in via `SHARPINFERENCE_VK_PUSH_DESCRIPTORS=1` for Phase E AMD/Intel testing.

FlashAttention is **not** a meaningful lever here — SDPA is 4.4-5% of host time.

## 6. Deviations

Track unanticipated issues in [`PHASE_3_5_DEVIATIONS.md`](PHASE_3_5_DEVIATIONS.md). Anticipated likely entries (fill in as encountered):

- Subgroup-size assumption bugs (cross-vendor 32 vs 64)
- Bank-conflict surprises in shared-memory tiles
- Mesa RADV pipeline-cache invalidation across driver upgrades
- Intel Arc variable-subgroup-size pipeline failures
- AMD wave64 vs wave32 perf cliffs in tiled GEMM
- ReBAR fast-path correctness vs perf wins
- Validation-layer false positives (rare, but worth documenting)

## 7. Review & Merge

- [ ] **Code review** — Vulkan handle-leak audit (every `vkCreate*` paired with `vkDestroy*` in `Dispose`); `[LibraryImport]` correctness; no allocations on hot path
- [ ] **Documentation update** — `docs/Design/CORE_DESIGN.md` Phase-3.5 status flips to Done; `docs/Agents/KERNEL.md` SPIR-V section gains "implemented kernels" table; `README.md` adds AMD/Intel install snippet
- [ ] **CI green** — Linux GPU runner passes Vulkan kernel tests + SD1.5 end-to-end
- [x] **Sample updated** — `samples/BasicImageGeneration` accepts `--backend cuda|vulkan|cpu`
- [ ] **Benchmark report committed** to `docs/Research/CUDA_PERFORMANCE.md` (or new `VULKAN_PERFORMANCE.md`) with measured numbers
- [ ] **Merge to main** with explicit changelog

---

## Acceptance Criteria

Phase 3.5 is **done** when:

1. `SharpInference.Vulkan` package builds and ships ~16 `.spv` kernels.
2. Every `IBackend` op has a Vulkan implementation that matches the CPU reference within documented tolerance.
3. Every `IBackend` op matches the CUDA reference within 1e-3 (FP16) on the same NVIDIA hardware.
4. SD1.5 512×512 generates the same image (SSIM > 0.99) on Vulkan and CUDA from the same seed on the same NVIDIA GPU.
5. SD1.5 512×512 runs on AMD RDNA2 or RDNA3 (Linux + Mesa RADV) and produces visually correct output.
6. RTX 3060 Vulkan SD1.5 ≤ 8 s (≤ 1.6× CUDA wall-clock).
7. No memory leaks: validation layer clean on shutdown; 100-step loop returns to baseline VRAM.
8. Diffusion package required **zero changes** beyond optional `weight.DataPointer` audit (validates `IBackend` abstraction).
