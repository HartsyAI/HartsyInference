# Phase 3.5 — Deviations from Design Plan

This document tracks every case where the Vulkan backend implementation diverged from the design plan in [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md), [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md), [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md), or from the CUDA backend's behavior. It is a debugging journal and a guide for future Vulkan work.

Format mirrors [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md): one entry per issue with **Design assumption → Deviation → How it was found → Fix → Impact**.

---

### 1. FP8 → F16 GEMM cast clobbered F32 output buffers

**Design assumption**: When the model is loaded with FP8 weights but a downstream op accumulates in F32, the GEMM kernel can be dispatched in whatever dtype is most convenient (F16 for FP8 inputs), and the output buffer's element layout doesn't matter as long as the byte count is right.

**Deviation**: `DispatchMatmul` was forcing `gemmDtype = DType.F16` when either input was FP8, but writing the GEMM result into the caller's pre-allocated F32 output buffer. The kernel writes 2 bytes per element, the consumer reads 4 — every other element is interpreted as garbage F32, which becomes NaN within one or two ops. The Flux Schnell pipeline produced a fully black image because NaNs propagated all the way through the VAE.

**How it was found**: First end-to-end Flux Schnell 4-step generation produced an all-zero RGB buffer. Backed up by checking activations after the first transformer block — they were already NaN. Disabled FP8 cast path locally and the NaNs went away, isolating the cast logic as the root cause.

**Fix**: Bind `gemmDtype` to `output.DType` directly, demoting only when capability flags require it ([VulkanBackend.cs:331-333](../../src/SharpInference.Vulkan/VulkanBackend.cs#L331-L333), mirrored at L436-438 for Conv2D and in `ScaledDotProductAttention`):

```csharp
DType gemmDtype = output.DType;
if (gemmDtype.IsFp8) gemmDtype = DType.F16;
if (gemmDtype == DType.F16 && !Capabilities.SupportsF16) gemmDtype = DType.F32;
```

The FP8 inputs are still cast up to F16 (or F32) via `CastIfNeeded` for the multiply, but the result lands in a buffer matching the output tensor's dtype.

**Impact**: Three tiny lines, but this is what unblocked end-to-end Flux generation. CUDA never had this bug because cuBLAS GEMM takes the output dtype as an explicit parameter. Lesson: kernel-level dtype selection in Vulkan must always be a function of the *output* tensor, not derived from the inputs.

---

### 2. Auto-flush mid-op freed transient upload buffers still in use

**Design assumption**: A periodic auto-flush every N dispatches keeps the GPU work queue from growing unbounded and lets the deferred-free list reclaim memory. The flush is a host-side wait + a drain of transient (cache-miss) upload buffers; ops below it don't care.

**Deviation**: `Dispatch` called `DrainTransients()` whenever `_dispatchesSinceSubmit` crossed `FLUSH_THRESHOLD`. But multi-dispatch ops — most consequentially `ScaledDotProductAttention` at Flux dimensions (24 heads × 3 dispatches per head = 72 dispatches) — share the *same* Q/K/V upload buffers across all per-head dispatches. When the flush fired mid-op, those Q/K/V buffers were tagged for deferred free, the *next* flush completed the deferred-free for them, and `vkDestroyBuffer` ran while heads ≥2 still had descriptor sets pointing at the (now-destroyed) buffers. Result on NVIDIA: silent garbage output for everything past head 1; on stricter drivers it would have validated a use-after-free.

**How it was found**: After fixing deviation #1, the image went from all-black (NaN) to **uniform gray** with no spatial structure. Wrote `Backend_SDPA_FluxShape_AllHeads_Match_Cpu` (H=24, S=64, D=128) which produced 2,788 mismatches starting precisely at head 2 — the head index where the auto-flush threshold was first crossed inside the SDPA dispatch loop.

**Fix**: Introduced an `OpScope` RAII counter to suppress mid-op drain, with explicit drain at op boundaries instead ([VulkanBackend.cs:140-174](../../src/SharpInference.Vulkan/VulkanBackend.cs#L140-L174)):

```csharp
private int _opNestingDepth;

private OpScope EnterOp() => new(this);
private readonly struct OpScope : IDisposable
{
    private readonly VulkanBackend _b;
    public OpScope(VulkanBackend b) { _b = b; b._opNestingDepth++; }
    public void Dispose()
    {
        _b._opNestingDepth--;
        if (_b._opNestingDepth == 0)
            _b.DrainAndFlush();
    }
}
```

`Dispatch` now checks `_opNestingDepth == 0` before auto-flushing. Public ops (`MatMul`, `Linear`, `Conv2D`, `ScaledDotProductAttention`) are wrapped in `using OpScope _ = EnterOp();` so all of an op's dispatches share a single transient lifetime.

**Impact**: Flux Schnell went from uniform gray to real photographic content matching the prompt. RGB std dev jumped from ~5 to ~60, all 256 byte values present, recognizable spatial structure. Lesson: any "every-N-dispatches" auto-flush in Vulkan must respect op atomicity — the only safe drain points are op boundaries, not dispatch boundaries.

---

### 3. NVIDIA Linux blob lies about promoted core features

**Design assumption**: `vkGetPhysicalDeviceFeatures2` with a chained `VkPhysicalDeviceVulkan12Features` / `Vulkan13Features` returns the truth about what the driver supports. If `shaderFloat16 == 0`, FP16 is unavailable.

**Deviation**: NVIDIA Linux 535-series drivers (and likely older) advertise `apiVersion = 1.3.x` but return zeros for `shaderFloat16`, `timelineSemaphore`, `bufferDeviceAddress`, `synchronization2`, `subgroupSizeControl`, and `computeFullSubgroups` when queried through the *promoted* v1.2/v1.3 feature structs. Querying through the original extension structs (`VkPhysicalDeviceShaderFloat16Int8FeaturesKHR`, etc.) returns the right answer, but our codepath used the consolidated chain. The features are real and `vkCreateDevice` accepts them — only the feature *query* is wrong.

**How it was found**: `VulkanFeatureProbe.IsolateFeatureQuery` test ([tests/SharpInference.Vulkan.Tests/VulkanFeatureProbe.cs](../../tests/SharpInference.Vulkan.Tests/VulkanFeatureProbe.cs)) probed each promoted struct independently and saw all-zero results despite `apiVersion = 1.3.250`. The smoke tests then failed to enable any of the features the kernels needed.

**Fix**: Trust `apiVersion` as the source of truth for guaranteed-promoted core features on known vendors ([VulkanDevice.cs:195-211](../../src/SharpInference.Vulkan/VulkanDevice.cs#L195-L211)):

```csharp
uint api = props2.properties.apiVersion;
bool atLeast12 = (api >> 22 & 0x7F) > 1 || (((api >> 22) & 0x7F) == 1 && ((api >> 12) & 0x3FF) >= 2);
bool atLeast13 = (api >> 22 & 0x7F) > 1 || (((api >> 22) & 0x7F) == 1 && ((api >> 12) & 0x3FF) >= 3);

bool fp16  = f12.shaderFloat16 != 0     || (atLeast12 && vendorID is 0x10DE or 0x1002 or 0x8086);
bool ts    = f12.timelineSemaphore != 0 || atLeast12;
bool sync2 = f13.synchronization2 != 0  || atLeast13;
// etc.
```

The vendorID guard restricts the override to NVIDIA / AMD / Intel, where these features are guaranteed by the spec at api ≥ 1.2. We still pass them through `vkCreateDevice` and rely on driver acceptance there.

**Impact**: The backend boots and runs FP16 kernels on real NVIDIA hardware. Without this override, a driver that under-reports features would silently demote to FP32, halving throughput. Likely worth re-checking on driver upgrades — file an upstream NVIDIA bug if it persists.

---

### 4. Slab size reduced from 256 MB to 64 MB after OOM at allocation

**Design assumption**: 256 MB large slab + 16 MB small slab (per [VULKAN_MEMORY_MANAGEMENT.md § Block / page allocator](../Research/VULKAN_MEMORY_MANAGEMENT.md#block--page-allocator)) is a good fit for SDXL (~7 GB working set) and Flux (~12 GB) — keeps `vkAllocateMemory` count low and matches VMA defaults.

**Deviation**: At Flux Schnell FP8 working set (~12 GB on a 12 GB RTX 3060), the *first* 256 MB slab failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY` once the model was mid-load and the heap was already 90% full. The deferred-free list couldn't free anything because the working set was real, not stale. Dropped to 64 MB / 8 MB and OOM disappeared because the allocator had finer-grained room to fit allocations into post-deferred-free gaps.

**How it was found**: Loading Flux Schnell on RTX 3060 — the first transformer-block weight upload OOM'd ~70% through the load. CUDA backend handles the same weights without issue (it uses raw `cuMemAlloc`, no slabs).

**Fix**: [VulkanMemory.cs:160-161](../../src/SharpInference.Vulkan/VulkanMemory.cs#L160-L161):

```csharp
public const ulong SLAB_LARGE = 64UL * 1024 * 1024;   // was 256 MB
public const ulong SLAB_SMALL = 8UL  * 1024 * 1024;   // was 16 MB
```

Also added an `OnOutOfMemory` callback that drains the deferred-free list and retries before propagating the failure.

**Impact**: 64 MB slabs mean ~4× more `vkAllocateMemory` calls than the design budget targeted (it aimed for ≤ 64 device allocations; we're closer to ~200 on Flux). No measurable perf hit — `vkAllocateMemory` is one-shot at load time, and the working slab count is still well under the 4096 spec cap. Lesson: at near-VRAM-limit working sets, slab granularity matters more than allocation count.

---

### 5. Cache-miss upload buffers leaked across step boundaries

**Design assumption**: `VulkanGpuTransferHelper.CopyToDevice` either hits the weight cache (returns the cached `VulkanBuffer`) or uploads via the persistent staging buffer (transient device-side copy, freed on deferred-free tick). Either way, the helper owns the lifetime.

**Deviation**: Cache-miss buffers (typical for activations and any tensor that wasn't preloaded) were created in `CopyToDevice` and returned to the caller, but never tracked for cleanup. Each transformer step leaked one buffer per cache miss. On a 4-step Flux generation that's hundreds of leaked `VkBuffer` handles — the system noticed via VRAM pressure, not validation layer (Mesa's validation catches this; NVIDIA blob doesn't).

**How it was found**: VRAM usage climbing per step during the long pipeline integration runs. Process-RSS roughly stable but driver-side allocation count rising.

**Fix**: Added a `_transientBuffers` list and `DrainTransients()` method ([VulkanGpuTransferHelper.cs:22-75](../../src/SharpInference.Vulkan/VulkanGpuTransferHelper.cs#L22-L75)). Cache-miss buffers are appended to the list, then drained at op boundaries via `DrainAndFlush` in the backend. (See deviation #2 — the original drain-on-every-flush was too aggressive; deviation #2 fixed it to drain on op exit instead.)

**Impact**: Stable VRAM across multi-step generation. No buffer-leak validation errors on Mesa LLVMpipe. The `_transientBuffers` list is also what made deviation #2 visible — without it the SDPA dispatches would have leaked but completed fine.

---

### 6. Deferred-free tag must be `_value + 1`, not `_lastSubmitted`

**Design assumption**: When an op finishes recording dispatches and wants to free a temporary buffer, tag it with the timeline value of the last *submitted* batch. Once the GPU reaches that value, the buffer is safe to destroy.

**Deviation**: At the moment `DeferredFree(buf)` is called, the dispatch that consumed `buf` has been *recorded* but not yet submitted. The submit increments `_value` to the next tick — so the dispatch will signal value `_value + 1`, not `_lastSubmitted`. Tagging with `_lastSubmitted` makes the deferred-free list believe the dispatch is already done; the next `ReclaimUpTo(currentTimeline)` happily destroys the buffer while it's still pending in a command buffer. Use-after-free, manifest as garbage values in subsequent computations.

**How it was found**: Caught by inspection while debugging deviation #2 — when I traced the timeline counter against destroyed-buffer events the off-by-one was visible. No isolated test reproduced it because the dispatch usually completes before the next reclaim runs; it was a latent UAF that would bite under load.

**Fix**: Tag with `_value + 1` ([VulkanCommandStream.cs](../../src/SharpInference.Vulkan/VulkanCommandStream.cs) — `DeferredFree(alloc, _value + 1)`). `SubmitAndAdvance` increments `_value` first, then signals that value, so the tag matches.

**Impact**: Eliminates a class of latent UAF that's hard to reproduce but real. Lesson: deferred-free tags must reference the *next* tick to be reached when the recording op submits, not the most-recent already-submitted tick.

---

### 7. GLSL has no built-in `erf` — needed for exact GELU

**Design assumption**: Per [SPIRV_COMPUTE_SHADERS.md § Kernel catalog](../Research/SPIRV_COMPUTE_SHADERS.md), elementwise GELU has both a `tanh` approximation (default for compatibility with PyTorch) and an "exact" path using `erf`. We assumed `erf` was available in core GLSL like `tanh`.

**Deviation**: GLSL 4.50 / Vulkan compute does not provide `erf` as a built-in. `glslangValidator` rejects the shader.

**Fix**: Added an Abramowitz & Stegun 7.1.26 approximation (max error 1.5e-7, well below FP32 epsilon for this kernel) inline in [elementwise.comp.glsl](../../native/vulkan/shaders/elementwise.comp.glsl):

```glsl
float erf_approx(float x) { /* Abramowitz & Stegun 7.1.26 */ }
float gelu_exact(float x) { return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865475)); }
```

**Impact**: Exact-GELU path matches CUDA reference within 1e-6. Negligible. Worth flagging the missing built-in in the shader research doc so it's not re-discovered.

---

### 8. matmul C buffer can't be `writeonly` when beta ≠ 0

**Design assumption**: The tiled GEMM output buffer is write-only — every workgroup writes its own tile and never reads the C accumulator.

**Deviation**: When the kernel supports `C = alpha*A*B + beta*C` (used for residual-add fusion), the `beta` path *does* read C. A `writeonly` qualifier on the C SSBO causes glslangValidator to reject (or, depending on flags, miscompile to an undefined-behavior load).

**Fix**: Removed `writeonly` from the matmul C buffer binding. Kept `writeonly` everywhere else (including `elementwise.comp.glsl` where the output is genuinely write-only).

**Impact**: Negligible — no perf delta from removing the qualifier on this binding alone. Lesson: `writeonly` should be added per-kernel after verifying the kernel really doesn't need to read the binding for any spec-const path.

---

### 9. glslangValidator (older) doesn't accept `-O` for SPIR-V optimization

**Design assumption**: Build script invokes `glslangValidator -V -O ...` to produce SPIR-V optimized at compile time, matching the GLSL → SPIR-V → spirv-opt pipeline assumed in [SPIRV_COMPUTE_SHADERS.md § Toolchain](../Research/SPIRV_COMPUTE_SHADERS.md).

**Deviation**: The Ubuntu 22.04 `glslang-tools` package ships a glslangValidator that doesn't recognize `-O`. Build fails with `unknown argument`.

**Fix**: Removed `-O` from `native/vulkan/build.sh`. SPIR-V is shipped unoptimized; the Vulkan driver's own SPIR-V → ISA compiler does its own optimization, so the runtime cost is minimal but startup pipeline-build time may be slightly higher.

**Impact**: First-run pipeline-build time on cold pipeline cache is ~100 ms slower per kernel. Negligible after the persistent pipeline cache warms up. To revisit: add `spirv-opt` as a separate post-step in the build script if perf-critical.

---

## Anticipated Categories (kept for reference)

The CUDA backend's experience suggested the following classes of issues; mapping how each played out in Phase 3.5:

| Category | CUDA precedent | Phase 3.5 outcome |
|---|---|---|
| 64-bit indexing for 1024+ resolutions | PHASE_3_DEVIATIONS #12 | Used `GL_EXT_shader_explicit_arithmetic_types_int64` from day one in im2col; no incident |
| Last-dim split for gated activations | PHASE_3_DEVIATIONS #16 | GeGLU shader written with explicit decomposition; multi-row test passed first try |
| Stream / queue race with sync copies | PHASE_3_DEVIATIONS #19 | See deviation #2 — the Vulkan equivalent was the auto-flush draining transients mid-op |
| Async free deferred cleanup | PHASE_3_DEVIATIONS #18 | See deviation #6 — the off-by-one in the timeline tag |
| In-place ops invalidating activation cache | PHASE_3_DEVIATIONS #17 | Softmax pre-emptively split into separate `probsBuf` (defensive, no incident) |
| Weight `DataPointer` access after preload | PHASE_3_DEVIATIONS #14–15 | Diffusion package required no changes — `IBackend` abstraction held |
| Subgroup-size cross-vendor surprises | (new for Vulkan) | Not yet validated on AMD/Intel — flagged for cross-vendor verification |
