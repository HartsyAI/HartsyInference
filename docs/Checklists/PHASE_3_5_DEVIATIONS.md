# Phase 3.5 — Deviations from Design Plan

This document tracks every case where the Vulkan backend implementation diverged from the design plan in [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md), [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md), [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md), or from the CUDA backend's behavior. It is a debugging journal and a guide for future Vulkan work.

Format mirrors [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md): one entry per issue with **Design assumption → Deviation → How it was found → Fix → Impact**.

---

### 1. FP8 → F16 GEMM cast clobbered F32 output buffers

**Design assumption**: When the model is loaded with FP8 weights but a downstream op accumulates in F32, the GEMM kernel can be dispatched in whatever dtype is most convenient (F16 for FP8 inputs), and the output buffer's element layout doesn't matter as long as the byte count is right.

**Deviation**: `DispatchMatmul` was forcing `gemmDtype = DType.F16` when either input was FP8, but writing the GEMM result into the caller's pre-allocated F32 output buffer. The kernel writes 2 bytes per element, the consumer reads 4 — every other element is interpreted as garbage F32, which becomes NaN within one or two ops. The Flux Schnell pipeline produced a fully black image because NaNs propagated all the way through the VAE.

**How it was found**: First end-to-end Flux Schnell 4-step generation produced an all-zero RGB buffer. Backed up by checking activations after the first transformer block — they were already NaN. Disabled FP8 cast path locally and the NaNs went away, isolating the cast logic as the root cause.

**Fix**: Bind `gemmDtype` to `output.DType` directly, demoting only when capability flags require it ([VulkanBackend.cs:331-333](../../src/HartsyInference.Vulkan/VulkanBackend.cs#L331-L333), mirrored at L436-438 for Conv2D and in `ScaledDotProductAttention`):

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

**Fix**: Introduced an `OpScope` RAII counter to suppress mid-op drain, with explicit drain at op boundaries instead ([VulkanBackend.cs:140-174](../../src/HartsyInference.Vulkan/VulkanBackend.cs#L140-L174)):

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

**How it was found**: `VulkanFeatureProbe.IsolateFeatureQuery` test ([tests/HartsyInference.Vulkan.Tests/VulkanFeatureProbe.cs](../../tests/HartsyInference.Vulkan.Tests/VulkanFeatureProbe.cs)) probed each promoted struct independently and saw all-zero results despite `apiVersion = 1.3.250`. The smoke tests then failed to enable any of the features the kernels needed.

**Fix**: Trust `apiVersion` as the source of truth for guaranteed-promoted core features on known vendors ([VulkanDevice.cs:195-211](../../src/HartsyInference.Vulkan/VulkanDevice.cs#L195-L211)):

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

**Fix**: [VulkanMemory.cs:160-161](../../src/HartsyInference.Vulkan/VulkanMemory.cs#L160-L161):

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

**Fix**: Added a `_transientBuffers` list and `DrainTransients()` method ([VulkanGpuTransferHelper.cs:22-75](../../src/HartsyInference.Vulkan/VulkanGpuTransferHelper.cs#L22-L75)). Cache-miss buffers are appended to the list, then drained at op boundaries via `DrainAndFlush` in the backend. (See deviation #2 — the original drain-on-every-flush was too aggressive; deviation #2 fixed it to drain on op exit instead.)

**Impact**: Stable VRAM across multi-step generation. No buffer-leak validation errors on Mesa LLVMpipe. The `_transientBuffers` list is also what made deviation #2 visible — without it the SDPA dispatches would have leaked but completed fine.

---

### 6. Deferred-free tag must be `_value + 1`, not `_lastSubmitted`

**Design assumption**: When an op finishes recording dispatches and wants to free a temporary buffer, tag it with the timeline value of the last *submitted* batch. Once the GPU reaches that value, the buffer is safe to destroy.

**Deviation**: At the moment `DeferredFree(buf)` is called, the dispatch that consumed `buf` has been *recorded* but not yet submitted. The submit increments `_value` to the next tick — so the dispatch will signal value `_value + 1`, not `_lastSubmitted`. Tagging with `_lastSubmitted` makes the deferred-free list believe the dispatch is already done; the next `ReclaimUpTo(currentTimeline)` happily destroys the buffer while it's still pending in a command buffer. Use-after-free, manifest as garbage values in subsequent computations.

**How it was found**: Caught by inspection while debugging deviation #2 — when I traced the timeline counter against destroyed-buffer events the off-by-one was visible. No isolated test reproduced it because the dispatch usually completes before the next reclaim runs; it was a latent UAF that would bite under load.

**Fix**: Tag with `_value + 1` ([VulkanCommandStream.cs](../../src/HartsyInference.Vulkan/VulkanCommandStream.cs) — `DeferredFree(alloc, _value + 1)`). `SubmitAndAdvance` increments `_value` first, then signals that value, so the tag matches.

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

### 12. Push descriptors regressed perf on NVIDIA — default off

**Design assumption**: `VK_KHR_push_descriptor` (or core-1.4 `vkCmdPushDescriptorSet`) eliminates
the per-dispatch `vkAllocateDescriptorSets` + `vkUpdateDescriptorSets` round-trip by writing
descriptor bindings directly into the command buffer. Common guidance is that this saves the
host-time spent on pool allocation and update calls, especially on draw-heavy workloads.

**Deviation**: Measured on RTX 3060 / NVIDIA Linux 535 driver, push descriptors are *slower*
than the pool-ring path for our workload. Flux Schnell 4-step:
- Pool-ring path:   129.5 s wall-clock, Linear total 37.0 s
- Push-descriptor:  139.6 s wall-clock, Linear total 43.8 s (≈ 7% regression)

**How it was found**: After implementing push descriptors as a default-on optimization in Phase
C2 step 3, re-ran the Flux Schnell benchmark expecting ~10–30% improvement; saw a 7%
regression instead. Profile dump showed Linear avg/call rose from 17.0 ms → 20.1 ms.

**Probable cause**: NVIDIA's pool-ring implementation is highly tuned (descriptors live in a
pre-allocated pool, allocation is essentially a counter bump). Push descriptors in contrast
require copying the descriptor data (~32 bytes × 5 bindings = 160 B per dispatch × 11,319
dispatches = 1.8 MB) into the command buffer at recording time, which on this driver is more
expensive than the pre-allocated-pool path. Other vendors (AMD, Intel) may differ; the
extension is reportedly the recommended path on AMD RDNA per several published guides.

**Fix**: Push descriptors remain implemented in
[VulkanDescriptorManager.PushSet](../../src/HartsyInference.Vulkan/VulkanDescriptorManager.cs)
and in the
[VulkanBackend.Dispatch](../../src/HartsyInference.Vulkan/VulkanBackend.cs) branch, but are
**default-off**. Opt-in via `HARTSYINFERENCE_VK_PUSH_DESCRIPTORS=1`. To revisit on AMD when
Phase E hardware is available — if push descriptors are a win there, we'd want a per-vendor
default rather than a universal one.

**Impact**: Small — one config flip and one comment block. Important lesson: not every Vulkan
"best practice" generalizes; profile before defaulting on. The current default (pool path) is
the right choice for the only hardware we've measured.

---

### 11. Coopmat F16-output guard silently skipped Flux's F32-output Linears

**Design assumption**: `VK_KHR_cooperative_matrix` writes via `coopMatStore` from a fp16 Accumulator
coopmat to a `float16_t[]` buffer; the only safe output dtype for the coopmat path is therefore
F16. A guard `if (output.DType != DType.F16) return false;` would correctly route F32-output ops
to the tiled fallback.

**Deviation**: Flux's `FluxTransformer.cs` allocates *every* Linear output as `DType.F32`
([line 164](../../src/HartsyInference.Diffusion/Models/Denoisers/FluxTransformer.cs#L164),
[L169](../../src/HartsyInference.Diffusion/Models/Denoisers/FluxTransformer.cs#L169),
[L199](../../src/HartsyInference.Diffusion/Models/Denoisers/FluxTransformer.cs#L199),
[L219](../../src/HartsyInference.Diffusion/Models/Denoisers/FluxTransformer.cs#L219), and many
more) regardless of the input/weight dtype. With the F16-only guard, the coopmat path was never
actually hit during Flux Schnell generation — every Linear silently fell through to
`matmul_tiled`. Wall-clock improvement from "enabling" coopmat was 1.4% (138 s → 138 s within
noise), confirming the path was inert.

**How it was found**: After "enabling" coopmat with the F16-only guard, the Flux Schnell
benchmark showed essentially no change. Tracing `output.DType` in `Linear` calls inside the Flux
transformer showed they were all F32. Profile dump's `Linear` stats (count and dispatches/call)
were identical to the pre-coopmat run — definitive evidence the coopmat shader was never
launched.

**Fix**: Added an `OUTPUT_F32` spec const + a second binding (slot 4) in
[matmul_coopmat.comp.glsl](../../native/vulkan/shaders/matmul_coopmat.comp.glsl) so the
accumulator (already fp32 internally on every conformant impl) can be stored directly to a
fp32 buffer when the host requests it. Host guard relaxed to
`output.DType is DType.F16 or DType.F32`; binds the real output buffer to either slot 2
(fp16) or slot 4 (fp32) based on the spec const, with the unused slot bound to a placeholder
to keep descriptor count uniform.

**Impact**: Linear time dropped from 43.8 s → 37.0 s (15% faster) on Flux Schnell 4-step.
Wall-clock improved 138 s → 129.5 s (6%). Less than the 2-4× projected for coopmat in
isolation — see the perf measurements section below for why per-dispatch overhead, not GEMM
compute, dominates at Flux Linear shapes. Lesson: tensor-core paths must store to whatever
dtype the surrounding pipeline allocated; gating on fp16-only is correct but disables the path
on every realistic transformer model.

---

### 10. Cross-warp reduction overflow at small subgroup sizes (LLVMpipe / Intel iGPU)

**Design assumption**: The GroupNorm / LayerNorm / RmsNorm / Softmax cross-warp reductions
follow the standard pattern: each subgroup reduces internally via `subgroupAdd`, the first
lane stores the per-subgroup partial in `shared float warp_sum[N]`, then subgroup 0 loads
`warp_sum[invId]` and reduces. The final reduction lane mask is gated with
`gl_SubgroupInvocationID < gl_NumSubgroups`, which works correctly as long as
`gl_NumSubgroups ≤ subgroupSize`.

**Deviation**: On Mesa LLVMpipe (software Vulkan), subgroup size is small (often 4–8). With
`local_size_x = 256`, `gl_NumSubgroups = 32` or `64`, which is *larger* than the subgroup
size. The single-subgroup final reduction only covers the first `subgroupSize` partials and
silently drops the rest. Real AMD (subgroup 32 or 64 on RDNA) and NVIDIA (subgroup 32) are
unaffected because `local_size / subgroupSize` stays ≤ subgroupSize. Intel iGPU is the most
likely real-hardware target to hit this.

**How it was found**: Phase E pre-fly run of the Vulkan smoke suite under
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (Mesa LLVMpipe). 30/33 pass;
the 3 failures are all GroupNorm-shaped:
- `Backend_GroupNorm_Matches_Cpu_Sd15Shape`: 81,869 mismatches, max error 2.01
- `Backend_GroupNormSilu_Matches_Cpu`: error -0.11
- The other reduction kernels (LayerNorm / RmsNorm) pass because their workgroup sizes are
  small enough that `gl_NumSubgroups ≤ subgroupSize` holds even on LLVMpipe.

**Fix**: Not yet shipped. Two viable options:
1. **Pin `requiredSubgroupSize`** at pipeline creation via
   `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` to a value that satisfies
   `local_size_x / requiredSubgroupSize ≤ requiredSubgroupSize` (i.e. for `local_size_x =
   256`, need `requiredSubgroupSize ≥ 16`). Intel Arc, AMD RDNA, NVIDIA all support
   pinning ≥ 16. LLVMpipe doesn't, so it would still fail there — but LLVMpipe is a
   software validation tool, not a deployment target.
2. **Multi-pass reduction**: when `gl_NumSubgroups > subgroupSize`, do a second pass through
   `shared` memory with another subgroup-internal reduction. Slightly more expensive but
   driver-agnostic.

**Impact**: Real AMD RDNA2/3 and Intel Arc are unaffected on the typical
`local_size_x = 256` workgroups. Phase 3.5 acceptance gate #5 (SD1.5 visually correct on
AMD RDNA + Mesa RADV) is at risk only if RADV reports subgroup ≤ 8, which is unlikely.
Recommend implementing fix #1 (`requiredSubgroupSize`) before AMD acceptance run, and
upgrading to fix #2 if Intel iGPU support becomes a deployment target.

---

### 13. Wrong `VkAccessFlags2` sync2 constants weakened every compute barrier

**Found by**: the 2026-06-17 Vulkan bug-audit sweep (sibling of the CUDA cuBLASLt-constant find). Three `VkAccessFlags2` values in `VulkanEnums.cs` were wrong, and two of them are used by `RecordGlobalComputeBarrier`, the barrier emitted between every compute dispatch:

| Constant | Was | Should be (spec) | Note |
|---|---|---|---|
| `UniformRead` | `0x40` | `0x08` | `0x40` collides with `ShaderWrite`; unused, latent |
| `ShaderStorageRead` | `0x100000000` | `0x200000000` | `0x100000000` is actually `SHADER_SAMPLED_READ` |
| `ShaderStorageWrite` | `0x200000000` | `0x400000000` | `0x200000000` is actually `SHADER_STORAGE_READ` |

So the compute-to-compute barrier requested `srcAccess = SHADER_STORAGE_READ` (should be WRITE) and `dstAccess = SHADER_SAMPLED_READ` (should be STORAGE_READ) — the storage write-to-read hazard it exists to cover was not actually specified. NVIDIA's driver appears to over-flush on a compute barrier, so the 33-test suite passed both before and after, but it is a genuine spec violation that could surface as intermittent wrong results on a stricter driver or under load. Same wrong access masks also fed the transfer-to-compute barrier in `VulkanGpuTransferHelper`.

**Fix**: corrected all three to spec values, added `ShaderSampledRead = 0x100000000` for completeness, and pinned them with `VulkanConstantTests` (CPU-only regression). Validated: 33/33 Vulkan tests pass on the RTX 3060.

### 14. `ErrorOutOfPoolMemory` had the wrong VkResult value

`VkResult.ErrorOutOfPoolMemory` was `-1000257000` (an unrelated maintenance-range code); spec value is `-1000069000`. Still negative so `ThrowOnError` still threw, but the error would render as the raw int instead of the name. Also note `AllocateSet` compares against the literal `-1000257000` for the out-of-pool retry path, which therefore matched the (wrong) enum but not a real driver `VK_ERROR_OUT_OF_POOL_MEMORY`. Fixed the enum; the literal in `AllocateSet` also checks `r == VkResult.ErrorOutOfPoolMemory`, so the retry still fires.

### 15. `nonCoherentAtomSize` flush/invalidate range could miss the buffer tail

`FlushIfNonCoherent` / `InvalidateIfNonCoherent` aligned the offset down but computed `size` from `buffer.Size` rounded up independently, so for a buffer whose offset was not already atom-aligned, `alignedOffset + alignedSize` could end before the real data end, leaving the tail un-flushed (silent stale data). Fixed with `start = AlignDown(off, atom); end = AlignUp(off + size, atom); size = end - start`. Latent on the RTX 3060 (all host-visible memory types here are `HOST_COHERENT`, so both methods early-return), but a real corruption bug on a non-coherent driver/ReBAR path.

### 16. `VulkanBuffer.AsSpan<T>` truncated 64-bit sizes to int

`new Span<T>(ptr, (int)(Size / sizeof(T)))` silently truncated the element count for buffers above `int.MaxValue` elements (the Vulkan analogue of the CUDA im2col int32 overflow). Now throws a clear error above `int.MaxValue` instead of returning a partial view.

### 17. im2col 32-bit indexing reassessed (corrects the "no incident" claim below)

The "Anticipated Categories" table previously claimed im2col used 64-bit indexing "from day one". The audit found the shader's `uint64_t total` is only used for the bounds check; the actual index variables (`linear`, `perImage`, `inIdx`, `outIdx`) are 32-bit `uint`. Unlike CUDA this is **bounded by `maxStorageBufferRange`** (~4 GB on NVIDIA): a column buffer large enough to overflow a 32-bit element index cannot be fully addressed by a shader anyway, so it is a robustness gap rather than the silent-corruption-at-1024² that CUDA had. Mitigation shipped: a C# guard in `VulkanBackend` Conv2D that throws a clear `NotSupportedException` when `colElements > int.MaxValue` instead of silently corrupting. A full fix (widen the shader to 64-bit workgroup-derived indexing, `linear = uint64_t(gl_WorkGroupID.x) * local_size + gl_LocalInvocationID.x`) is deferred because no SPIR-V compiler was available in the audit environment to rebuild and validate the `.spv`.

### Deferred / noted (not fixed this pass)

- **Descriptor pool `FlipPool` resets without a timeline wait** (`VulkanDescriptorManager`). A real hazard only on the allocated-set fallback path; the hot path uses push descriptors (`PushDescriptorActive`, core on the 3060), so it does not trigger here. A correct fix threads the command-stream timeline into the descriptor manager and waits on the last-allocated tick before reset. Deferred.
- **coopmat dead-code ternaries** (`outputIsF32 ? outBuf.Handle : outBuf.Handle`) were simplified to a single output handle with a clarifying comment; functionally correct before and after.
- **Per-dispatch global barrier serializes independent SDPA heads** — a perf opportunity (scope the barrier to the written buffer), not a correctness bug. See [`../Research/VULKAN_OPTIMIZATION.md`](../Research/VULKAN_OPTIMIZATION.md) §2.3.

---

### 18. Cooperative-matrix sType constants were swapped (and the path wasn't shape-gated)

**Found by**: the 2026-06-17 cross-vendor optimization pass. Two issues in the coopmat matmul path, both cross-vendor correctness:

1. **Swapped sType values.** `VkStructureType.PhysicalDeviceCooperativeMatrixFeaturesKHR` was `1000506000` and `CooperativeMatrixPropertiesKHR` was `1000506001`; the spec is the reverse (`COOPERATIVE_MATRIX_PROPERTIES_KHR = 1000506000`, `PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR = 1000506001`). So at device creation the coopmat *features* struct was chained with the *properties* sType. NVIDIA's driver enables coopmat leniently regardless, so the 3060 tests passed, but a strict driver / AMD / Intel would not recognize the feature struct and fail to enable `cooperativeMatrix`. Fixed both (and added `PhysicalDeviceCooperativeMatrixPropertiesKHR = 1000506002`).

2. **No shape enumeration.** `HasCooperativeMatrix` was set purely from the extension *name* being present. But the `matmul_coopmat` shader hard-codes one configuration: 16x16x16, FP16 A/B, FP32 accumulate (C/Result), subgroup scope. NVIDIA always reports that combo; AMD RDNA3 / Intel Arc advertise different sets, so blindly assuming it would fail pipeline creation or miscompute. Added `CoopMatShapeSupported` which loads `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` via `vkGetInstanceProcAddr`, enumerates `VkCooperativeMatrixPropertiesKHR`, and enables coopmat only when the exact combo is present.

**Validated**: device-info reports `CoopMat=True` on the RTX 3060 (the enumeration finds the combo, confirming the new struct layout + `VkComponentTypeKHR`/`VkScopeKHR` enum values are correct) and `CoopMat=False` on llvmpipe (no coopmat, correctly disabled, falls back to tiled). The matmul-vs-CPU FP16 tests pass with coopmat enabled.

### 19. Cross-subgroup reduction dropped partials when gl_NumSubgroups > subgroupSize

The layernorm / rmsnorm / groupnorm / groupnorm_silu / softmax shaders already used subgroup reductions, but the second-stage cross-subgroup combine read `warp[gl_SubgroupInvocationID]` guarded by `< gl_NumSubgroups`. When a device's subgroup size is smaller than the subgroup count, the final-stage subgroup has fewer lanes than there are partials, so partials beyond `subgroupSize` were silently dropped. With `local_size = 256`: safe on NVIDIA (subgroup 32 -> 8 subgroups) and AMD (wave32/64), but **wrong on small-subgroup devices** (llvmpipe / older Intel: subgroup 8 -> 32 subgroups drops partials 8..31). The previous deviation #10 flagged the array-sizing risk; this is the read-side correctness bug.

**Fix**: replaced the guarded single read with a strided fold (`for k = gl_SubgroupInvocationID; k < gl_NumSubgroups; k += gl_SubgroupSize`) so each final-stage lane accumulates its strided share of all partials, then one subgroup reduction combines them. Correct for any subgroup size. Also bumped the `warp_*` shared arrays from `[32]` to `[64]` to cover the worst case (local 256 with a pinned subgroup as small as 4).

**Validated cross-vendor**: a new `VulkanCrossVendorTests` runs LayerNorm and RmsNorm at D=1024 (every subgroup holds real data — the discriminating case). On the RTX 3060 (subgroup 32) maxErr ~1e-7; **on llvmpipe (subgroup 8 -> 32 subgroups) maxErr ~1e-7 as well** — the exact configuration the old code got wrong now matches the CPU reference. Full suite: 38/38 on the 3060; on llvmpipe all compute tests pass (one pre-existing pipeline-cache test fails because software lavapipe returns an empty cache blob, unrelated to this work).

### 20. INT8 dot-product GEMM (cross-vendor DP4a/IMMA path) + NVIDIA lies about `shaderIntegerDotProduct`

Added a `matmul_int8` shader and `VulkanBackend.MatMulInt8` using the `GL_EXT_integer_dot_product` `dotPacked4x8` instruction — the cross-vendor equivalent of CUDA's INT8 tensor cores (DP4a / IMMA), HW-accelerated on NVIDIA (`integerDotProduct4x8BitPackedSignedAccelerated`), AMD, and Intel. Computes `C[M,N] = (A_i8 @ B_i8^T) * scaleA * scaleB`: int8 operands read as `int[]` (4 packed int8 per word, no 8-bit-storage extension needed), accumulated exactly in int32, then dequantized to F32.

Two findings worth recording:

1. **`dotPacked4x8` overload by sign.** With `uint` operands the function returns `uint` (the unsigned overload); for signed int8 the operands must be typed `int[]` so the SIGNED overload (`int` result) is selected. Same bytes, different interpretation — a reinterpret, not a conversion.

2. **NVIDIA lies about `shaderIntegerDotProduct` too** (extends deviation #3). The 3060 reported `int8dot=False` through `VkPhysicalDeviceVulkan13Features` even though the feature is real (gpuinfo confirms it, and the GPU has had DP4a since Pascal). Applied the same vendor + apiVersion fallback used for fp16/timeline/sync2: enable when the flag is set OR (apiVersion >= 1.3 AND vendor is NVIDIA/AMD/Intel). Verified the feature is then accepted on device-create (no `VK_ERROR_FEATURE_NOT_PRESENT`).

**Validated**: `VulkanInt8GemmTests` runs the kernel against an exact int64 reference (the integer dot is exact; only the final float scale rounds). On the RTX 3060: **maxErr = 0.000** (bit-exact) across 32x48x64, 17x20x64 (unaligned, exercises the bounds check), and 8x8x256 (large K), plus a K%4 rejection test. On llvmpipe (vendor not in the fallback, no dot-product support) the tests skip gracefully and device creation is unaffected — the correct "use where supported, absent cleanly elsewhere" cross-vendor behavior. Full suite 42/42 on the 3060. The kernel is the correct one-invocation-per-output baseline (like the CUDA MMA kernel started naive); tiling for bandwidth can layer onto the verified dot-product path later, and pipeline integration (quantized weight loading) is the next step.

### Note: subgroup reductions and coopmat were already implemented

The research survey ([`../Research/VULKAN_OPTIMIZATION.md`](../Research/VULKAN_OPTIMIZATION.md)) listed "implement subgroup reductions" and "make coopmat the default matmul" as top recommendations, but both were already present in the backend (the research agents inferred shared-memory trees without reading the shaders). The actual high-value work was the cross-vendor *correctness* of those existing paths (#18, #19), not new implementation.

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
