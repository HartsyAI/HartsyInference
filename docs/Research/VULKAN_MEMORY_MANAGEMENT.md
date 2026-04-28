# Vulkan Memory Management — Research Notes

CUDA hides nearly all of memory management behind `cuMemAlloc` / `cuMemFree` (or `cuMemAllocAsync` on a stream). Vulkan does not. The application is responsible for: (a) discovering memory types and heaps, (b) selecting the right type for each use case, (c) sub-allocating from large `VkDeviceMemory` blocks (because the spec only guarantees 4096 simultaneous allocations), (d) handling host-visible vs device-local correctly, (e) flushing/invalidating non-coherent ranges, (f) tracking buffer lifetimes through the synchronization graph so we never free memory still referenced by an in-flight command buffer, and (g) recycling descriptor sets and command buffers across frames.

This document specifies how SharpInference's `VulkanMemoryAllocator`, `VulkanDescriptorManager`, `VulkanCommandPool`, and `VulkanGpuTransferHelper` are structured so the inference pipeline matches CUDA semantically: lazy-sync activation cache, GPU weight preloading, OOM retry, stream-ordered free.

The reference implementation patterns here are inspired by AMD's [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (the de-facto standard C++ allocator) but rewritten for the C# memory model and tailored to **inference workloads** (large weights uploaded once, activations recycled per step, no images/textures/sparse memory).

Sources: [Vulkan 1.3 Spec — Resources](https://docs.vulkan.org/spec/latest/chapters/resources.html), [Vulkan Memory Allocator (AMD)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator), [VMA documentation site](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/), [Vulkan Memory Types Tutorial](https://docs.vulkan.org/guide/latest/memory_allocation.html), [VK_EXT_memory_budget proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_memory_budget.html), [Adam Sawicki — Vulkan Memory Heaps Cheatsheet](https://gpuopen.com/learn/vulkan-device-memory/), [llama.cpp Vulkan allocator](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

---

## Memory Hierarchy on Real GPUs

### Common heap layouts

| Vendor / GPU | Heap 0 (Device-local) | Heap 1 (Host-visible) | ReBAR / Smart Access |
|---|---|---|---|
| NVIDIA discrete | VRAM (10–24 GB) | RAM | 256 MB pinned region (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) on most consumer cards; full VRAM on some Quadro |
| AMD discrete (no ReBAR) | VRAM | RAM | 256 MB pinned region |
| AMD discrete (ReBAR / SAM enabled) | VRAM | RAM | full VRAM mapped (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`) |
| Intel Arc (ReBAR) | VRAM | RAM | full VRAM mapped |
| Intel iGPU | shared system RAM | shared system RAM | most types are `DEVICE_LOCAL | HOST_VISIBLE` because the iGPU and CPU share memory |
| AMD APU | shared system RAM | shared system RAM | as above |
| Apple M-series (MoltenVK) | unified memory | unified memory | as above |

### Memory types per vendor (typical)

NVIDIA RTX 30/40 series:

| Type idx | Heap idx | Property bits |
|---|---|---|
| 0 | 1 (RAM) | `HOST_VISIBLE | HOST_COHERENT` |
| 1 | 1 (RAM) | `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` |
| 2 | 0 (VRAM) | `DEVICE_LOCAL` |
| 3 | 0 (VRAM, 256 MB ReBAR window) | `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` |

AMD RDNA3 with ReBAR:

| Type idx | Heap idx | Property bits |
|---|---|---|
| 0 | 0 (VRAM) | `DEVICE_LOCAL` |
| 1 | 0 (VRAM) | `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` |
| 2 | 1 (RAM) | `HOST_VISIBLE | HOST_COHERENT` |
| 3 | 1 (RAM) | `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` |

The exact layout varies; **always discover** at startup via `vkGetPhysicalDeviceMemoryProperties2`.

---

## Memory Type Selection Algorithm

Given a (`memoryTypeBits`, requested `propertyFlags`) pair, find the best memory type:

```csharp
public uint FindMemoryType(uint memoryTypeBitsMask, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0)
{
    uint best = uint.MaxValue;
    int bestScore = int.MinValue;
    for (uint i = 0; i < _memProps.memoryTypeCount; i++)
    {
        if ((memoryTypeBitsMask & (1u << (int)i)) == 0) continue;
        var t = _memProps.memoryTypes[i];
        if ((t.propertyFlags & required) != required) continue;

        int score = 0;
        score += BitsSet(t.propertyFlags & preferred);
        score -= BitsSet(t.propertyFlags & ~required & ~preferred);  // penalize extra bits
        // Tie-break: smaller heap idx (usually = main VRAM) wins for DEVICE_LOCAL
        if ((required & DEVICE_LOCAL) != 0)
            score += (int)(_memProps.memoryHeaps[t.heapIndex].size / (1L << 30));   // bigger heap better

        if (score > bestScore) { bestScore = score; best = i; }
    }
    return best;
}
```

### Selection rules per use-case

| Use-case | required | preferred | Notes |
|---|---|---|---|
| **Weights (preloaded, GPU-only)** | `DEVICE_LOCAL` | (none) | Large allocations → pure VRAM heap |
| **Activations (GPU-resident)** | `DEVICE_LOCAL` | (none) | Same as weights but recycled |
| **Staging upload (CPU → GPU, write-once)** | `HOST_VISIBLE | HOST_COHERENT` | (none) | Avoid `HOST_CACHED` (write-combined is faster for write-only) |
| **Staging readback (GPU → CPU)** | `HOST_VISIBLE | HOST_COHERENT` | `HOST_CACHED` | CPU reads benefit from cache |
| **Hot ReBAR upload (skip staging)** | `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` | (none) | Falls back to staging if not present |
| **Tiny constants (push consts < 128 B)** | n/a — push constants live in command buffer | | |

### `DeviceCoherentAMD`

Some AMD configs offer `0x40 DEVICE_COHERENT_AMD` — fine-grained CPU↔GPU coherence. Useful for atomic counters but not for tensor data. Ignore in v1.

---

## Sub-Allocation Strategy

The spec guarantees only 4096 simultaneous `VkDeviceMemory` allocations. SDXL has thousands of weight tensors; we cannot allocate per tensor.

### Allocation count budget

- 4096 total cap — but the *practical* cap is lower because each `vkAllocateMemory` is slow (~50–500 µs) and fragments VRAM.
- Original target: ≤ 64 device allocations. **Actual after Phase 3.5 implementation:** ~200 on a 12 GB Flux Schnell working set. Still well under spec cap; see deviation note below.

### Block / page allocator

We use a **slab allocator** with two block sizes:

| Pool | Block size | Use |
|---|---|---|
| Large (`SLAB_LARGE = 64 MB`) | 64 MB | Weights, large activations |
| Small (`SLAB_SMALL = 8 MB`) | 8 MB | Per-tensor temporaries, small tensors (< 1 MB) |

> **Note (Phase 3.5):** The original design specified 256 MB / 16 MB slabs to match VMA defaults. Loading Flux Schnell FP8 (~12 GB) on a 12 GB RTX 3060 OOM'd at ~70% load with the larger slabs because the deferred-free list couldn't fit reclaimed allocations into the gaps left in a near-full heap. Reducing to 64 MB / 8 MB resolves the OOM at the cost of more `vkAllocateMemory` calls. See [PHASE_3_5_DEVIATIONS.md #4](../Checklists/PHASE_3_5_DEVIATIONS.md). Allocator also gained an `OnOutOfMemory` callback that drains the deferred-free list and retries.

A pool tracks free regions inside its blocks via a sorted list (or tree) of `(offset, size)` pairs. Allocation = first-fit (or best-fit if fragmentation grows). Free = merge with adjacent regions.

```csharp
internal sealed class VulkanMemoryBlock : IDisposable
{
    public ulong DeviceMemory;          // VkDeviceMemory handle
    public nint  MappedPointer;         // 0 if not host-visible
    public ulong Size;
    public uint  MemoryTypeIndex;
    public uint  HeapIndex;
    public List<FreeRegion> FreeList;   // sorted by offset
    public int   AllocCount;            // number of live sub-allocations
}

internal readonly record struct FreeRegion(ulong Offset, ulong Size);
```

Allocator API:

```csharp
public sealed class VulkanMemoryAllocator
{
    public VulkanAllocation Allocate(
        ulong size, ulong alignment,
        uint memoryTypeBits,
        VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred = 0);

    public void Free(VulkanAllocation alloc);

    public ulong UsedBytes(uint heapIndex);
    public ulong BudgetBytes(uint heapIndex);   // VK_EXT_memory_budget
}

public readonly record struct VulkanAllocation(
    ulong DeviceMemory,    // backing VkDeviceMemory
    ulong Offset,          // byte offset within the block
    ulong Size,            // requested size (after alignment)
    nint  MappedPointer,   // 0 if device-only
    uint  MemoryTypeIndex,
    int   BlockId);        // for free()
```

When a free request would leave the block 100% free, the block is returned to the OS (`vkFreeMemory`) — except for the first block of each pool, which is kept resident to avoid alloc thrash on the next allocation.

### Alignment rules

Each `vkGetBufferMemoryRequirements` returns a (size, alignment) pair specific to that buffer's usage flags. Sub-allocator must:

1. Round offset up to `alignment`.
2. Verify `(memReq.memoryTypeBits & (1u << blockTypeIndex)) != 0` — buffer-compatible type.
3. Round offset up to `nonCoherentAtomSize` if memory is non-coherent (so flush ranges are valid).

### Buffer wrapping

Each tensor allocation creates one `VkBuffer` bound to a region of the block:

```csharp
public sealed class VulkanBuffer : IDisposable
{
    public ulong Handle;                  // VkBuffer
    public VulkanAllocation Backing;
    public ulong Size;
    public VkBufferUsageFlags Usage;

    public unsafe Span<T> AsSpan<T>() where T : unmanaged
        => new((void*)(Backing.MappedPointer + (nint)Backing.Offset), (int)(Size / (ulong)sizeof(T)));
}
```

`vkBindBufferMemory(buffer, alloc.DeviceMemory, alloc.Offset)` ties the buffer to its slab region. Multiple `VkBuffer` handles can alias overlapping regions of the same `VkDeviceMemory` — useful for views — though we don't rely on this in v1.

---

## GPU Weight Cache (Vulkan port)

The CUDA backend's `GpuTransferHelper` keeps a `Dictionary<Tensor, ulong>` mapping each weight `Tensor` to its uploaded device pointer. The Vulkan equivalent maps to `VulkanBuffer`:

```csharp
public sealed class VulkanGpuTransferHelper
{
    private readonly Dictionary<Tensor, VulkanBuffer> _weightCache = new();
    private readonly VulkanMemoryAllocator _alloc;
    private readonly VulkanBackend _backend;

    public VulkanBuffer CopyToDevice(Tensor cpuTensor)
    {
        if (_weightCache.TryGetValue(cpuTensor, out var cached)) return cached;

        var dst = AllocateDevice(cpuTensor.ByteSize, cpuTensor.Alignment);
        UploadWithStaging(dst, cpuTensor);
        _weightCache[cpuTensor] = dst;
        return dst;
    }

    public void PreloadWeight(Tensor weight)  => _ = CopyToDevice(weight);
    public void PreloadWeights(IEnumerable<Tensor> weights) {
        // Group into one staging buffer per ~64 MB batch for efficient PCIe upload
        ...
    }

    public void FreeWeights(IEnumerable<Tensor> weights) {
        foreach (var w in weights)
            if (_weightCache.Remove(w, out var buf)) buf.Dispose();
    }
}
```

This mirrors `GpuTransferHelper.CopyToDevice` with reference equality semantics — works on CPU `Tensor`s that have been disposed (memory freed) because the dictionary lookup uses object identity. The cache hit path returns instantly with no H2D transfer, exactly like the CUDA path.

### Upload via staging buffer

```csharp
private unsafe void UploadWithStaging(VulkanBuffer dst, Tensor src)
{
    // Hot path: ReBAR direct write, no staging
    if (dst.Backing.MappedPointer != 0)
    {
        Buffer.MemoryCopy(src.DataPointer, (void*)(dst.Backing.MappedPointer + (nint)dst.Backing.Offset),
                          dst.Size, (long)src.ByteSize);
        if (!IsCoherent(dst)) FlushRange(dst);
        return;
    }

    // Staging path
    using var stage = AllocateStaging(src.ByteSize);
    Buffer.MemoryCopy(src.DataPointer, (void*)stage.MappedPointer, stage.Size, (long)src.ByteSize);
    if (!IsCoherent(stage)) FlushRange(stage);

    var cb = _backend.AcquireOneShotCommandBuffer();
    var copy = new VkBufferCopy { srcOffset = 0, dstOffset = 0, size = src.ByteSize };
    fixed (VkBufferCopy* p = &copy)
        VulkanApi.vkCmdCopyBuffer(cb, stage.Handle, dst.Handle, 1, (nint)p);

    EmitBarrier(cb, dst.Handle, srcStage: COPY, srcAccess: TRANSFER_WRITE,
                                  dstStage: COMPUTE_SHADER, dstAccess: SHADER_STORAGE_READ);
    _backend.SubmitOneShotAndWait(cb);
}
```

### Batched upload optimization

For `PreloadWeights(IEnumerable<Tensor>)` we coalesce multiple tensors into one staging buffer per 64 MB batch:

```
Layout in staging:
  [tensor_0 data | pad | tensor_1 data | pad | ... | tensor_K data]
Single vkCmdCopyBuffer with multiple VkBufferCopy regions (one per tensor) → multiple device buffers
```

Cuts setup overhead for SDXL (3-7k weight tensors) from minutes to seconds. Same trick the CUDA path uses for `PreloadWeights`.

---

## GPU-Resident Activation Cache (Vulkan port)

The CUDA backend's lazy-sync activation cache lives in `CacheActivation()` / `_gpuSyncCallback` / `_gpuDisposeCallback` — see [docs/Agents/AGENTS.md § GPU Activation Cache Rules](../Agents/AGENTS.md#gpu-activation-cache-rules) and [CUDA_PERFORMANCE.md](CUDA_PERFORMANCE.md). Port directly:

```csharp
internal static class VulkanActivationCache
{
    // Same fields as CUDA: _gpuSyncCallback, _gpuDisposeCallback on Tensor
    // Only the implementation changes — pointers are VulkanBuffer not ulong.

    public static void Cache(Tensor activation, VulkanBuffer buffer,
                             VkPipelineStageFlags2 producedAtStage,
                             VkAccessFlags2 producedAtAccess)
    {
        // Stage/Access are stored alongside the buffer for the next consumer's barrier
        // ...
    }
}
```

Per-op `cuStreamSynchronize` was removed in CUDA Phase 2; the Vulkan equivalent is **never call `vkQueueWaitIdle` per op**. Use the timeline-semaphore counter and `VkBufferMemoryBarrier2` chains to express dependencies (see [VULKAN_COMPUTE_API.md § Pipeline Barriers](VULKAN_COMPUTE_API.md#pipeline-barriers-synchronization-2)).

### Stream-ordered free (the `cuMemFreeAsync` equivalent)

Vulkan doesn't have stream-ordered memory operations natively. Approach: keep a **deferred free list** per timeline-semaphore tick:

```csharp
private readonly Dictionary<ulong /* timelineValue */, List<VulkanAllocation>> _deferredFrees = new();

public void FreeAsync(VulkanAllocation a, ulong currentTimelineValue)
{
    _deferredFrees.GetOrAdd(currentTimelineValue, _ => new()).Add(a);
}

public void Reclaim(ulong completedTimelineValue)
{
    foreach (var (k, list) in _deferredFrees)
        if (k <= completedTimelineValue) {
            foreach (var a in list) _alloc.Free(a);
            _deferredFrees.Remove(k);
        }
}
```

Call `Reclaim()` at pipeline-step boundaries (or when allocation pressure hits a watermark). Matches the CUDA `cuMemFreeAsync` deferred-cleanup semantics — see the OOM retry pattern below.

---

## OOM Handling

Mirror the CUDA pattern:

```csharp
public VulkanAllocation Allocate(ulong size, /* ... */)
{
    try
    {
        return AllocateInternal(...);
    }
    catch (VulkanException e) when (e.Result == VkResult.ERROR_OUT_OF_DEVICE_MEMORY)
    {
        // Flush deferred frees (analogue of cuStreamSynchronize then retry)
        _backend.WaitForCompletedTimelineValue(out var v);
        Reclaim(v);
        return AllocateInternal(...);  // one retry
    }
}
```

Surface as `OutOfVramException` to caller. Use `VK_EXT_memory_budget` (when available) to *predict* OOM:

```csharp
public ulong AvailableBytes(uint heapIndex)
{
    if (_hasMemoryBudget)
    {
        var budget = QueryMemoryBudget();   // VkPhysicalDeviceMemoryBudgetPropertiesEXT
        return budget.heapBudget[heapIndex] - budget.heapUsage[heapIndex];
    }
    return _memProps.memoryHeaps[heapIndex].size - UsedBytes(heapIndex);
}
```

---

## Descriptor Set Management

Descriptor pools have a fixed `maxSets` and a fixed budget per `VkDescriptorType`. Per-frame creation/destruction is cheap if the pool is reset rather than freed.

### Pool ring strategy

Two pools, alternating per "phase boundary" (e.g. UNet→VAE):

```csharp
public sealed class VulkanDescriptorManager
{
    private readonly ulong[] _pools = new ulong[2];     // VkDescriptorPool x2
    private int _activePool = 0;
    private uint _setsAllocated;

    public ulong AllocateSet(ulong layout)
    {
        if (_setsAllocated >= MAX_SETS_PER_POOL)
            FlipPool();
        // vkAllocateDescriptorSets(...)
    }

    public void FlipPool()
    {
        _activePool = 1 - _activePool;
        VulkanApi.vkResetDescriptorPool(_device, _pools[_activePool], 0);
        _setsAllocated = 0;
    }
}
```

**Sizing:**

```
poolSize = {
    { type = STORAGE_BUFFER,  count = 16 * MAX_SETS_PER_POOL },  // up to 16 SSBO bindings per set
    { type = UNIFORM_BUFFER,  count =  4 * MAX_SETS_PER_POOL },
};
maxSets = 4096;
```

This holds ~4096 dispatch worth. A SDXL UNet step does ~5000 dispatches → flip once per step.

### Push descriptors path

When `VK_KHR_push_descriptor` is supported, **skip the pool entirely**. Push descriptors are written directly into the command buffer via `vkCmdPushDescriptorSetKHR`. No allocation, no reset, simpler lifetime. We prefer this whenever the device offers it; pool path is the fallback for old or restricted drivers.

```csharp
public unsafe void BindBuffersPushDescriptor(
    nint cb, ulong layout, ReadOnlySpan<ulong> bufferHandles)
{
    Span<VkDescriptorBufferInfo> infos = stackalloc VkDescriptorBufferInfo[bufferHandles.Length];
    Span<VkWriteDescriptorSet>   wr    = stackalloc VkWriteDescriptorSet  [bufferHandles.Length];
    for (int i = 0; i < bufferHandles.Length; i++)
    {
        infos[i] = new() { buffer = bufferHandles[i], offset = 0, range = ulong.MaxValue };
        wr[i]    = new()
        {
            sType = VkStructureType.WriteDescriptorSet,
            dstBinding = (uint)i,
            descriptorCount = 1,
            descriptorType = (uint)VkDescriptorType.StorageBuffer,
            pBufferInfo = (nint)Unsafe.AsPointer(ref infos[i]),
        };
    }
    fixed (VkWriteDescriptorSet* p = wr)
        _pfnPushDescriptorSet(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, (uint)wr.Length, (nint)p);
}
```

### Layout dedup

Pre-build one layout per binding shape:

| Layout | Bindings | Used by |
|---|---|---|
| `L_2SSBO` | 2 storage buffers | unary ops (silu, gelu, scale, copy, transpose) |
| `L_3SSBO` | 3 storage buffers | binary ops (add, mul, broadcast_add, geglu) |
| `L_4SSBO` | 4 storage buffers | linear (x, w, b, y), groupnorm (x, w, b, y) |
| `L_5SSBO` | 5 storage buffers | groupnorm + bias (x, w, b, y, optional cache) |
| `L_3SSBO_QKV` | 3 storage buffers | sdpa attention (Q, K, V → out) |

Cap at ~10–12 distinct layouts. One pipeline layout per descriptor layout × push-constant-range pair.

---

## Command Pool & Command Buffer Lifecycle

```csharp
public sealed class VulkanCommandPool : IDisposable
{
    private readonly nint _device;
    private readonly ulong _pool;
    private readonly Stack<nint> _free  = new();   // VkCommandBuffer
    private readonly List<nint>  _inUse = new();

    public VulkanCommandPool(nint device, uint queueFamily)
    {
        _device = device;
        var ci = new VkCommandPoolCreateInfo
        {
            sType = VkStructureType.CommandPoolCreateInfo,
            queueFamilyIndex = queueFamily,
            flags = (uint)VkCommandPoolCreateFlags.RESET_COMMAND_BUFFER_BIT,
        };
        VulkanApi.vkCreateCommandPool(_device, in ci, 0, out _pool).ThrowOnError();
    }

    public nint Acquire()
    {
        if (_free.TryPop(out var cb)) { _inUse.Add(cb); return cb; }
        // Allocate a new one (in batches of 8 to amortize)
        var alloc = new VkCommandBufferAllocateInfo {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _pool, level = 0, commandBufferCount = 8 };
        Span<nint> handles = stackalloc nint[8];
        fixed (nint* p = handles)
            VulkanApi.vkAllocateCommandBuffers(_device, in alloc, (nint)p).ThrowOnError();
        for (int i = 0; i < 8; i++) _free.Push(handles[i]);
        return Acquire();
    }

    public void RecycleAllInUse()
    {
        foreach (var cb in _inUse) {
            VulkanApi.vkResetCommandBuffer(cb, 0).ThrowOnError();
            _free.Push(cb);
        }
        _inUse.Clear();
    }
}
```

**Reset, don't free.** `vkResetCommandBuffer` is O(1); `vkFreeCommandBuffers` returns the slot but is more expensive. Per-step we accumulate ~5000 command buffers; reset all at the next step boundary.

`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` lets us reset individual command buffers; without it we'd have to reset the entire pool at once (also fine for our access pattern, slightly faster).

---

## Synchronization Strategy

The CUDA backend runs everything on a single blocking stream (after Phase 2 fix). Vulkan equivalent: single timeline semaphore + single command-buffer-recording context.

```csharp
public sealed class VulkanCommandStream
{
    private ulong _timeline;        // VkSemaphore (timeline)
    private ulong _value;           // monotonic counter
    private nint  _currentCb;

    public void RecordOp(Action<nint> record)
    {
        record(_currentCb);
        EmitBarrierForLastOpOutput();
    }

    public ulong SubmitAndAdvance()
    {
        VulkanApi.vkEndCommandBuffer(_currentCb).ThrowOnError();

        var cbInfo  = new VkCommandBufferSubmitInfo { sType = ..., commandBuffer = _currentCb };
        var signal  = new VkSemaphoreSubmitInfo {
            sType = ..., semaphore = _timeline, value = ++_value,
            stageMask = (ulong)VkPipelineStageFlags2.AllCommands };
        var submit  = new VkSubmitInfo2 {
            sType = ...,
            commandBufferInfoCount = 1, pCommandBufferInfos  = ...,
            signalSemaphoreInfoCount = 1, pSignalSemaphoreInfos = ... };
        VulkanApi.vkQueueSubmit2(_queue, 1, in submit, /*fence*/ 0).ThrowOnError();

        // Recycle command buffer for next batch
        _currentCb = _commandPool.Acquire();
        BeginCommandBuffer(_currentCb);
        return _value;
    }

    public void WaitTimeline(ulong target, ulong timeoutNs = ulong.MaxValue)
    {
        var info = new VkSemaphoreWaitInfo {
            sType = ..., semaphoreCount = 1,
            pSemaphores = ..., pValues = ... };
        VulkanApi.vkWaitSemaphores(_device, in info, timeoutNs).ThrowOnError();
    }

    public ulong CurrentValue() {
        VulkanApi.vkGetSemaphoreCounterValue(_device, _timeline, out var v).ThrowOnError();
        return v;
    }
}
```

### Barrier emission rules

| Producer op | Consumer op | Barrier |
|---|---|---|
| Compute | Compute | `srcStage=COMPUTE, srcAccess=SHADER_STORAGE_WRITE; dstStage=COMPUTE, dstAccess=SHADER_STORAGE_READ` |
| H2D copy | Compute | `srcStage=COPY, srcAccess=TRANSFER_WRITE; dstStage=COMPUTE, dstAccess=SHADER_STORAGE_READ` |
| Compute | D2H copy | `srcStage=COMPUTE, srcAccess=SHADER_STORAGE_WRITE; dstStage=COPY, dstAccess=TRANSFER_READ` |
| Compute | Host map read | `srcStage=COMPUTE, srcAccess=SHADER_STORAGE_WRITE; dstStage=HOST, dstAccess=HOST_READ` |

Always scope the barrier to the *output buffer* of the last op (`VkBufferMemoryBarrier2`), not a global memory barrier. Only use `srcAccessMask = ALL_COMMANDS` at known sync points (pipeline stage transitions: UNet→VAE, end-of-step).

---

## Mapping to `IBackend.Sync()` and `FreeWeights()`

| `IBackend` method | CUDA impl | Vulkan impl |
|---|---|---|
| `Sync()` | `cuStreamSynchronize` | `vkWaitSemaphores(_timeline, currentValue)` then `Reclaim()` deferred frees |
| `FreeWeights(IEnumerable<Tensor>)` | drop from `Dictionary<Tensor, ulong>`, `cuMemFreeAsync` | drop from `Dictionary<Tensor, VulkanBuffer>`, return slab regions to allocator |
| `CopyTo(dst, src)` | `cuMemcpyDtoH` / `HtoD` | record `vkCmdCopyBuffer` + barrier or memcpy via mapped pointer |
| Allocate device tensor | `cuMemAlloc` | `_alloc.Allocate(...)` + `vkCreateBuffer` + `vkBindBufferMemory` |

These are the sync points. Between them, lazy execution streams a command buffer.

---

## VRAM Budget & Reporting

```csharp
public sealed class VulkanCapabilities
{
    public string DeviceName;
    public uint   VendorId, DeviceId;
    public ulong  TotalVramBytes;
    public ulong  AvailableVramBytes;
    public uint   SubgroupSize;
    public uint   MinSubgroupSize, MaxSubgroupSize;
    public bool   SupportsFp16;
    public bool   SupportsCoopMatrix;
    public bool   HasReBAR;          // DEVICE_LOCAL | HOST_VISIBLE memory type present
    public bool   HasMemoryBudget;
    public bool   HasPushDescriptor;
    public uint   MaxComputeSharedMem;
}
```

Surface to `BackendCapabilities` so model code can probe (e.g., FP16 path on/off).

`VK_EXT_memory_budget` lets us query `heapBudget[]` and `heapUsage[]` on every step boundary — present this in the OOM error message exactly like CUDA does (`OutOfVramException("requested 6.4 GB, available 2.1 GB")`).

---

## Testing Strategy

### Unit tests (`SharpInference.Vulkan.Tests`)

| Test | What it validates |
|---|---|
| `MemoryAllocatorTests.AllocFree_RoundTrip` | Block sub-alloc / free / re-alloc; alignment honored |
| `MemoryAllocatorTests.LargeAllocBeyondBlock` | Tensor > `SLAB_LARGE_BLOCK` triggers a dedicated allocation |
| `MemoryAllocatorTests.OutOfMemory_RetriesAfterReclaim` | Deferred-free reclamation triggered by alloc pressure |
| `DescriptorManagerTests.PoolFlip` | Reaching `MAX_SETS_PER_POOL` flips to next pool and resets |
| `CommandPoolTests.Recycle` | Reset path returns command buffers to free list |
| `WeightCacheTests.ReferenceEqualityHits` | Same `Tensor` → same `VulkanBuffer` (no re-upload) |
| `WeightCacheTests.AfterCpuDispose` | Cache hit works on disposed CPU tensors (parity with CUDA) |
| `StagingUploadTests.Coherent_NoFlushNeeded` | `HOST_COHERENT` → no `vkFlushMappedMemoryRanges` |
| `StagingUploadTests.NonCoherent_FlushSucceeds` | `HOST_VISIBLE` only → manual flush, GPU sees data |
| `StagingUploadTests.ReBAR_FastPath` | When `DEVICE_LOCAL | HOST_VISIBLE` exists, no staging buffer used |

### Integration tests

| Test | What it validates |
|---|---|
| `VulkanKernelTests.MatMul_Vs_Cpu` | Tiled GEMM matches CPU within 1e-3 |
| `VulkanKernelTests.GroupNorm_Vs_Cpu` | within 1e-3 |
| `VulkanKernelTests.Conv2D_Vs_Cpu` | within 1e-3 |
| `VulkanKernelTests.AllOps_Vs_Cuda` | every op matches CUDA result within 1e-3 |
| `VulkanPipelineTests.Sd15_512_Matches_Cuda` | full SD1.5 pipeline, same seed, SSIM > 0.99 vs CUDA reference |
| `VulkanPipelineTests.MemoryLeak_Multistep` | 100 step cycles → device alloc count returns to baseline |

Tests for `Vulkan.Tests` mark `[Fact(Skip = "no vulkan device")]` when no Vulkan device available; CI runs on a Mesa LLVMpipe software fallback for at least the unit tests.

---

## Mapping CUDA `Phase_3_Deviations.md` Issues Forward

The CUDA backend has 22+ documented bugs. Most are PTX-specific but several apply directly:

| Deviation | Vulkan impact |
|---|---|
| #12 64-bit indexing | Same in GLSL; use `int64_t` extension |
| #14–15 weight DataPointer access | Same — model code must never deref `weight.DataPointer`; Vulkan path goes through `IBackend` ops |
| #16 Last-dim split for GEGLU | Same; tested case with multi-row inputs |
| #17 In-place ops + activation cache | Same — clear sync/dispose callbacks before re-caching the same buffer |
| #18 `cuMemFreeAsync` deferred cleanup | Direct port: deferred-free list keyed by timeline value |
| #19 Non-blocking streams + sync `cuMemcpy` race | Vulkan: don't record copies on a different queue without barriers |

---

## Open Questions

- [ ] Whether to expose `VulkanBuffer` slicing (sub-views) or always one buffer per tensor. Initial: one buffer per tensor (simpler).
- [ ] Whether to use a B-tree or interval-tree for the free list inside large blocks (sorted list is fine up to ~10 K live regions).
- [ ] Whether `VK_KHR_buffer_device_address` is worth requiring — saves descriptor binding for scratch buffers but adds a feature dep.
- [ ] Whether to expose multiple queues (compute + transfer) for concurrent H2D + compute. Phase 7 optimization.
- [ ] Heuristic for when to coalesce small tensor allocations into one block vs separate blocks (defragmentation cost vs simplicity).

---

## Implementation Notes

1. **Two block sizes only** — 64 MB large pool, 8 MB small pool. More flexibility = more bugs, less win. (Originally 256 / 16 MB; reduced after near-VRAM-limit OOM on Flux Schnell — see deviation note above.)
2. **Don't allocate per tensor** — `vkAllocateMemory` count is the binding constraint. Sub-allocate from pre-allocated blocks.
3. **Persistent mapping** — host-visible blocks are mapped once at create time; never `vkMapMemory` per write.
4. **ReBAR fast path** — detect `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` memory type at startup; route weight uploads there when total weight size fits the heap.
5. **Coalesce weight uploads** — one staging buffer per ~64 MB batch, multi-region `vkCmdCopyBuffer`. Cuts SDXL preload from minutes to seconds.
6. **Timeline semaphores over fence + binary semaphore** — single 64-bit counter, simpler lifetime, matches CUDA stream-event mental model.
7. **Push descriptors when supported** — `VK_KHR_push_descriptor` removes pool churn; pool path is fallback only.
8. **Reset, don't free** — command buffers and descriptor pools are reset between phases, not freed.
9. **Deferred-free list keyed by timeline value** — analogue of `cuMemFreeAsync`; reclaim when GPU passes that value. **Tag with `currentTick + 1`, not `lastSubmittedTick`** — at the time `DeferredFree` is called from inside a recording op, the consuming dispatch hasn't been submitted yet, so its signal value is the next tick. See [PHASE_3_5_DEVIATIONS.md #6](../Checklists/PHASE_3_5_DEVIATIONS.md).
10. **Track cache-miss upload buffers explicitly** — `_transientBuffers` list in `VulkanGpuTransferHelper`, drained at op boundaries (not dispatch boundaries — multi-dispatch ops like SDPA share Q/K/V across heads, draining mid-op causes UAF). See [PHASE_3_5_DEVIATIONS.md #2, #5](../Checklists/PHASE_3_5_DEVIATIONS.md).
10. **OOM retry** — flush deferred frees + retry once before surfacing `OutOfVramException`.
11. **Memory budget** via `VK_EXT_memory_budget` for live VRAM telemetry; degrade gracefully when extension absent.
12. **Per-buffer barriers** — never global; respect the activation-cache producer/consumer dependency graph.
13. **Validation at every level** — unit tests for allocator; integration tests vs CPU and CUDA references.

---

## References

- [Vulkan Memory Allocator (AMD)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) — production C++ reference; many of our patterns derive from VMA's `VmaAllocator` design
- [VMA Documentation](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/) — the "Choosing memory type" and "Allocation algorithm" pages are the canonical guide
- [Adam Sawicki — Vulkan device memory cheatsheet](https://gpuopen.com/learn/vulkan-device-memory/)
- [Vulkan Guide — Memory Allocation](https://docs.vulkan.org/guide/latest/memory_allocation.html)
- [Vulkan Spec § Resources](https://docs.vulkan.org/spec/latest/chapters/resources.html)
- [Vulkan Spec § Memory Allocation](https://docs.vulkan.org/spec/latest/chapters/memory.html)
- [VK_EXT_memory_budget](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_memory_budget.html)
- [VK_KHR_push_descriptor](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_push_descriptor.html)
- [VK_KHR_timeline_semaphore](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_timeline_semaphore.html)
- [VK_KHR_synchronization2](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_synchronization2.html)
- [llama.cpp Vulkan allocator](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) — production reference for inference workloads
- [NCNN Vulkan allocator](https://github.com/Tencent/ncnn/blob/master/src/allocator.cpp) — mobile-focused, uses block sub-allocation
- [Granite Vulkan allocator](https://github.com/Themaister/Granite/blob/master/granite/vulkan/memory_allocator.cpp) — modern C++ Vulkan engine; clean reference
