using System.Runtime.InteropServices;

namespace SharpInference.Vulkan;

/// <summary>One logical compute "stream" backed by a single timeline semaphore + a recycled command-buffer chain on the device's compute queue. Replaces CudaStream + per-op cuStreamSynchronize. Batches recorded ops into one command buffer, submitting once per <see cref="Sync"/> / <see cref="SubmitAndAdvance"/>. Buffers freed during recording stay live until the timeline counter passes the recording's tick — analogous to CUDA's <c>cuMemFreeAsync</c>.</summary>
public sealed class VulkanCommandStream : IDisposable
{
    private readonly nint _device;
    private readonly nint _queue;
    private readonly uint _queueFamily;

    private ulong _commandPool;
    private nint _activeCmd;
    private bool _activeCmdRecording;

    private ulong _timeline;
    private ulong _value;       // monotonic; submitted ops signal value+1
    private ulong _lastSubmitted;

    // free list keyed by tick at which it was scheduled — released once timeline >= tick
    private readonly SortedDictionary<ulong, List<VulkanBuffer>> _deferredFrees = new();
    // command buffers recycled at next reset
    private readonly Stack<nint> _freeCmds = new();
    private readonly List<nint> _allocatedCmds = new();

    public ulong CurrentTick => _value;
    public ulong LastSubmittedTick => _lastSubmitted;

    public VulkanCommandStream(nint device, nint queue, uint queueFamilyIndex)
    {
        _device = device;
        _queue = queue;
        _queueFamily = queueFamilyIndex;

        VkCommandPoolCreateInfo poolCi = new()
        {
            sType = VkStructureType.CommandPoolCreateInfo,
            flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
            queueFamilyIndex = queueFamilyIndex,
        };
        VulkanApi.vkCreateCommandPool(device, in poolCi, 0, out _commandPool)
            .ThrowOnError("vkCreateCommandPool");

        unsafe
        {
            VkSemaphoreTypeCreateInfo typeCi = new()
            {
                sType = VkStructureType.SemaphoreTypeCreateInfo,
                semaphoreType = VkSemaphoreType.Timeline,
                initialValue = 0,
            };
            VkSemaphoreCreateInfo semCi = new()
            {
                sType = VkStructureType.SemaphoreCreateInfo,
                pNext = (nint)(&typeCi),
            };
            VulkanApi.vkCreateSemaphore(device, in semCi, 0, out _timeline)
                .ThrowOnError("vkCreateSemaphore (timeline)");
        }
    }

    /// <summary>Returns the active command buffer (creating + beginning one if needed). Use for direct-record ops.</summary>
    public nint AcquireRecording()
    {
        if (_activeCmdRecording) return _activeCmd;

        if (!_freeCmds.TryPop(out nint cb))
        {
            // Allocate a batch of 8 to amortize
            unsafe
            {
                VkCommandBufferAllocateInfo ai = new()
                {
                    sType = VkStructureType.CommandBufferAllocateInfo,
                    commandPool = _commandPool,
                    level = VkCommandBufferLevel.Primary,
                    commandBufferCount = 8,
                };
                Span<nint> handles = stackalloc nint[8];
                fixed (nint* p = handles)
                {
                    VulkanApi.vkAllocateCommandBuffers(_device, in ai, (nint)p)
                        .ThrowOnError("vkAllocateCommandBuffers");
                }
                for (int i = 0; i < 8; i++) { _freeCmds.Push(handles[i]); _allocatedCmds.Add(handles[i]); }
            }
            cb = _freeCmds.Pop();
        }

        VkCommandBufferBeginInfo bi = new()
        {
            sType = VkStructureType.CommandBufferBeginInfo,
            flags = VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        VulkanApi.vkBeginCommandBuffer(cb, in bi).ThrowOnError("vkBeginCommandBuffer");
        _activeCmd = cb;
        _activeCmdRecording = true;
        return cb;
    }

    /// <summary>Records a buffer-scoped sync2 barrier on the active command buffer.</summary>
    public unsafe void RecordBufferBarrier(
        ulong buffer, ulong offset, ulong size,
        ulong srcStage, ulong srcAccess,
        ulong dstStage, ulong dstAccess)
    {
        nint cb = AcquireRecording();
        VkBufferMemoryBarrier2 b = new()
        {
            sType = VkStructureType.BufferMemoryBarrier2,
            srcStageMask = srcStage, srcAccessMask = srcAccess,
            dstStageMask = dstStage, dstAccessMask = dstAccess,
            srcQueueFamilyIndex = VkConstants.QueueFamilyIgnored,
            dstQueueFamilyIndex = VkConstants.QueueFamilyIgnored,
            buffer = buffer, offset = offset, size = size,
        };
        VkDependencyInfo dep = new()
        {
            sType = VkStructureType.DependencyInfo,
            bufferMemoryBarrierCount = 1,
            pBufferMemoryBarriers = (nint)(&b),
        };
        VulkanApi.vkCmdPipelineBarrier2(cb, in dep);
    }

    /// <summary>Records a global compute->compute memory barrier (covers all buffers). Cheap fallback when per-buffer scope is unwieldy.</summary>
    public unsafe void RecordGlobalComputeBarrier()
    {
        nint cb = AcquireRecording();
        VkMemoryBarrier2 mb = new()
        {
            sType = VkStructureType.MemoryBarrier2,
            srcStageMask = VkPipelineStageFlags2.ComputeShader,
            srcAccessMask = VkAccessFlags2.ShaderStorageWrite,
            dstStageMask = VkPipelineStageFlags2.ComputeShader,
            dstAccessMask = VkAccessFlags2.ShaderStorageRead,
        };
        VkDependencyInfo dep = new()
        {
            sType = VkStructureType.DependencyInfo,
            memoryBarrierCount = 1,
            pMemoryBarriers = (nint)(&mb),
        };
        VulkanApi.vkCmdPipelineBarrier2(cb, in dep);
    }

    /// <summary>Records a buffer→buffer copy followed by a barrier transitioning to the requested consumer stage/access.</summary>
    public unsafe void RecordCopyAndBarrier(ulong src, ulong dst, ulong size, ulong postStage, ulong postAccess)
    {
        nint cb = AcquireRecording();
        VkBufferCopy region = new() { srcOffset = 0, dstOffset = 0, size = size };
        VulkanApi.vkCmdCopyBuffer(cb, src, dst, 1, (nint)(&region));

        VkBufferMemoryBarrier2 b = new()
        {
            sType = VkStructureType.BufferMemoryBarrier2,
            srcStageMask = VkPipelineStageFlags2.Copy,
            srcAccessMask = VkAccessFlags2.TransferWrite,
            dstStageMask = postStage,
            dstAccessMask = postAccess,
            srcQueueFamilyIndex = VkConstants.QueueFamilyIgnored,
            dstQueueFamilyIndex = VkConstants.QueueFamilyIgnored,
            buffer = dst, offset = 0, size = size,
        };
        VkDependencyInfo dep = new()
        {
            sType = VkStructureType.DependencyInfo,
            bufferMemoryBarrierCount = 1,
            pBufferMemoryBarriers = (nint)(&b),
        };
        VulkanApi.vkCmdPipelineBarrier2(cb, in dep);
    }

    /// <summary>Ends the active command buffer and submits it, signalling timeline += 1. Returns the new tick value.</summary>
    public unsafe ulong SubmitAndAdvance()
    {
        if (!_activeCmdRecording) return _value;

        nint cb = _activeCmd;
        VulkanApi.vkEndCommandBuffer(cb).ThrowOnError("vkEndCommandBuffer");
        _activeCmdRecording = false;

        ulong target = ++_value;

        VkCommandBufferSubmitInfo cbInfo = new()
        {
            sType = VkStructureType.CommandBufferSubmitInfo,
            commandBuffer = cb,
        };
        VkSemaphoreSubmitInfo sigInfo = new()
        {
            sType = VkStructureType.SemaphoreSubmitInfo,
            semaphore = _timeline,
            value = target,
            stageMask = VkPipelineStageFlags2.AllCommands,
        };
        VkSubmitInfo2 si = new()
        {
            sType = VkStructureType.SubmitInfo2,
            commandBufferInfoCount = 1,
            pCommandBufferInfos = (nint)(&cbInfo),
            signalSemaphoreInfoCount = 1,
            pSignalSemaphoreInfos = (nint)(&sigInfo),
        };
        VulkanApi.vkQueueSubmit2(_queue, 1, (nint)(&si), 0)
            .ThrowOnError("vkQueueSubmit2");

        _lastSubmitted = target;

        // Opportunistic reclaim — non-blocking peek at the timeline. Whatever the GPU has
        // already finished, free now so the slab pool doesn't hold deferred buffers across
        // many submits. Without this, large models exhaust device heaps before any explicit
        // wait happens.
        VulkanApi.vkGetSemaphoreCounterValue(_device, _timeline, out ulong nowTick);
        if (nowTick > 0) ReclaimUpTo(nowTick);

        // CB will be recycled once we know its tick has completed.
        // Conservative: don't reuse until WaitTimeline(target).
        // We recycle by tying CB recycling to the deferred-free list: when ReclaimUpTo runs, if cb's tick has passed, push back on _freeCmds.
        ScheduleCmdRecycle(cb, target);

        return target;
    }

    private readonly Dictionary<ulong, List<nint>> _cmdRecycle = new();
    private void ScheduleCmdRecycle(nint cb, ulong tick)
    {
        if (!_cmdRecycle.TryGetValue(tick, out List<nint>? list))
        { list = new(); _cmdRecycle[tick] = list; }
        list.Add(cb);
    }

    /// <summary>Returns true when the GPU has completed up to <paramref name="tick"/>.</summary>
    public bool TimelineReached(ulong tick)
    {
        VulkanApi.vkGetSemaphoreCounterValue(_device, _timeline, out ulong v)
            .ThrowOnError("vkGetSemaphoreCounterValue");
        return v >= tick;
    }

    /// <summary>Blocks the calling thread until the GPU reaches the given tick. Free deferred buffers afterwards.</summary>
    public unsafe void WaitTimeline(ulong tick, ulong timeoutNs = ulong.MaxValue)
    {
        if (tick == 0 || TimelineReached(tick)) { ReclaimUpTo(GetTimelineNow()); return; }
        ulong t = tick;
        ulong sem = _timeline;
        VkSemaphoreWaitInfo wi = new()
        {
            sType = VkStructureType.SemaphoreWaitInfo,
            semaphoreCount = 1,
            pSemaphores = (nint)(&sem),
            pValues = (nint)(&t),
        };
        VulkanApi.vkWaitSemaphores(_device, in wi, timeoutNs).ThrowOnError("vkWaitSemaphores");
        ReclaimUpTo(GetTimelineNow());
    }

    /// <summary>Submits any pending recording, then blocks until the GPU is fully idle on this stream's timeline.</summary>
    public void WaitIdleHost()
    {
        ulong tick = SubmitAndAdvance();
        WaitTimeline(tick);
    }

    /// <summary>Schedules a buffer for free once the next-to-be-submitted tick completes. Using
    /// _value+1 (the tick that will be assigned to the active command buffer when it next submits)
    /// guarantees any in-flight commands referencing this buffer are done before destroy. Tagging
    /// at _lastSubmitted (already-completed) would risk use-after-free for buffers used by the
    /// currently-recording command buffer.</summary>
    public void DeferredFree(VulkanBuffer buffer)
    {
        ulong tick = _value + 1;
        DeferredFreeAt(tick, buffer);
    }

    /// <summary>Schedules a buffer for free once the given tick completes.</summary>
    public void DeferredFreeAt(ulong tick, VulkanBuffer buffer)
    {
        if (!_deferredFrees.TryGetValue(tick, out List<VulkanBuffer>? list))
        { list = new(); _deferredFrees[tick] = list; }
        list.Add(buffer);
    }

    /// <summary>Releases all deferred-free buffers whose tick has been reached. Also recycles command buffers.</summary>
    public void ReclaimUpTo(ulong completedTick)
    {
        List<ulong> done = new();
        foreach ((ulong tick, List<VulkanBuffer> bufs) in _deferredFrees)
        {
            if (tick > completedTick) break;
            foreach (VulkanBuffer b in bufs) b.Dispose();
            done.Add(tick);
        }
        foreach (ulong t in done) _deferredFrees.Remove(t);

        List<ulong> doneCb = new();
        foreach ((ulong tick, List<nint> cbs) in _cmdRecycle)
        {
            if (tick > completedTick) break;
            foreach (nint cb in cbs)
            {
                VulkanApi.vkResetCommandBuffer(cb, 0).ThrowOnError("vkResetCommandBuffer");
                _freeCmds.Push(cb);
            }
            doneCb.Add(tick);
        }
        foreach (ulong t in doneCb) _cmdRecycle.Remove(t);
    }

    private ulong GetTimelineNow()
    {
        VulkanApi.vkGetSemaphoreCounterValue(_device, _timeline, out ulong v)
            .ThrowOnError("vkGetSemaphoreCounterValue");
        return v;
    }

    public void Dispose()
    {
        if (_activeCmdRecording) try { VulkanApi.vkEndCommandBuffer(_activeCmd); } catch { /* ignore */ }
        try { VulkanApi.vkQueueWaitIdle(_queue); } catch { /* ignore */ }

        // Drain deferred frees
        foreach ((_, List<VulkanBuffer> bufs) in _deferredFrees)
            foreach (VulkanBuffer b in bufs) b.Dispose();
        _deferredFrees.Clear();
        _cmdRecycle.Clear();

        if (_allocatedCmds.Count > 0)
        {
            unsafe
            {
                fixed (nint* p = CollectionsMarshal.AsSpan(_allocatedCmds))
                    VulkanApi.vkFreeCommandBuffers(_device, _commandPool, (uint)_allocatedCmds.Count, (nint)p);
            }
            _allocatedCmds.Clear();
            _freeCmds.Clear();
        }

        if (_timeline != 0) { VulkanApi.vkDestroySemaphore(_device, _timeline, 0); _timeline = 0; }
        if (_commandPool != 0) { VulkanApi.vkDestroyCommandPool(_device, _commandPool, 0); _commandPool = 0; }
        GC.SuppressFinalize(this);
    }

    ~VulkanCommandStream()
    {
        if (_timeline != 0) VulkanApi.vkDestroySemaphore(_device, _timeline, 0);
        if (_commandPool != 0) VulkanApi.vkDestroyCommandPool(_device, _commandPool, 0);
    }
}
