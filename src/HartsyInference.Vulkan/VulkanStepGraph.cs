namespace HartsyInference.Vulkan;

/// <summary>Vulkan-native analog of <c>HartsyInference.Cuda.CudaGraph</c>: records a fixed op sequence into a single, persistent <c>VkCommandBuffer</c> once, then replays it with a plain resubmit — collapsing the per-op host dispatch/descriptor/submit overhead of a hand-written-kernel backend into one host call per step. See <c>docs/Checklists/ROADMAP.md</c>'s Phase 6e/7 entries for why this needed real design work (a pool-allocated descriptor set can be invalidated by a later pool reset while a replayed command buffer still references it — undefined behavior) rather than a direct port of CUDA Graph capture (which has no descriptor-binding concept at all, and whose allocation-node semantics have no Vulkan equivalent).</summary>
/// <remarks>
/// <para><b>Descriptor safety</b>: every dispatch recorded during capture uses <c>vkCmdPushDescriptorSet</c>
/// (via <see cref="VulkanDescriptorManager.PushSet"/>, always, regardless of the process's default eager-mode
/// descriptor strategy) against a push-descriptor-flavored pipeline (<c>VulkanKernelRegistry.Get(..., forCapture:
/// true)</c>). Push descriptors write the buffer bindings directly into the command stream — there is no
/// external <c>VkDescriptorSet</c> object a later pool flip could invalidate.</para>
/// <para><b>Buffer lifetime</b>: a captured command buffer bakes device buffer addresses at record time. Every
/// transient buffer the model's normal per-op allocate/dispose cycle touches during capture is redirected by
/// <see cref="VulkanGpuTransferHelper.CapturingStepGraph"/> to a retain list instead of being freed, and stays
/// alive for the graph's entire lifetime (freed in bulk by <see cref="VulkanGpuTransferHelper.ReleaseStepGraphRetained"/>
/// when the graph is reset). There is no bump-pointer arena or address-remapping — buffers simply aren't freed
/// until the graph itself is torn down, which is both simpler and correct: the exact same allocation sequence
/// happens on every real forward pass, so "don't free between replays" is sufficient.</para>
/// <para><b>Command buffer flags</b>: recorded with plain (no <c>ONE_TIME_SUBMIT</c>, no <c>SIMULTANEOUS_USE</c>)
/// begin flags, replayed via a fence-waited <c>vkQueueSubmit2</c> per launch. This is correct for a strictly
/// serial step loop (one launch fully completes before the next begins) but is a deliberate v1 floor, not an
/// oversight: it gives the host-overhead win without submission/execution overlap across steps. Pipelining
/// replays (double-buffered command buffers + a semaphore instead of a blocking fence wait) is a follow-up if
/// profiling shows fence-wait latency is itself a meaningful fraction of the per-step cost.</para>
/// <para><b>No cross-graph reuse of the fixed pool/pipeline layouts</b>: the push-descriptor pipeline variant a
/// capture uses is cached by <see cref="VulkanKernelRegistry"/> exactly like the normal-mode one, keyed by
/// <c>KernelKey.ForCapture</c> — no extra bookkeeping needed here.</para>
/// </remarks>
public sealed unsafe class VulkanStepGraph : IDisposable
{
    private readonly nint _device;
    private readonly nint _queue;
    private readonly uint _queueFamily;

    private ulong _commandPool;
    private ulong _fence;
    private nint _commandBuffer;

    private bool _recording;
    private bool _ready;

    /// <summary>True once a capture has been ended and launched at least once — the graph can be replayed via <see cref="Launch"/>.</summary>
    public bool IsReady => _ready && !_recording;

    public VulkanStepGraph(nint device, nint queue, uint queueFamilyIndex)
    {
        _device = device;
        _queue = queue;
        _queueFamily = queueFamilyIndex;
    }

    private void EnsureResources()
    {
        if (_commandPool != 0) return;

        VkCommandPoolCreateInfo poolCi = new()
        {
            sType = VkStructureType.CommandPoolCreateInfo,
            // No RESET_COMMAND_BUFFER flag: this pool ever holds exactly one long-lived command buffer that
            // we reset explicitly (vkResetCommandBuffer) only when re-capturing, not per-submit like the
            // normal stream's recycled pool.
            flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
            queueFamilyIndex = _queueFamily,
        };
        VulkanApi.vkCreateCommandPool(_device, in poolCi, 0, out _commandPool).ThrowOnError("vkCreateCommandPool (step-graph)");

        VkCommandBufferAllocateInfo ai = new()
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        nint cb;
        VulkanApi.vkAllocateCommandBuffers(_device, in ai, (nint)(&cb)).ThrowOnError("vkAllocateCommandBuffers (step-graph)");
        _commandBuffer = cb;

        VkFenceCreateInfo fenceCi = new() { sType = VkStructureType.FenceCreateInfo, flags = 0 };
        VulkanApi.vkCreateFence(_device, in fenceCi, 0, out _fence).ThrowOnError("vkCreateFence (step-graph)");
    }

    /// <summary>Begins recording. Subsequent dispatches must be recorded onto <see cref="RecordingBuffer"/>.</summary>
    public void BeginCapture()
    {
        EnsureResources();
        if (_ready)
        {
            // Re-capture: the previous graph's command buffer must be reset before re-recording.
            VulkanApi.vkResetCommandBuffer(_commandBuffer, 0).ThrowOnError("vkResetCommandBuffer (step-graph recapture)");
            _ready = false;
        }
        VkCommandBufferBeginInfo bi = new()
        {
            sType = VkStructureType.CommandBufferBeginInfo,
            // Plain flags: no ONE_TIME_SUBMIT (this buffer IS resubmitted many times) and no SIMULTANEOUS_USE
            // (replays are strictly serial — Launch fence-waits before the next Launch/EndAndLaunch call can
            // begin), which is legal without that bit per the Vulkan spec.
            flags = VkCommandBufferUsageFlags.None,
        };
        VulkanApi.vkBeginCommandBuffer(_commandBuffer, in bi).ThrowOnError("vkBeginCommandBuffer (step-graph)");
        _recording = true;
    }

    /// <summary>The command buffer to record dispatches onto while <see cref="BeginCapture"/>...<see cref="EndCaptureAndLaunch"/> is active.</summary>
    public nint RecordingBuffer => _commandBuffer;

    /// <summary>Ends recording and launches the graph once (capture records without executing — this call runs the step).</summary>
    public void EndCaptureAndLaunch()
    {
        if (!_recording) throw new InvalidOperationException("VulkanStepGraph.EndCaptureAndLaunch called without a matching BeginCapture.");
        VulkanApi.vkEndCommandBuffer(_commandBuffer).ThrowOnError("vkEndCommandBuffer (step-graph)");
        _recording = false;
        _ready = true;
        SubmitAndWait();
    }

    /// <summary>Replays the captured command buffer (one launch call).</summary>
    public void Launch()
    {
        if (!IsReady) throw new InvalidOperationException("VulkanStepGraph.Launch called with no captured graph ready.");
        SubmitAndWait();
    }

    private void SubmitAndWait()
    {
        ulong fence = _fence;
        VulkanApi.vkResetFences(_device, 1, (nint)(&fence)).ThrowOnError("vkResetFences (step-graph)");

        nint cb = _commandBuffer;
        VkCommandBufferSubmitInfo cbInfo = new() { sType = VkStructureType.CommandBufferSubmitInfo, commandBuffer = cb };
        VkSubmitInfo2 si = new()
        {
            sType = VkStructureType.SubmitInfo2,
            commandBufferInfoCount = 1,
            pCommandBufferInfos = (nint)(&cbInfo),
        };
        VulkanApi.vkQueueSubmit2(_queue, 1, (nint)(&si), fence).ThrowOnError("vkQueueSubmit2 (step-graph)");
        VulkanApi.vkWaitForFences(_device, 1, (nint)(&fence), 1, ulong.MaxValue).ThrowOnError("vkWaitForFences (step-graph)");
    }

    /// <summary>Aborts an in-flight capture (if any) and drops the captured graph. Does NOT free retained buffers — the caller (<c>VulkanBackend.StepGraphReset</c>) does that via <see cref="VulkanGpuTransferHelper.ReleaseStepGraphRetained"/> after this returns.</summary>
    public void Reset()
    {
        if (_recording)
        {
            // A partially-recorded command buffer must still be ended before it can be reset/reused —
            // vkResetCommandBuffer on a buffer stuck mid-recording is undefined. The recorded-but-unexecuted
            // commands are simply discarded (never submitted).
            try { VulkanApi.vkEndCommandBuffer(_commandBuffer); } catch { /* best-effort on an already-invalid capture */ }
            _recording = false;
        }
        _ready = false;
    }

    public void Dispose()
    {
        Reset();
        if (_fence != 0) { VulkanApi.vkDestroyFence(_device, _fence, 0); _fence = 0; }
        if (_commandPool != 0) { VulkanApi.vkDestroyCommandPool(_device, _commandPool, 0); _commandPool = 0; }
        GC.SuppressFinalize(this);
    }
}
