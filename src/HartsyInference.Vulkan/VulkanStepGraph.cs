namespace HartsyInference.Vulkan;

/// <summary>Vulkan analog of <c>CudaGraph</c>: records a fixed op sequence into one persistent
/// <c>VkCommandBuffer</c> and replays it with a resubmit, collapsing per-op host overhead into one call per step.</summary>
/// <remarks>Capture uses push descriptors against a push-descriptor pipeline variant, because a pool-allocated
/// set can be invalidated by a later pool reset while a replayed buffer still references it. A captured buffer
/// also bakes device addresses, so transients allocated during capture are retained, not freed, until the graph
/// is reset. Replay is a fence-waited submit per launch, correct only for a strictly serial step loop.</remarks>
public sealed unsafe class VulkanStepGraph(nint device, nint queue, uint queueFamilyIndex) : IDisposable
{
    private readonly nint _device = device;
    private readonly nint _queue = queue;
    private readonly uint _queueFamily = queueFamilyIndex;

    private ulong _commandPool;
    private ulong _fence;
    private nint _commandBuffer;

    private bool _recording;
    private bool _ready;

    /// <summary>True once a capture has been ended and launched at least once — the graph can be replayed via <see cref="Launch"/>.</summary>
    public bool IsReady => _ready && !_recording;

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
