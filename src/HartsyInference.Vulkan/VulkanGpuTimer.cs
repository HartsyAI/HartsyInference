namespace HartsyInference.Vulkan;

/// <summary>GPU-side timestamp instrumentation via a 2-slot <c>VkQueryPool</c> (TIMESTAMP type) — the portable, always-available Vulkan mechanism for measuring exact GPU execution time, used here because neither Nsight Compute (CUDA-only — confirmed empirically, no Vulkan/graphics API support in its `--help` output) nor Nsight Graphics (not apt-installable, needs a separate NVIDIA download) were available on this box (2026-07-31). Diagnostic-only: not used on any production dispatch path. See docs/Checklists/TROUBLESHOOTING.md for what this was built to answer (is the coopmat gap actually GPU execution time, or is host/submission overhead the confound in wall-clock Stopwatch measurements).</summary>
internal sealed class VulkanGpuTimer : IDisposable
{
    private readonly nint _device;
    private readonly ulong _queryPool;
    private readonly float _periodNs;
    private bool _disposed;

    internal unsafe VulkanGpuTimer(nint device, float timestampPeriodNs)
    {
        _device = device;
        _periodNs = timestampPeriodNs;
        VkQueryPoolCreateInfo ci = new()
        {
            sType = VkStructureType.QueryPoolCreateInfo,
            queryType = 2,   // VK_QUERY_TYPE_TIMESTAMP
            queryCount = 2,  // slot 0 = region start, slot 1 = region end
        };
        VulkanApi.vkCreateQueryPool(device, in ci, 0, out _queryPool).ThrowOnError("vkCreateQueryPool");
    }

    /// <summary>Must be recorded before <see cref="RecordStart"/> on the SAME command buffer (a query must be reset before its first use in a submission, or vkCmdWriteTimestamp2 is invalid usage).</summary>
    internal void RecordReset(nint cb) => VulkanApi.vkCmdResetQueryPool(cb, _queryPool, 0, 2);

    /// <summary>Writes the region-start timestamp at the ALL_COMMANDS stage — the GPU only writes this once every command recorded before it in submission order has fully completed, giving an accurate lower bound for "everything after this point on the GPU timeline".</summary>
    internal void RecordStart(nint cb) => VulkanApi.vkCmdWriteTimestamp2(cb, VkPipelineStageFlags2.AllCommands, _queryPool, 0);

    /// <summary>Writes the region-end timestamp. May be recorded on a DIFFERENT command buffer than <see cref="RecordStart"/> (e.g. if an auto-flush happened in between) — this is still valid, since query pools are device-level resources and in-order queue submission preserves the ordering that makes the elapsed delta meaningful.</summary>
    internal void RecordEnd(nint cb) => VulkanApi.vkCmdWriteTimestamp2(cb, VkPipelineStageFlags2.AllCommands, _queryPool, 1);

    /// <summary>Blocks until both timestamps are available (VK_QUERY_RESULT_WAIT_BIT) and returns the elapsed GPU time in milliseconds. Caller must have already ensured RecordEnd's command buffer was actually submitted (e.g. via a host wait) — this does NOT submit anything itself.</summary>
    internal unsafe double ReadElapsedMs()
    {
        ulong* results = stackalloc ulong[2];
        const uint QUERY_RESULT_64_BIT = 1;
        const uint QUERY_RESULT_WAIT_BIT = 2;
        VulkanApi.vkGetQueryPoolResults(_device, _queryPool, 0, 2, (nuint)(2 * sizeof(ulong)), (nint)results,
            sizeof(ulong), QUERY_RESULT_64_BIT | QUERY_RESULT_WAIT_BIT).ThrowOnError("vkGetQueryPoolResults");
        ulong deltaTicks = results[1] - results[0];
        return deltaTicks * _periodNs / 1_000_000.0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        VulkanApi.vkDestroyQueryPool(_device, _queryPool, 0);
    }
}
