namespace HartsyInference.Vulkan;

/// <summary>GPU-side time per op name: a timestamp pair around every dispatch, resolved in batches.</summary>
/// <remarks>Host wall time charges a stall to whichever op happened to be waiting when the queue backed up; these
/// timestamps charge each dispatch its own execution. Diagnostic-only (<c>diagnostics.vkProfileGpu</c>).</remarks>
internal sealed class VulkanGpuOpTimer : IDisposable
{
    private const int Capacity = 16384;
    private readonly nint _device;
    private readonly ulong _queryPool;
    private readonly double _periodNs;
    private readonly string[] _ops = new string[Capacity];
    private readonly Dictionary<string, (double Ms, long Dispatches)> _totals = new(StringComparer.Ordinal);
    private int _used;
    private bool _needsReset = true;
    private bool _disposed;

    internal VulkanGpuOpTimer(nint device, float timestampPeriodNs)
    {
        _device = device;
        _periodNs = timestampPeriodNs;
        VkQueryPoolCreateInfo ci = new()
        {
            sType = VkStructureType.QueryPoolCreateInfo,
            queryType = 2, // VK_QUERY_TYPE_TIMESTAMP
            queryCount = Capacity * 2,
        };
        VulkanApi.vkCreateQueryPool(device, in ci, 0, out _queryPool).ThrowOnError("vkCreateQueryPool");
    }

    /// <summary>Whether the pool must be resolved (after a drain) before the next dispatch can be timed.</summary>
    internal bool Full => _used >= Capacity;

    /// <summary>Records the start timestamp for a dispatch belonging to <paramref name="opName"/>.</summary>
    internal void Begin(nint cb, string opName)
    {
        if (_needsReset)
        {
            VulkanApi.vkCmdResetQueryPool(cb, _queryPool, 0, Capacity * 2);
            _needsReset = false;
        }
        _ops[_used] = opName;
        VulkanApi.vkCmdWriteTimestamp2(cb, VkPipelineStageFlags2.AllCommands, _queryPool, (uint)(_used * 2));
    }

    /// <summary>Records the end timestamp for the dispatch opened by <see cref="Begin"/>.</summary>
    internal void End(nint cb)
    {
        VulkanApi.vkCmdWriteTimestamp2(cb, VkPipelineStageFlags2.AllCommands, _queryPool, (uint)((_used * 2) + 1));
        _used++;
    }

    /// <summary>Folds every recorded pair into the per-op totals. The caller must have drained the queue first.</summary>
    internal unsafe void Resolve()
    {
        if (_used == 0)
        {
            return;
        }
        const uint Result64 = 1;
        const uint ResultWait = 2;
        ulong[] results = new ulong[_used * 2];
        fixed (ulong* p = results)
        {
            VulkanApi.vkGetQueryPoolResults(_device, _queryPool, 0, (uint)(_used * 2), (nuint)(results.Length * sizeof(ulong)),
                (nint)p, sizeof(ulong), Result64 | ResultWait).ThrowOnError("vkGetQueryPoolResults");
        }
        for (int k = 0; k < _used; k++)
        {
            double ms = (results[(2 * k) + 1] - results[2 * k]) * _periodNs / 1_000_000.0;
            (double total, long count) = _totals.TryGetValue(_ops[k], out (double, long) v) ? v : (0.0, 0L);
            _totals[_ops[k]] = (total + ms, count + 1);
        }
        _used = 0;
        _needsReset = true;
    }

    /// <summary>Writes the per-op GPU time table, largest first.</summary>
    internal void Dump(TextWriter writer)
    {
        double all = _totals.Values.Sum(v => v.Ms);
        writer.WriteLine();
        writer.WriteLine("=== VulkanBackend GPU time per op (timestamp queries) ===");
        writer.WriteLine($"  Total GPU: {all:F1}ms over {_totals.Values.Sum(v => v.Dispatches):N0} dispatches");
        writer.WriteLine($"{"Op",-36} {"Dispatches",11} {"GPU(ms)",11} {"Avg(ms)",9} {"%",6}");
        writer.WriteLine(new string('-', 78));
        foreach (KeyValuePair<string, (double Ms, long Dispatches)> kvp in _totals.OrderByDescending(p => p.Value.Ms).Take(30))
        {
            double pct = all > 0 ? 100.0 * kvp.Value.Ms / all : 0.0;
            writer.WriteLine($"{kvp.Key,-36} {kvp.Value.Dispatches,11:N0} {kvp.Value.Ms,11:F1} {kvp.Value.Ms / kvp.Value.Dispatches,9:F3} {pct,5:F1}%");
        }
        writer.WriteLine();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        VulkanApi.vkDestroyQueryPool(_device, _queryPool, 0);
    }
}
