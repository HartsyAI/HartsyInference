using HartsyInference.LLM.Kimi;

namespace HartsyInference.Engine.Services;

/// <summary>The placements the Kimi K3.1 Text route refuses by name: a request that names fewer ranks than the sharding needs, a rank count the
/// decoder does not split across, and a device listed twice. A multi-device request that passes these checks is sharded over its devices and
/// never falls back to one. Image refusals are in <c>RunKimiVision</c> and the vision input, before the vision tower loads.</summary>
internal static class KimiK31Refusals
{
    /// <summary>Accepts a single-device request and a multi-device request the decoder runs; refuses any other by name.</summary>
    /// <param name="deviceKey">The device the request names, possibly a composite such as <c>cuda:0+cuda:1</c>.</param>
    /// <param name="rankDevices">The devices the request shards over; empty when it names one device.</param>
    /// <param name="tensorParallelDegree">The engine's tensor-parallel degree.</param>
    /// <param name="shardRatios">The engine's uneven shard ratios, or null for an even split.</param>
    /// <param name="config">The checkpoint's decoder config.</param>
    /// <exception cref="NotSupportedException">The placement is one the route does not run.</exception>
    internal static void RequireRunnablePlacement(string deviceKey, IReadOnlyList<string> rankDevices, int tensorParallelDegree,
        IReadOnlyList<float>? shardRatios, KimiK31Config config)
    {
        int ranks = rankDevices.Count;
        if (ranks == 0 && tensorParallelDegree <= 1) return;
        if (ranks < 2)
        {
            string named = ranks == 0 ? "names one device" : $"names {ranks} device(s)";
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement needs at least 2 ranks: the request for '{deviceKey}' {named} "
                + $"and the engine's tensor-parallel degree is {tensorParallelDegree}. It is refused rather than loaded on one device.");
        }
        if (tensorParallelDegree > 1 && tensorParallelDegree != ranks)
        {
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement: the engine's tensor-parallel degree is {tensorParallelDegree}, "
                + $"but the request for '{deviceKey}' names {ranks} devices.");
        }
        if (rankDevices.Select(DeviceIdentity).Distinct(StringComparer.Ordinal).Count() != ranks)
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement: the request for '{deviceKey}' lists a device more than once.");
        if (shardRatios is not null)
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement splits the decoder evenly; uneven ShardRatios are not supported for '{deviceKey}'.");
        try
        {
            KimiK31Sharding.Require(config, ranks);
        }
        catch (ArgumentException ex)
        {
            throw new NotSupportedException($"Kimi K3.1 multi-GPU placement over {ranks} ranks is not supported: {ex.Message}", ex);
        }
    }

    // 'cuda' and 'cuda:0' name one device. A bare Vulkan selector ranks the devices rather than naming index 0, so it stays as written.
    private static string DeviceIdentity(string device)
    {
        string s = device.Trim().ToLowerInvariant();
        return BackendFactory.IsValidSelector(s) && BackendFactory.Kind(s) == "cuda" ? $"cuda:{BackendFactory.ParseOrdinal(s)}" : s;
    }
}
