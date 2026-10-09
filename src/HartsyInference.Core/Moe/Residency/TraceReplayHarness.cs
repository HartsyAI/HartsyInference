using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe.Telemetry;

namespace HartsyInference.Core.Moe.Residency;

/// <summary>Replays a routing trace through a policy with a fixed slot budget and reports the hit rate.</summary>
public static class TraceReplayHarness
{
    /// <summary>Replays <paramref name="trace"/> in order. The policy must be fresh (or reset) for a comparable result.</summary>
    public static ReplayResult Replay(ExpertTraceRecord[] trace, IAdaptiveResidencyPolicy policy, int slotBudget)
    {
        ArgumentNullException.ThrowIfNull(trace);
        ArgumentNullException.ThrowIfNull(policy);

        SlotCacheSimulator simulator = new(policy, slotBudget);
        foreach (ExpertTraceRecord record in trace)
        {
            simulator.Access(new ExpertKey(record.Layer, record.Expert), record.Bytes);
        }

        return new ReplayResult(
            policy.Name,
            slotBudget,
            trace.Length,
            simulator.Hits,
            simulator.Misses,
            simulator.Evictions,
            simulator.BytesMoved,
            simulator.Digest);
    }
}
