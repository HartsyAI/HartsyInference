using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>One routed expert access as kept in the telemetry ring.</summary>
public readonly record struct RoutingStepRecord(long Step, ExpertKey Key, long BytesMoved, RoutingEventFlags Flags)
{
    /// <summary>True when the expert was resident when routed.</summary>
    public bool IsHit => (Flags & RoutingEventFlags.Hit) != 0;

    /// <summary>True when the expert ran on a GPU.</summary>
    public bool RanOnGpu => (Flags & RoutingEventFlags.GpuRun) != 0;
}
