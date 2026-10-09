namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>Outcome bits of one routed expert access.</summary>
[Flags]
public enum RoutingEventFlags : byte
{
    /// <summary>No bits set: a miss that ran on the CPU.</summary>
    None = 0,

    /// <summary>The expert was resident when routed.</summary>
    Hit = 1,

    /// <summary>The expert ran on a GPU; clear means it ran on the CPU.</summary>
    GpuRun = 2,
}
