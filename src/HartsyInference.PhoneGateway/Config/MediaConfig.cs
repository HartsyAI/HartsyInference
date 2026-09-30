namespace HartsyInference.PhoneGateway.Config;

/// <summary>Tick thread scheduling (<c>media</c> section).</summary>
public sealed record MediaConfig
{
    /// <summary><c>SCHED_FIFO</c> priority for the tick thread (needs <c>LimitRTPRIO</c>); zero never asks.</summary>
    public int FifoPriority { get; set; } = 50;

    /// <summary>CPU to pin the tick thread to; -1 for none.</summary>
    public int TickCpu { get; set; } = -1;

    public int WarmUpTicks { get; set; } = 200;
}
