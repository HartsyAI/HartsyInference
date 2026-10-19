namespace HartsyInference.Core.Rope;

/// <summary>Reusable lease bookkeeping. <see cref="Ticket"/> is 0 when free and is consumed by exactly one release.</summary>
internal sealed class RopeLeaseSlot
{
    public ulong Ticket;

    public RopeHostTable? Host { get; set; }
}
