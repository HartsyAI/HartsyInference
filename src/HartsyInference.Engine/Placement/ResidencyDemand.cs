using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Placement;

/// <summary>What one load asks of memory, in bytes. The working set is charged to a <see cref="ResidencyLedger"/> up front; the stored weights are mapped and
/// judged against what the working set leaves.</summary>
public sealed record ResidencyDemand
{
    /// <summary>Stored bytes of the dense, embedding and head weights.</summary>
    public required long DenseBytes { get; init; }

    /// <summary>Stored bytes of the routed experts.</summary>
    public required long ExpertBytes { get; init; }

    /// <summary>Stored bytes of the Engram table, read from storage by row.</summary>
    public required long EngramBytes { get; init; }

    /// <summary>Anonymous bytes by account, excluding headroom.</summary>
    public required IReadOnlyDictionary<ResidencyAccount, long> WorkingSet { get; init; }

    /// <summary>Free memory the load keeps beyond its working set.</summary>
    public required long HeadroomBytes { get; init; }
}
