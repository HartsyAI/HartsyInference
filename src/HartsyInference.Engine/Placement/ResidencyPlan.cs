using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Planning.Memory;

namespace HartsyInference.Engine.Placement;

/// <summary>The fit verdict for one load and where each component is placed, judged before anything is mapped.</summary>
public sealed record ResidencyPlan
{
    /// <summary>Resident when everything fits in memory, Streamed when the working set fits but the mapped weights must be read from storage, Infeasible when it does not fit.</summary>
    public required MemoryFitVerdict Verdict { get; init; }

    /// <summary>Each component's placement.</summary>
    public required IReadOnlyList<ComponentResidency> Components { get; init; }

    /// <summary>Free memory the plan was judged against, in bytes.</summary>
    public required long AvailableBytes { get; init; }

    /// <summary>Anonymous working set, excluding headroom, in bytes.</summary>
    public required long WorkingSetBytes { get; init; }

    /// <summary>Headroom the plan kept free, in bytes.</summary>
    public required long HeadroomBytes { get; init; }

    /// <summary>Stored weights that are mapped rather than charged, in bytes.</summary>
    public required long MappedBytes { get; init; }

    /// <summary>The reservations the ledger holds for this plan, by account.</summary>
    public required IReadOnlyDictionary<ResidencyAccount, long> ReservedByAccount { get; init; }

    /// <summary>Bytes each account asked for that the ledger refused; empty unless the plan is infeasible.</summary>
    public required IReadOnlyDictionary<ResidencyAccount, long> RefusedByAccount { get; init; }

    /// <summary>A one-line explanation of the verdict.</summary>
    public required string Reason { get; init; }
}
