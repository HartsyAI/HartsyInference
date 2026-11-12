namespace HartsyInference.Engine.Planning;

/// <summary>A rough cost of one default generation, from the measurements recorded in the model status document; null where none was made.</summary>
public sealed record MeshCost
{
    /// <summary>Peak device memory in GiB, whole-card, of a default request.</summary>
    public double? PeakVramGiB { get; init; }

    /// <summary>Wall-clock seconds of a default request on a 12 GB card.</summary>
    public int? TypicalSeconds { get; init; }

    /// <summary>What was measured, in one sentence, so a host does not read the numbers as a guarantee.</summary>
    public required string Basis { get; init; }
}
