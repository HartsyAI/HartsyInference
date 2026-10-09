namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Thresholds for <see cref="SpeculationSelector"/>.
/// The defaults disable a provider after eight consecutive rounds that each accept under 30% of their draft.</summary>
public sealed record SpeculationSelectorOptions
{
    /// <summary>A round that proposed tokens is below threshold when its acceptance rate (accepted / proposed)
    /// is under this value. Range 0 to 1.</summary>
    public double AcceptanceThreshold { get; init; } = 0.3;

    /// <summary>Consecutive below-threshold rounds that disable a provider for the rest of the selector's life. At least 1.</summary>
    public int DisableAfterRounds { get; init; } = 8;

    /// <summary>Weight of the newest throughput sample in the moving average of tokens per millisecond.
    /// Range (0, 1].</summary>
    public double ThroughputSmoothing { get; init; } = 0.25;
}
