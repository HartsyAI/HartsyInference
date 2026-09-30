namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark suite definition.</summary>
public sealed record SuiteDefinition
{
    public int SchemaVersion { get; init; } = 1;
    public required string Id { get; init; }
    public int Sessions { get; init; } = 3;
    public int Warmups { get; init; } = 2;
    public bool Publishable { get; init; }
    /// <summary>Sampling period requested from nvidia-smi. Frozen here rather than in the sampler so the
    /// validator's coverage bound cannot drift without changing the suite hash.</summary>
    public int TelemetryCadenceMs { get; init; } = 100;
    /// <summary>Share of samples that may report a disqualifying throttle reason before a session stops being
    /// comparable. Zero would reject a run for a single transient sample.</summary>
    public double MaxThrottledSampleFraction { get; init; } = 0.02;
    public required CaseDefinition[] Cases { get; init; }
}
