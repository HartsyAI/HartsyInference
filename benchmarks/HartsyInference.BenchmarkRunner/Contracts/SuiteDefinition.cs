namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark suite definition.</summary>
public sealed record SuiteDefinition
{
    public int SchemaVersion { get; init; } = 1;
    public required string Id { get; init; }
    public int Sessions { get; init; } = 3;
    public int Warmups { get; init; } = 2;
    public bool Publishable { get; init; }
    public required CaseDefinition[] Cases { get; init; }
}
