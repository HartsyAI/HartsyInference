namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark case definition.</summary>
public sealed record CaseDefinition
{
    public required string Id { get; init; }
    public required string Adapter { get; init; }
    public required string Model { get; init; }
    public required AssetDefinition Asset { get; init; }
    public required string[] Inputs { get; init; }
    public string InputPrefix { get; init; } = "";
    public int PrefixRepeats { get; init; }
    public int Width { get; init; } = 512;
    public int Height { get; init; } = 512;
    public int Steps { get; init; } = 20;
    public int MaxTokens { get; init; } = 128;
    public long Seed { get; init; } = 42;
    public int TimeoutSeconds { get; init; } = 600;
}
