namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark asset definition.</summary>
public sealed record AssetDefinition
{
    public required string Repository { get; init; }
    public required string Revision { get; init; }
    public required string File { get; init; }
    public required string Sha256 { get; init; }
    public required long Bytes { get; init; }
}
