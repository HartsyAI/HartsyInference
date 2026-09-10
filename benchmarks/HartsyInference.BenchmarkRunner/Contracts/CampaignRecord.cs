namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark campaign record.</summary>
public sealed record CampaignRecord
{
    public int SchemaVersion { get; init; } = 1;
    public required string SuiteId { get; init; }
    public required string SuiteSha256 { get; init; }
    public required DateTimeOffset CreatedUtc { get; init; }
    public required EnvironmentRecord Environment { get; init; }
    public required SessionRecord[] Sessions { get; init; }
    public int BudgetMinutes { get; init; } = 30;
}
