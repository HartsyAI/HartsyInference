namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark submission record.</summary>
public sealed record SubmissionRecord
{
    public int SchemaVersion { get; init; } = 1;
    public required string Id { get; init; }
    public required string CampaignSha256 { get; init; }
    public required string BundleSha256 { get; init; }
    public required long BundleBytes { get; init; }
    public required string StagingUrl { get; init; }
    public required string ArchiveUrl { get; init; }
}
