namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark review record.</summary>
public sealed record ReviewRecord
{
    public required DateTimeOffset CreatedUtc { get; init; }
    public required string BundleSha256 { get; init; }
    public required string SubmissionId { get; init; }
    public required string Status { get; init; }
    public required string Reviewer { get; init; }
    public required string PullRequest { get; init; }
    public required string VerifiedHead { get; init; }
    public string? Reproduces { get; init; }
    public required string Reason { get; init; }
}
