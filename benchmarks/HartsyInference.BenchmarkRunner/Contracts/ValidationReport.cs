namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark validation report.</summary>
public sealed record ValidationReport
{
    public required bool Valid { get; init; }
    public required bool HeadlineEligible { get; init; }
    public required string[] Errors { get; init; }
    public required string[] Notes { get; init; }
}
