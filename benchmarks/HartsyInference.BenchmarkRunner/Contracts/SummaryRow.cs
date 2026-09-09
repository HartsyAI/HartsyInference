namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark summary row.</summary>
public sealed record SummaryRow
{
    public required string CaseId { get; init; }
    public required string SuiteId { get; init; }
    public required string EngineRevision { get; init; }
    public required string Gpu { get; init; }
    public required string Backend { get; init; }
    public required string Driver { get; init; }
    public required string Configuration { get; init; }
    public long DeviceMemoryBytes { get; init; }
    public string OperatingSystem { get; init; } = "";
    public string Runtime { get; init; } = "";
    public string NativeConfiguration { get; init; } = "";
    public double? MedianFirstRequestMs { get; init; }
    public double? MedianFirstTokenMs { get; init; }
    public double? MedianDecodeTokensPerSecond { get; init; }
    public double? MedianOutputTokens { get; init; }
    public required double MedianMs { get; init; }
    public required double MeanMs { get; init; }
    public required double IntervalLowMs { get; init; }
    public required double IntervalHighMs { get; init; }
    public required int Machines { get; init; }
    public required int Sessions { get; init; }
    public required string[] Evidence { get; init; }
}
