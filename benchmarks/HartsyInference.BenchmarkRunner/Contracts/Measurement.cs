namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark measurement.</summary>
public sealed record Measurement
{
    public required int Input { get; init; }
    public required string Lane { get; init; }
    public required double ElapsedMs { get; init; }
    public long StopwatchFrequency { get; init; }
    public long StartedTicks { get; init; }
    public long RequestStartedTicks { get; init; }
    public long CompletedTicks { get; init; }
    public long PrefillTicks { get; init; }
    public long[] TokenTimestamps { get; init; } = [];
    public double? FirstTokenMs { get; init; }
    public double? DecodeTokensPerSecond { get; init; }
    public int PromptTokens { get; init; }
    public int CompletionTokens { get; init; }
    public string StopReason { get; init; } = "completed";
    public required string Output { get; init; }
    public required string OutputSha256 { get; init; }
    public required bool QualityPassed { get; init; }
    public required string QualityDetail { get; init; }
    public long HostPeakBytes { get; init; }
    public long? SampledUsedDeviceBytes { get; init; }
    public string MemorySource { get; init; } = "unavailable";
}
