namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark session record.</summary>
public sealed record SessionRecord
{
    public required string CaseId { get; init; }
    public required int Session { get; init; }
    public required int Attempt { get; init; }
    public required string Status { get; init; }
    public required DateTimeOffset StartedUtc { get; init; }
    public required Measurement[] Measurements { get; init; }
    public DeviceRecord? ActualDevice { get; init; }
    public SortedDictionary<string, string> NativeLibraries { get; init; } = new(StringComparer.Ordinal);
    /// <summary>Other compute processes seen on the device when this session started. Nonzero only after the
    /// operator overrode the pre-flight refusal, or when a tenant appeared mid-campaign.</summary>
    public int SharedProcessCount { get; init; }
    /// <summary>Why telemetry is missing or partial, when it is.</summary>
    public string? TelemetryNote { get; init; }
    public string? Failure { get; init; }
}
