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
    public string? Failure { get; init; }
}
