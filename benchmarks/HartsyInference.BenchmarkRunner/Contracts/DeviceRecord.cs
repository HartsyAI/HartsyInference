namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark device record.</summary>
public sealed record DeviceRecord
{
    public required string Selector { get; init; }
    public required string Name { get; init; }
    public required string Identity { get; init; }
    public string HardwareKind { get; init; } = "unknown";
    public long MemoryBytes { get; init; }
    public required string Driver { get; init; }
    public string Capabilities { get; init; } = "";
    public string? Error { get; init; }
}
