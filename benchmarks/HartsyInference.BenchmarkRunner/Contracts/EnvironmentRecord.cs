namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Versioned benchmark environment record.</summary>
public sealed record EnvironmentRecord
{
    public required string MachineId { get; init; }
    public required string EngineRevision { get; init; }
    public required string EngineVersion { get; init; }
    public required string OperatingSystem { get; init; }
    public required string Runtime { get; init; }
    public required string Architecture { get; init; }
    public int CpuCount { get; init; }
    public required DeviceRecord Device { get; init; }
    public required SortedDictionary<string, string> Binaries { get; init; }
    public required SortedDictionary<string, string> Settings { get; init; }
    public SortedDictionary<string, string> NativeLibraries { get; init; } = new(StringComparer.Ordinal);
    public string PowerProfile { get; init; } = "unreported";
}
