namespace HartsyInference.BenchmarkRunner.Contracts;
/// <summary>Device configuration and exclusivity as attested before the first session. Carries no PIDs or
/// process paths: the operator sees those on the console, the exported record sees only how many and how much.</summary>
public sealed record AttestationRecord
{
    public const string Unavailable = "unavailable";
    public const string Smi = "nvidia-smi";

    public required string Source { get; init; }
    /// <summary>Raw nvidia-smi values keyed by query field. Stored as printed, including <c>[N/A]</c>, because
    /// a consumer-class card legitimately reports several of them that way.</summary>
    public SortedDictionary<string, string> Fields { get; init; } = new(StringComparer.Ordinal);
    public int SharedProcessCount { get; init; }
    public long SharedProcessBytes { get; init; }
    /// <summary>True when the operator overrode the exclusivity refusal with <c>--allow-shared-device</c>.</summary>
    public bool SharedDeviceAllowed { get; init; }
}
