using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HartsyInference.OpParity;

/// <summary>Everything a fixture needs to be trusted later: where it came from, on what, with which kernels and settings.</summary>
public sealed class FixtureManifest
{
    public required string Family { get; init; }
    public required string Case { get; init; }
    public required string Backend { get; init; }
    public required string Device { get; init; }
    public required string Commit { get; init; }
    public required string RecordedUtc { get; init; }
    /// <summary><c>recorded</c> or <c>unsupported</c> (the baseline backend does not serve the case).</summary>
    public required string Status { get; init; }
    public string? SettingsFile { get; init; }
    public Dictionary<string, string> KernelHashes { get; init; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> FileHashes { get; init; } = new(StringComparer.Ordinal);
    public required string AcceptanceKind { get; init; }
    public double MaxAbs { get; init; }
    public double MaxRel { get; init; }
    public string Reason { get; init; } = "";
    public List<string> MayBeNonFinite { get; init; } = [];
    public List<string> RequiredBackends { get; init; } = [];

    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, Options));

    public static FixtureManifest Load(string path) => JsonSerializer.Deserialize<FixtureManifest>(File.ReadAllText(path), Options)
        ?? throw new InvalidDataException($"{path} is not a fixture manifest.");

    /// <summary>SHA-256 of a file, lower-case hex.</summary>
    public static string HashFile(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    /// <summary>Hashes of every kernel artifact in <paramref name="dir"/>, keyed by file name; empty when the directory is absent.</summary>
    public static Dictionary<string, string> HashKernels(string dir)
    {
        Dictionary<string, string> hashes = new(StringComparer.Ordinal);
        if (!Directory.Exists(dir)) return hashes;
        foreach (string file in Directory.EnumerateFiles(dir).Order(StringComparer.Ordinal))
            hashes[Path.GetFileName(file)] = HashFile(file);
        return hashes;
    }
}
