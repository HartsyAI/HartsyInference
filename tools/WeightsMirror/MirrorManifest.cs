using System.Text.Json;

namespace HartsyInference.Tools.WeightsMirror;

/// <summary>What the repack run recorded: one entry per repacked file with its source, its hashes, the tensor-for-tensor verdict and its licence.</summary>
internal sealed class MirrorManifest
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };

    public int Version { get; init; } = 1;

    public List<MirrorManifestItem> Items { get; set; } = [];

    public static MirrorManifest LoadOrNew(string path) =>
        File.Exists(path) ? JsonSerializer.Deserialize<MirrorManifest>(File.ReadAllText(path), Options) ?? new MirrorManifest() : new MirrorManifest();

    public void Save(string path)
    {
        Items = [.. Items.OrderBy(i => i.Id, StringComparer.Ordinal)];
        File.WriteAllText(path, JsonSerializer.Serialize(this, Options) + "\n");
    }
}

/// <summary>One repacked file.</summary>
internal sealed class MirrorManifestItem
{
    public string Id { get; init; } = "";

    public string Model { get; init; } = "";

    public string DateUtc { get; init; } = "";

    public string Tool { get; init; } = "";

    public ManifestSource Source { get; init; } = new();

    public ManifestRepack Repack { get; init; } = new();

    public ManifestVerification Verification { get; init; } = new();

    public ManifestLicense License { get; init; } = new();

    public string Notes { get; init; } = "";
}

/// <summary>The upstream pickle the repack came from.</summary>
internal sealed class ManifestSource
{
    public string Repo { get; init; } = "";

    public string Path { get; init; } = "";

    public string Revision { get; init; } = "";

    public string Url { get; init; } = "";

    public long Size { get; init; }

    public string Sha256 { get; init; } = "";

    /// <summary>True when the local copy hashed to the upstream pin before it was repacked.</summary>
    public bool LocalCopyMatchesUpstream { get; init; }
}

/// <summary>The safetensors file a host would serve.</summary>
internal sealed class ManifestRepack
{
    /// <summary>The path under the staging folder and the path a mirror serves it at.</summary>
    public string Path { get; init; } = "";

    public long Size { get; init; }

    public string Sha256 { get; init; } = "";

    public int TensorCount { get; init; }

    public long TensorBytes { get; init; }

    public Dictionary<string, int> Dtypes { get; init; } = [];

    public bool Streamed { get; init; }
}

/// <summary>The comparison of the repack with its source, each read through its own loader.</summary>
internal sealed class ManifestVerification
{
    public bool Equal { get; init; }

    public int TensorsCompared { get; init; }

    public string Method { get; init; } = "";

    /// <summary>Checks made outside this tool, recorded with <c>annotate</c>.</summary>
    public List<string> ExternalChecks { get; init; } = [];
}

/// <summary>The licence that travels with the file.</summary>
internal sealed class ManifestLicense
{
    public string Id { get; init; } = "";

    public string Name { get; init; } = "";

    public string Spdx { get; init; } = "";

    public string HostObligations { get; init; } = "";

    public List<MirrorLicenseFile> TextFiles { get; init; } = [];
}
