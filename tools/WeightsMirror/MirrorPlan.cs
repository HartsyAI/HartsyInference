using System.Text.Json;

namespace HartsyInference.Tools.WeightsMirror;

/// <summary>The checked-in list of files to repack for hosting: where each comes from, what it must hash to, and the licence that travels with it.</summary>
internal sealed class MirrorPlan
{
    private static readonly JsonSerializerOptions Options = new() { PropertyNameCaseInsensitive = true, ReadCommentHandling = JsonCommentHandling.Skip };

    public List<MirrorLicense> Licenses { get; init; } = [];

    public List<MirrorItem> Items { get; init; } = [];

    public static MirrorPlan Load(string path) =>
        JsonSerializer.Deserialize<MirrorPlan>(File.ReadAllText(path), Options) ?? throw new InvalidDataException($"'{path}' is not a mirror plan.");

    public MirrorLicense LicenseOf(MirrorItem item) =>
        Licenses.FirstOrDefault(l => l.Id == item.License) ?? throw new InvalidDataException($"Item '{item.Id}' names an unknown licence '{item.License}'.");
}

/// <summary>A licence and the text files a host must ship beside the weights it covers.</summary>
internal sealed class MirrorLicense
{
    public string Id { get; init; } = "";

    public string Name { get; init; } = "";

    public string Spdx { get; init; } = "";

    /// <summary>What a host has to do, in a sentence, taken from the licence text.</summary>
    public string HostObligations { get; init; } = "";

    public List<MirrorLicenseFile> TextFiles { get; init; } = [];
}

/// <summary>One licence or notice file in its upstream repository: its size and git blob id, so the copy a host ships can be checked.</summary>
internal sealed class MirrorLicenseFile
{
    public string Name { get; init; } = "";

    public string Url { get; init; } = "";

    public long Size { get; init; }

    public string GitBlobSha1 { get; init; } = "";

    /// <summary>SHA-256 of the copy placed in the staging folder, recorded by the <c>licenses</c> command once the file matches <see cref="GitBlobSha1"/>.</summary>
    public string Sha256 { get; set; } = "";

    /// <summary>When the copy was first verified, UTC.</summary>
    public string VerifiedUtc { get; set; } = "";
}

/// <summary>One pickle to repack.</summary>
internal sealed class MirrorItem
{
    public string Id { get; init; } = "";

    /// <summary>The catalog model id; the first folder of the hosted path.</summary>
    public string Model { get; init; } = "";

    public string SourceRepo { get; init; } = "";

    public string SourcePath { get; init; } = "";

    public string Revision { get; init; } = "";

    public string SourceUrl { get; init; } = "";

    public long SourceSize { get; init; }

    public string SourceSha256 { get; init; } = "";

    /// <summary>The local copy, relative to the models root. Only ever read.</summary>
    public string LocalSource { get; init; } = "";

    /// <summary>The repack's path under the staging folder: <c>&lt;model&gt;/&lt;the source's path with .safetensors&gt;</c>.</summary>
    public string Output { get; init; } = "";

    /// <summary>True for a root that is a dict of state dicts (a Lightning checkpoint).</summary>
    public bool Recursive { get; init; }

    public string License { get; init; } = "";

    public string Notes { get; init; } = "";
}
