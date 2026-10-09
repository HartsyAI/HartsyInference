using System.Text.Json.Serialization;

namespace HartsyInference.API;

/// <summary>The <c>/admin/packages</c> response: the text model packages found under one directory.</summary>
public sealed class PackageListResponse
{
    [JsonPropertyName("root")] public required string Root { get; init; }
    [JsonPropertyName("packages")] public required IReadOnlyList<PackageDto> Packages { get; init; }

    /// <summary>Each directory the scan skipped because it could not be read, with the reason. Its contents are not in <see cref="Packages"/>.</summary>
    [JsonPropertyName("problems")] public required IReadOnlyList<string> Problems { get; init; }
}

/// <summary>One text model package on disk, as <c>TextPackageDiscovery</c> found it. Nothing is loaded to produce it.</summary>
public sealed class PackageDto
{
    [JsonPropertyName("id")] public required string Id { get; init; }

    /// <summary><c>safetensors_shards</c> (a checkpoint directory), <c>gguf</c> (one file) or <c>split_gguf</c> (every part of a llama.cpp split).</summary>
    [JsonPropertyName("format")] public required string Format { get; init; }

    /// <summary>The entry point: the checkpoint directory, the GGUF file, or the first part of a split.</summary>
    [JsonPropertyName("path")] public required string Path { get; init; }

    [JsonPropertyName("files")] public required IReadOnlyList<string> Files { get; init; }
    [JsonPropertyName("total_bytes")] public required long TotalBytes { get; init; }
    [JsonPropertyName("family")] public string? Family { get; init; }
    [JsonPropertyName("quant")] public string? Quant { get; init; }
    [JsonPropertyName("sidecars")] public required IReadOnlyList<SidecarDto> Sidecars { get; init; }
    [JsonPropertyName("capabilities")] public required CapabilitiesDto Capabilities { get; init; }

    /// <summary>What is wrong with the package as found, such as missing split parts. Empty when nothing is.</summary>
    [JsonPropertyName("problems")] public required IReadOnlyList<string> Problems { get; init; }
}

/// <summary>A companion file that goes with a package but is not a model of its own, such as a vision tower.</summary>
public sealed class SidecarDto
{
    [JsonPropertyName("role")] public required string Role { get; init; }
    [JsonPropertyName("path")] public required string Path { get; init; }
}

/// <summary>What a text model can do beyond chat. <c>speculation</c> is DSpark draft-head speculation; when it is false, <c>speculation_reason</c> says why.</summary>
public sealed class CapabilitiesDto
{
    [JsonPropertyName("speculation")] public required bool Speculation { get; init; }
    [JsonPropertyName("speculation_reason")] public required string SpeculationReason { get; init; }
}
