using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;

namespace HartsyInference.API.Endpoints;

/// <summary>Client-safe checkpoint profile without host paths, hashes, or raw artifact metadata.</summary>
public sealed class NativeVideoModelProfile
{
    /// <summary>Stable model-profile identifier accepted by generation requests.</summary>
    public required string Id { get; init; }

    /// <summary>Operator-facing checkpoint or composition name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Engine recipe family that executes this profile.</summary>
    public required string FamilyId { get; init; }

    /// <summary>Conditioning family declared by the resolved artifacts.</summary>
    public required VideoTaskFamily Task { get; init; }

    /// <summary>Checkpoint-bound acceleration semantics.</summary>
    public required VideoAccelerationKind Acceleration { get; init; }

    /// <summary>Attention implementation required by the checkpoint.</summary>
    public required VideoAttentionKind Attention { get; init; }

    /// <summary>Hash-bound sampling defaults and locked fields.</summary>
    public required VideoDefaults Defaults { get; init; }

    /// <summary>Conditioning inputs the profile can consume.</summary>
    public required VideoFeatures Features { get; init; }

    /// <summary>Weight format inferred from tensor descriptors.</summary>
    public string? Quantization { get; init; }

    /// <summary>Whether the profile came from Hartsy's built-in manifest.</summary>
    public bool IsBuiltIn { get; init; }

    /// <summary>Whether the profile came from a local hash-bound sidecar.</summary>
    public bool IsSidecar { get; init; }

    /// <summary>Artifact provenance page shown to operators.</summary>
    public string? ProvenanceUrl { get; init; }

    /// <summary>Artifact or base-model license page shown to operators.</summary>
    public string? LicenseUrl { get; init; }

    /// <summary>Projects an internal profile onto the fields safe for remote callers.</summary>
    internal static NativeVideoModelProfile Create(VideoModelProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return new NativeVideoModelProfile
        {
            Id = profile.Id,
            DisplayName = profile.DisplayName,
            FamilyId = profile.FamilyId,
            Task = profile.Task,
            Acceleration = profile.Acceleration,
            Attention = profile.Attention,
            Defaults = profile.Defaults,
            Features = profile.Features,
            Quantization = profile.Quantization,
            IsBuiltIn = profile.IsBuiltIn,
            IsSidecar = profile.IsSidecar,
            // Sidecars are operator-local policy documents. Their free-form URLs can contain private hosts,
            // filesystem URIs, or signed query credentials, so they never cross the remote API boundary.
            ProvenanceUrl = profile.IsSidecar ? null : ManifestHttpUrl(profile.ProvenanceUrl),
            LicenseUrl = profile.IsSidecar ? null : ManifestHttpUrl(profile.LicenseUrl),
        };
    }

    private static string? ManifestHttpUrl(string? value)
    {
        if (!Uri.TryCreate(value, UriKind.Absolute, out Uri? uri)
            || uri.Scheme is not ("http" or "https")
            || !string.IsNullOrEmpty(uri.UserInfo))
        {
            return null;
        }
        return uri.AbsoluteUri;
    }
}
