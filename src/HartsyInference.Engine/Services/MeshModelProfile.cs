using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>What the engine knows about one 3D model without loading it: its inputs and outputs, how it treats each tunable request field, its licence and cost.</summary>
internal sealed record MeshModelProfile
{
    /// <summary>The request fields a profile describes for every model, so a field added to <see cref="MeshRequest"/> cannot ship undescribed.</summary>
    public static readonly IReadOnlyList<string> TunableFields =
    [
        nameof(MeshRequest.Steps), nameof(MeshRequest.GridResolution), nameof(MeshRequest.FieldOfViewRadians), nameof(MeshRequest.Seed),
        nameof(MeshRequest.Samples), nameof(MeshRequest.Colors), nameof(MeshRequest.Chains), nameof(MeshRequest.Rigged),
        nameof(MeshRequest.Texture), nameof(MeshRequest.TextureSize), nameof(MeshRequest.Preprocess), nameof(MeshRequest.PreviewTriangles), nameof(MeshRequest.Frame),
    ];

    /// <summary>The catalog id.</summary>
    public required string Id { get; init; }

    /// <summary>What the model takes.</summary>
    public required MeshInputKind Input { get; init; }

    /// <summary>What the model returns.</summary>
    public required MeshOutputKinds Outputs { get; init; }

    /// <summary>One entry per <see cref="TunableFields"/> name.</summary>
    public required IReadOnlyList<MeshFieldSpec> Fields { get; init; }

    /// <summary>The terms of use, without the gated repositories, which come from the catalog.</summary>
    public required MeshLicence Licence { get; init; }

    /// <summary>A rough cost.</summary>
    public required MeshCost Cost { get; init; }

    /// <summary>Whole-card peak device memory in GiB of a request, from a recorded measurement; null when none was made for this geometry.</summary>
    public Func<MeshRequest, double?>? PeakGiB { get; init; }

    /// <summary>A line of what to know before using the model.</summary>
    public string? Note { get; init; }
}
