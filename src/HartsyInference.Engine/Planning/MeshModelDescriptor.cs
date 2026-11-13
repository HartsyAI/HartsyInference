using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Planning;

/// <summary>What one 3D model takes, returns and accepts, answered without loading weights: a host builds its form from this.</summary>
public sealed record MeshModelDescriptor
{
    /// <summary>The catalog id.</summary>
    public required string Id { get; init; }

    /// <summary>The name for menus.</summary>
    public required string DisplayName { get; init; }

    /// <summary>What the model takes as input.</summary>
    public required MeshInputKind Input { get; init; }

    /// <summary>The results the model can return; <see cref="MeshOutputKinds.TexturedMesh"/> appears for models whose texture is optional too.</summary>
    public required MeshOutputKinds Outputs { get; init; }

    /// <summary>The catalog's maturity of the model; the status document has the evidence.</summary>
    public required ModelStatus Status { get; init; }

    /// <summary>Every <see cref="Requests.MeshRequest"/> field this model treats specially, keyed by property name; a field not listed is read as the request carries it.</summary>
    public required IReadOnlyDictionary<string, MeshFieldSpec> Fields { get; init; }

    /// <summary>The terms the weights are used under.</summary>
    public required MeshLicence Licence { get; init; }

    /// <summary>A rough cost of one default request.</summary>
    public required MeshCost Cost { get; init; }

    /// <summary>One line of what to know before using the model (a limit, a manual file), or null.</summary>
    public string? Note { get; init; }
}
