namespace HartsyInference.Engine.Planning;

/// <summary>One <see cref="Requests.MeshRequest"/> field as one model treats it: access, default, and the legal values or range.</summary>
public sealed record MeshFieldSpec
{
    /// <summary>The request property this describes, for example <c>GridResolution</c>.</summary>
    public required string Name { get; init; }

    /// <summary>Whether the model reads, ignores or rejects the field.</summary>
    public required MeshFieldAccess Access { get; init; }

    /// <summary>The value a request that leaves the field unset runs with, rendered invariantly; null when the model's checkpoint decides it.</summary>
    public string? Default { get; init; }

    /// <summary>The legal values when they are a closed set (grid resolutions); null otherwise.</summary>
    public IReadOnlyList<string>? Values { get; init; }

    /// <summary>The smallest legal value when the field is a range. A request that leaves the field at 0 (or unset) takes <see cref="Default"/> wherever the note says so; the range is for values the caller sets.</summary>
    public double? Min { get; init; }

    /// <summary>The largest legal value when the field is a range.</summary>
    public double? Max { get; init; }

    /// <summary>The unit of the value (<c>texels</c>, <c>radians</c>), when it has one.</summary>
    public string? Unit { get; init; }

    /// <summary>What the field means for this model, in one sentence.</summary>
    public string? Note { get; init; }
}
