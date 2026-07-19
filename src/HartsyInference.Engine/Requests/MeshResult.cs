namespace HartsyInference.Engine.Requests;

/// <summary>The result of a 3D generation: the encoded mesh bytes and their container format.</summary>
public sealed record MeshResult
{
    /// <summary>Encoded mesh bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container/format of <see cref="Data"/> (e.g. "glb").</summary>
    public string Format { get; init; } = "glb";
}
