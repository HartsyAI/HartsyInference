namespace HartsyInference.Engine.Requests;

/// <summary>An encoded triangle scene mesh for scene-decomposition models: its container bytes and format (<c>obj</c> or <c>glb</c>).</summary>
public sealed record SceneMeshData
{
    /// <summary>Encoded scene mesh bytes.</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container format of <see cref="Data"/>: <c>obj</c> or <c>glb</c>.</summary>
    public required string Format { get; init; }
}
