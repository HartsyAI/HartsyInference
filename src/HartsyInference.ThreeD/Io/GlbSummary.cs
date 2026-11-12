namespace HartsyInference.ThreeD.Io;

/// <summary>What a GLB's JSON chunk says about it, read without touching the binary chunk.</summary>
/// <param name="MeshCount">Meshes.</param>
/// <param name="HasNormals">Some primitive carries a NORMAL attribute.</param>
/// <param name="HasTexture">The file has at least one texture.</param>
/// <param name="HasSkin">The file has at least one skin.</param>
/// <param name="AnimationNames">The names of its animations, in file order (an unnamed one is <c>animation_i</c>).</param>
/// <param name="Min">The component-wise minimum of every POSITION accessor, or null when none declares one.</param>
/// <param name="Max">The component-wise maximum of every POSITION accessor, or null when none declares one.</param>
public sealed record GlbSummary(
    int MeshCount, bool HasNormals, bool HasTexture, bool HasSkin, IReadOnlyList<string> AnimationNames, float[]? Min, float[]? Max);
