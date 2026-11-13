namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>A decimated, UV-unwrapped mesh with its baked PBR material.</summary>
/// <param name="Mesh">Mesh with positions, normals and UVs.</param>
/// <param name="Material">Baked material aligned to <see cref="Mesh"/>'s UVs.</param>
/// <param name="CoveredTexels">Texels the bake wrote directly (the rest were inpainted).</param>
public sealed record PbrPostProcessResult(Mesh Mesh, PbrMaterial Material, int CoveredTexels)
{
    /// <summary>What the caller should know about the result (an input that unwrapped into confetti, for one); empty when nothing is wrong.</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];
}
