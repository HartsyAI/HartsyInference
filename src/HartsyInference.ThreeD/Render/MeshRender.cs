using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Render;

/// <summary>A rendered picture.</summary>
/// <param name="Image">RGBA with a transparent background, RGB with a coloured one.</param>
/// <param name="Notes">What was not drawn as the file describes it (a texture that could not be decoded).</param>
public sealed record MeshRender(TextureImage Image, IReadOnlyList<string> Notes);
