namespace HartsyInference.ThreeD.Geometry;

/// <summary>A metallic-roughness material described per quantity: RGB base color, optional alpha, metallic and roughness. Textures are
/// aligned to the same UV set; <see cref="Io.PbrGlbWriter"/> packs them into glTF's two textures.</summary>
public sealed record PbrMaterial
{
    /// <summary>Material name.</summary>
    public string Name { get; init; } = "material";

    /// <summary>Three-channel base color, sRGB-encoded as glTF expects.</summary>
    public required TextureImage BaseColor { get; init; }

    /// <summary>One-channel alpha, or null for an opaque material.</summary>
    public TextureImage? Alpha { get; init; }

    /// <summary>One-channel metallic.</summary>
    public required TextureImage Metallic { get; init; }

    /// <summary>One-channel roughness.</summary>
    public required TextureImage Roughness { get; init; }
}
