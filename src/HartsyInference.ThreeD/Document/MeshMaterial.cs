namespace HartsyInference.ThreeD.Document;

/// <summary>A metallic-roughness material as a file carries it: factors and the encoded images of each slot. Every image is read through UV set 0; a slot that named another set is left out and noted on the document.</summary>
public sealed record MeshMaterial
{
    /// <summary>Material name; may be empty.</summary>
    public string Name { get; init; } = "";

    /// <summary>Base colour factor, linear RGBA in [0, 1].</summary>
    public float[] BaseColorFactor { get; init; } = [1f, 1f, 1f, 1f];

    /// <summary>Base colour image (sRGB), or null.</summary>
    public MeshImage? BaseColor { get; init; }

    /// <summary>Metallic factor.</summary>
    public float Metallic { get; init; } = 1f;

    /// <summary>Roughness factor.</summary>
    public float Roughness { get; init; } = 1f;

    /// <summary>The image glTF packs roughness (green) and metallic (blue) into, or null.</summary>
    public MeshImage? MetallicRoughness { get; init; }

    /// <summary>Tangent-space normal map, or null.</summary>
    public MeshImage? Normal { get; init; }

    /// <summary>Occlusion map (red channel), or null.</summary>
    public MeshImage? Occlusion { get; init; }

    /// <summary>Emissive colour factor, linear RGB.</summary>
    public float[] EmissiveFactor { get; init; } = [0f, 0f, 0f];

    /// <summary>Emissive image, or null.</summary>
    public MeshImage? Emissive { get; init; }

    /// <summary>How alpha is used.</summary>
    public MeshAlphaMode AlphaMode { get; init; }

    /// <summary>Cutoff for <see cref="MeshAlphaMode.Mask"/>.</summary>
    public float AlphaCutoff { get; init; } = 0.5f;

    /// <summary>Whether back faces are drawn.</summary>
    public bool DoubleSided { get; init; }

    /// <summary>Every image the material holds, in slot order.</summary>
    public IEnumerable<MeshImage> Images => new[] { BaseColor, MetallicRoughness, Normal, Occlusion, Emissive }.OfType<MeshImage>();
}
