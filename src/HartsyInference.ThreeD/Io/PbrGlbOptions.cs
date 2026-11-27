using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Options for <see cref="PbrGlbWriter"/>.</summary>
public sealed record PbrGlbOptions
{
    /// <summary>Rotate Z-up input to glTF Y-up with <c>(x, y, z) -> (x, z, -y)</c> on positions and normals (TRELLIS.2 export).</summary>
    public bool ConvertZUpToYUp { get; init; }

    /// <summary>Write <c>v' = 1 - v</c> for UVs with v pointing up; the bake path already uses glTF's convention (row 0 at v = 0).</summary>
    public bool FlipV { get; init; }

    /// <summary>glTF alpha mode written on the material.</summary>
    public PbrAlphaMode AlphaMode { get; init; } = PbrAlphaMode.Opaque;

    /// <summary>Alpha cutoff used with <see cref="PbrAlphaMode.Mask"/>.</summary>
    public float AlphaCutoff { get; init; } = 0.5f;

    /// <summary>Render both faces of every triangle.</summary>
    public bool DoubleSided { get; init; }

    /// <summary>Node and mesh name.</summary>
    public string MeshName { get; init; } = "mesh";
}
