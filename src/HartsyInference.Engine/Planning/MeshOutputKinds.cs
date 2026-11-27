namespace HartsyInference.Engine.Planning;

/// <summary>The kinds of result a 3D model can return.</summary>
[Flags]
public enum MeshOutputKinds
{
    /// <summary>None.</summary>
    None = 0,

    /// <summary>A glb triangle mesh without a texture.</summary>
    Mesh = 1,

    /// <summary>A glb mesh with a PBR or albedo texture.</summary>
    TexturedMesh = 2,

    /// <summary>A glb with a skeleton and skin weights.</summary>
    RiggedMesh = 4,

    /// <summary>A glb with one node per segmented part.</summary>
    PartMesh = 8,

    /// <summary>A ply of Gaussian splats.</summary>
    Splats = 16,
}
