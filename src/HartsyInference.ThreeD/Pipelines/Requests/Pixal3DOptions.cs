using HartsyInference.ThreeD.Models.Pixal3D;

namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Options of one Pixal3D request: a shape-only mesh, or with <see cref="Texture"/> a textured PBR model.</summary>
public sealed record Pixal3DOptions
{
    /// <summary>Final dual-grid resolution: 1024 or 1536 (Pixal3D has no 512 pipeline type). 0 uses <see cref="Pixal3DImageTo3DPipeline.DefaultGrid"/>.</summary>
    public int Grid { get; init; }

    /// <summary>Euler steps of both samplers. 0 keeps the <c>pipeline.json</c> value (12).</summary>
    public int Steps { get; init; }

    /// <summary>Noise seed; null draws one. Must not be negative.</summary>
    public int? Seed { get; init; }

    /// <summary>Horizontal field of view of the photo in radians, in (0, π/2). Null uses <see cref="Pixal3DView.DefaultFovRadians"/>: the
    /// port does not run MoGe, which upstream uses to estimate it, so a photo with strong perspective needs this set.</summary>
    public double? FovRadians { get; init; }

    /// <summary>Run the texture path (<see cref="Pixal3DImageTo3DPipeline.GenerateTextured"/>); <see cref="Pixal3DImageTo3DPipeline.Generate"/> ignores it.</summary>
    public bool Texture { get; init; }

    /// <summary>Edge of the baked base-color, metallic and roughness textures in texels. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultTextureSize"/>.</summary>
    public int TextureSize { get; init; }

    /// <summary>Face count the textured mesh is decimated to. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultDecimationTarget"/>.</summary>
    public int DecimationTarget { get; init; }

    /// <summary>Repair the shape-only mesh (<see cref="Geometry.Ops.ShapeMeshFinisher"/>): weld, non-manifold split, specks removed, small holes closed, winding unified.
    /// The textured path always repairs through its post-processor. Boundary edges along larger openings remain either way.</summary>
    public bool CleanMesh { get; init; } = true;

    /// <summary>Upstream <c>mesh_scale</c>: the object's frame relative to the unit grid; must be positive.</summary>
    public double MeshScale { get; init; } = 1.0;
}
