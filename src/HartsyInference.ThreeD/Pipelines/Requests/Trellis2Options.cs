namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Options of one TRELLIS.2 request: a shape-only mesh, or with <see cref="Texture"/> a textured PBR model.</summary>
public sealed record Trellis2Options
{
    /// <summary>Final dual-grid resolution: 512, 1024 or 1536. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultGrid"/>.</summary>
    public int Grid { get; init; }

    /// <summary>Euler steps of both samplers. 0 keeps the <c>pipeline.json</c> value (12).</summary>
    public int Steps { get; init; }

    /// <summary>Noise seed; null draws one. Must not be negative.</summary>
    public int? Seed { get; init; }

    /// <summary>Run the texture path (<see cref="Trellis2ImageTo3DPipeline.GenerateTextured"/>); <see cref="Trellis2ImageTo3DPipeline.Generate"/> ignores it.</summary>
    public bool Texture { get; init; }

    /// <summary>Edge of the baked base-color, metallic and roughness textures in texels. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultTextureSize"/>.</summary>
    public int TextureSize { get; init; }

    /// <summary>Face count the textured mesh is decimated to. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultDecimationTarget"/>.</summary>
    public int DecimationTarget { get; init; }

    /// <summary>Repair the shape-only mesh (<see cref="Geometry.Ops.ShapeMeshFinisher"/>): weld, non-manifold split, specks removed, small holes closed, winding unified.
    /// The textured path always repairs through its post-processor. Boundary edges along larger openings remain either way.</summary>
    public bool CleanMesh { get; init; } = true;

    /// <summary>With <see cref="CleanMesh"/>, also close the thin slits and gaps the repair leaves: every boundary loop that triangulates with no new edge longer than
    /// <see cref="Geometry.Ops.ShapeMeshFinisher.DefaultOpeningWidthRatio"/> of the bounding-box diagonal (<see cref="Geometry.Ops.HoleFiller"/>); wider openings stay open. Off by default.</summary>
    public bool CloseOpenings { get; init; }
}
