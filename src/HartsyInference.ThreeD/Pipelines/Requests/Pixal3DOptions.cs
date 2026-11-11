using HartsyInference.ThreeD.Models.Pixal3D;

namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Options of one Pixal3D shape request.</summary>
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

    /// <summary>Upstream <c>mesh_scale</c>: the object's frame relative to the unit grid; must be positive.</summary>
    public double MeshScale { get; init; } = 1.0;
}
