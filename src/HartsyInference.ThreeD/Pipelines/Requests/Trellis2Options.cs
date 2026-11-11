namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>Options of one TRELLIS.2 shape request.</summary>
public sealed record Trellis2Options
{
    /// <summary>Final dual-grid resolution: 512, 1024 or 1536. 0 uses <see cref="Trellis2ImageTo3DPipeline.DefaultGrid"/>.</summary>
    public int Grid { get; init; }

    /// <summary>Euler steps of both samplers. 0 keeps the <c>pipeline.json</c> value (12).</summary>
    public int Steps { get; init; }

    /// <summary>Noise seed; null draws one. Must not be negative.</summary>
    public int? Seed { get; init; }
}
