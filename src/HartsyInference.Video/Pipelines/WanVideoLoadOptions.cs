namespace HartsyInference.Video.Pipelines;

/// <summary>Optional knobs for <see cref="WanVideoLoader.Load"/>. Everything else is auto-detected.</summary>
public record WanVideoLoadOptions
{
    /// <summary>Second (low-noise) expert for Wan2.2 A14B MoE packaging; enables the expert swap at <see cref="BoundaryRatio"/>.</summary>
    public string? LowNoiseDitPath { get; init; }

    /// <summary>MoE noise boundary (Wan2.2 A14B ships with 0.875 for T2V / 0.9 for I2V).</summary>
    public float BoundaryRatio { get; init; } = 0.875f;

    /// <summary>umT5-XXL safetensors for <see cref="WanVideoLoader.EncodePrompts"/> (fp8-scaled supported).</summary>
    public string? Umt5Path { get; init; }

    /// <summary>Overrides the flow-match shift (resolution-dependent: 3.0 @ 480p, 5.0 @ 720p; not a weight fact).</summary>
    public float? FlowShift { get; init; }

    /// <summary>Overrides the auto CFG renormalization (fp8 → 0.7, else 0).</summary>
    public float? CfgRescale { get; init; }
}
