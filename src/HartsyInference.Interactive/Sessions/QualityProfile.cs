namespace HartsyInference.Interactive.Sessions;

/// <summary>Speed/quality trade-off applied to a running session. Models interpret the knobs they support and ignore
/// the rest (e.g. Matrix-Game 3.0 distilled runs 3 steps; its 50-step base profile is offline-render only).</summary>
public sealed record QualityProfile
{
    /// <summary>Denoise steps per frame/segment.</summary>
    public int DenoiseSteps { get; init; } = 3;

    /// <summary>Classifier-free guidance scale (1.0 disables the second forward pass).</summary>
    public float GuidanceScale { get; init; } = 1.0f;

    /// <summary>Distilled real-time preset (3-step, no CFG).</summary>
    public static QualityProfile Low => new();

    /// <summary>Distilled with CFG.</summary>
    public static QualityProfile Medium => new() { DenoiseSteps = 4, GuidanceScale = 5.0f };

    /// <summary>Offline-render quality (full 50-step base schedule).</summary>
    public static QualityProfile High => new() { DenoiseSteps = 50, GuidanceScale = 5.0f };
}
