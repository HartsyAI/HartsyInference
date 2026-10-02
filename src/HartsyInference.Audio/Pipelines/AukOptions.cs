namespace HartsyInference.Audio.Pipelines;

/// <summary>AuK generation knobs; AuK-Flash pins its distilled 4-step recipe and ignores <see cref="Steps"/>, <see cref="CfgScale"/> and <see cref="SwayCoef"/>.</summary>
public sealed record AukOptions
{
    /// <summary>Noise seed for the sampler and the reference-encoder sampling.</summary>
    public int Seed { get; init; }

    /// <summary>Base-model Euler steps (NFE); null uses 32.</summary>
    public int? Steps { get; init; }

    /// <summary>Base-model guidance strength; null uses 2.0, and 0 skips the unconditional forward.</summary>
    public float? CfgScale { get; init; }

    /// <summary>Sway-sampling coefficient of the base schedule.</summary>
    public float SwayCoef { get; init; } = -1f;

    /// <summary>Speech speed divisor of the text-length duration heuristic.</summary>
    public double Speed { get; init; } = 1.0;

    /// <summary>Text to be spoken; with <see cref="RefText"/> it scales the reference duration when no explicit duration is given.</summary>
    public string? GenText { get; init; }

    /// <summary>Transcript of the reference clip, for the duration heuristic.</summary>
    public string? RefText { get; init; }

    /// <summary>Keep only one stage's weights on the device at a time (audio tower, thinker, DiT, VAE decoder); false preloads everything and frees nothing.</summary>
    public bool SequentialResidency { get; init; } = true;
}
