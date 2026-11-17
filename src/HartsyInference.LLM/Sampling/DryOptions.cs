namespace HartsyInference.LLM.Sampling;

/// <summary>Parameters of <see cref="DryPenaltyStep"/>. The defaults are the ones Haru's reference sampler uses.</summary>
public sealed record DryOptions
{
    /// <summary>Scales the penalty; the penalty of a repeat of the allowed length.</summary>
    public float Multiplier { get; init; } = 0.8f;

    /// <summary>Growth of the penalty per extra matched token.</summary>
    public float Base { get; init; } = 1.75f;

    /// <summary>Matched tokens that cost nothing extra: the penalty is <c>Multiplier * Base^(match - AllowedLength)</c>.</summary>
    public int AllowedLength { get; init; } = 1;

    /// <summary>How many tokens back an earlier occurrence is looked for.</summary>
    public int Lookback { get; init; } = 1024;

    /// <summary>Longest match that is measured; longer repeats are capped at this length.</summary>
    public int MaxMatchLength { get; init; } = 32;

    /// <summary>Upper bound of a penalty.</summary>
    public float MaxPenalty { get; init; } = 32f;
}
