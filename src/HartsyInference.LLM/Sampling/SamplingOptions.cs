namespace HartsyInference.LLM.Sampling;

/// <summary>Configuration for autoregressive token sampling: temperature, top-k, top-p (nucleus), min-p, and repetition penalty, plus the RNG seed and a greedy toggle.</summary>
public sealed record SamplingOptions
{
    /// <summary>Logit scaling divisor applied before softmax; 1.0 leaves logits unchanged.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Keeps only the highest-scoring tokens; 0 disables the filter.</summary>
    public int TopK { get; init; } = 0;

    /// <summary>Nucleus cumulative-probability cutoff; 1.0 disables the filter.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Drops tokens whose probability is below this fraction of the top token's probability; 0.0 disables the filter.</summary>
    public float MinP { get; init; } = 0.0f;

    /// <summary>Penalty applied to logits of previously generated tokens (HF convention); 1.0 disables the penalty.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>RNG seed for the multinomial draw; 0 falls back to a fixed reproducible constant.</summary>
    public ulong Seed { get; init; } = 0;

    /// <summary>When true, always selects the argmax token and ignores every other setting.</summary>
    public bool Greedy { get; init; } = false;

    /// <summary>Default sampling options (all filters disabled, stochastic draw at temperature 1.0).</summary>
    public static SamplingOptions Default { get; } = new();

    /// <summary>Convenience preset that selects the most likely token deterministically.</summary>
    public static SamplingOptions GreedyPreset => new() { Greedy = true };
}
