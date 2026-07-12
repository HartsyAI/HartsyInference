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

    /// <summary>When true, every generated token is masked so only syntactically-valid JSON can be produced
    /// (see <see cref="JsonGrammarStep"/>) — a hard structural constraint, not a probability-shaping setting,
    /// so unlike <see cref="Temperature"/>/<see cref="TopK"/>/etc. it still applies even when
    /// <see cref="Greedy"/> is true (matches how <see cref="RepetitionPenalty"/> already behaves under
    /// greedy — see <see cref="SamplerChain.Next"/>'s doc comment).</summary>
    public bool JsonMode { get; init; } = false;

    /// <summary>When set to a non-empty literal (e.g. <c>"&lt;tool_call&gt;"</c>), grammar masking
    /// (see <see cref="SentinelJsonGrammarStep"/>) activates only for the span starting right after this text
    /// is emitted, deactivating the instant that JSON value completes — everywhere else generation is
    /// completely unconstrained plain text. Mutually exclusive with <see cref="JsonMode"/> in intent (that
    /// one constrains the entire response from token 0); if both are set, <see cref="JsonMode"/> wins and
    /// this is ignored, since a whole-response constraint already subsumes any narrower one.</summary>
    public string? JsonModeSentinel { get; init; } = null;

    /// <summary>True if either JSON constraint is active — the single check call sites that need to gate an
    /// incompatible fast path (CUDA-graph decode, speculative decode) on "some form of JSON masking is live"
    /// should use.</summary>
    public bool HasJsonConstraint => JsonMode || !string.IsNullOrEmpty(JsonModeSentinel);

    /// <summary>Default sampling options (all filters disabled, stochastic draw at temperature 1.0).</summary>
    public static SamplingOptions Default { get; } = new();

    /// <summary>Convenience preset that selects the most likely token deterministically.</summary>
    public static SamplingOptions GreedyPreset => new() { Greedy = true };
}
