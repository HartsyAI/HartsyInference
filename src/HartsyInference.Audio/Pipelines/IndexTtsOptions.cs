namespace HartsyInference.Audio.Pipelines;

/// <summary>Generation knobs for <see cref="IndexTtsPipeline"/>. Autoregressive code sampling, not flow-matching —
/// no step count or CFG scale.</summary>
public sealed record IndexTtsOptions
{
    /// <summary>Sampling temperature; ≤0 selects greedy argmax instead.</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Top-k candidates kept before sampling.</summary>
    public int TopK { get; init; } = 30;

    /// <summary>CTRL-style repetition penalty over already-generated mel codes; 1.0 disables it.</summary>
    public float RepetitionPenalty { get; init; } = 2.0f;

    /// <summary>Hard cap on generated mel-code steps; null uses the model's own <c>max_mel_tokens</c> (800 for IndexTTS-1.5).</summary>
    public int? MaxMelTokens { get; init; }

    /// <summary>Seed for the sampling RNG.</summary>
    public ulong Seed { get; init; } = 42;
}
