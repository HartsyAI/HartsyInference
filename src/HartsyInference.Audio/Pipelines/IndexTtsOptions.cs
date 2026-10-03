namespace HartsyInference.Audio.Pipelines;

/// <summary>Generation knobs for <see cref="IndexTtsPipeline"/>. Autoregressive code sampling, not flow-matching —
/// no step count or CFG scale.</summary>
/// <remarks>The reference CLI's defaults also include <c>top_p=0.8</c> (nucleus sampling) and <c>num_beams=3</c>
/// (beam search); neither is implemented here — this is single-sequence top-k-with-temperature decoding only, a
/// Phase-1 simplification, not a claim of matching upstream's search behavior exactly.</remarks>
public sealed record IndexTtsOptions
{
    /// <summary>Sampling temperature; ≤0 selects greedy argmax instead.</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Top-k candidates kept before sampling.</summary>
    public int TopK { get; init; } = 30;

    /// <summary>CTRL-style repetition penalty over already-generated mel codes; 1.0 disables it. Matches the
    /// reference CLI's default (verified: <c>indextts/infer.py</c> defaults <c>repetition_penalty</c> to 10.0).</summary>
    public float RepetitionPenalty { get; init; } = 10.0f;

    /// <summary>Hard cap on generated mel-code steps; null uses the model's own <c>max_mel_tokens</c> (800 for IndexTTS-1.5).</summary>
    public int? MaxMelTokens { get; init; }

    /// <summary>Seed for the sampling RNG.</summary>
    public ulong Seed { get; init; } = 42;
}
