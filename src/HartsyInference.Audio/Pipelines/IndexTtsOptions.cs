namespace HartsyInference.Audio.Pipelines;

/// <summary>Generation knobs for <see cref="IndexTtsPipeline"/>. Autoregressive code sampling, not flow-matching —
/// no step count or CFG scale.</summary>
/// <remarks>The reference CLI's default also includes <c>num_beams=3</c> (beam search combined with sampling, via
/// HF <c>generate()</c>); not implemented here — this is single-sequence temperature/top-k/top-p decoding only, a
/// Phase-1 simplification, not a claim of matching upstream's beam search exactly.</remarks>
public sealed record IndexTtsOptions
{
    /// <summary>Sampling temperature; ≤0 selects greedy argmax instead.</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Top-k candidates kept before nucleus filtering.</summary>
    public int TopK { get; init; } = 30;

    /// <summary>Nucleus (top-p) sampling threshold, applied after top-k — matches the reference CLI's default
    /// (<c>indextts/infer.py</c>'s <c>infer()</c> defaults <c>top_p</c> to 0.8).</summary>
    public float TopP { get; init; } = 0.8f;

    /// <summary>CTRL-style repetition penalty over already-generated mel codes; 1.0 disables it. Matches the
    /// reference CLI's default (verified: <c>indextts/infer.py</c> defaults <c>repetition_penalty</c> to 10.0).</summary>
    public float RepetitionPenalty { get; init; } = 10.0f;

    /// <summary>Hard cap on generated mel-code steps; null uses the model's own <c>max_mel_tokens</c> (800 for IndexTTS-1.5).</summary>
    public int? MaxMelTokens { get; init; }

    /// <summary>Seed for the sampling RNG.</summary>
    public ulong Seed { get; init; } = 42;
}
