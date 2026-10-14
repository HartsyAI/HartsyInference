namespace HartsyInference.Audio.Pipelines;

/// <summary>Generation knobs for <see cref="IndexTtsPipeline"/>. Defaults are upstream <c>infer.py</c>'s
/// <c>generate()</c> arguments: beam sampling with 3 beams, temperature 1.0, top-k 30, top-p 0.8, repetition penalty
/// 10, length penalty 0, at most 600 mel tokens.</summary>
public sealed record IndexTtsOptions
{
    /// <summary>Sampling temperature; ≤0 decodes without sampling (greedy, or greedy beam search).</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Top-k candidates kept before nucleus filtering.</summary>
    public int TopK { get; init; } = 30;

    /// <summary>Nucleus (top-p) sampling threshold, applied after top-k — matches the reference CLI's default
    /// (<c>indextts/infer.py</c>'s <c>infer()</c> defaults <c>top_p</c> to 0.8).</summary>
    public float TopP { get; init; } = 0.8f;

    /// <summary>CTRL-style repetition penalty over already-generated mel codes; 1.0 disables it. Matches the
    /// reference CLI's default (verified: <c>indextts/infer.py</c> defaults <c>repetition_penalty</c> to 10.0).</summary>
    public float RepetitionPenalty { get; init; } = 10.0f;

    /// <summary>Hard cap on generated mel-code steps; null uses upstream <c>infer()</c>'s <c>max_mel_tokens</c> (600),
    /// never more than the model's own limit.</summary>
    public int? MaxMelTokens { get; init; }

    /// <summary>HF <c>num_beams</c>: 1 decodes a single sequence, more runs <see cref="HartsyInference.Audio.Sampling.BeamSearch"/>
    /// (about that many times the GPT decode compute).</summary>
    public int NumBeams { get; init; } = 3;

    /// <summary>HF <c>do_sample</c>: sample (beam sampling when <see cref="NumBeams"/> &gt; 1) instead of decoding greedily.</summary>
    public bool DoSample { get; init; } = true;

    /// <summary>HF <c>length_penalty</c> for beam search: finished beams are scored by <c>sum / length^penalty</c>.</summary>
    public float LengthPenalty { get; init; }

    /// <summary>Seed for the sampling RNG.</summary>
    public ulong Seed { get; init; } = 42;
}
