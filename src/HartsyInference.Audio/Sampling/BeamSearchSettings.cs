namespace HartsyInference.Audio.Sampling;

/// <summary>Hugging Face <c>generate()</c> arguments that <see cref="BeamSearch"/> honours, with HF's meaning.</summary>
public sealed record BeamSearchSettings
{
    /// <summary><c>num_beams</c>; at least 2 (HF decodes one beam with plain greedy or sampling, not beam search).</summary>
    public required int NumBeams { get; init; }

    /// <summary><c>max_new_tokens</c>: a beam that reaches this many generated tokens is finished.</summary>
    public required int MaxNewTokens { get; init; }

    /// <summary><c>eos_token_id</c>.</summary>
    public required int EosToken { get; init; }

    /// <summary><c>do_sample</c>: sample continuations (beam sampling) instead of taking the top ones.</summary>
    public bool DoSample { get; init; }

    /// <summary><c>temperature</c>, applied only when sampling; 1 disables it.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary><c>top_k</c>, applied only when sampling; 0 disables it.</summary>
    public int TopK { get; init; }

    /// <summary><c>top_p</c>, applied only when sampling; 1 disables it.</summary>
    public float TopP { get; init; } = 1f;

    /// <summary><c>repetition_penalty</c> over each beam's own prompt and generated tokens; 1 disables it.</summary>
    public float RepetitionPenalty { get; init; } = 1f;

    /// <summary><c>length_penalty</c>: finished scores are divided by <c>generatedLength ^ LengthPenalty</c>.</summary>
    public float LengthPenalty { get; init; } = 1f;

    /// <summary><c>min_new_tokens</c>: the EOS token is masked until this many tokens are generated.</summary>
    public int MinNewTokens { get; init; }
}
