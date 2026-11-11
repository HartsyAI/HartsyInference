namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Skeleton generation parameters; defaults are upstream's <c>ar_inference_articulationxl</c> values.</summary>
public sealed record UniRigSamplingOptions
{
    /// <summary>Cap on generated tokens (BOS and class excluded).</summary>
    public int MaxNewTokens { get; init; } = 2048;

    public float Temperature { get; init; } = 1.5f;

    public int TopK { get; init; } = 5;

    public float TopP { get; init; } = 0.95f;

    /// <summary>Hugging Face repetition penalty over generated tokens; applied to log-probabilities.</summary>
    public float RepetitionPenalty { get; init; } = 3.0f;

    /// <summary>Seed of the draw and of the encoder's point subset.</summary>
    public int Seed { get; init; } = 12345;

    /// <summary>Whole-run retries with the next seed when a draw does not end in a decodable skeleton.</summary>
    public int MaxAttempts { get; init; } = 4;
}
