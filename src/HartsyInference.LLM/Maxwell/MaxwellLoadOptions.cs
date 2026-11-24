namespace HartsyInference.LLM.Maxwell;

/// <summary>How <see cref="MaxwellModel"/> holds its weights and how long its sequences may be.</summary>
public sealed record MaxwellLoadOptions
{
    /// <summary>Longest sequence (prompt plus generation); sizes the RoPE table and caps every sequence state. The host caches are F32, so at the
    /// card's 262,144 tokens they take about 45 GB across the 45 layers (K, V and the indexer keys) on top of the RoPE table.</summary>
    public required int MaxSequenceTokens { get; init; }

    /// <summary>Builds every layer once at load and keeps it. When false each layer is read and dropped on every forward, which keeps host memory low
    /// and costs a full read of the checkpoint per token.</summary>
    public bool ResidentWeights { get; init; } = true;
}
