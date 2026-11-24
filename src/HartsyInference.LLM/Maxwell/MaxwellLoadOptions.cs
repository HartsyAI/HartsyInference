namespace HartsyInference.LLM.Maxwell;

/// <summary>How <see cref="MaxwellModel"/> holds its weights and how long its sequences may be.</summary>
public sealed record MaxwellLoadOptions
{
    /// <summary>Longest sequence (prompt plus generation); sizes the RoPE table and caps every sequence state.</summary>
    public required int MaxSequenceTokens { get; init; }

    /// <summary>Builds every layer once at load and keeps it. When false each layer is read and dropped on every forward, which keeps host memory low
    /// and costs a full read of the checkpoint per token.</summary>
    public bool ResidentWeights { get; init; }
}
