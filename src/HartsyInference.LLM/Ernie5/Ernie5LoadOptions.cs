namespace HartsyInference.LLM.Ernie5;

/// <summary>How <see cref="Ernie5Model"/> holds its weights and how long its sequences may be.</summary>
public sealed record Ernie5LoadOptions
{
    /// <summary>Longest sequence (prompt plus generation); sizes the rope tables and caps every sequence state.</summary>
    public required int MaxSequenceTokens { get; init; }

    /// <summary>Builds every layer once at load and keeps it. When false each layer is read, dequantized and dropped on every forward,
    /// which bounds host memory to one layer's F32 weights at the cost of re-reading them per call.</summary>
    public bool ResidentWeights { get; init; }
}
