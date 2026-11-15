namespace HartsyInference.LLM.Ernie5;

/// <summary>How <see cref="Ernie5Model"/> holds its weights and how long its sequences may be.</summary>
public sealed record Ernie5LoadOptions
{
    /// <summary>Longest sequence (prompt plus generation); sizes the rope tables and caps every sequence state.</summary>
    public required int MaxSequenceTokens { get; init; }

    /// <summary>Builds every layer once at load and keeps it; on a GPU the load also preloads each layer's per-token weights while the device
    /// has room. Weights stay mapped views of the checkpoint, but every layer keeps the tensors it decoded to F32 at build (norms, the grouped
    /// output projection, the hyper-connection mappers), so host memory grows by all layers' copies at once. When false each layer is read,
    /// dequantized and dropped on every forward, and no weight stays on the device between forwards.</summary>
    public bool ResidentWeights { get; init; }

    /// <summary>Maps and loads the multi-token-prediction layers so <see cref="Ernie5Model.MtpForward"/> can draft tokens, and gives every
    /// sequence state their caches. When false their keys are skipped (<see cref="Ernie5WeightCoverage.SkippedMtp"/>). Without
    /// <see cref="ResidentWeights"/> every draft call reads and dequantizes the whole MTP layer again, its routed experts included.</summary>
    public bool Mtp { get; init; }
}
