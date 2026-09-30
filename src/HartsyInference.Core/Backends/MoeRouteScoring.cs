namespace HartsyInference.Core.Backends;

/// <summary>How <see cref="IBackend.MoeRoute"/> turns router logits into per-expert scores.</summary>
public enum MoeRouteScoring
{
    /// <summary>Softmax over all experts (OLMoE, Qwen, Mixtral).</summary>
    Softmax,

    /// <summary>Independent logistic per expert (DeepSeek-V3).</summary>
    Sigmoid,

    /// <summary><c>sqrt(softplus(x))</c> per expert (DeepSeek-V4.1).</summary>
    SqrtSoftplus,
}
