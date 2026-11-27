using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>One encoder block: the fused q, k and v split into three <c>[QkvHidden, Hidden]</c> projections, the output projection, two RMS norms and the GELU feed-forward.</summary>
public sealed record KimiVisionBlockWeights(Tensor Q, Tensor K, Tensor V, Tensor Out, Tensor Norm0, Tensor Norm1, Tensor Fc0, Tensor Fc1);
