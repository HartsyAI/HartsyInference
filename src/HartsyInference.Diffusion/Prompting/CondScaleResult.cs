using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>What <see cref="CondTokenWeights"/> produced: replacement tensors the caller now OWNS and must dispose.
/// A null member means the corresponding input was left untouched and must keep being used as-is — the originals are
/// never mutated, because several pipelines cache conditioning under a token-id key that weighting does not change.</summary>
public readonly record struct CondScaleResult(Tensor? Cond, Tensor? Pooled);
