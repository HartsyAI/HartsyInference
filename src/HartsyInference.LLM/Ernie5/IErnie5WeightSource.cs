using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Where <see cref="Ernie5Model"/> reads its weights: a checkpoint on disk (<see cref="Ernie5CheckpointWeights"/>) or any other
/// store with the checkpoint's key names.</summary>
public interface IErnie5WeightSource
{
    /// <summary>Every logical weight name; an FP8 weight's <c>_scale_inv</c> companion is folded into its weight and not listed.</summary>
    IEnumerable<string> Names { get; }

    /// <summary>The weight as a tensor the caller disposes, in a dtype <c>IBackend.Linear</c> takes: an FP8 block weight dequantized to F32
    /// with its scale, any float weight in its stored dtype. It may be a view of storage the source keeps alive.</summary>
    /// <exception cref="KeyNotFoundException">The source has no weight <paramref name="name"/>.</exception>
    Tensor Read(string name);
}
