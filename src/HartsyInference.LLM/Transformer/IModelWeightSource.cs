using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Where a model reads its weights by checkpoint key: a checkpoint on disk (<see cref="BlockFp8CheckpointWeights"/>) or any other store
/// with the checkpoint's key names.</summary>
public interface IModelWeightSource
{
    /// <summary>Every logical weight name; an FP8 weight's <c>_scale_inv</c> companion is folded into its weight and not listed.</summary>
    IEnumerable<string> Names { get; }

    /// <summary>The weight as a tensor the caller disposes, in a dtype <c>IBackend.Linear</c> takes: an FP8 block weight stays E4M3 with its scale
    /// grid on <see cref="Tensor.QuantInfo"/>, any float weight is in its stored dtype. It may be a view of storage the source keeps alive.</summary>
    /// <exception cref="KeyNotFoundException">The source has no weight <paramref name="name"/>.</exception>
    Tensor Read(string name);
}
