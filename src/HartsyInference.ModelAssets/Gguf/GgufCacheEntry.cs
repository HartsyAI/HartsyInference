using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>One tensor of a disk quant cache (<see cref="GgufQuantCache"/>): the engine-order shape and dtype it is stored as,
/// and how to produce its data when the writer reaches it.</summary>
/// <param name="Name">Tensor name in the cache.</param>
/// <param name="Shape">Engine-order shape; the read-back restores it.</param>
/// <param name="DType">Stored dtype.</param>
/// <param name="Produce">Returns a tensor of <paramref name="Shape"/> and <paramref name="DType"/> the writer owns and disposes
/// (see <see cref="GgufWriter.AddTensor(string, TensorShape, DType, Func{Tensor})"/>).</param>
public sealed record GgufCacheEntry(string Name, TensorShape Shape, DType DType, Func<Tensor> Produce);
