using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi;

/// <summary>A weight source that reads a row or column range of a tensor without reading the rest of it, so a rank loads only its share.</summary>
public interface IKimiK31RankWeightSource : IKimiK31WeightSource
{
    /// <summary>The range <c>[start, start + count)</c> along <paramref name="dim"/> of the tensor, as a new tensor the caller disposes, in the
    /// stored dtype. Only the bytes of that range are read.</summary>
    /// <param name="dim">0 for rows; 1 for columns of a rank-2 tensor.</param>
    /// <exception cref="KeyNotFoundException">The source has no tensor <paramref name="name"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range does not lie inside the tensor, or <paramref name="dim"/> is not a dimension of it.</exception>
    Tensor ReadSlice(string name, int dim, long start, long count);
}
