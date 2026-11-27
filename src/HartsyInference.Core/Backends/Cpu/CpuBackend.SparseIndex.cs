using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu;

public sealed partial class CpuBackend
{
    /// <inheritdoc />
    public void SparseTopK(Tensor indices, Tensor scores, Tensor limits, int topK, int offset)
    {
        ThrowIfDisposed();
        SparseTopKReference.Apply(indices, scores, limits, topK, offset);
    }

    /// <inheritdoc />
    public void MergeWindowTopK(Tensor rows, Tensor topk, Tensor positions, int window)
    {
        ThrowIfDisposed();
        WindowTopKMergeReference.Apply(rows, topk, positions, window);
    }
}
