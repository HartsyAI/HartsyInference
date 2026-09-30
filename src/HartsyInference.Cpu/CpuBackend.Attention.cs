using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cpu;

public sealed partial class CpuBackend
{
    /// <inheritdoc />
    public void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale)
    {
        ThrowIfDisposed();
        SparseLatentAttentionReference.Apply(output, query, window, main, indices, windowSlots, sink, scale);
    }

    /// <inheritdoc />
    public void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates, float scale)
    {
        ThrowIfDisposed();
        IndexerScoresReference.Apply(scores, query, keys, headWeights, compressLens, candidates, scale);
    }

    /// <inheritdoc />
    public void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters, float eps)
    {
        ThrowIfDisposed();
        HcReference.SplitSinkhorn(pre, post, comb, mixes, scale, bias, hc, iters, eps);
    }

    /// <inheritdoc />
    public void HcPreMix(Tensor output, Tensor x, Tensor pre)
    {
        ThrowIfDisposed();
        HcReference.PreMix(output, x, pre);
    }

    /// <inheritdoc />
    public void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb)
    {
        ThrowIfDisposed();
        HcReference.PostMix(output, x, residual, post, comb);
    }

    /// <inheritdoc />
    public void QuantizeLatentRows(in LatentSource dest, Tensor rows, Tensor physicalRows)
    {
        ThrowIfDisposed();
        LatentQuantReference.QuantizeRows(dest, rows, physicalRows);
    }

    /// <inheritdoc />
    public void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding)
    {
        ThrowIfDisposed();
        LatentQuantReference.ActQuantDequantInPlace(x, encoding);
    }

    /// <inheritdoc />
    public void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos)
    {
        ThrowIfDisposed();
        WindowIndicesReference.Apply(indices, windowSize, seqLen, startPos);
    }

    /// <inheritdoc />
    public void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset)
    {
        ThrowIfDisposed();
        RopeInterleavedOffsetReference.Apply(x, cos, sin, rotaryDim, dimOffset);
    }
}
