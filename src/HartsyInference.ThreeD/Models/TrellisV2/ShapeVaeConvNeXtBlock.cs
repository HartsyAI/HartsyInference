using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Upstream <c>SparseConvNeXtBlock3d</c>: submanifold conv → affine norm → Linear·SiLU·Linear MLP → residual add.</summary>
internal sealed class ShapeVaeConvNeXtBlock : IDisposable
{
    private readonly Tensor _normW, _normB, _convB, _mlp0W, _mlp0B, _mlp2W, _mlp2B;
    private readonly Tensor[] _convSlices;   // owned: per-offset GEMM weights built from the spconv weight
    private readonly int _channels;
    private bool _disposed;

    internal ShapeVaeConvNeXtBlock(IReadOnlyDictionary<string, Tensor> w, string p, int channels)
    {
        _channels = channels;
        _normW = w[$"{p}.norm.weight"];
        _normB = w[$"{p}.norm.bias"];
        _convSlices = SparseOps.ConvWeightSlices(w[$"{p}.conv.weight"]);
        _convB = w[$"{p}.conv.bias"];
        _mlp0W = w[$"{p}.mlp.0.weight"];
        _mlp0B = w[$"{p}.mlp.0.bias"];
        _mlp2W = w[$"{p}.mlp.2.weight"];
        _mlp2B = w[$"{p}.mlp.2.bias"];
    }

    internal SparseTensor Forward(IBackend backend, SparseTensor x)
    {
        int n = x.Count;
        SparseTensor conv = SparseOps.SubmanifoldConv3dSparse(backend, x, _convSlices, _convB);
        Tensor normed = ShapeVaeOps.LayerNorm(backend, conv.Feats, _normW, _normB, n, _channels, ShapeVaeOps.BlockNormEps);
        conv.Feats.Dispose();

        Tensor hidden = ShapeVaeOps.Linear(backend, normed, _mlp0W, _mlp0B, n, _channels * ShapeVaeKeyMap.ConvNeXtMlpRatio);
        normed.Dispose();
        Tensor activated = ShapeVaeOps.Silu(backend, hidden, n, _channels * ShapeVaeKeyMap.ConvNeXtMlpRatio);
        hidden.Dispose();
        Tensor projected = ShapeVaeOps.Linear(backend, activated, _mlp2W, _mlp2B, n, _channels);
        activated.Dispose();

        Tensor sum = ShapeVaeOps.Add(backend, projected, x.Feats, n, _channels);
        projected.Dispose();
        return x.Replace(sum);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor slice in _convSlices) slice.Dispose();
    }
}
