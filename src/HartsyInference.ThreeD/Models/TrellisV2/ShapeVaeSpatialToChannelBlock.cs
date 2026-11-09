using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>SparseResBlockS2C3d</c>: norm·SiLU·conv at full resolution, space-to-channel merge, then norm·SiLU·conv.
/// The skip is the same merge of the input, averaged down to <c>cout</c> channels.
/// </summary>
internal sealed class ShapeVaeSpatialToChannelBlock : IDisposable
{
    private readonly Tensor _norm1W, _norm1B, _conv1B, _conv2B;
    private readonly Tensor[] _conv1Slices, _conv2Slices;   // owned
    private readonly int _cin, _cout;
    private bool _disposed;

    internal ShapeVaeSpatialToChannelBlock(IReadOnlyDictionary<string, Tensor> w, string p, int cin, int cout)
    {
        _cin = cin;
        _cout = cout;
        _norm1W = w[$"{p}.norm1.weight"];
        _norm1B = w[$"{p}.norm1.bias"];
        _conv1Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv1.weight"]);
        _conv1B = w[$"{p}.conv1.bias"];
        _conv2Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv2.weight"]);
        _conv2B = w[$"{p}.conv2.bias"];
    }

    /// <summary>Downsamples <paramref name="x"/> onto the grid described by <paramref name="plan"/>, planned from <c>x.Coords</c>.</summary>
    internal SparseTensor Forward(IBackend backend, SparseTensor x, SparseSpatialChannel.Plan plan)
    {
        int n = x.Count, m = plan.Coords.Length / 4, childCh = _cout / ShapeVaeConfigRules.ChildGroup;

        Tensor normed = ShapeVaeOps.LayerNorm(backend, x.Feats, _norm1W, _norm1B, n, _cin, ShapeVaeOps.BlockNormEps);
        Tensor activated = ShapeVaeOps.Silu(backend, normed, n, _cin);
        normed.Dispose();
        SparseTensor conv1 = SparseOps.SubmanifoldConv3dSparse(backend, x.Replace(activated), _conv1Slices, _conv1B);
        activated.Dispose();

        Tensor merged = SparseSpatialChannel.SpatialToChannelFeats(backend, conv1.Feats, plan, childCh);
        conv1.Feats.Dispose();
        Tensor skipMerged = SparseSpatialChannel.SpatialToChannelFeats(backend, x.Feats, plan, _cin);
        Tensor skip = SparseSpatialChannel.MeanChannelGroups(skipMerged, _cout);
        skipMerged.Dispose();

        Tensor normed2 = ShapeVaeOps.LayerNormNoAffine(backend, merged, m, _cout, ShapeVaeOps.BlockNormEps);
        merged.Dispose();
        Tensor activated2 = ShapeVaeOps.Silu(backend, normed2, m, _cout);
        normed2.Dispose();
        SparseTensor coarse = new(activated2, plan.Coords, plan.Resolution);
        SparseTensor conv2 = SparseOps.SubmanifoldConv3dSparse(backend, coarse, _conv2Slices, _conv2B);
        activated2.Dispose();

        Tensor sum = ShapeVaeOps.Add(backend, conv2.Feats, skip, m, _cout);
        conv2.Feats.Dispose();
        skip.Dispose();
        return new SparseTensor(sum, plan.Coords, plan.Resolution);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Tensor slice in _conv1Slices) slice.Dispose();
        foreach (Tensor slice in _conv2Slices) slice.Dispose();
    }
}
