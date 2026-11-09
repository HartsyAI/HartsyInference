using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>SparseResBlockC2S3d</c> with <c>pred_subdiv=True</c>: a subdivision head picks which children exist,
/// conv1 emits <c>8·cout</c> channels that are split onto those children, and the skip repeats the input's channels.
/// </summary>
internal sealed class ShapeVaeChannelToSpatialBlock : IDisposable
{
    private readonly Tensor _norm1W, _norm1B, _conv1B, _conv2B, _subdivW, _subdivB;
    private readonly Tensor[] _conv1Slices, _conv2Slices;   // owned
    private readonly int _cin, _cout;
    private bool _disposed;

    internal ShapeVaeChannelToSpatialBlock(IReadOnlyDictionary<string, Tensor> w, string p, int cin, int cout)
    {
        _cin = cin;
        _cout = cout;
        _norm1W = w[$"{p}.norm1.weight"];
        _norm1B = w[$"{p}.norm1.bias"];
        _conv1Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv1.conv.weight"]);
        _conv1B = w[$"{p}.conv1.conv.bias"];
        _conv2Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv2.conv.weight"]);
        _conv2B = w[$"{p}.conv2.conv.bias"];
        _subdivW = w[$"{p}.to_subdiv.weight"];
        _subdivB = w[$"{p}.to_subdiv.bias"];
    }

    internal SparseTensor Forward(IBackend backend, SparseTensor x)
    {
        int n = x.Count, children = ShapeVaeConfigRules.ChildGroup;

        Tensor logits = ShapeVaeOps.Linear(backend, x.Feats, _subdivW, _subdivB, n, children);
        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanChannelToSpatial(x.Coords, x.Resolution, 2, logits);
        logits.Dispose();
        int m = plan.Coords.Length / 4;
        if (m == 0)
        {
            throw new InvalidOperationException("Channel-to-spatial subdivision kept no children: every to_subdiv logit is <= 0 at this level.");
        }


        Tensor normed = ShapeVaeOps.LayerNorm(backend, x.Feats, _norm1W, _norm1B, n, _cin, ShapeVaeOps.BlockNormEps);
        Tensor activated = ShapeVaeOps.Silu(backend, normed, n, _cin);
        normed.Dispose();
        SparseTensor conv1 = SparseOps.SubmanifoldConv3dSparse(backend, x.Replace(activated), _conv1Slices, _conv1B);
        activated.Dispose();

        Tensor expanded = SparseSpatialChannel.ChannelToSpatialFeats(backend, conv1.Feats, plan, _cout);
        conv1.Feats.Dispose();
        Tensor skipExpanded = SparseSpatialChannel.ChannelToSpatialFeats(backend, x.Feats, plan, _cin / children);
        Tensor skip = ShapeVaeOps.Alloc(m, _cout);
        StridedCopies.RepeatInterleave(backend, skip, skipExpanded, 2, _cout / (_cin / children));
        skipExpanded.Dispose();

        Tensor normed2 = ShapeVaeOps.LayerNormNoAffine(backend, expanded, m, _cout, ShapeVaeOps.BlockNormEps);
        expanded.Dispose();
        Tensor activated2 = ShapeVaeOps.Silu(backend, normed2, m, _cout);
        normed2.Dispose();
        SparseTensor up = new(activated2, plan.Coords, plan.Resolution);
        SparseTensor conv2 = SparseOps.SubmanifoldConv3dSparse(backend, up, _conv2Slices, _conv2B);
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
