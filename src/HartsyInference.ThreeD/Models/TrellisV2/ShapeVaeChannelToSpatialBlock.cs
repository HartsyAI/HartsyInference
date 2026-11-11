using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Upstream <c>SparseResBlockC2S3d</c>: a subdivision head (<c>pred_subdiv=True</c>) or a caller-supplied guide (<c>pred_subdiv=False</c>)
/// picks which children exist, conv1 emits <c>8·cout</c> channels that are split onto those children, and the skip repeats the
/// input's channels.
/// </summary>
internal sealed class ShapeVaeChannelToSpatialBlock : IDisposable
{
    private readonly Tensor _norm1W, _norm1B, _conv1B, _conv2B;
    private readonly Tensor? _subdivW, _subdivB;
    private readonly Tensor[] _conv1Slices, _conv2Slices;   // owned
    private readonly int _cin, _cout;
    private bool _disposed;

    internal ShapeVaeChannelToSpatialBlock(IReadOnlyDictionary<string, Tensor> w, string p, int cin, int cout, bool predictSubdivision = true)
    {
        _cin = cin;
        _cout = cout;
        _norm1W = w[$"{p}.norm1.weight"];
        _norm1B = w[$"{p}.norm1.bias"];
        _conv1Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv1.weight"]);
        _conv1B = w[$"{p}.conv1.bias"];
        _conv2Slices = SparseOps.ConvWeightSlices(w[$"{p}.conv2.weight"]);
        _conv2B = w[$"{p}.conv2.bias"];
        if (predictSubdivision)
        {
            _subdivW = w[$"{p}.to_subdiv.weight"];
            _subdivB = w[$"{p}.to_subdiv.bias"];
        }
    }

    /// <summary>Runs the block. A predicting block derives the child logits itself and hands them to <paramref name="record"/> when
    /// given; a guided block needs <paramref name="guide"/>, the logits of the same voxels from a predicting decoder.</summary>
    internal SparseTensor Forward(IBackend backend, SparseTensor x, Tensor? guide = null, ShapeVaeSubdivisions? record = null)
    {
        int n = x.Count, children = ShapeVaeConfigRules.ChildGroup;

        Tensor logits;
        bool predicted = _subdivW is not null;
        if (predicted)
        {
            logits = ShapeVaeOps.Linear(backend, x.Feats, _subdivW!, _subdivB!, n, children);
        }
        else
        {
            if (guide is null)
                throw new ArgumentException("A decoder without subdivision heads needs the guide logits of every upsample level.", nameof(guide));
            if (guide.ElementCount != (long)n * children)
                throw new ArgumentException(
                    $"Guide logits hold {guide.ElementCount} values; this level has {n} voxels and needs {(long)n * children}. "
                    + "The guide must come from a decoder run over the same coordinates.",
                    nameof(guide));
            logits = guide;
        }

        SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanChannelToSpatial(x.Coords, x.Resolution, 2, logits);
        if (predicted)
        {
            if (record is not null)
                record.Add(logits);
            else
                logits.Dispose();
        }

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
