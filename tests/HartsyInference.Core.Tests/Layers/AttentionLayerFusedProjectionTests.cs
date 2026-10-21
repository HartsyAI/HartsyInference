using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The fused <c>[q|k|v]</c> projections: one Linear, then the split (with the per-head norm in the same op for
/// the split-norm kind), recorded without computing.</summary>
public sealed class AttentionLayerFusedProjectionTests
{
    private const int Heads = 2, HeadDim = 4, Inner = Heads * HeadDim;

    private static Tensor W(int rows) => new(new TensorShape(rows, Inner), DType.F32);

    private static AttentionLayerConfig Config(AttentionProjection projection) => new()
    {
        Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, Projection = projection, QkNorm = AttentionQkNorm.PerHead,
    };

    private static AttentionLayerWeights Weights(float keyEps = 1e-6f) => new()
    {
        Qkv = W(110), Output = W(104), QueryNorm = W(106), KeyNorm = W(107), QueryNormEps = 1e-6f, KeyNormEps = keyEps,
    };

    [Fact]
    public void SplitNorm_RunsOneLinearAndOneSplitNorm_AndNoSeparateNorm()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Config(AttentionProjection.FusedSplitNorm), Weights());
        Tensor x = new(new TensorShape(1, 5, Inner), DType.F32);

        layer.Forward(backend, x, x, new AttentionCallContext { ReleaseInputs = true });

        Assert.Equal(["Linear w110", "QkvSplitNorm w106,w107 [1, 5, 24]", "Permute0213 5,2,4"], backend.Ops.Take(3));
        Assert.DoesNotContain(backend.Ops, op => op.StartsWith("RmsNorm", StringComparison.Ordinal));
        Assert.Equal("Linear w104", backend.Ops[^1]);
    }

    [Fact]
    public void Slice_SplitsThenNormsEachHead()
    {
        AttentionTraceBackend backend = new();
        AttentionLayer layer = new(Config(AttentionProjection.FusedSlice), Weights());
        Tensor x = new(new TensorShape(1, 5, Inner), DType.F32);

        (Tensor q, Tensor k, Tensor v) = layer.ProjectSelf(backend, x, releaseInput: false);
        layer.NormalizeQk(backend, ref q, ref k);

        Assert.Equal(["Linear w110", "SliceLastDim @0", "SliceLastDim @8", "SliceLastDim @16", "RmsNorm w106", "RmsNorm w107"],
            backend.Ops);
        Assert.Equal(new TensorShape(1, 5, Heads, HeadDim), v.Shape);
    }

    [Fact]
    public void FusedProjections_AreRefusedWhereTheyCannotRun()
    {
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(AttentionProjection.FusedSplitNorm) with { KvHeads = 1 }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(AttentionProjection.FusedSplitNorm) with
        {
            QkNorm = AttentionQkNorm.None,
        }));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(AttentionProjection.FusedSplitNorm), Weights(keyEps: 1e-5f)));
        Assert.Throws<ArgumentException>(() => new AttentionLayer(Config(AttentionProjection.FusedSlice), Weights() with { Qkv = null }));
        AttentionLayer layer = new(Config(AttentionProjection.FusedSlice), Weights());
        Tensor q = new(new TensorShape(1, 5, Inner), DType.F32), kv = new(new TensorShape(1, 3, Inner), DType.F32);
        Assert.Throws<NotSupportedException>(() => layer.Forward(new AttentionTraceBackend(), q, kv, default));
    }
}
