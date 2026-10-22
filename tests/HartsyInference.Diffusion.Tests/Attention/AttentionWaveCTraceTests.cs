using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>The attention call each migrated Wave C block issues through its layer: ops, route, <c>allowF16</c>, scale and that
/// the legacy entry point can carry it. The layers come from the blocks themselves.</summary>
public sealed class AttentionWaveCTraceTests
{
    private static unsafe bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    private static void AssertDenseCall(AttentionCall call, bool allowF16, float scale)
    {
        Assert.Equal(AttentionRoute.Dense, call.Route);
        Assert.Equal(allowF16, call.Numerics.LegacyAssertedBounded);
        Assert.Equal(scale, call.Scale);
        Assert.Null(call.LegacyGap());
    }

    [Fact]
    public void SeedVr2_WindowGroup_PermutesAttendsAndPermutesBackWithoutAllowF16()
    {
        const int heads = 2, headDim = 4, g = 2, s = 3, c = heads * headDim;
        SeedVr2Config config = new()
        {
            VidDim = c, TxtInDim = 8, EmbDim = 6 * c, Heads = heads, HeadDim = headDim, MlpDim = 16, NumLayers = 2, MmLayers = 1,
            InChannels = 4, OutChannels = 4, WindowCounts = (2, 2, 2),
        };
        SeedVr2DitBlock block = new(config, 0);
        AttentionTraceBackend backend = new();
        Tensor Rows() => new(new TensorShape(g, s, c), DType.F32);
        Tensor q = Rows(), k = Rows(), v = Rows();

        Tensor attn = block.AttentionStage.AttendFlat(backend, q, k, v, new TensorShape(g, s, c), default);

        Assert.Equal(["Permute0213 3,2,4", "Permute0213 3,2,4", "Permute0213 3,2,4", "Attention Dense heads=2/2 bounded=False", "Permute0213 2,3,4"],
            backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        AssertDenseCall(call, allowF16: false, 1f / MathF.Sqrt(headDim));
        Assert.Null(call.Mask.Bias);
        Assert.Equal(new TensorShape(g, s, c), attn.Shape);
        Assert.Equal(DType.F32, attn.DType);
        Assert.All((Tensor[])[q, k, v], t => Assert.True(IsDisposed(t)));
    }

    [Fact]
    public void H3_DenseAttention_IssuesOneAttentionCallAndLeavesTheOperandsToTheCaller()
    {
        const int heads = 2, headDim = 16, seq = 5;
        MiniMaxH3Transformer dit = new(new MiniMaxH3Config { NumAttentionHeads = heads, AttentionHeadDim = headDim });
        AttentionTraceBackend backend = new();
        Tensor Headed() => new(new TensorShape(1, heads, seq, headDim), DType.F32);
        Tensor q = Headed(), k = Headed(), v = Headed();

        Tensor attn = dit.AttentionStage.AttendHeadMajorOperands(backend, q, k, v, default, OperandOwnership.None);

        Assert.Equal(["Attention Dense heads=2/2 bounded=False"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        AssertDenseCall(call, allowF16: false, 1f / MathF.Sqrt(headDim));
        Assert.Null(call.Mask.Bias);
        Assert.Equal(new TensorShape(1, heads, seq, headDim), attn.Shape);
        Assert.Equal(DType.F32, attn.DType);
        Assert.All((Tensor[])[q, k, v], t => Assert.False(IsDisposed(t)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Oasis_SpatialAndTemporalAttention_IssueOneBoundedCallWithTheirBias(bool temporal)
    {
        const int dim = 32, heads = 4, headDim = dim / heads, frames = 3, sp = 2;
        OasisSpatioTemporalBlock block = new(new OasisDitConfig { HiddenSize = dim, NumHeads = heads });
        AttentionTraceBackend backend = new();
        int batch = temporal ? sp : frames, seq = temporal ? frames : sp;
        Tensor Headed() => new(new TensorShape(batch, heads, seq, headDim), DType.F32);
        Tensor q = Headed(), k = Headed(), v = Headed();
        Tensor? mask = temporal ? new Tensor(new TensorShape(1, 1, frames, frames), DType.F32) : null;

        Tensor attn = block.AttentionStage.AttendHeadMajorOperands(backend, q, k, v, new AttentionCallContext { Bias = mask });

        Assert.Equal(["Attention Dense heads=4/4 bounded=True"], backend.Ops);
        AttentionCall call = Assert.Single(backend.Calls);
        AssertDenseCall(call, allowF16: true, 1f / MathF.Sqrt(headDim));
        Assert.Same(mask, call.Mask.Bias);
        Assert.Equal(new TensorShape(batch, heads, seq, headDim), attn.Shape);
        Assert.Equal(DType.F32, attn.DType);
        Assert.All((Tensor[])[q, k, v], t => Assert.True(IsDisposed(t)));
    }
}
