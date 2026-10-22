using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>The attention configs the Wan block, the S2V audio injector and the Matrix-Game 3 action module build, which the CPU
/// goldens cannot see: <c>allowF16</c>, the compute dtype, the dense route and the QK norm. The block's F16 stream attends through
/// F16 twins of its self and cross stages; the image branch has none, since an image context keeps the stream F32.</summary>
public sealed class WanAttentionConfigPinTests
{
    private const int Heads = 2, HeadDim = 16, Dim = Heads * HeadDim;

    private static Tensor T(params long[] shape) => new(new TensorShape(shape), DType.F32);

    private static WanVideoBlock Block(bool image)
    {
        Dictionary<string, Tensor> w = new() { ["b.scale_shift_table"] = T(1, 6, Dim) };
        foreach (string a in new[] { "attn1", "attn2" })
        {
            foreach (string p in new[] { "to_q", "to_k", "to_v", "to_out.0" }) w[$"b.{a}.{p}.weight"] = T(Dim, Dim);
            w[$"b.{a}.norm_q.weight"] = T(Dim);
            w[$"b.{a}.norm_k.weight"] = T(Dim);
        }
        if (image)
        {
            w["b.attn2.add_k_proj.weight"] = T(Dim, Dim);
            w["b.attn2.add_v_proj.weight"] = T(Dim, Dim);
            w["b.attn2.norm_added_k.weight"] = T(Dim);
        }
        w["b.ffn.net.0.proj.weight"] = T(64, Dim);
        w["b.ffn.net.2.weight"] = T(Dim, 64);
        WanVideoBlock block = new(new WanVideoConfig { NumHeads = Heads, HeadDim = HeadDim, FfnDim = 64 }, crossAttnNorm: false);
        block.LoadWeights(w, "b");
        return block;
    }

    private static void AssertWanConfig(AttentionLayerConfig? config, DType compute)
    {
        Assert.NotNull(config);
        Assert.Equal(Heads, config.Heads);
        Assert.Equal(Heads, config.KvHeads);
        Assert.Equal(HeadDim, config.HeadDim);
        Assert.Equal(AttentionQkNorm.FullWidthRms, config.QkNorm);
        Assert.Equal(AttentionProjection.PerWeight, config.Projection);
        Assert.Equal(AttentionRoute.Dense, config.TokenMajorRoute);
        Assert.Equal(compute, config.ComputeDType);
        Assert.True(config.AllowF16);
        Assert.Equal(0f, config.Scale);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheWanBlockAttendsDenseWithAllowF16InF32AndThroughF16Twins(bool image)
    {
        WanVideoBlock block = Block(image);

        AssertWanConfig(block.SelfAttentionStage?.Config, DType.F32);
        AssertWanConfig(block.CrossAttentionStage?.Config, DType.F32);
        AssertWanConfig(block.SelfAttentionF16Stage?.Config, DType.F16);
        AssertWanConfig(block.CrossAttentionF16Stage?.Config, DType.F16);
        Assert.Equal(image, block.ImageAttentionStage is not null);
        if (image) AssertWanConfig(block.ImageAttentionStage?.Config, DType.F32);
    }

    [Fact]
    public void TheS2VInjectorKeepsItsDefaultFalseAllowF16()
    {
        AttentionLayerConfig config = new WanS2VAudioInjector(1, Dim, Heads, HeadDim).AttentionStage.Config;

        Assert.False(config.AllowF16);
        Assert.Equal(AttentionQkNorm.None, config.QkNorm);
        Assert.Equal(AttentionRoute.Dense, config.TokenMajorRoute);
        Assert.Equal(DType.F32, config.ComputeDType);
        Assert.Equal(Heads, config.Heads);
        Assert.Equal(HeadDim, config.HeadDim);
    }

    [Fact]
    public void TheMatrixGame3ActionModulePinsAllowF16()
    {
        AttentionLayerConfig config = new MatrixGame3ActionModule(16, headsNum: 4, streamHiddenDim: 32, ropeTDim: 4, ropeHDim: 2, ropeWDim: 2).AttentionStage.Config;

        Assert.True(config.AllowF16);
        Assert.Equal(AttentionQkNorm.None, config.QkNorm);
        Assert.Equal(AttentionRoute.Dense, config.TokenMajorRoute);
        Assert.Equal(DType.F32, config.ComputeDType);
        Assert.Equal(4, config.Heads);
        Assert.Equal(8, config.HeadDim);
    }
}
