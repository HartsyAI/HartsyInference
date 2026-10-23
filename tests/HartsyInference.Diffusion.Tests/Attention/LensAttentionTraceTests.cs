using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>Lens's joint attention, driven through the block's own forward: one head-major call that asserts nothing about F16, grants
/// BF16 storage only under <c>numerics.lensBf16</c>, and has no scaling of V or of the attention output around it.</summary>
public sealed class LensAttentionTraceTests
{
    private const int Heads = 2, HeadDim = 8, Hidden = Heads * HeadDim, Mlp = 32, Txt = 3, GridH = 2, GridW = 3;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheJointAttention_GrantsBf16OnlyWhenAsked_NeverF16_AndScalesNothing(bool bf16)
    {
        LensTransformerBlock block = new(Hidden, Heads, HeadDim, Mlp);
        Weights w = new(51);
        w.Linear("b.img_mod.1", 6 * Hidden, Hidden, bias: true);
        w.Linear("b.txt_mod.1", 6 * Hidden, Hidden, bias: true);
        foreach (string n in new[] { "img_norm1", "img_norm2", "txt_norm1", "txt_norm2" })
            w.AddNorm($"b.{n}.weight", Hidden);
        foreach (string n in new[] { "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out" })
            w.Linear($"b.attn.{n}", Hidden, Hidden, bias: true);
        w.Linear("b.attn.to_out.0", Hidden, Hidden, bias: true);
        foreach (string n in new[] { "norm_q", "norm_k", "norm_added_q", "norm_added_k" })
            w.AddNorm($"b.attn.{n}.weight", HeadDim);
        foreach (string stream in new[] { "img_mlp", "txt_mlp" })
        {
            w.Linear($"b.{stream}.w1", Mlp, Hidden);
            w.Linear($"b.{stream}.w3", Mlp, Hidden);
            w.Linear($"b.{stream}.w2", Hidden, Mlp);
        }
        block.LoadWeights(w, "b");
        using LensRope rope = new([2, 2, 4]);
        AttentionTraceBackend backend = new();
        using IDisposable knob = KnobProfile.Create("lens-bf16").With(EngineKnobs.LensBf16, bf16).Push();

        block.Forward(backend, AttentionGolden.Seeded(new TensorShape(1, GridH * GridW, Hidden), 1),
            AttentionGolden.Seeded(new TensorShape(1, Txt, Hidden), 2), AttentionGolden.Seeded(new TensorShape(1, Hidden), 3),
            rope, GridH, GridW, txtPositionStart: 2);

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(AttentionRoute.Dense, call.Route);
        Assert.Equal(bf16 ? AttentionNumerics.Bf16Storage : default, call.Numerics);
        // The legacy flag would admit the saturating F16 row, which clips Lens's V.
        Assert.False(call.Numerics.LegacyAssertedBounded);
        Assert.Equal(1f / MathF.Sqrt(HeadDim), call.Scale);
        Assert.Null(call.Mask.Bias);
        Assert.DoesNotContain("Scale", backend.Ops);
    }
}
