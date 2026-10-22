using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Diffusion.Tests.Attention;

/// <summary>The sites the Phase 3 numerical step moves onto the token-major entry, driven through their own forward passes:
/// each issues one token-major attention call and no head permute, and the configurations that keep head-major still do.</summary>
public sealed class TokenMajorFlipTraceTests
{
    // Single-stream geometry: head dim 64 is one the Sage row takes; 3 text rows ahead of a 2x3 image grid.
    private const int Heads = 2, HeadDim = 64, Hidden = Heads * HeadDim, Mlp = 64, Txt = 3, GridH = 2, GridW = 3, Seq = Txt + GridH * GridW;

    private static int Permutes(AttentionTraceBackend backend) =>
        backend.Ops.Count(op => op.StartsWith("Permute0213", StringComparison.Ordinal));

    [Theory]
    [InlineData(true, AttentionRoute.TokenMajor, 0)]
    [InlineData(false, AttentionRoute.Dense, 4)]
    public void ZImageBlock_TurboAttendsTokenMajor_BaseStaysHeadMajor(bool turbo, AttentionRoute route, int permutes)
    {
        const int hidden = 32, heads = 4, ffn = 48, embed = 16, seq = 9;
        ZImageBlock block = new(hidden, heads, ffn, 1e-5f, useF16SandwichDamp: false, allowF16Attention: turbo);
        Weights w = new(41);
        w.Add("b.adaLN_modulation.0.weight", 4 * hidden, embed);
        w.Add("b.attention.qkv.weight", 3 * hidden, hidden);
        w.Add("b.attention.out.weight", hidden, hidden);
        w.AddNorm("b.attention.q_norm.weight", hidden / heads);
        w.AddNorm("b.attention.k_norm.weight", hidden / heads);
        foreach (string n in new[] { "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" })
            w.AddNorm($"b.{n}.weight", hidden);
        w.Add("b.feed_forward.w1.weight", ffn, hidden);
        w.Add("b.feed_forward.w2.weight", hidden, ffn);
        w.Add("b.feed_forward.w3.weight", ffn, hidden);
        block.LoadWeights(w, "b");
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };

        block.Forward(backend, new Tensor(new TensorShape(1, seq, hidden), DType.F32), new Tensor(new TensorShape(1, embed), DType.F32),
            rope: null);

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(route, call.Route);
        Assert.Equal(turbo, call.Numerics.LegacyAssertedBounded);
        Assert.Equal(permutes, Permutes(backend));
    }

    [Theory]
    [InlineData(2, AttentionRoute.TokenMajorGqa)]
    [InlineData(4, AttentionRoute.TokenMajor)]
    public void Krea2Attention_GroupedHeadsKeepTheGroupedEntry_EqualHeadsTakeThePlainOne(int kvHeads, AttentionRoute route)
    {
        Krea2Attention attention = new(16, 4, kvHeads);
        Weights w = new(42);
        AddKrea2Attention(w, "a", 16, 4, kvHeads);
        attention.LoadWeights(w, "a");
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true, SupportsTokenMajorGqaAttention = true };

        attention.Forward(backend, new Tensor(new TensorShape(1, 6, 16), DType.F16), null, 1, 6);

        Assert.Equal(route, Assert.Single(backend.Calls).Route);
        Assert.Equal(0, Permutes(backend));
    }

    [Fact]
    public void Krea2TextFusion_RefinerAttendsTokenMajor_LayerAxisBlocksStayHeadMajor()
    {
        const int dim = 16, heads = 2, layers = 3, seq = 4, inter = 24;
        Krea2TextFusion fusion = new(layers, dim, heads, heads, inter, numLayerwise: 1, numRefiner: 1, eps: 1e-5f);
        Weights w = new(43);
        foreach (string b in new[] { "f.layerwise_blocks.0", "f.refiner_blocks.0" })
        {
            w.AddNorm($"{b}.norm1.weight", dim, zeroCentered: true);
            w.AddNorm($"{b}.norm2.weight", dim, zeroCentered: true);
            AddKrea2Attention(w, $"{b}.attn", dim, heads, heads);
            w.Add($"{b}.ff.gate.weight", inter, dim);
            w.Add($"{b}.ff.up.weight", inter, dim);
            w.Add($"{b}.ff.down.weight", dim, inter);
        }
        w.Add("f.projector.weight", 1, layers);
        fusion.LoadWeights(w, "f");
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true, SupportsTokenMajorGqaAttention = true };

        fusion.Forward(backend, new Tensor(new TensorShape(1, seq, layers, dim), DType.F32), 1, seq);

        // The layer-axis block attends one sequence per token, so it keeps the head-major entry and its four permutes.
        Assert.Equal([AttentionRoute.Dense, AttentionRoute.TokenMajor], backend.Calls.Select(c => c.Route));
        Assert.Equal(4, Permutes(backend));
    }

    [Theory]
    [InlineData(1, AttentionRoute.TokenMajor, 0)]
    [InlineData(2, AttentionRoute.Dense, 4)]
    public void UNetSelfAttention_TakesTheTokenMajorEntryAtOneBatchItem(int batch, AttentionRoute route, int permutes)
    {
        const int channels = 16, heads = 2, seq = 6;
        TransformerSubBlock self = SubBlock(channels, heads, channels);
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };
        Tensor hidden = new(new TensorShape(batch, seq, channels), DType.F32);

        self.Forward(backend, hidden, hidden);

        Assert.Equal(route, Assert.Single(backend.Calls).Route);
        Assert.Equal(permutes, Permutes(backend));
    }

    [Fact]
    public void UNetCrossAttention_StaysHeadMajorOverItsCachedKeysAndValues()
    {
        const int channels = 16, heads = 2, contextDim = 12;
        TransformerSubBlock cross = SubBlock(channels, heads, contextDim);
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true };

        cross.Forward(backend, new Tensor(new TensorShape(1, 6, channels), DType.F32), new Tensor(new TensorShape(1, 5, contextDim), DType.F32));

        Assert.Equal(AttentionRoute.Dense, Assert.Single(backend.Calls).Route);
        Assert.Equal(4, Permutes(backend));
    }

    /// <summary>The single-stream blocks attend through <c>Attend</c>: one batch item takes the token-major entry and writes the
    /// caller's <c>[1, S, hidden]</c>. A batch, and an F16 call the Sage row would take (the knob lowered to this sequence),
    /// keep the head-major call and its four permutes; a bias rules Sage out, so Chroma's masked call still flips.</summary>
    [Theory]
    [InlineData("flux", 1, false, false, AttentionRoute.TokenMajor)]
    [InlineData("flux", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("flux", 1, true, true, AttentionRoute.Dense)]
    [InlineData("flux", 2, false, false, AttentionRoute.Dense)]
    [InlineData("chroma", 1, false, false, AttentionRoute.TokenMajor)]
    [InlineData("chroma", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("chroma", 1, true, true, AttentionRoute.Dense)]
    [InlineData("chroma", 2, false, false, AttentionRoute.Dense)]
    [InlineData("chroma-fused", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("chroma-masked", 1, true, true, AttentionRoute.TokenMajor)]
    [InlineData("chroma-masked", 2, false, false, AttentionRoute.Dense)]
    [InlineData("flux2", 1, false, false, AttentionRoute.TokenMajor)]
    [InlineData("flux2", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("flux2", 1, true, true, AttentionRoute.Dense)]
    [InlineData("flux2", 2, false, false, AttentionRoute.Dense)]
    [InlineData("flux3", 1, false, false, AttentionRoute.TokenMajor)]
    [InlineData("flux3", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("flux3", 1, true, true, AttentionRoute.Dense)]
    [InlineData("hunyuanimage", 1, false, false, AttentionRoute.TokenMajor)]
    [InlineData("hunyuanimage", 1, true, false, AttentionRoute.TokenMajor)]
    [InlineData("hunyuanimage", 1, true, true, AttentionRoute.Dense)]
    [InlineData("hunyuanimage", 2, false, false, AttentionRoute.Dense)]
    public void SingleStreamBlocks_AttendTokenMajorAtOneBatchItem(string site, int batch, bool f16, bool sageLength, AttentionRoute route)
    {
        AttentionTraceBackend backend = new() { SupportsTokenMajorAttention = true, SkipAdaLayerNorm = true };
        try
        {
            if (sageLength) KnobStore.Set(EngineKnobs.SageF16MinSkv, Seq);
            RunSingleStream(backend, site, batch, f16 ? DType.F16 : DType.F32);
        }
        catch (NotSupportedException) when (backend.Calls.Count == 1)
        {
            // The F16 gated residual past the attention has no stub body; the attention is recorded by then.
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageF16MinSkv);
        }

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Equal(route, call.Route);
        Assert.True(call.Numerics.LegacyAssertedBounded);
        Assert.Equal(site == "chroma-masked", call.Mask.Bias is not null);
        Assert.Equal(route == AttentionRoute.Dense ? 4 : 0, Permutes(backend));
        if (route == AttentionRoute.TokenMajor) Assert.Equal(new TensorShape(1, Seq, Hidden), call.Output.Storage.Shape);
    }

    private static void RunSingleStream(AttentionTraceBackend backend, string site, int batch, DType act)
    {
        Tensor Rows(int rows) => new(new TensorShape(batch, rows, Hidden), act);
        Tensor Mod() => new(new TensorShape(batch, Hidden), DType.F32);
        Weights w = new(45);
        switch (site)
        {
            case "flux":
            {
                FluxSingleStreamBlock block = new(Hidden, Heads, Mlp);
                w.Linear("b.norm.linear", 3 * Hidden, Hidden, true);
                AddPerHeadNorms(w, "b.attn.norm_q", "b.attn.norm_k");
                foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, true);
                w.Linear("b.proj_mlp", Mlp, Hidden, true);
                w.Linear("b.proj_out", Hidden, Hidden + Mlp, true);
                block.LoadWeights(w, "b");
                using FluxRope rope = Rope(FluxRope.BuildPositionIds(Txt, GridH, GridW), [16, 24, 24]);
                block.Forward(backend, Rows(Seq), Mod(), rope);
                break;
            }
            case "chroma" or "chroma-fused" or "chroma-masked":
            {
                const int chromaMlp = 4 * Hidden;
                ChromaSingleStreamBlock block = new(Hidden, Heads, HeadDim);
                AddPerHeadNorms(w, "b.attn.norm_q", "b.attn.norm_k");
                if (site == "chroma-fused")
                {
                    w.Linear("b.linear1", 3 * Hidden + chromaMlp, Hidden, true);
                }
                else
                {
                    foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, true);
                    w.Linear("b.proj_mlp", chromaMlp, Hidden, true);
                }
                w.Linear("b.proj_out", Hidden, Hidden + chromaMlp, true);
                block.LoadWeights(w, "b");
                using FluxRope rope = Rope(FluxRope.BuildPositionIds(Txt, GridH, GridW), [16, 24, 24]);
                Tensor? mask = site == "chroma-masked" ? new Tensor(new TensorShape(batch, 1, Seq, Seq), DType.F32) : null;
                block.Forward(backend, Rows(Seq), new Tensor(new TensorShape(batch, 3, Hidden), DType.F32), rope, mask);
                break;
            }
            case "flux2":
            {
                Flux2SingleBlock block = new(Hidden, Heads, Mlp);
                foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden);
                w.Linear("b.attn.linear_in_gate", Mlp, Hidden);
                w.Linear("b.attn.linear_in_up", Mlp, Hidden);
                w.Linear("b.attn.to_out", Hidden, Hidden + Mlp);
                AddPerHeadNorms(w, "b.attn.norm_q", "b.attn.norm_k");
                block.LoadWeights(w, "b");
                using FluxRope rope = Rope(Flux2PosEmbed.BuildPositionIds(Txt, GridH, GridW), [16, 16, 16, 16]);
                block.Forward(backend, Rows(Seq), [Mod(), Mod(), Mod()], rope);
                break;
            }
            case "flux3":
            {
                Flux3Block block = new(Hidden, Heads, Mlp, 1e-6f, 1e-6f);
                foreach (string n in new[] { "q_proj", "k_proj", "v_proj" }) w.Linear($"b.{n}", Hidden, Hidden);
                w.Linear("b.mlp_in", 2 * Mlp, Hidden);
                w.Linear("b.attn_out", Hidden, Hidden);
                w.Linear("b.mlp_out", Hidden, Mlp);
                AddPerHeadNorms(w, "b.norm.query_norm", "b.norm.key_norm");
                block.LoadWeights(w, "b");
                Tensor text = Flux3PosEmbed.TextIds(Txt), image = Flux3PosEmbed.ImageIds(GridH, GridW, 0f, 1f);
                using FluxRope rope = Rope(Flux3PosEmbed.Concat(text, image), [16, 16, 16, 16]);
                text.Dispose();
                image.Dispose();
                using Flux3Modulation modulation = Flux3Modulation.Uniform(Mod(), Mod(), Mod());
                block.Forward(backend, Rows(Seq), modulation, rope);
                break;
            }
            case "hunyuanimage":
            {
                HunyuanImageSingleBlock block = new(Hidden, Heads, HeadDim, Mlp);
                w.Linear("b.norm.linear", 3 * Hidden, Hidden, true);
                foreach (string n in new[] { "to_q", "to_k", "to_v" }) w.Linear($"b.attn.{n}", Hidden, Hidden, true);
                AddPerHeadNorms(w, "b.attn.norm_q", "b.attn.norm_k");
                w.Linear("b.proj_mlp", Mlp, Hidden, true);
                w.Linear("b.proj_out", Hidden, Hidden + Mlp, true);
                block.LoadWeights(w, "b");
                using HunyuanImageRope rope = new([32, 32]);
                block.Forward(backend, Rows(GridH * GridW), Rows(Txt), Mod(), rope, GridH, GridW);
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(site));
        }
    }

    private static FluxRope Rope(Tensor ids, int[] axes)
    {
        FluxRope rope = new(axes);
        rope.Precompute(ids);
        ids.Dispose();
        return rope;
    }

    private static void AddPerHeadNorms(Weights w, string query, string key)
    {
        w.AddNorm($"{query}.weight", HeadDim);
        w.AddNorm($"{key}.weight", HeadDim);
    }

    private static TransformerSubBlock SubBlock(int channels, int heads, int contextDim)
    {
        TransformerSubBlock block = new(channels, heads, contextDim);
        Weights w = new(44);
        w.AddNorm("n.weight", channels);
        w.AddBias("n.bias", channels);
        w.Add("a.to_q.weight", channels, channels);
        w.Add("a.to_k.weight", channels, contextDim);
        w.Add("a.to_v.weight", channels, contextDim);
        w.Add("a.to_out.0.weight", channels, channels);
        w.AddBias("a.to_out.0.bias", channels);
        block.LoadWeights(w, "a", "n");
        return block;
    }

    private static void AddKrea2Attention(Weights w, string p, int hidden, int heads, int kvHeads)
    {
        int headDim = hidden / heads;
        w.Add($"{p}.to_q.weight", hidden, hidden);
        w.Add($"{p}.to_k.weight", kvHeads * headDim, hidden);
        w.Add($"{p}.to_v.weight", kvHeads * headDim, hidden);
        w.Add($"{p}.to_gate.weight", hidden, hidden);
        w.Add($"{p}.to_out.0.weight", hidden, hidden);
        w.AddNorm($"{p}.norm_q.weight", headDim, zeroCentered: true);
        w.AddNorm($"{p}.norm_k.weight", headDim, zeroCentered: true);
    }
}
