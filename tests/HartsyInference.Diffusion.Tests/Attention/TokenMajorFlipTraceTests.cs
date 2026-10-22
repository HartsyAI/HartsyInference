using HartsyInference.Core.Backends;
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
