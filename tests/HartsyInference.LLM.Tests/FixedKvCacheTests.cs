using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The fixed-capacity in-place KV cache must produce identical transformer output to the Concat-grown
/// <see cref="KvCache"/> (it's the O(n²)→O(n) / bounded-VRAM replacement, so it must be behavior-preserving).</summary>
public sealed unsafe class FixedKvCacheTests
{
    private static uint _rng = 0x2BadCafeu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Embeds(int t, int h) => Fill(new Tensor(new TensorShape(1, t, h), DType.F32));

    [Fact]
    public void FixedKvCache_MatchesConcatKvCache()
    {
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
        };
        Dictionary<string, Tensor> w = Weights(cfg);

        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        using KvCache concat = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        using FixedKvCache @fixed = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxSequenceLength: 16);

        // Prefill (t=4) then two decode steps, identical inputs into each cache.
        using (Tensor e = Embeds(4, cfg.HiddenSize))
        using (Tensor a = model.ForwardEmbeds(backend, e, 4, 0, concat))
        using (Tensor b = model.ForwardEmbeds(backend, e, 4, 0, @fixed))
            AssertClose(a, b, "prefill");
        Assert.Equal(concat.CurrentLength, @fixed.CurrentLength);

        for (int step = 0; step < 2; step++)
        {
            using Tensor e = Embeds(1, cfg.HiddenSize);
            using Tensor a = model.ForwardEmbeds(backend, e, 1, concat.CurrentLength, concat);
            using Tensor b = model.ForwardEmbeds(backend, e, 1, @fixed.CurrentLength, @fixed);
            AssertClose(a, b, $"decode-{step}");
            Assert.Equal(concat.CurrentLength, @fixed.CurrentLength);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>A cache grown chunk by chunk must be indistinguishable from one allocated at the cap: the growth
    /// copy runs mid-sequence and a prefix it drops or misplaces produces plausible-looking output, not a crash.</summary>
    [Fact]
    public void GrownCache_MatchesPreallocatedCache()
    {
        TransformerConfig cfg = new()
        {
            HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
            IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
        };
        Dictionary<string, Tensor> w = Weights(cfg);

        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");

        const int cap = 32;
        using FixedKvCache prealloc = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxSequenceLength: cap);
        using FixedKvCache grown = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxSequenceLength: cap, kvDtype: null, growthChunk: 3);

        using (Tensor e = Embeds(4, cfg.HiddenSize))
        using (Tensor a = model.ForwardEmbeds(backend, e, 4, 0, prealloc))
        using (Tensor b = model.ForwardEmbeds(backend, e, 4, 0, grown))
            AssertExact(a, b, "prefill");

        for (int step = 0; step < 12; step++)
        {
            using Tensor e = Embeds(1, cfg.HiddenSize);
            using Tensor a = model.ForwardEmbeds(backend, e, 1, prealloc.CurrentLength, prealloc);
            using Tensor b = model.ForwardEmbeds(backend, e, 1, grown.CurrentLength, grown);
            AssertExact(a, b, $"decode-{step}");
            Assert.Equal(prealloc.CurrentLength, grown.CurrentLength);
        }

        Assert.Equal(cap, prealloc.LayerCapacity(0));
        Assert.Equal(0, prealloc.GrowthEpoch);
        Assert.Equal(18, grown.LayerCapacity(0));
        Assert.True(grown.GrowthEpoch > 1, $"expected several reallocations, saw {grown.GrowthEpoch}.");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = F2(c.VocabSize, h), ["model.norm.weight"] = Ones(h) };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    private static void AssertExact(Tensor a, Tensor b, string label)
    {
        Assert.Equal(a.Shape, b.Shape);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < a.ElementCount; i++)
            Assert.True(pa[i] == pb[i], $"{label}: mismatch at {i} ({pa[i]} vs {pb[i]})");
    }

    private static void AssertClose(Tensor a, Tensor b, string label)
    {
        Assert.Equal(a.Shape, b.Shape);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        for (long i = 0; i < a.ElementCount; i++)
            Assert.True(MathF.Abs(pa[i] - pb[i]) <= 1e-5f, $"{label}: mismatch at {i} ({pa[i]} vs {pb[i]})");
    }
}
