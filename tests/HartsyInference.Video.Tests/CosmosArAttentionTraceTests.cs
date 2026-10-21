using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tests.Common.Attention;
using HartsyInference.Video.Models.Cosmos;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>The attention calls of a Cosmos AR block through the layer's stage: causal self-attention over the KV cache (permute q, k, v; append; attend; permute back), then the non-causal T5 cross-attention over
/// freshly projected keys and values, with the head counts, key length, mask and scale.</summary>
public sealed unsafe class CosmosArAttentionTraceTests
{
    private const int Dim = 32, Heads = 4, KvHeads = 2, HeadDim = 8, Ffn = 48, ContextDim = 24, ContextLen = 6;

    private static Tensor Seeded(string name, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = 2166136261u;
        foreach (char c in name) state = (state ^ c) * 16777619u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * 0.5f;
        }
        return tensor;
    }

    private static string Flash(int kv, int qOffset, bool causal) =>
        $"Attention Flash heads=4/2 kv={kv} q@{qOffset} {(causal ? "causal" : "full")} window=0 softcap=0";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ABlockAttendsCausallyOverItsCacheThenOverTheContext(bool cross)
    {
        CosmosArConfig cfg = new()
        {
            NumLayers = 1, Dim = Dim, NumHeads = Heads, NumKvHeads = KvHeads, HeadDim = HeadDim, FfnHiddenSize = Ffn, ContextDim = ContextDim,
            QkNorm = true, InsertCrossAttn = cross,
        };
        Dictionary<string, Tensor> w = new()
        {
            ["layers.0.attention_norm.weight"] = Seeded("an", Dim),
            ["layers.0.attention.wq.weight"] = Seeded("wq", Heads * HeadDim, Dim),
            ["layers.0.attention.wk.weight"] = Seeded("wk", KvHeads * HeadDim, Dim),
            ["layers.0.attention.wv.weight"] = Seeded("wv", KvHeads * HeadDim, Dim),
            ["layers.0.attention.wo.weight"] = Seeded("wo", Dim, Heads * HeadDim),
            ["layers.0.attention.q_norm.weight"] = Seeded("qn", HeadDim),
            ["layers.0.attention.k_norm.weight"] = Seeded("kn", HeadDim),
            ["layers.0.ffn_norm.weight"] = Seeded("fn", Dim),
            ["layers.0.feed_forward.w1.weight"] = Seeded("w1", Ffn, Dim),
            ["layers.0.feed_forward.w2.weight"] = Seeded("w2", Dim, Ffn),
            ["layers.0.feed_forward.w3.weight"] = Seeded("w3", Ffn, Dim),
        };
        if (cross)
        {
            w["layers.0.cross_attention_norm.weight"] = Seeded("cn", Dim);
            w["layers.0.cross_attention.wq.weight"] = Seeded("cwq", Heads * HeadDim, Dim);
            w["layers.0.cross_attention.wk.weight"] = Seeded("cwk", KvHeads * HeadDim, ContextDim);
            w["layers.0.cross_attention.wv.weight"] = Seeded("cwv", KvHeads * HeadDim, ContextDim);
            w["layers.0.cross_attention.wo.weight"] = Seeded("cwo", Dim, Heads * HeadDim);
        }
        AttentionTraceBackend trace = new() { FlashAsAttention = true };
        CosmosArBlock block = new(cfg, 0);
        block.LoadWeights(w, "layers.0");
        using FixedKvCache cache = new(1, 1, KvHeads, HeadDim, 16);
        using Tensor context = Seeded("context", 1, ContextLen, ContextDim);
        int position = 0;
        try
        {
            foreach (int t in new[] { 5, 1 })
            {
                trace.Ops.Clear();
                using Tensor hidden = Seeded($"hidden{t}", 1, t, Dim);
                using Tensor cos = Seeded($"cos{t}", 1, t, HeadDim);
                using Tensor sin = Seeded($"sin{t}", 1, t, HeadDim);
                using Tensor result = block.Forward(trace, hidden, t, position, cache, 0, cos, sin, cross ? context : null, ContextLen);

                List<string> expected = [$"Permute0213 {t},4,8", $"Permute0213 {t},2,8", $"Permute0213 {t},2,8", Flash(position + t, position, causal: true), $"Permute0213 4,{t},8"];
                if (cross)
                    expected.AddRange([$"Permute0213 {t},4,8", $"Permute0213 {ContextLen},2,8", $"Permute0213 {ContextLen},2,8", Flash(ContextLen, 0, causal: false), $"Permute0213 4,{t},8"]);
                Assert.Equal(expected, trace.Ops.Where(op => op.StartsWith("Permute0213", StringComparison.Ordinal) || op.StartsWith("Attention", StringComparison.Ordinal)));
                Assert.All(trace.Calls, call =>
                {
                    Assert.Equal(1f / MathF.Sqrt(HeadDim), call.Scale);
                    Assert.Null(call.LegacyGap());
                });
                trace.Calls.Clear();
                cache.AdvanceLength(t);
                position += t;
            }
        }
        finally
        {
            foreach (Tensor t in w.Values) t.Dispose();
        }
    }
}
