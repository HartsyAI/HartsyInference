using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>An untied head narrower than the embedding table (mllama: 128256 head rows, 128264 table rows) sets the logit
/// width; a wider logits row holds entries no GEMM writes, which greedy decoding then compares as real scores.</summary>
public sealed unsafe class HeadVocabWidthTests
{
    private const int Hidden = 16, Heads = 4, KvHeads = 2, Ffn = 32, HeadRows = 24, TableRows = HeadRows + 8;

    private static uint _rng = 0x51ED270Bu;
    private static float Rand()
    {
        _rng ^= _rng << 13;
        _rng ^= _rng >> 17;
        _rng ^= _rng << 5;
        return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f;
    }

    private static Tensor Random(params int[] dims)
    {
        Tensor t = new(new TensorShape(dims.Select(d => (long)d).ToArray()), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        new Span<float>((float*)t.DataPointer, n).Fill(1f);
        return t;
    }

    private static GgufMetadata Metadata()
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "llama");
        m.Add("llama.block_count", 1u);
        m.Add("llama.embedding_length", (uint)Hidden);
        m.Add("llama.attention.head_count", (uint)Heads);
        m.Add("llama.attention.head_count_kv", (uint)KvHeads);
        m.Add("llama.feed_forward_length", (uint)Ffn);
        return m;
    }

    private static Dictionary<string, Tensor> Weights(bool untiedHead)
    {
        int headDim = Hidden / Heads, kvDim = KvHeads * headDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = Random(TableRows, Hidden),
            ["model.norm.weight"] = Ones(Hidden),
            ["model.layers.0.input_layernorm.weight"] = Ones(Hidden),
            ["model.layers.0.post_attention_layernorm.weight"] = Ones(Hidden),
            ["model.layers.0.self_attn.q_proj.weight"] = Random(Hidden, Hidden),
            ["model.layers.0.self_attn.k_proj.weight"] = Random(kvDim, Hidden),
            ["model.layers.0.self_attn.v_proj.weight"] = Random(kvDim, Hidden),
            ["model.layers.0.self_attn.o_proj.weight"] = Random(Hidden, Hidden),
            ["model.layers.0.mlp.gate_proj.weight"] = Random(Ffn, Hidden),
            ["model.layers.0.mlp.up_proj.weight"] = Random(Ffn, Hidden),
            ["model.layers.0.mlp.down_proj.weight"] = Random(Hidden, Ffn),
        };
        if (untiedHead) w["lm_head.weight"] = Random(HeadRows, Hidden);
        return w;
    }

    private static void DisposeAll(Dictionary<string, Tensor> w)
    {
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void UntiedHead_SetsTheLogitWidth()
    {
        Dictionary<string, Tensor> w = Weights(untiedHead: true);
        try
        {
            Assert.Equal(HeadRows, GgufConfigFactory.FromGguf(Metadata(), w).VocabSize);
        }
        finally
        {
            DisposeAll(w);
        }
    }

    [Fact]
    public void TiedHead_KeepsTheTableWidth()
    {
        Dictionary<string, Tensor> w = Weights(untiedHead: false);
        try
        {
            Assert.Equal(TableRows, GgufConfigFactory.FromGguf(Metadata(), w).VocabSize);
        }
        finally
        {
            DisposeAll(w);
        }
    }

    [Fact]
    public void ProjectLogits_ReturnsExactlyTheHeadRows()
    {
        Dictionary<string, Tensor> w = Weights(untiedHead: true);
        using GenericTransformer model = new(GgufConfigFactory.FromGguf(Metadata(), w));
        using CpuBackend backend = new();
        try
        {
            model.LoadWeights(w, "model");
            using Tensor hidden = Random(1, 1, Hidden);
            using Tensor logits = model.ProjectLogits(backend, hidden, 1);
            Assert.Equal(HeadRows, (int)logits.Shape[2]);
            float* h = (float*)hidden.DataPointer, head = (float*)w["lm_head.weight"].DataPointer, l = (float*)logits.DataPointer;
            for (int row = 0; row < HeadRows; row++)
            {
                float expected = 0f;
                for (int k = 0; k < Hidden; k++) expected += h[k] * head[row * Hidden + k];
                Assert.True(MathF.Abs(expected - l[row]) <= 1e-5f, $"logit {row}: {l[row]} vs {expected}");
            }
        }
        finally
        {
            DisposeAll(w);
        }
    }

    [Fact]
    public void EmbedLookup_ReadsTableRowsPastTheLogitWidth()
    {
        Dictionary<string, Tensor> w = Weights(untiedHead: true);
        using GenericTransformer model = new(GgufConfigFactory.FromGguf(Metadata(), w));
        try
        {
            model.LoadWeights(w, "model");
            using Tensor embeds = new(new TensorShape(1, 1, Hidden), DType.F32);
            model.EmbedLookup(embeds, [TableRows - 1]);
            float* row = (float*)w["model.embed_tokens.weight"].DataPointer + (TableRows - 1) * Hidden;
            Assert.Equal(new Span<float>(row, Hidden).ToArray(), new Span<float>((float*)embeds.DataPointer, Hidden).ToArray());
            Assert.Throws<ArgumentException>(() => model.EmbedLookup(embeds, [TableRows]));
        }
        finally
        {
            DisposeAll(w);
        }
    }
}
