using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.Video.Models.Cosmos;

namespace HartsyInference.Video.Tests;

/// <summary>Structural unit tests (CPU, synthetic weights) for the Cosmos AR backbone: 3D RoPE table layout, and a
/// tiny-config prefill + per-token decode forward proving the self-attn / T5 cross-attn / SwiGLU / abs-pos / KV-cache
/// path composes and yields the right shapes. Real-weight layer-diff parity is env-gated (integration).</summary>
public unsafe class CosmosArTransformerTests
{
    private readonly ITestOutputHelper _output;
    public CosmosArTransformerTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Cosmos3DRoPE_TableLayout_IsDuplicatedHalfAndAxisPartitioned()
    {
        int headDim = 8;
        Cosmos3DRoPE rope = new(headDim, theta: 10000.0, section: (2, 1, 1));
        (Tensor cos, Tensor sin) = rope.BuildTable(2, 2, 2);
        try
        {
            Assert.Equal(new[] { 1L, 8, headDim }, new[] { cos.Shape[0], cos.Shape[1], cos.Shape[2] });
            float* cp = (float*)cos.DataPointer;
            float* sp = (float*)sin.DataPointer;
            int half = headDim / 2;

            // Token 0 = grid (0,0,0): every angle is 0 → cos=1, sin=0.
            for (int k = 0; k < headDim; k++)
            {
                Assert.Equal(1f, cp[k], 5);
                Assert.Equal(0f, sp[k], 5);
            }

            // Duplicated-half layout: channel k and k+half share the same cos/sin for every token.
            for (int s = 0; s < 8; s++)
                for (int k = 0; k < half; k++)
                {
                    long b = (long)s * headDim;
                    Assert.Equal(cp[b + k], cp[b + k + half], 5);
                    Assert.Equal(sp[b + k], sp[b + k + half], 5);
                }

            // Token index 5 = (t=1,h=0,w=1) with H=W=2: temporal channels (k=0,1) rotate by pos=1, width
            // channel (k=3) rotates by pos=1, height channel (k=2) by pos=0 → its sin stays 0.
            long row5 = 5L * headDim;
            Assert.Equal(0f, sp[row5 + 2], 5);   // height channel, h=0
            Assert.NotEqual(0f, sp[row5 + 0]);   // temporal channel, t=1
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
    }

    [Fact]
    public void CosmosArTransformer_TinyConfig_PrefillAndDecode_ProduceRightShapes()
    {
        CpuBackend backend = new();
        CosmosArConfig cfg = new()
        {
            NumLayers = 2, Dim = 32, NumHeads = 4, NumKvHeads = 2, HeadDim = 8,
            FfnHiddenSize = 64, VocabSize = 100, ContextDim = 16,
            RopeSection = (2, 1, 1), RopeTheta = 10000f,
        };
        using CosmosArTransformer model = new(cfg);
        model.LoadWeights(BuildSyntheticWeights(cfg));

        int latentT = 1, latentH = 2, latentW = 2, n = latentT * latentH * latentW;
        model.SetGrid(latentT, latentH, latentW);
        Assert.Equal(n, model.SequenceLength);

        using Tensor context = Rand(new TensorShape(1, 3, cfg.ContextDim), 0.05f, 0x55u);
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, n);

        // Prefill one conditioning token.
        int[] prefix = [7];
        using (Tensor hidden = model.Forward(backend, prefix, 0, cache, context, 3))
        {
            Assert.Equal(new[] { 1L, 1, cfg.Dim }, new[] { hidden.Shape[0], hidden.Shape[1], hidden.Shape[2] });
            using Tensor logits = model.ProjectLogits(backend, hidden, 1);
            Assert.Equal(new[] { 1L, 1, cfg.VocabSize }, new[] { logits.Shape[0], logits.Shape[1], logits.Shape[2] });
        }
        Assert.Equal(1, cache.CurrentLength);

        // One decode step at the next position.
        int[] step = [13];
        using (Tensor hidden = model.Forward(backend, step, 1, cache, context, 3))
        using (Tensor logits = model.ProjectLogits(backend, hidden, 1))
        {
            Assert.Equal(cfg.VocabSize, (int)logits.Shape[2]);
            float* lp = (float*)logits.DataPointer;
            for (int i = 0; i < cfg.VocabSize; i++) Assert.False(float.IsNaN(lp[i]) || float.IsInfinity(lp[i]));
        }
        Assert.Equal(2, cache.CurrentLength);
    }

    private static Dictionary<string, Tensor> BuildSyntheticWeights(CosmosArConfig cfg)
    {
        int dim = cfg.Dim, hq = cfg.NumHeads, hkv = cfg.NumKvHeads, d = cfg.HeadDim, ff = cfg.FfnHiddenSize;
        int qDim = hq * d, kvDim = hkv * d, ctx = cfg.ContextDim;
        uint seed = 0x100u;
        Dictionary<string, Tensor> w = new()
        {
            ["tok_embeddings.weight"] = Rand(new TensorShape(cfg.VocabSize, dim), 0.05f, seed++),
            ["norm.weight"] = Ones(new TensorShape(dim)),
            ["output.weight"] = Rand(new TensorShape(cfg.VocabSize, dim), 0.05f, seed++),
        };
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"layers.{i}";
            w[$"{p}.attention_norm.weight"] = Ones(new TensorShape(dim));
            w[$"{p}.attention.wq.weight"] = Rand(new TensorShape(qDim, dim), 0.05f, seed++);
            w[$"{p}.attention.wk.weight"] = Rand(new TensorShape(kvDim, dim), 0.05f, seed++);
            w[$"{p}.attention.wv.weight"] = Rand(new TensorShape(kvDim, dim), 0.05f, seed++);
            w[$"{p}.attention.wo.weight"] = Rand(new TensorShape(dim, qDim), 0.05f, seed++);
            w[$"{p}.attention.q_norm.weight"] = Ones(new TensorShape(d));
            w[$"{p}.attention.k_norm.weight"] = Ones(new TensorShape(d));
            w[$"{p}.cross_attention_norm.weight"] = Ones(new TensorShape(dim));
            w[$"{p}.cross_attention.wq.weight"] = Rand(new TensorShape(qDim, dim), 0.05f, seed++);
            w[$"{p}.cross_attention.wk.weight"] = Rand(new TensorShape(kvDim, ctx), 0.05f, seed++);
            w[$"{p}.cross_attention.wv.weight"] = Rand(new TensorShape(kvDim, ctx), 0.05f, seed++);
            w[$"{p}.cross_attention.wo.weight"] = Rand(new TensorShape(dim, qDim), 0.05f, seed++);
            w[$"{p}.ffn_norm.weight"] = Ones(new TensorShape(dim));
            w[$"{p}.feed_forward.w1.weight"] = Rand(new TensorShape(ff, dim), 0.05f, seed++);
            w[$"{p}.feed_forward.w2.weight"] = Rand(new TensorShape(dim, ff), 0.05f, seed++);
            w[$"{p}.feed_forward.w3.weight"] = Rand(new TensorShape(ff, dim), 0.05f, seed++);
        }
        return w;
    }

    private static Tensor Ones(TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++) p[i] = 1f;
        return t;
    }

    private static Tensor Rand(TensorShape shape, float scale, uint seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        uint s = seed == 0 ? 1u : seed;
        for (long i = 0; i < shape.ElementCount; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            p[i] = ((s >> 8) * (1f / 16777216f) - 0.5f) * 2f * scale;
        }
        return t;
    }
}
