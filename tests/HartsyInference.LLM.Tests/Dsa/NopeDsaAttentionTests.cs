using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using Xunit;

namespace HartsyInference.LLM.Tests.Dsa;

/// <summary>Kimi-shaped multi-head latent attention with no position encoding (NoPE) on the shared DSA layer: one call over a sequence
/// must equal the same tokens split across two calls that share the latent and indexer cache. Synthetic dimensions; the top-k is
/// wide enough to keep every causal key, so the selection is exact.</summary>
public sealed class NopeDsaAttentionTests
{
    private const int Dim = 16, Heads = 2, QLoraRank = 8, KvLoraRank = 6, QkNope = 4, QkRope = 4, VDim = 4;
    private const int IndexHeads = 2, IndexHeadDim = 8, IndexRope = 2, TopK = 16, Tokens = 6;

    private static DsaAttentionSettings Settings() => new()
    {
        Dim = Dim, Heads = Heads, QLoraRank = QLoraRank, KvLoraRank = KvLoraRank, QkNopeDim = QkNope, QkRopeDim = QkRope,
        VDim = VDim, Window = 0, NormEps = 1e-6f,
        Indexer = new DsaIndexerSettings
        {
            Heads = IndexHeads, HeadDim = IndexHeadDim, RopeDim = IndexRope, TopK = TopK,
            Hadamard = false, QueryEncoding = LatentEncoding.F32, KeyEncoding = LatentEncoding.F32,
        },
    };

    [Fact]
    public void Nope_OneCallEqualsTwoCallsWithCarriedCache()
    {
        DsaAttentionSettings s = Settings();
        Random rng = new(5);
        DsaAttentionWeights w = RandomWeights(rng, s);
        float[] x = Floats(rng, Tokens * Dim, -1f, 1f);
        IBackend backend = new CpuBackend();

        float[] whole = Run(backend, s, w, x, split: null);
        float[] halves = Run(backend, s, w, x, split: 2);
        for (int i = 0; i < whole.Length; i++)
        {
            Assert.True(float.IsFinite(whole[i]));
            Assert.Equal(whole[i], halves[i], 4);
        }
    }

    private static float[] Run(IBackend backend, DsaAttentionSettings s, DsaAttentionWeights w, float[] x, int? split)
    {
        using DsaAttention layer = new(backend, s, w, new IdentityRopeRows(QkRope / 2, Tokens), new IdentityRopeRows(IndexRope / 2, Tokens));
        using DsaAttentionState state = new(Tokens, s);
        float[] result = new float[Tokens * Dim];
        int first = split ?? Tokens;
        Chunk(layer, state, x, 0, first, result);
        if (split is int k) Chunk(layer, state, x, k, Tokens - k, result);
        return result;
    }

    private static void Chunk(DsaAttention layer, DsaAttentionState state, float[] x, int start, int count, float[] result)
    {
        layer.Forward(x.AsSpan(start * Dim, count * Dim), count, start, state, result.AsSpan(start * Dim, count * Dim));
    }

    private static DsaAttentionWeights RandomWeights(Random rng, DsaAttentionSettings s)
    {
        int qkHead = s.QkHeadDim, latent = s.LatentDim;
        return new DsaAttentionWeights
        {
            QA = Make(Floats(rng, QLoraRank * Dim, -0.3f, 0.3f), QLoraRank, Dim),
            KvA = Make(Floats(rng, latent * Dim, -0.3f, 0.3f), latent, Dim),
            QANorm = Make(Ones(QLoraRank), QLoraRank),
            KvANorm = Make(Ones(KvLoraRank), KvLoraRank),
            QB = Make(Floats(rng, Heads * qkHead * QLoraRank, -0.3f, 0.3f), Heads * qkHead, QLoraRank),
            KvB = Make(Floats(rng, Heads * (QkNope + VDim) * KvLoraRank, -0.3f, 0.3f), Heads * (QkNope + VDim), KvLoraRank),
            WoB = Make(Floats(rng, Dim * Heads * VDim, -0.3f, 0.3f), Dim, Heads * VDim),
            Indexer = new DsaIndexerWeights
            {
                WqB = Make(Floats(rng, IndexHeads * IndexHeadDim * QLoraRank, -0.3f, 0.3f), IndexHeads * IndexHeadDim, QLoraRank),
                Wk = Make(Floats(rng, IndexHeadDim * Dim, -0.3f, 0.3f), IndexHeadDim, Dim),
                KeyNormWeight = Make(Ones(IndexHeadDim), IndexHeadDim),
                WeightsProj = Make(Floats(rng, IndexHeads * Dim, -0.3f, 0.3f), IndexHeads, Dim),
            },
        };
    }

    private static float[] Ones(int n)
    {
        float[] values = new float[n];
        Array.Fill(values, 1f);
        return values;
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
