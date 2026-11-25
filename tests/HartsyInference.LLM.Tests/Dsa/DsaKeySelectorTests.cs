using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using Xunit;

namespace HartsyInference.LLM.Tests.Dsa;

/// <summary>The pluggable key selector of <see cref="DsaAttention"/>: the window selector's rows, and the new constructor against the token top-k one it replaced inside.</summary>
public sealed class DsaKeySelectorTests
{
    private const int Dim = 16, Heads = 2, QLoraRank = 8, KvLoraRank = 6, QkNope = 4, QkRope = 4, VDim = 4, Tokens = 6;

    [Fact]
    public void WindowSelector_AttendsTheLastRowsIncludingItself()
    {
        using DsaWindowSelector selector = new(3);
        using Tensor unused = new(new TensorShape(8, 1), DType.F32);
        DsaKeySelection s = selector.Select([], [], [0, 1, 5], 6, unused);
        Assert.Equal([1, 2, 3], s.Counts);
        Assert.Equal([0, -1, -1], s.Rows.AsSpan(0, 3).ToArray());
        Assert.Equal([0, 1, -1], s.Rows.AsSpan(3, 3).ToArray());
        Assert.Equal([3, 4, 5], s.Rows.AsSpan(6, 3).ToArray());
    }

    [Fact]
    public void WindowSelectorAsTheSelector_EqualsTheTokenTopKPathWhenTheWindowCoversTheContext()
    {
        DsaAttentionSettings s = new()
        {
            Dim = Dim, Heads = Heads, QLoraRank = QLoraRank, KvLoraRank = KvLoraRank, QkNopeDim = QkNope, QkRopeDim = QkRope, VDim = VDim,
            Window = Tokens, NormEps = 1e-6f,
            Indexer = new DsaIndexerSettings
            {
                Heads = 2, HeadDim = 8, RopeDim = 2, TopK = 4, Hadamard = false, QueryEncoding = LatentEncoding.F32, KeyEncoding = LatentEncoding.F32,
            },
        };
        Random rng = new(11);
        DsaAttentionWeights w = Weights(rng, s);
        float[] x = Floats(rng, Tokens * Dim);
        IBackend backend = new CpuBackend();

        float[] legacy = Run(new DsaAttention(backend, s, w, new IdentityRopeRows(QkRope / 2, Tokens), new IdentityRopeRows(1, Tokens)), s, 8, x);
        float[] pluggable = Run(new DsaAttention(backend, s, w, new IdentityRopeRows(QkRope / 2, Tokens), new DsaWindowSelector(Tokens)), s, 0, x);
        for (int i = 0; i < legacy.Length; i++) Assert.Equal(legacy[i], pluggable[i], 5);
    }

    private static float[] Run(DsaAttention layer, DsaAttentionSettings s, int keyWidth, float[] x)
    {
        using (layer)
        {
            using DsaAttentionState state = new(Tokens, s, keyWidth);
            float[] result = new float[Tokens * Dim];
            layer.Forward(x, Tokens, 0, state, result);
            return result;
        }
    }

    private static DsaAttentionWeights Weights(Random rng, DsaAttentionSettings s) => new()
    {
        QA = Make(Floats(rng, QLoraRank * Dim), QLoraRank, Dim),
        KvA = Make(Floats(rng, s.LatentDim * Dim), s.LatentDim, Dim),
        QANorm = Make(Ones(QLoraRank), QLoraRank),
        KvANorm = Make(Ones(KvLoraRank), KvLoraRank),
        QB = Make(Floats(rng, Heads * s.QkHeadDim * QLoraRank), Heads * s.QkHeadDim, QLoraRank),
        KvB = Make(Floats(rng, Heads * (QkNope + VDim) * KvLoraRank), Heads * (QkNope + VDim), KvLoraRank),
        WoB = Make(Floats(rng, Dim * Heads * VDim), Dim, Heads * VDim),
        Indexer = new DsaIndexerWeights
        {
            WqB = Make(Floats(rng, 2 * 8 * QLoraRank), 16, QLoraRank),
            Wk = Make(Floats(rng, 8 * Dim), 8, Dim),
            KeyNormWeight = Make(Ones(8), 8),
            WeightsProj = Make(Floats(rng, 2 * Dim), 2, Dim),
        },
    };

    private static float[] Ones(int n) => [.. Enumerable.Repeat(1f, n)];

    private static float[] Floats(Random rng, int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 0.6 - 0.3))];

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
