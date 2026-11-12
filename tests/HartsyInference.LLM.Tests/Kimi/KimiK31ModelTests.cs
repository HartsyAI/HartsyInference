using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A tiny synthetic Kimi K3.1 decoder with the checkpoint's structure (KDA and NoPE MLA layers, a dense first layer, latent SiTU
/// MoE layers, block attention residuals that cross a block boundary) on the CPU backend: logits are finite, deterministic, and the
/// state advances one token per call.</summary>
public sealed class KimiK31ModelTests
{
    internal static readonly KimiK31Config Config = new()
    {
        Hidden = 16, Vocab = 20, Layers = 4, DenseLayers = 1, DenseIntermediate = 24,
        Heads = 2, KdaHeadDim = 8, KdaForgetRank = 4, FullAttentionLayers = new HashSet<int> { 1, 3 },
        QLoraRank = 8, KvLoraRank = 6, QkNopeDim = 4, QkRopeDim = 4, VDim = 4,
        IndexHeads = 2, IndexHeadDim = 8, IndexTopK = 16,
        Experts = 4, ExpertsPerToken = 2, MoeIntermediate = 8, SharedExperts = 1, LatentSize = 8,
        AttnResBlock = 2,
    };

    [Fact]
    public void ForwardToken_ProducesFiniteLogits_AndAdvancesState()
    {
        using KimiK31Model model = new(new CpuBackend(), Config, RandomWeights(new Random(9)), maxPositions: 16);
        using KimiK31State state = new(Config, capacity: 16);
        float[] logits = model.Forward([1, 5, 7, 2, 11, 3], state);

        Assert.Equal(Config.Vocab, logits.Length);
        foreach (float v in logits) Assert.True(float.IsFinite(v));
        Assert.Equal(6, state.Position);
    }

    [Fact]
    public void Forward_IsDeterministic_OnAFreshState()
    {
        KimiK31Weights weights = RandomWeights(new Random(9));
        int[] tokens = [4, 4, 9, 0];
        float[] first = Run(weights, tokens);
        float[] second = Run(weights, tokens);
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i], 6);
    }

    [Fact]
    public void LaterToken_DependsOnItsOwnEmbedding_NotTheFirstTokens()
    {
        // The same first token, a different second token: the second token's logits must differ.
        KimiK31Weights weights = RandomWeights(new Random(9));
        float[] a = Run(weights, [4, 1]);
        float[] b = Run(weights, [4, 7]);
        bool differs = false;
        for (int i = 0; i < a.Length; i++) differs |= MathF.Abs(a[i] - b[i]) > 1e-5f;
        Assert.True(differs, "the second token's logits did not depend on the second token");
    }

    private static float[] Run(KimiK31Weights weights, int[] tokens)
    {
        using KimiK31Model model = new(new CpuBackend(), Config, weights, maxPositions: 16);
        using KimiK31State state = new(Config, capacity: 16);
        return model.Forward(tokens, state);
    }

    internal static KimiK31Weights RandomWeights(Random rng) => RandomWeights(rng, Config);

    internal static KimiK31Weights RandomWeights(Random rng, KimiK31Config c)
    {
        int keyDim = c.Heads * c.KdaHeadDim;
        List<KimiK31LayerWeights> layers = new();
        for (int i = 0; i < c.Layers; i++)
        {
            bool mla = c.IsMla(i), dense = c.IsDense(i);
            layers.Add(new KimiK31LayerWeights
            {
                InputNorm = Ones(rng, c.Hidden),
                PostNorm = Ones(rng, c.Hidden),
                AttnResAttnQuery = Make(Floats(rng, c.Hidden, -0.2f, 0.2f), c.Hidden, 1),
                AttnResAttnNorm = Ones(rng, c.Hidden),
                AttnResMlpQuery = Make(Floats(rng, c.Hidden, -0.2f, 0.2f), c.Hidden, 1),
                AttnResMlpNorm = Ones(rng, c.Hidden),
                Kda = mla ? null : KdaWeights(rng, c, keyDim),
                Mla = mla ? MlaWeights(rng, c) : null,
                DenseGate = dense ? Make(Floats(rng, c.DenseIntermediate * c.Hidden, -0.3f, 0.3f), c.DenseIntermediate, c.Hidden) : null,
                DenseUp = dense ? Make(Floats(rng, c.DenseIntermediate * c.Hidden, -0.3f, 0.3f), c.DenseIntermediate, c.Hidden) : null,
                DenseDown = dense ? Make(Floats(rng, c.Hidden * c.DenseIntermediate, -0.3f, 0.3f), c.Hidden, c.DenseIntermediate) : null,
                Moe = dense ? null : MoeWeights(rng, c, $"layer{i}"),
                MoePrefix = $"layer{i}",
            });
        }
        return new KimiK31Weights
        {
            Embedding = Make(Floats(rng, c.Vocab * c.Hidden, -0.5f, 0.5f), c.Vocab, c.Hidden),
            FinalNorm = Ones(rng, c.Hidden),
            LmHead = Make(Floats(rng, c.Vocab * c.Hidden, -0.5f, 0.5f), c.Vocab, c.Hidden),
            OutputResQuery = Make(Floats(rng, c.Hidden, -0.2f, 0.2f), c.Hidden, 1),
            OutputResNorm = Ones(rng, c.Hidden),
            Layers = layers,
        };
    }

    private static KdaWeights KdaWeights(Random rng, KimiK31Config c, int keyDim) => new(
        QkvProj: Make(Floats(rng, 3 * keyDim * c.Hidden, -0.3f, 0.3f), 3 * keyDim, c.Hidden),
        ConvTaps: Make(Floats(rng, 3 * keyDim * c.ConvKernel, -0.5f, 0.5f), 3 * keyDim, c.ConvKernel),
        ForgetDown: Make(Floats(rng, c.KdaForgetRank * c.Hidden, -0.3f, 0.3f), c.KdaForgetRank, c.Hidden),
        ForgetUp: Make(Floats(rng, keyDim * c.KdaForgetRank, -0.3f, 0.3f), keyDim, c.KdaForgetRank),
        ForgetBias: Make(Floats(rng, keyDim, -0.2f, 0.2f), keyDim),
        BetaProj: Make(Floats(rng, c.Heads * c.Hidden, -0.3f, 0.3f), c.Heads, c.Hidden),
        OutGateProj: Make(Floats(rng, keyDim * c.Hidden, -0.3f, 0.3f), keyDim, c.Hidden),
        OutNorm: Ones(rng, c.KdaHeadDim),
        OutProj: Make(Floats(rng, c.Hidden * keyDim, -0.3f, 0.3f), c.Hidden, keyDim));

    private static DsaAttentionWeights MlaWeights(Random rng, KimiK31Config c)
    {
        int qk = c.QkNopeDim + c.QkRopeDim, latent = c.KvLoraRank + c.QkRopeDim;
        return new DsaAttentionWeights
        {
            QA = Make(Floats(rng, c.QLoraRank * c.Hidden, -0.3f, 0.3f), c.QLoraRank, c.Hidden),
            KvA = Make(Floats(rng, latent * c.Hidden, -0.3f, 0.3f), latent, c.Hidden),
            QANorm = Ones(rng, c.QLoraRank),
            KvANorm = Ones(rng, c.KvLoraRank),
            QB = Make(Floats(rng, c.Heads * qk * c.QLoraRank, -0.3f, 0.3f), c.Heads * qk, c.QLoraRank),
            KvB = Make(Floats(rng, c.Heads * (c.QkNopeDim + c.VDim) * c.KvLoraRank, -0.3f, 0.3f), c.Heads * (c.QkNopeDim + c.VDim), c.KvLoraRank),
            WoB = Make(Floats(rng, c.Hidden * c.Heads * c.VDim, -0.3f, 0.3f), c.Hidden, c.Heads * c.VDim),
            Indexer = new DsaIndexerWeights
            {
                WqB = Make(Floats(rng, c.IndexHeads * c.IndexHeadDim * c.QLoraRank, -0.3f, 0.3f), c.IndexHeads * c.IndexHeadDim, c.QLoraRank),
                Wk = Make(Floats(rng, c.IndexHeadDim * c.Hidden, -0.3f, 0.3f), c.IndexHeadDim, c.Hidden),
                KeyNormWeight = Ones(rng, c.IndexHeadDim),
                WeightsProj = Make(Floats(rng, c.IndexHeads * c.Hidden, -0.3f, 0.3f), c.IndexHeads, c.Hidden),
            },
        };
    }

    internal static Dictionary<string, Tensor> MoeWeights(Random rng, KimiK31Config c, string p)
    {
        string mlp = $"{p}.mlp";
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [$"{mlp}.gate.weight"] = Make(Floats(rng, c.Experts * c.Hidden, -0.3f, 0.3f), c.Experts, c.Hidden),
            [$"{mlp}.gate.e_score_correction_bias"] = Make(Floats(rng, c.Experts, -0.1f, 0.1f), c.Experts),
            [$"{mlp}.pre_expert_proj.weight"] = Make(Floats(rng, c.LatentSize * c.Hidden, -0.3f, 0.3f), c.LatentSize, c.Hidden),
            [$"{mlp}.pre_expert_norm.weight"] = Ones(rng, c.LatentSize),
            [$"{mlp}.post_expert_proj.weight"] = Make(Floats(rng, c.Hidden * c.LatentSize, -0.3f, 0.3f), c.Hidden, c.LatentSize),
        };
        int shared = c.SharedExperts * c.MoeIntermediate;
        w[$"{mlp}.shared_expert.gate_proj.weight"] = Make(Floats(rng, shared * c.Hidden, -0.3f, 0.3f), shared, c.Hidden);
        w[$"{mlp}.shared_expert.up_proj.weight"] = Make(Floats(rng, shared * c.Hidden, -0.3f, 0.3f), shared, c.Hidden);
        w[$"{mlp}.shared_expert.down_proj.weight"] = Make(Floats(rng, c.Hidden * shared, -0.3f, 0.3f), c.Hidden, shared);
        for (int e = 0; e < c.Experts; e++)
        {
            w[$"{mlp}.experts.{e}.gate_proj.weight"] = Make(Floats(rng, c.MoeIntermediate * c.LatentSize, -0.3f, 0.3f), c.MoeIntermediate, c.LatentSize);
            w[$"{mlp}.experts.{e}.up_proj.weight"] = Make(Floats(rng, c.MoeIntermediate * c.LatentSize, -0.3f, 0.3f), c.MoeIntermediate, c.LatentSize);
            w[$"{mlp}.experts.{e}.down_proj.weight"] = Make(Floats(rng, c.LatentSize * c.MoeIntermediate, -0.3f, 0.3f), c.LatentSize, c.MoeIntermediate);
        }
        return w;
    }

    private static Tensor Ones(Random rng, int n)
    {
        float[] values = new float[n];
        Array.Fill(values, 1f);
        return Make(values, n);
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
