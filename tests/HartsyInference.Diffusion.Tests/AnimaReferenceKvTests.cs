using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The reference K/V path: the rewrite that joins a cached reference to a block's live tokens, the placement of a reference on
/// the RoPE axes, the prepass that caches a reference's keys and values, and a tiny Anima transformer that joins a reference into every
/// block. These run on the CPU; the parity against ComfyUI's reference adapter is a GPU check.</summary>
public sealed class AnimaReferenceKvTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 4, Ffn = 16, Layers = 2, TextWidth = 8, EmbeddedDim = 8, AdaRank = 4;
    private const int TargetH = 6, TargetW = 10, RefH = 4, RefW = 6, Text = 7;

    [Fact]
    public void Rewrite_Append_AndPrepend_KeepTheStatedOrder()
    {
        using CpuBackend cpu = new();
        using Tensor cachedK = Values(new TensorShape(1, 1, 1, 2), 101f);
        using Tensor cachedV = Values(new TensorShape(1, 1, 1, 2), 201f);

        (Tensor appendK, Tensor appendV) = new ReferenceKvRewrite(cachedK, cachedV, ReferenceKvOrder.Append)
            .Rewrite(cpu, Values(new TensorShape(1, 2, 1, 2), 1f), Values(new TensorShape(1, 2, 1, 2), 11f));
        Assert.Equal(new TensorShape(1, 3, 1, 2), appendK.Shape);
        Assert.Equal(new float[] { 1, 2, 3, 4, 101, 102 }, appendK.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(new float[] { 11, 12, 13, 14, 201, 202 }, appendV.AsReadOnlySpan<float>().ToArray());
        appendK.Dispose();
        appendV.Dispose();

        (Tensor prependK, Tensor prependV) = new ReferenceKvRewrite(cachedK, cachedV, ReferenceKvOrder.Prepend)
            .Rewrite(cpu, Values(new TensorShape(1, 2, 1, 2), 1f), Values(new TensorShape(1, 2, 1, 2), 11f));
        Assert.Equal(new float[] { 101, 102, 1, 2, 3, 4 }, prependK.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(new float[] { 201, 202, 11, 12, 13, 14 }, prependV.AsReadOnlySpan<float>().ToArray());
        prependK.Dispose();
        prependV.Dispose();
    }

    [Theory]
    [InlineData(0.5f, 3, 2)]
    [InlineData(0.5f, 5, 2)]
    [InlineData(0.5f, 7, 4)]
    [InlineData(1.0f, 4, 4)]
    [InlineData(0f, 9, 0)]
    public void Placement_RowOffset_RoundsHalfToEven(float heights, int targetHeight, int expected)
    {
        Assert.Equal(expected, new ReferencePlacement(1, heights, 0).RowOffset(targetHeight));
    }

    [Fact]
    public void Placement_RefusesANegativeFrameOrOffset()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(-1, 1f, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(1, -0.5f, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReferencePlacement(1, 1f, -1));
    }

    [Fact]
    public void Prepass_SkipsTheListedLayers_AndCachesTheRest()
    {
        using CpuBackend cpu = new();
        using Tensor embedded = Ramp(new TensorShape(1, 3, 4), 0f);
        int calls = 0;
        ReferenceKvCache cache = ReferenceKvPrepass.Build(cpu, 3, [1], embedded,
            (int layer, Tensor stream, out Tensor keys, out Tensor values) =>
            {
                calls++;
                keys = Values(new TensorShape(1, 3, 1, 2), layer + 1f);
                values = Values(new TensorShape(1, 3, 1, 2), 10f + layer);
                return Ramp(stream.Shape, 0f);
            },
            fingerprint: 9, family: "test");
        using ReferenceKvCache owned = cache;

        Assert.Equal(2, calls);
        Assert.Equal(3, owned.Layers);
        Assert.Equal(3, owned.Tokens);
        Assert.Equal(9, owned.Fingerprint);
        Assert.Null(owned.Keys(1));
        Assert.Equal(3f, owned.Keys(2)!.AsReadOnlySpan<float>()[0]);
        Assert.Equal(10f, owned.Values(0)!.AsReadOnlySpan<float>()[0]);
    }

    [Fact]
    public void Fingerprint_TracksEachInputOfTheCache()
    {
        ReferencePlacement placement = new(1, 1.0f, 0);
        long baseline = AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, placement);

        Assert.Equal(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(43, 2, 3, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 4, 3, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 5, placement));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, new ReferencePlacement(2, 1.0f, 0)));
        Assert.NotEqual(baseline, AnimaTransformer.ReferenceFingerprint(42, 2, 3, 3, new ReferencePlacement(1, 1.0f, 1)));
    }

    [Fact]
    public void Transformer_ReferenceJoin_ChangesTheVelocity_AndTheCacheIsReusable()
    {
        using CpuBackend cpu = new();
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        using Tensor latent = Ramp(new TensorShape(1, 2, TargetH, TargetW), 0.3f);
        using Tensor text = Ramp(new TensorShape(1, Text, TextWidth), 0.7f);
        using Tensor reference = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);

        using Tensor baseline = transformer.Forward(cpu, latent, 0.625f, text);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cpu, reference, TargetH / 2,
            new ReferencePlacement(1, 1.0f, 0), identity: 42);
        using Tensor first = transformer.Forward(cpu, latent, 0.625f, text, null, cache);
        using Tensor second = transformer.Forward(cpu, latent, 0.625f, text, null, cache);

        Assert.Equal(baseline.Shape, first.Shape);
        Assert.Equal(first.AsReadOnlySpan<float>().ToArray(), second.AsReadOnlySpan<float>().ToArray());
        Assert.True(MaxAbsDifference(baseline, first) > 1e-4f, "The reference left the velocity unchanged.");
    }

    [Fact]
    public void Transformer_ReferenceForABatch_IsRefused()
    {
        using CpuBackend cpu = new();
        AnimaConfig config = TinyConfig();
        using AnimaTransformer transformer = new(config);
        transformer.LoadWeights(Weights(config));
        using Tensor reference = Ramp(new TensorShape(1, 2, RefH, RefW), 1.1f);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cpu, reference, TargetH / 2,
            new ReferencePlacement(1, 1.0f, 0), identity: 1);
        using Tensor latent = Ramp(new TensorShape(2, 2, TargetH, TargetW), 0.3f);
        using Tensor text = Ramp(new TensorShape(2, Text, TextWidth), 0.7f);

        Assert.Throws<NotSupportedException>(() => transformer.Forward(cpu, latent, 0.625f, text, null, cache));
    }

    private static float MaxAbsDifference(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(x[i] - y[i]));
        }
        return max;
    }

    private static AnimaConfig TinyConfig() => new()
    {
        HiddenSize = Hidden,
        NumHeads = Heads,
        HeadDim = HeadDim,
        NumLayers = Layers,
        MlpRatio = 2.0f,
        InChannels = 2,
        OutChannels = 3,
        PatchSize = (1, 2, 2),
        MaxSize = (1, 8, 8),
        RopeScale = (1.0f, 4.0f, 4.0f),
        RopeTheta = 10_000.0f,
        ConcatPaddingMask = true,
        ConditionDim = 3 * Hidden,
        EmbeddedTimestepDim = EmbeddedDim,
        AdaLnLoraDim = AdaRank,
        RmsNormEps = 1e-6f,
        QkNormEps = 1e-6f,
        LlmAdapter = new AnimaLlmAdapterConfig
        {
            HiddenSize = TextWidth,
            NumHeads = Heads,
            HeadDim = HeadDim,
            FfnHiddenSize = Ffn,
            NumLayers = 0,
            CodebookVocab = 32,
            RmsNormEps = 1e-6f,
            QkNormEps = 1e-6f,
        },
    };

    /// <summary>Random weights for every key the tiny transformer and its blocks read, under the names the converter produces.</summary>
    private static Dictionary<string, Tensor> Weights(AnimaConfig config)
    {
        int patchFeatures = (config.InChannels + 1) * config.PatchSize.H * config.PatchSize.W;
        int finalFeatures = config.OutChannels * config.PatchSize.H * config.PatchSize.W;
        Random rng = new(1234);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            ["x_embedder.proj.1.weight"] = Noise(rng, new TensorShape(Hidden, patchFeatures), 0.2f),
            ["t_embedder.1.linear_1.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f),
            ["t_embedder.1.linear_2.weight"] = Noise(rng, new TensorShape(config.ConditionDim, Hidden), 0.2f),
            ["t_embedding_norm.weight"] = Noise(rng, new TensorShape(Hidden), 0.05f, center: 1f),
            ["final_layer.adaln_modulation.1.weight"] = Noise(rng, new TensorShape(AdaRank, Hidden), 0.2f),
            ["final_layer.adaln_modulation.2.weight"] = Noise(rng, new TensorShape(2 * Hidden, AdaRank), 0.2f),
            ["final_layer.linear.weight"] = Noise(rng, new TensorShape(finalFeatures, Hidden), 0.2f),
        };
        for (int i = 0; i < config.NumLayers; i++)
        {
            string p = $"blocks.{i}";
            // adaLN-LoRA per modulated sub-block: [embeddedDim → rank → 3·hidden].
            foreach (string site in new[] { "self_attn", "cross_attn", "mlp" })
            {
                weights[$"{p}.adaln_modulation_{site}.1.weight"] = Noise(rng, new TensorShape(AdaRank, EmbeddedDim), 0.2f);
                weights[$"{p}.adaln_modulation_{site}.2.weight"] = Noise(rng, new TensorShape(3 * Hidden, AdaRank), 0.2f);
            }
            weights[$"{p}.self_attn.q_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.self_attn.k_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.self_attn.v_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.self_attn.output_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.self_attn.q_norm.weight"] = Noise(rng, new TensorShape(HeadDim), 0.05f, center: 1f);
            weights[$"{p}.self_attn.k_norm.weight"] = Noise(rng, new TensorShape(HeadDim), 0.05f, center: 1f);
            weights[$"{p}.cross_attn.q_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.cross_attn.k_proj.weight"] = Noise(rng, new TensorShape(Hidden, TextWidth), 0.2f);
            weights[$"{p}.cross_attn.v_proj.weight"] = Noise(rng, new TensorShape(Hidden, TextWidth), 0.2f);
            weights[$"{p}.cross_attn.output_proj.weight"] = Noise(rng, new TensorShape(Hidden, Hidden), 0.2f);
            weights[$"{p}.cross_attn.q_norm.weight"] = Noise(rng, new TensorShape(HeadDim), 0.05f, center: 1f);
            weights[$"{p}.cross_attn.k_norm.weight"] = Noise(rng, new TensorShape(HeadDim), 0.05f, center: 1f);
            weights[$"{p}.mlp.layer1.weight"] = Noise(rng, new TensorShape(Ffn, Hidden), 0.2f);
            weights[$"{p}.mlp.layer2.weight"] = Noise(rng, new TensorShape(Hidden, Ffn), 0.2f);
        }
        return weights;
    }

    private static Tensor Noise(Random rng, TensorShape shape, float scale, float center = 0f)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = center + ((float)rng.NextDouble() * 2f - 1f) * scale;
        }
        return t;
    }

    private static Tensor Ramp(TensorShape shape, float phase)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f + phase);
        }
        return t;
    }

    private static Tensor Values(TensorShape shape, float start)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = start + i;
        }
        return t;
    }
}
