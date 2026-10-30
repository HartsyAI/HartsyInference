using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Tests.Common;

/// <summary>A tiny Anima transformer with random weights, for the reference K/V path. The sizes are the smallest that exercise every
/// block key the transformer reads. The CPU and CUDA test projects share this config and these weights.</summary>
public static class AnimaSyntheticWeights
{
    /// <summary>Hidden width of the blocks.</summary>
    public const int Hidden = 8;

    /// <summary>Attention heads; <see cref="HeadDim"/> is the width of each.</summary>
    public const int Heads = 2;

    /// <summary>Width of one attention head.</summary>
    public const int HeadDim = 4;

    /// <summary>Width of the feed-forward hidden layer (twice <see cref="Hidden"/>).</summary>
    public const int Ffn = 16;

    /// <summary>Width of the text features the cross-attention reads (the LLM-adapter output).</summary>
    public const int TextWidth = 8;

    /// <summary>Width of the timestep embedding the adaLN-LoRA modulators read.</summary>
    public const int EmbeddedDim = 8;

    /// <summary>Rank of the adaLN-LoRA bottleneck.</summary>
    public const int AdaRank = 4;

    /// <summary>The config: two blocks, the sizes above, a 2-channel latent, and 2×2 patches.</summary>
    public static AnimaConfig TinyConfig() => new()
    {
        HiddenSize = Hidden,
        NumHeads = Heads,
        HeadDim = HeadDim,
        NumLayers = 2,
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
    public static Dictionary<string, Tensor> Weights(AnimaConfig config)
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

    /// <summary>A deterministic sine ramp, for latents, text and references; repeated calls give the same values.</summary>
    public static Tensor Ramp(TensorShape shape, float phase)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Sin(i * 0.37f + phase);
        }
        return t;
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
}
