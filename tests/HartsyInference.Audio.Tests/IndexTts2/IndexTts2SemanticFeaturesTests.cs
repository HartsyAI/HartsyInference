using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.Wav2Vec2Bert;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random weights through <see cref="IndexTts2SemanticFeatures"/>: confirms the
/// <c>(hidden - mean) / sqrt(var)</c> normalization is applied correctly on top of
/// <see cref="Wav2Vec2BertExtractor"/>'s own (already-tested) front end. Says nothing about parity with the
/// real <c>facebook/w2v-bert-2.0</c> checkpoint or <c>wav2vec2bert_stats.pt</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2SemanticFeaturesTests
{
    [Fact]
    public unsafe void Forward_AppliesMeanVarianceNormalization_OnTopOfTheRawExtractor()
    {
        const int hidden = 1024;
        Random rng = new(1);

        Dictionary<string, Tensor> w = [];
        Wav2Vec2BertConfig cfg = Wav2Vec2BertConfig.V2(17);
        w["feature_projection.layer_norm.weight"] = Rand(rng, cfg.FeatureProjectionInputDim);
        w["feature_projection.layer_norm.bias"] = Rand(rng, cfg.FeatureProjectionInputDim);
        w["feature_projection.projection.weight"] = Rand(rng, hidden, cfg.FeatureProjectionInputDim);
        w["feature_projection.projection.bias"] = Rand(rng, hidden);
        for (int i = 0; i < 17; i++) AddConformerLayer(w, rng, $"encoder.layers.{i}", hidden);

        Tensor mean = Rand(rng, hidden);
        Tensor variance = new(new TensorShape(hidden), DType.F32);
        foreach (ref float v in variance.AsSpan<float>()) v = (float)(rng.NextDouble() * 0.5 + 0.5);   // > 0

        using IndexTts2SemanticFeatures features = new();
        features.LoadWeights(w, "", mean, variance);

        float[] audio = new float[16_000];   // 1 second of silence-ish noise at 16kHz.
        for (int i = 0; i < audio.Length; i++) audio[i] = (float)((rng.NextDouble() * 2 - 1) * 0.01);

        using CpuBackend backend = new();
        using Tensor result = features.Forward(backend, audio);

        Assert.Equal(hidden, (int)result.Shape[2]);
        foreach (float v in result.AsSpan<float>()) Assert.True(float.IsFinite(v));

        foreach (Tensor t in w.Values) t.Dispose();
        mean.Dispose();
        variance.Dispose();
    }

    private static Tensor Rand(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * 0.05);
        return t;
    }

    private static void AddConformerLayer(Dictionary<string, Tensor> w, Random rng, string prefix, int hidden)
    {
        int intermediate = 4096, convKernel = 31, heads = 16;
        foreach (string ffn in new[] { "ffn1", "ffn2" })
        {
            w[$"{prefix}.{ffn}_layer_norm.weight"] = Rand(rng, hidden);
            w[$"{prefix}.{ffn}_layer_norm.bias"] = Rand(rng, hidden);
            w[$"{prefix}.{ffn}.intermediate_dense.weight"] = Rand(rng, intermediate, hidden);
            w[$"{prefix}.{ffn}.intermediate_dense.bias"] = Rand(rng, intermediate);
            w[$"{prefix}.{ffn}.output_dense.weight"] = Rand(rng, hidden, intermediate);
            w[$"{prefix}.{ffn}.output_dense.bias"] = Rand(rng, hidden);
        }
        w[$"{prefix}.self_attn_layer_norm.weight"] = Rand(rng, hidden);
        w[$"{prefix}.self_attn_layer_norm.bias"] = Rand(rng, hidden);
        foreach (string proj in new[] { "linear_q", "linear_k", "linear_v", "linear_out" })
        {
            w[$"{prefix}.self_attn.{proj}.weight"] = Rand(rng, hidden, hidden);
            w[$"{prefix}.self_attn.{proj}.bias"] = Rand(rng, hidden);
        }
        w[$"{prefix}.self_attn.distance_embedding.weight"] = Rand(rng, 64 + 8 + 1, hidden / heads);
        w[$"{prefix}.conv_module.layer_norm.weight"] = Rand(rng, hidden);
        w[$"{prefix}.conv_module.layer_norm.bias"] = Rand(rng, hidden);
        w[$"{prefix}.conv_module.pointwise_conv1.weight"] = Rand(rng, 2 * hidden, hidden, 1);
        w[$"{prefix}.conv_module.depthwise_conv.weight"] = Rand(rng, hidden, 1, convKernel);
        w[$"{prefix}.conv_module.depthwise_layer_norm.weight"] = Rand(rng, hidden);
        w[$"{prefix}.conv_module.depthwise_layer_norm.bias"] = Rand(rng, hidden);
        w[$"{prefix}.conv_module.pointwise_conv2.weight"] = Rand(rng, hidden, hidden, 1);
        w[$"{prefix}.final_layer_norm.weight"] = Rand(rng, hidden);
        w[$"{prefix}.final_layer_norm.bias"] = Rand(rng, hidden);
    }
}
