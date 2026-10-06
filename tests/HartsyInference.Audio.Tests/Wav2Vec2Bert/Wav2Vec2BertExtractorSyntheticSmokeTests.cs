using HartsyInference.Audio.Models.Wav2Vec2Bert;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests.Wav2Vec2Bert;

/// <summary>Tiny random weights through the full <see cref="Wav2Vec2BertExtractor"/> pipeline (fbank → per-bin
/// normalize → stride-2 stack → feature_projection → Conformer stack): shapes, finiteness and determinism only.
/// Says nothing about parity with the real <c>facebook/w2v-bert-2.0</c> checkpoint.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class Wav2Vec2BertExtractorSyntheticSmokeTests : IDisposable
{
    private const int Hidden = 16;
    private const int Intermediate = 32;
    private const int Heads = 2;
    private const int ConvKernel = 5;
    private const int LeftMax = 4;
    private const int RightMax = 2;
    private const int FeatProjInput = 160; // fixed: numMelBins(80) * stride(2), not config-driven.
    private const int NumLayers = 2;

    private static readonly Wav2Vec2BertConfig Cfg = new()
    {
        HiddenSize = Hidden,
        IntermediateSize = Intermediate,
        NumAttentionHeads = Heads,
        ConvDepthwiseKernelSize = ConvKernel,
        LeftMaxPositionEmbeddings = LeftMax,
        RightMaxPositionEmbeddings = RightMax,
        FeatureProjectionInputDim = FeatProjInput,
        NumLayersToRun = NumLayers,
    };

    private readonly List<Tensor> _owned = [];
    private readonly CpuBackend _backend = new();

    private Tensor Rand(Random rng, double amp, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * amp);
        _owned.Add(t);
        return t;
    }

    private Tensor NormWeight(Random rng, long dim)
    {
        Tensor t = Rand(rng, 0.1, dim);
        foreach (ref float v in t.AsSpan<float>()) v += 1f;
        return t;
    }

    private Dictionary<string, Tensor> BuildWeights(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        w["feature_projection.layer_norm.weight"] = NormWeight(rng, FeatProjInput);
        w["feature_projection.layer_norm.bias"] = Rand(rng, 0.02, FeatProjInput);
        w["feature_projection.projection.weight"] = Rand(rng, 0.1, Hidden, FeatProjInput);
        w["feature_projection.projection.bias"] = Rand(rng, 0.02, Hidden);

        int numPositions = LeftMax + RightMax + 1;
        int headDim = Hidden / Heads;
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"encoder.layers.{i}";
            foreach (string ffn in new[] { "ffn1", "ffn2" })
            {
                w[$"{p}.{ffn}_layer_norm.weight"] = NormWeight(rng, Hidden);
                w[$"{p}.{ffn}_layer_norm.bias"] = Rand(rng, 0.02, Hidden);
                w[$"{p}.{ffn}.intermediate_dense.weight"] = Rand(rng, 0.1, Intermediate, Hidden);
                w[$"{p}.{ffn}.intermediate_dense.bias"] = Rand(rng, 0.02, Intermediate);
                w[$"{p}.{ffn}.output_dense.weight"] = Rand(rng, 0.1, Hidden, Intermediate);
                w[$"{p}.{ffn}.output_dense.bias"] = Rand(rng, 0.02, Hidden);
            }

            w[$"{p}.self_attn_layer_norm.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.self_attn_layer_norm.bias"] = Rand(rng, 0.02, Hidden);
            foreach (string qkvo in new[] { "q", "k", "v", "out" })
            {
                w[$"{p}.self_attn.linear_{qkvo}.weight"] = Rand(rng, 0.1, Hidden, Hidden);
                w[$"{p}.self_attn.linear_{qkvo}.bias"] = Rand(rng, 0.02, Hidden);
            }
            w[$"{p}.self_attn.distance_embedding.weight"] = Rand(rng, 0.1, numPositions, headDim);

            w[$"{p}.conv_module.layer_norm.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.conv_module.layer_norm.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.conv_module.pointwise_conv1.weight"] = Rand(rng, 0.1, 2 * Hidden, Hidden, 1);
            w[$"{p}.conv_module.depthwise_conv.weight"] = Rand(rng, 0.1, Hidden, 1, ConvKernel);
            w[$"{p}.conv_module.depthwise_layer_norm.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.conv_module.depthwise_layer_norm.bias"] = Rand(rng, 0.02, Hidden);
            w[$"{p}.conv_module.pointwise_conv2.weight"] = Rand(rng, 0.1, Hidden, Hidden, 1);

            w[$"{p}.final_layer_norm.weight"] = NormWeight(rng, Hidden);
            w[$"{p}.final_layer_norm.bias"] = Rand(rng, 0.02, Hidden);
        }
        return w;
    }

    private static float[] SineWave(int samples, double freqHz, int sampleRate)
    {
        float[] a = new float[samples];
        for (int i = 0; i < samples; i++) a[i] = 0.2f * MathF.Sin((float)(2 * Math.PI * freqHz * i / sampleRate));
        return a;
    }

    [Fact]
    public void Forward_ProducesFiniteOutput_OfExpectedShape()
    {
        Random rng = new(1234);
        Dictionary<string, Tensor> weights = BuildWeights(rng);
        using Wav2Vec2BertExtractor extractor = new(Cfg);
        extractor.LoadWeights(weights);

        float[] audio = SineWave(16_000, 220.0, 16_000); // 1 s @ 16 kHz
        using Tensor output = extractor.Forward(_backend, audio);

        Assert.Equal(3, output.Shape.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(Hidden, output.Shape[2]);
        Assert.True(output.Shape[1] > 0);
        foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Forward_IsDeterministic_AcrossRepeatedCalls()
    {
        Random rng = new(99);
        Dictionary<string, Tensor> weights = BuildWeights(rng);
        using Wav2Vec2BertExtractor extractor = new(Cfg);
        extractor.LoadWeights(weights);

        float[] audio = SineWave(16_000, 440.0, 16_000);
        using Tensor a = extractor.Forward(_backend, audio);
        using Tensor b = extractor.Forward(_backend, audio);

        float[] sa = a.AsSpan<float>().ToArray(), sb = b.AsSpan<float>().ToArray();
        Assert.Equal(sa.Length, sb.Length);
        for (int i = 0; i < sa.Length; i++) Assert.Equal(sa[i], sb[i], 6);
    }

    [Fact]
    public void Forward_StackedFrameCount_IsHalfTheFbankFrameCount()
    {
        // 1 s @ 16 kHz, 25 ms window / 10 ms shift -> (16000-400)/160 + 1 = 98 fbank frames -> 49 stacked frames.
        Random rng = new(7);
        Dictionary<string, Tensor> weights = BuildWeights(rng);
        using Wav2Vec2BertExtractor extractor = new(Cfg);
        extractor.LoadWeights(weights);

        float[] audio = SineWave(16_000, 150.0, 16_000);
        using Tensor output = extractor.Forward(_backend, audio);
        Assert.Equal(49, (int)output.Shape[1]);
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Loads the real <c>facebook/w2v-bert-2.0</c> <c>model.safetensors</c> (too large to bundle; point
/// <c>WAV2VEC2_BERT_SAFETENSORS_PATH</c> at a local copy to exercise it). A clean <c>LoadWeights</c> (no missing
/// key) is itself a strong signal every key name in <see cref="Wav2Vec2BertExtractor"/>/
/// <see cref="Wav2Vec2BertConformerLayer"/> matches the real checkpoint; shape/finite are checked past that.</summary>
[Trait("Category", "Integration")]
public sealed class Wav2Vec2BertExtractorRealWeightTests
{
    [Fact]
    public void LoadWeights_AndForward_SucceedAgainstRealCheckpoint()
    {
        string? path = Environment.GetEnvironmentVariable("WAV2VEC2_BERT_SAFETENSORS_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;   // resource-gated: skip without the real checkpoint file

        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        try
        {
            // Only the first 17 layers are ever read downstream (IndexTTS-2's hidden_states[17]); loading fewer
            // than the real 24 is the point of NumLayersToRun, not a shortfall.
            Wav2Vec2BertConfig cfg = Wav2Vec2BertConfig.V2(numLayersToRun: 17);
            using Wav2Vec2BertExtractor extractor = new(cfg);
            extractor.LoadWeights(weights);

            float[] audio = new float[16_000];
            Random rng = new(42);
            for (int i = 0; i < audio.Length; i++) audio[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

            using CpuBackend backend = new();
            using Tensor output = extractor.Forward(backend, audio);
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(cfg.HiddenSize, (int)output.Shape[2]);
            Assert.True(output.Shape[1] > 0);
            foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
