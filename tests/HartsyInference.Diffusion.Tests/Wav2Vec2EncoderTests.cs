using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat Avatar 1.0 options of <see cref="Wav2Vec2Encoder"/> on a tiny random-weight model: the feature-rate
/// hook, the exact-GELU and biased-variance switches, and that the defaults the S2V presets use are untouched.</summary>
public sealed class Wav2Vec2EncoderTests
{
    private static Wav2Vec2EncoderConfig Tiny => new()
    {
        ConvDims = [8, 8, 8],
        ConvStrides = [2, 2, 2],
        ConvKernels = [3, 3, 2],
        Hidden = 16,
        NumLayers = 2,
        NumHeads = 2,
        Intermediate = 32,
        PosConvKernel = 4,
        PosConvGroups = 2,
        NormalizeInput = true,
    };

    private static (Wav2Vec2Encoder Encoder, List<Tensor> Owned) Build(Wav2Vec2EncoderConfig cfg)
    {
        Random rng = new(11);
        List<Tensor> owned = [];
        Dictionary<string, Tensor> w = new();

        void Make(string name, params long[] dims)
        {
            Tensor t = new(new TensorShape(dims), DType.F32);
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5) * 0.5f;
            w[name] = t;
            owned.Add(t);
        }

        void Norm(string name, long d)
        {
            Make(name + ".weight", d);
            w[name + ".weight"].AsSpan<float>().Fill(1f);
            Make(name + ".bias", d);
            w[name + ".bias"].AsSpan<float>().Clear();
        }

        int prev = 1;
        for (int i = 0; i < cfg.ConvDims.Length; i++)
        {
            Make($"feature_extractor.conv_layers.{i}.conv.weight", cfg.ConvDims[i], prev, cfg.ConvKernels[i]);
            prev = cfg.ConvDims[i];
        }
        Norm("feature_extractor.conv_layers.0.layer_norm", cfg.ConvDims[0]);
        Norm("feature_projection.layer_norm", prev);
        Make("feature_projection.projection.weight", cfg.Hidden, prev);
        Make("feature_projection.projection.bias", cfg.Hidden);
        Make("encoder.pos_conv_embed.conv.weight", cfg.Hidden, cfg.Hidden / cfg.PosConvGroups, cfg.PosConvKernel);
        Make("encoder.pos_conv_embed.conv.bias", cfg.Hidden);
        Norm("encoder.layer_norm", cfg.Hidden);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"encoder.layers.{i}";
            foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            {
                Make($"{p}.attention.{proj}.weight", cfg.Hidden, cfg.Hidden);
                Make($"{p}.attention.{proj}.bias", cfg.Hidden);
            }
            Norm($"{p}.layer_norm", cfg.Hidden);
            Make($"{p}.feed_forward.intermediate_dense.weight", cfg.Intermediate, cfg.Hidden);
            Make($"{p}.feed_forward.intermediate_dense.bias", cfg.Intermediate);
            Make($"{p}.feed_forward.output_dense.weight", cfg.Hidden, cfg.Intermediate);
            Make($"{p}.feed_forward.output_dense.bias", cfg.Hidden);
            Norm($"{p}.final_layer_norm", cfg.Hidden);
        }

        Wav2Vec2Encoder encoder = new(cfg);
        encoder.LoadWeights(w);
        return (encoder, owned);
    }

    private static float[] Speech(int n)
    {
        float[] s = new float[n];
        for (int i = 0; i < n; i++) s[i] = 0.4f * MathF.Sin(i * 0.05f) + 0.1f * MathF.Sin(i * 0.31f) + 0.05f;
        return s;
    }

    private static float[] Encode(Wav2Vec2EncoderConfig cfg, float[] speech, int featureFrames, out long[] shape)
    {
        (Wav2Vec2Encoder encoder, List<Tensor> owned) = Build(cfg);
        using CpuBackend backend = new();
        using Tensor states = encoder.EncodeAllLayers(backend, speech, featureFrames);
        shape = [states.Shape[0], states.Shape[1], states.Shape[2]];
        float[] result = states.AsReadOnlySpan<float>().ToArray();
        foreach (Tensor t in owned) t.Dispose();
        return result;
    }

    // Conv frames of the tiny stack: ((n-3)/2+1 -3)/2+1 -2)/2+1.
    private static int ConvFrames(int samples)
    {
        int t = (samples - 3) / 2 + 1;
        t = (t - 3) / 2 + 1;
        return (t - 2) / 2 + 1;
    }

    [Fact]
    public void FeatureFrames_AtTheConvRate_IsTheDefaultPath()
    {
        float[] speech = Speech(400);
        int natural = ConvFrames(speech.Length);

        float[] plain = Encode(Tiny, speech, 0, out long[] plainShape);
        float[] same = Encode(Tiny, speech, natural, out _);

        Assert.Equal([natural, 3, 16], plainShape);
        Assert.Equal(plain, same);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(40)]
    public void FeatureFrames_SetsTheOutputRate(int frames)
    {
        float[] states = Encode(Tiny, Speech(400), frames, out long[] shape);

        Assert.Equal([frames, 3, 16], shape);
        Assert.All(states, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void ExactGelu_ChangesTheOutputByTheTanhApproximationGapOnly()
    {
        float[] speech = Speech(400);
        float[] tanh = Encode(Tiny, speech, 0, out _);
        float[] erf = Encode(Tiny with { ExactGelu = true }, speech, 0, out _);

        Assert.NotEqual(tanh, erf);
        double sumSq = 0, errSq = 0;
        for (int i = 0; i < tanh.Length; i++)
        {
            sumSq += (double)erf[i] * erf[i];
            errSq += (double)(tanh[i] - erf[i]) * (tanh[i] - erf[i]);
        }
        Assert.True(Math.Sqrt(errSq / sumSq) < 0.05, $"relRMS {Math.Sqrt(errSq / sumSq)}");
    }

    [Fact]
    public void BiasedVariance_DiffersFromUnbiasedOnlyByTheSampleCountFactor()
    {
        float[] speech = Speech(400);
        float[] unbiased = Encode(Tiny, speech, 0, out _);
        float[] biased = Encode(Tiny with { BiasedNormalizeVariance = true }, speech, 0, out _);

        Assert.NotEqual(unbiased, biased);
        double sumSq = 0, errSq = 0;
        for (int i = 0; i < unbiased.Length; i++)
        {
            sumSq += (double)unbiased[i] * unbiased[i];
            errSq += (double)(unbiased[i] - biased[i]) * (unbiased[i] - biased[i]);
        }
        Assert.True(Math.Sqrt(errSq / sumSq) < 0.02, $"relRMS {Math.Sqrt(errSq / sumSq)}");
    }

    [Fact]
    public void TheAvatarV10Preset_IsTheBaseShapeWithTheHfFrontEndOptions()
    {
        Wav2Vec2EncoderConfig preset = Wav2Vec2EncoderConfig.LongCatAvatarV10;

        Assert.Equal(768, preset.Hidden);
        Assert.Equal(12, preset.NumLayers);
        Assert.True(preset.NormalizeInput && preset.BiasedNormalizeVariance && preset.ExactGelu);
        Assert.False(Wav2Vec2EncoderConfig.Base.ExactGelu);
        Assert.False(Wav2Vec2EncoderConfig.Large.ExactGelu);
        Assert.False(Wav2Vec2EncoderConfig.Large.BiasedNormalizeVariance);
    }
}
