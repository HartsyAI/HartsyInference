using HartsyInference.Audio.Models.StyleTts2;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Synthetic-weights tests for StyleTTS 2's diffusion network — a <see cref="StyleTransformer1d"/>
/// forward (noisy style + sigma + PLBERT context → finite denoised) and a full <see cref="StyleDenoiser"/>
/// EDM/CFG step. Tiny dims, random weights, CpuBackend — no checkpoint required. Mirrors
/// <see cref="StyleTts2SamplerTests"/>.</summary>
public sealed unsafe class StyleTransformer1dTests
{
    private const int Channels = 16;
    private const int NumHeads = 2;
    private const int HeadFeatures = 8;     // inner = 16
    private const int ContextDim = 16;
    private const int Multiplier = 2;       // ff dim = 32
    private const int FfDim = Channels * Multiplier;
    private const int InnerDim = NumHeads * HeadFeatures;

    private static uint _rng = 0x1234_5678u;

    private static float NextUniform()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return (_rng / (float)uint.MaxValue - 0.5f) * 0.2f;
    }

    private static Tensor Rand(params int[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            1 => new TensorShape(dims[0]),
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            _ => throw new ArgumentException("Rand supports 1-3 dims.", nameof(dims)),
        };
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = NextUniform();
        return t;
    }

    private static StyleTransformerLayer BuildLayer() => new()
    {
        AdaLnW = Rand(2 * Channels, ContextDim), AdaLnB = Rand(2 * Channels),
        SelfNormW = Rand(Channels), SelfNormB = Rand(Channels),
        SelfQW = Rand(InnerDim, Channels), SelfQB = Rand(InnerDim),
        SelfKW = Rand(InnerDim, Channels), SelfKB = Rand(InnerDim),
        SelfVW = Rand(InnerDim, Channels), SelfVB = Rand(InnerDim),
        SelfOW = Rand(Channels, InnerDim), SelfOB = Rand(Channels),
        CrossNormW = Rand(Channels), CrossNormB = Rand(Channels),
        CrossQW = Rand(InnerDim, Channels), CrossQB = Rand(InnerDim),
        CrossKW = Rand(InnerDim, ContextDim), CrossKB = Rand(InnerDim),
        CrossVW = Rand(InnerDim, ContextDim), CrossVB = Rand(InnerDim),
        CrossOW = Rand(Channels, InnerDim), CrossOB = Rand(Channels),
        FfNormW = Rand(Channels), FfNormB = Rand(Channels),
        FfW1 = Rand(FfDim, Channels), FfB1 = Rand(FfDim),
        FfW2 = Rand(Channels, FfDim), FfB2 = Rand(Channels),
    };

    private static StyleTransformer1d BuildNet(int numLayers = 2)
    {
        StyleTransformerLayer[] layers = new StyleTransformerLayer[numLayers];
        for (int i = 0; i < numLayers; i++) layers[i] = BuildLayer();
        return new StyleTransformer1d(
            Channels, NumHeads, HeadFeatures, ContextDim, Multiplier, layers,
            Rand(Channels, Channels), Rand(Channels), Rand(Channels, Channels), Rand(Channels),
            Rand(Channels, Channels), Rand(Channels), Rand(Channels, Channels), Rand(Channels));
    }

    [Fact]
    public void Forward_WithContext_ProducesFiniteDenoised_OfRightShape()
    {
        using CpuBackend backend = new();
        StyleTransformer1d net = BuildNet();
        using Tensor noisy = Rand(1, 1, Channels);
        using Tensor context = Rand(1, 7, ContextDim);
        using Tensor refStyle = Rand(1, 1, Channels);
        net.SetReferenceStyle(refStyle);

        using Tensor outT = net.Forward(backend, noisy, sigma: 0.5f, cond: context);
        Assert.Equal(1, (int)outT.Shape[0]);
        Assert.Equal(1, (int)outT.Shape[1]);
        Assert.Equal(Channels, (int)outT.Shape[2]);
        float* p = (float*)outT.DataPointer;
        for (int i = 0; i < Channels; i++) Assert.True(float.IsFinite(p[i]), $"out[{i}] must be finite");
    }

    [Fact]
    public void Forward_WithoutContext_IsFinite()
    {
        using CpuBackend backend = new();
        StyleTransformer1d net = BuildNet();
        using Tensor noisy = Rand(1, 1, Channels);
        using Tensor outT = net.Forward(backend, noisy, sigma: 1.0f, cond: null);
        float* p = (float*)outT.DataPointer;
        for (int i = 0; i < Channels; i++) Assert.True(float.IsFinite(p[i]), $"out[{i}] must be finite");
    }

    [Fact]
    public void Forward_NoWeights_ReturnsZeroFallback()
    {
        using CpuBackend backend = new();
        StyleTransformer1d net = new(
            Channels, NumHeads, HeadFeatures, ContextDim, Multiplier,
            [BuildLayer()], null, null, null, null, null, null, null, null);
        Assert.False(net.HasWeights);
        using Tensor noisy = Rand(1, 1, Channels);
        using Tensor outT = net.Forward(backend, noisy, sigma: 0.7f, cond: null);
        float* p = (float*)outT.DataPointer;
        for (int i = 0; i < Channels; i++) Assert.Equal(0f, p[i]);
    }

    [Fact]
    public void DenoiserStep_WithNetwork_IsFinite()
    {
        using CpuBackend backend = new();
        StyleTts2Config cfg = StyleTts2Config.LibriTts with { EmbeddingScale = 1.5f };
        StyleDenoiser denoiser = new(cfg);
        StyleTransformer1d net = BuildNetForDim(cfg.StyleDim);
        denoiser.SetNetwork(net);

        using Tensor textEmb = Rand(1, 5, cfg.StyleDim);
        using Tensor refStyle = Rand(1, cfg.StyleDim);
        denoiser.Prepare(textEmb, refStyle);

        using Tensor x = Rand(1, 1, cfg.StyleDim);
        using Tensor d = denoiser.Denoise(backend, x, sigma: 0.8f);
        Assert.Equal(cfg.StyleDim, (int)d.Shape[2]);
        float* p = (float*)d.DataPointer;
        for (int i = 0; i < cfg.StyleDim; i++) Assert.True(float.IsFinite(p[i]), $"D[{i}] must be finite");
    }

    [Fact]
    public void DenoiserStep_NoNetwork_FallsBackToCSkipOnly()
    {
        using CpuBackend backend = new();
        StyleTts2Config cfg = StyleTts2Config.LibriTts;
        StyleDenoiser denoiser = new(cfg);
        using Tensor x = Rand(1, 1, cfg.StyleDim);
        using Tensor d = denoiser.Denoise(backend, x, sigma: 0.5f);
        float* xp = (float*)x.DataPointer;
        float* dp = (float*)d.DataPointer;
        float sd = cfg.SigmaData;
        float cSkip = sd * sd / (0.5f * 0.5f + sd * sd);
        for (int i = 0; i < cfg.StyleDim; i++)
        {
            Assert.True(float.IsFinite(dp[i]));
            Assert.True(MathF.Abs(dp[i] - cSkip * xp[i]) < 1e-4f, $"D[{i}] should be c_skip·x with a zero network");
        }
    }

    private static StyleTransformer1d BuildNetForDim(int dim)
    {
        int innerDim = NumHeads * HeadFeatures;
        int ffDim = dim * Multiplier;
        StyleTransformerLayer layer = new()
        {
            AdaLnW = Rand(2 * dim, dim), AdaLnB = Rand(2 * dim),
            SelfNormW = Rand(dim), SelfNormB = Rand(dim),
            SelfQW = Rand(innerDim, dim), SelfQB = Rand(innerDim),
            SelfKW = Rand(innerDim, dim), SelfKB = Rand(innerDim),
            SelfVW = Rand(innerDim, dim), SelfVB = Rand(innerDim),
            SelfOW = Rand(dim, innerDim), SelfOB = Rand(dim),
            CrossNormW = Rand(dim), CrossNormB = Rand(dim),
            CrossQW = Rand(innerDim, dim), CrossQB = Rand(innerDim),
            CrossKW = Rand(innerDim, dim), CrossKB = Rand(innerDim),
            CrossVW = Rand(innerDim, dim), CrossVB = Rand(innerDim),
            CrossOW = Rand(dim, innerDim), CrossOB = Rand(dim),
            FfNormW = Rand(dim), FfNormB = Rand(dim),
            FfW1 = Rand(ffDim, dim), FfB1 = Rand(ffDim),
            FfW2 = Rand(dim, ffDim), FfB2 = Rand(dim),
        };
        return new StyleTransformer1d(
            dim, NumHeads, HeadFeatures, dim, Multiplier, [layer],
            Rand(dim, dim), Rand(dim), Rand(dim, dim), Rand(dim),
            Rand(dim, dim), Rand(dim), Rand(dim, dim), Rand(dim));
    }
}
