using SharpInference.Audio.Models.StyleTts2;
using SharpInference.Core.Backends;
using SharpInference.Cpu;
using SharpInference.Core.Tensors;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Checkpoint-free tests for StyleTTS 2's diffusion style sampler — the Karras sigma schedule
/// (exact endpoints + monotonicity) and the ADPM2 sampler loop (deterministic, finite, right shape).
/// The StyleTransformer1d denoiser network + speech encoders need <c>StyleTTS2-LibriTTS</c>.</summary>
public sealed unsafe class StyleTts2SamplerTests
{
    private sealed class ZeroDenoiser : IStyleDenoiser
    {
        public Tensor Denoise(IBackend backend, Tensor x, float sigma) => new(x.Shape, DType.F32);
    }

    [Fact]
    public void KarrasSchedule_HasExactEndpoints_AndIsDescending()
    {
        float[] s = StyleDiffusionSampler.KarrasSchedule(5, 1e-4f, 3.0f, 9.0f);
        Assert.Equal(6, s.Length);
        Assert.True(MathF.Abs(s[0] - 3.0f) < 1e-3f, $"σ_0 should be σ_max=3.0, got {s[0]}");
        Assert.True(MathF.Abs(s[4] - 1e-4f) < 1e-5f, $"σ_(n-1) should be σ_min=1e-4, got {s[4]}");
        Assert.Equal(0f, s[5]);
        for (int i = 0; i < 5; i++) Assert.True(s[i] > s[i + 1], $"schedule must strictly descend at {i}");
    }

    [Fact]
    public void Sample_ProducesStyleVector_OfRightShapeAndFinite()
    {
        using CpuBackend backend = new();
        StyleDiffusionSampler sampler = new(StyleTts2Config.LibriTts);
        using Tensor s = sampler.Sample(backend, new ZeroDenoiser(), seed: 11);
        Assert.Equal(1, (int)s.Shape[0]);
        Assert.Equal(1, (int)s.Shape[1]);
        Assert.Equal(256, (int)s.Shape[2]);
        float* p = (float*)s.DataPointer;
        for (int i = 0; i < 256; i++) Assert.True(float.IsFinite(p[i]), $"style[{i}] must be finite");
    }

    [Fact]
    public void Sample_IsDeterministic_ForFixedSeed()
    {
        using CpuBackend backend = new();
        StyleDiffusionSampler sampler = new(StyleTts2Config.LibriTts);
        using Tensor a = sampler.Sample(backend, new ZeroDenoiser(), seed: 5);
        using Tensor b = sampler.Sample(backend, new ZeroDenoiser(), seed: 5);
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        for (int i = 0; i < 256; i++) Assert.Equal(ap[i], bp[i]);
    }

    [Fact]
    public void Config_StyleDim_Is256()
    {
        Assert.Equal(256, StyleTts2Config.LibriTts.StyleDim);
        Assert.False(StyleTts2Config.LjSpeech.MultiSpeaker);
    }
}
