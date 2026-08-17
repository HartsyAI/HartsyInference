using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Pins the pure math of the LTX-2.5 two-stage (latent-upsample + refine) flow against the shipped
/// <c>video_ltx2_5_t2v</c> template. Each of these fails silently end to end: an inverted normalization, a
/// transposed repack and a re-noise with the blend the wrong way round all produce a plausible video.</summary>
public sealed unsafe class LtxVideo2TwoStageTests
{
    /// <summary>The template's stage-2 ManualSigmas node, verbatim — including its literal 0.4219 (not 0.421875).
    /// It is NOT a tail of the distilled 8-step schedule — that one's corresponding entry is 0.909375, and using
    /// it would start the refine 6% further from the data.</summary>
    [Fact]
    public void RefineSigmasMatchTheShippedTemplate()
    {
        float[] expected = [0.85f, 0.725f, 0.4219f, 0.0f];
        Assert.Equal<IEnumerable<float>>(expected, LtxVideo2Config.Ltx25TwoStageRefineSigmas);
        Assert.NotEqual(LtxVideo2Config.Ltx25DistilledSigmas[5], LtxVideo2Config.Ltx25TwoStageRefineSigmas[0]);
    }

    /// <summary>A shared array would let one in-place edit anywhere corrupt every config built afterwards.</summary>
    [Fact]
    public void RefineSigmasAreAFreshArrayPerCall()
    {
        float[] first = LtxVideo2Config.Ltx25TwoStageRefineSigmas;
        first[0] = -1f;
        Assert.Equal(0.85f, LtxVideo2Config.Ltx25TwoStageRefineSigmas[0]);
    }

    /// <summary>ComfyUI <c>ModelSamplingDiscreteFlow.noise_scaling</c>: <c>sigma·noise + (1−sigma)·latent</c>. At
    /// sigma 0.85 the stage-1 result keeps 15% of its weight, so a swapped blend is a nearly-clean re-generation
    /// that still looks like a video.</summary>
    [Fact]
    public void RenoiseBlendsTowardNoiseBySigma()
    {
        using Tensor x = new Tensor(new TensorShape(2, 2), DType.F32);
        using Tensor noise = new Tensor(new TensorShape(2, 2), DType.F32);
        float* xp = (float*)x.DataPointer, np = (float*)noise.DataPointer;
        for (int i = 0; i < 4; i++) { xp[i] = 4f; np[i] = -2f; }

        using Tensor mixed = LtxVideo2Pipeline.Renoise(x, noise, 0.85f);
        float* mp = (float*)mixed.DataPointer;
        for (int i = 0; i < 4; i++)
            Assert.Equal(0.85f * -2f + 0.15f * 4f, mp[i], 5);
        // The inputs must survive: the audio latent is device-resident and an in-place host write would go stale
        // against its GPU cache.
        Assert.Equal(4f, xp[0]);
    }

    /// <summary>Sigma 0 is the identity and sigma 1 discards the latent entirely.</summary>
    [Theory]
    [InlineData(0f, 4f)]
    [InlineData(1f, -2f)]
    public void RenoiseEndpointsAreExact(float sigma, float expected)
    {
        using Tensor x = new Tensor(new TensorShape(1, 1), DType.F32);
        using Tensor noise = new Tensor(new TensorShape(1, 1), DType.F32);
        *(float*)x.DataPointer = 4f;
        *(float*)noise.DataPointer = -2f;
        using Tensor mixed = LtxVideo2Pipeline.Renoise(x, noise, sigma);
        Assert.Equal(expected, *(float*)mixed.DataPointer, 5);
    }

    /// <summary>The upsampler is defined on UN-normalized latents, so the transition un-normalizes in and
    /// re-normalizes out. A round-trip test passes with BOTH directions swapped — these are absolute pins against
    /// ComfyUI's <c>per_channel_statistics.un_normalize</c> (<c>x·std + mean</c>).</summary>
    [Fact]
    public void LatentNormalizationDirectionsArePinnedAbsolutely()
    {
        float[] mean = [1f, 1f], std = [2f, 2f];
        using Tensor latent = new Tensor(new TensorShape([1L, 2L, 1L, 1L, 1L]), DType.F32);
        float* p = (float*)latent.DataPointer;
        p[0] = 3f; p[1] = 3f;

        using Tensor denorm = LtxVideo2VaeDecoder.DenormalizeLatent(latent, mean, std);
        Assert.Equal(7f, ((float*)denorm.DataPointer)[0], 5);

        using Tensor renorm = LtxVideo2VaeDecoder.NormalizeLatent(denorm, mean, std);
        Assert.Equal(3f, ((float*)renorm.DataPointer)[0], 5);
    }

    /// <summary>The repack is the inverse of the unpack the VAE decode already uses, in (f,h,w) token order with
    /// channel last. A transposed H/W would survive a square grid, so this uses 2x3x4.</summary>
    [Fact]
    public void PackIsTheInverseOfUnpack()
    {
        const int t = 2, h = 3, w = 4, c = 5;
        using Tensor tokens = new Tensor(new TensorShape(t * h * w, c), DType.F32);
        float* tp = (float*)tokens.DataPointer;
        for (long i = 0; i < tokens.ElementCount; i++) tp[i] = i * 0.25f;

        using Tensor volume = LtxVideo2Pipeline.UnpackVideoLatents(tokens, t, h, w, c);
        Assert.Equal(c, (int)volume.Shape[1]);
        Assert.Equal(t, (int)volume.Shape[2]);
        Assert.Equal(h, (int)volume.Shape[3]);
        Assert.Equal(w, (int)volume.Shape[4]);
        using Tensor roundTrip = LtxVideo2Pipeline.PackVideoLatents(volume, c);
        float* rp = (float*)roundTrip.DataPointer;
        for (long i = 0; i < tokens.ElementCount; i++) Assert.Equal(tp[i], rp[i]);
    }

    /// <summary>Stage 1 is the request HALVED then snapped down to the 32-px latent grid, and stage 2 doubles it —
    /// so a half-size that is not a whole number of cells costs a latent row. 1280x736 renders 1280x704.</summary>
    [Theory]
    [InlineData(1280, 736, 20, 11, 1280, 704)]
    [InlineData(1280, 720, 20, 11, 1280, 704)]
    [InlineData(768, 512, 12, 8, 768, 512)]
    [InlineData(512, 320, 8, 5, 512, 320)]
    public void TwoStageGridSnapsDownAtTheHalfResolution(int width, int height,
        int expectStage1W, int expectStage1H, int expectWidth, int expectHeight)
    {
        (int h1, int w1, int h2, int w2) = LtxVideo2Pipeline.TwoStageGrid(width, height, 32);
        Assert.Equal(expectStage1H, h1);
        Assert.Equal(expectStage1W, w1);
        Assert.Equal(expectWidth, w2 * 32);
        Assert.Equal(expectHeight, h2 * 32);
    }

    /// <summary>Below two latent cells on an axis there is no half-resolution grid to denoise.</summary>
    [Fact]
    public void TwoStageGridRejectsGeometryBelowTwoLatentCells()
        => Assert.Throws<ArgumentException>(() => LtxVideo2Pipeline.TwoStageGrid(32, 512, 32));

    /// <summary>Ancestral (eta=1) coefficients, against values read out of ComfyUI's own
    /// <c>sample_euler_ancestral_RF</c> expressions at the two schedules this pipeline runs. The deterministic
    /// step goes to <c>sigma_down = s1²/s0</c>, NOT to <c>s1</c> — using the plain <c>s0−s1</c> delta alongside the
    /// noise injection would over-denoise and over-noise at every step.</summary>
    [Theory]
    [InlineData(0.85f, 0.725f, 0.231617647f, 0.720616570f, 0.571883618f)]
    [InlineData(0.725f, 0.421875f, 0.479512392f, 0.766223333f, 0.377620885f)]
    [InlineData(1.0f, 0.99375f, 0.012460937f, 0.501567398f, 0.861510149f)]
    public void AncestralCoefficientsMatchComfyUi(float s0, float s1, float delta, float zScale, float noiseScale)
    {
        (float gotDelta, float gotZ, float gotNoise) = LtxVideo2Pipeline.AncestralCoefficients(s0, s1);
        Assert.Equal(delta, gotDelta, 5);
        Assert.Equal(zScale, gotZ, 5);
        Assert.Equal(noiseScale, gotNoise, 5);
    }

    /// <summary>The terminal pair degenerates to plain Euler — same delta, identity blend, no noise. Without this
    /// the last step would inject noise into the final latent.</summary>
    [Fact]
    public void AncestralTerminalStepIsPlainEuler()
    {
        (float delta, float zScale, float noiseScale) = LtxVideo2Pipeline.AncestralCoefficients(0.421875f, 0f);
        Assert.Equal(0.421875f, delta, 6);
        Assert.Equal(1f, zScale, 6);
        Assert.Equal(0f, noiseScale, 6);
    }

    /// <summary>The property that makes the injection marginal-preserving: the deterministic step's residual noise
    /// scaled by the blend, plus the injected noise, sums in quadrature back to <c>s1</c>.</summary>
    [Theory]
    [InlineData(0.85f, 0.725f)]
    [InlineData(0.725f, 0.421875f)]
    [InlineData(0.975f, 0.909375f)]
    public void AncestralInjectionRestoresTheTargetSigma(float s0, float s1)
    {
        (float delta, float zScale, float noiseScale) = LtxVideo2Pipeline.AncestralCoefficients(s0, s1);
        float sigmaDown = s0 - delta;
        float restored = zScale * sigmaDown * (zScale * sigmaDown) + noiseScale * noiseScale;
        Assert.True(System.MathF.Abs(restored - s1 * s1) < 1e-6f, $"restored {restored:F8} vs s1² {s1 * s1:F8}");
    }
}
