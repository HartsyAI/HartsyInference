using HartsyInference.Audio.Models.Denoise;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Covers RNNoise's band/DCT front-end, whose tables are computed from closed-form formulas rather than
/// shipped as data. These fail silently if wrong: a mis-scaled DCT or a non-power-complementary window still
/// produces plausible audio, just with the wrong feature scaling, and the only symptom is a model that denoises
/// slightly worse than it should.</summary>
public sealed class RnnoiseBandsTests
{
    [Fact]
    public void FeatureLayout_Is_TwoBandsPlusPitch()
    {
        Assert.Equal(32, RnnoiseBands.BandCount);
        Assert.Equal(65, RnnoiseBands.FeatureCount);
        Assert.Equal(2 * RnnoiseBands.BandCount + 1, RnnoiseBands.FeatureCount);
        // 481 bins is a 960-point transform's half-spectrum; the band edges are indices into it.
        Assert.Equal(481, RnnoiseBands.FreqSize);
    }

    /// <summary>The analysis window is applied twice — once on analysis, once on synthesis — so at 50% overlap it
    /// must satisfy <c>w[i]² + w[i+N]² = 1</c> for overlap-add to reconstruct unity. A plain Hann does not have
    /// this property under double application, which is exactly why RNNoise uses the vorbis form.</summary>
    [Fact]
    public void Window_Is_PowerComplementary()
    {
        const int N = 480;
        float[] w = RnnoiseBands.BuildWindow(N);
        Assert.Equal(2 * N, w.Length);
        for (int i = 0; i < N; i++)
        {
            float sum = w[i] * w[i] + w[i + N] * w[i + N];
            Assert.True(MathF.Abs(sum - 1f) < 1e-5f, $"w[{i}]^2 + w[{i + N}]^2 = {sum}, expected 1");
        }
    }

    [Fact]
    public void Window_Is_Symmetric_And_Rises_From_Near_Zero()
    {
        const int N = 480;
        float[] w = RnnoiseBands.BuildWindow(N);
        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(w[i] - w[2 * N - 1 - i]) < 1e-6f, $"asymmetry at {i}");
        Assert.True(w[0] < 0.01f, $"window should start near zero, got {w[0]}");
        Assert.True(w[N - 1] > 0.99f, $"window should reach unity at the midpoint, got {w[N - 1]}");
    }

    /// <summary>A constant input has all its DCT energy in bin 0; anything else means the basis or the j=0
    /// sqrt(0.5) scaling is wrong.</summary>
    [Fact]
    public void Dct_Of_Constant_Concentrates_In_First_Bin()
    {
        float[] input = new float[RnnoiseBands.BandCount];
        Array.Fill(input, 1f);
        float[] output = new float[RnnoiseBands.BandCount];
        RnnoiseBands.Dct(input, output);

        Assert.True(MathF.Abs(output[0]) > 1f, $"bin 0 should carry the energy, got {output[0]}");
        for (int i = 1; i < output.Length; i++)
            Assert.True(MathF.Abs(output[i]) < 1e-4f, $"bin {i} should be ~0 for constant input, got {output[i]}");
    }

    /// <summary>Uniform band gains must interpolate to a flat response across the modelled range, and the bins
    /// above the last band edge must stay zero — the model does not score 20-24 kHz, and passing that tail
    /// through unmodified would leak exactly the noise this is meant to remove.</summary>
    [Fact]
    public void InterpolateBandGain_Is_Flat_For_Uniform_Gains_And_Zero_Above_The_Last_Band()
    {
        float[] bands = new float[RnnoiseBands.BandCount];
        Array.Fill(bands, 0.5f);
        float[] bins = new float[RnnoiseBands.FreqSize];
        RnnoiseBands.InterpolateBandGain(bands, bins);

        for (int k = 0; k < 400; k++)
            Assert.True(MathF.Abs(bins[k] - 0.5f) < 1e-5f, $"bin {k} = {bins[k]}, expected flat 0.5");
        for (int k = 400; k < RnnoiseBands.FreqSize; k++)
            Assert.Equal(0f, bins[k]);
    }

    /// <summary>Band energy must be scale-exact in power: doubling the spectrum quadruples every band.</summary>
    [Fact]
    public void BandEnergy_Scales_As_Power()
    {
        int bins = RnnoiseBands.FreqSize;
        float[] re = new float[bins];
        float[] im = new float[bins];
        Random rng = new Random(11);
        for (int k = 0; k < bins; k++)
        {
            re[k] = (float)(rng.NextDouble() - 0.5);
            im[k] = (float)(rng.NextDouble() - 0.5);
        }
        float[] baseline = new float[RnnoiseBands.BandCount];
        RnnoiseBands.ComputeBandEnergy(re, im, baseline);

        for (int k = 0; k < bins; k++) { re[k] *= 2f; im[k] *= 2f; }
        float[] doubled = new float[RnnoiseBands.BandCount];
        RnnoiseBands.ComputeBandEnergy(re, im, doubled);

        for (int i = 0; i < RnnoiseBands.BandCount; i++)
            Assert.True(MathF.Abs(doubled[i] - 4f * baseline[i]) < 1e-3f * MathF.Max(1f, doubled[i]),
                $"band {i}: {doubled[i]} vs 4x{baseline[i]}");
    }

    /// <summary>Correlating a spectrum with itself is its own energy — the cross-term path must reduce to the
    /// energy path, or the pitch-correlation feature is measuring something other than correlation.</summary>
    [Fact]
    public void BandCorrelation_With_Itself_Equals_BandEnergy()
    {
        int bins = RnnoiseBands.FreqSize;
        float[] re = new float[bins];
        float[] im = new float[bins];
        Random rng = new Random(23);
        for (int k = 0; k < bins; k++)
        {
            re[k] = (float)(rng.NextDouble() - 0.5);
            im[k] = (float)(rng.NextDouble() - 0.5);
        }
        float[] energy = new float[RnnoiseBands.BandCount];
        float[] correlation = new float[RnnoiseBands.BandCount];
        RnnoiseBands.ComputeBandEnergy(re, im, energy);
        RnnoiseBands.ComputeBandCorrelation(re, im, re, im, correlation);

        for (int i = 0; i < RnnoiseBands.BandCount; i++)
            Assert.Equal(energy[i], correlation[i], precision: 5);
    }
}
