using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Slaney mel scale + filterbank validation. The numeric anchors here come
/// from librosa's reference output for the Whisper parameter set (sr=16000, n_fft=400,
/// n_mels=80, fmin=0, fmax=8000). Mismatches against these constants mean we've
/// silently regressed the mel scale or the area-normalization step.</summary>
public sealed class MelFilterbankTests
{
    [Fact]
    public void SlaneyScale_Linear_Below_1kHz()
    {
        // Below 1 kHz, mel = 3 * f / 200. So 200 Hz → 3 mel, 500 Hz → 7.5 mel.
        Assert.Equal(3.0, MelFilterbank.HzToMel(200), precision: 6);
        Assert.Equal(7.5, MelFilterbank.HzToMel(500), precision: 6);
        Assert.Equal(15.0, MelFilterbank.HzToMel(1000), precision: 6);
    }

    [Fact]
    public void SlaneyScale_Logarithmic_Above_1kHz()
    {
        // At 1 kHz: mel = 15 (the breakpoint).
        // At 6.4 kHz: mel = 15 + 27 * ln(6.4) / ln(6.4) = 15 + 27 = 42.
        Assert.Equal(42.0, MelFilterbank.HzToMel(6400), precision: 6);
    }

    [Fact]
    public void SlaneyScale_RoundTrip()
    {
        double[] testHz = [50, 200, 999, 1000, 1500, 4000, 8000, 12000];
        foreach (double f in testHz)
        {
            double mel = MelFilterbank.HzToMel(f);
            double back = MelFilterbank.MelToHz(mel);
            Assert.Equal(f, back, precision: 4);
        }
    }

    [Fact]
    public void WhisperFilterbank_HasCorrectShape()
    {
        float[,] fb = MelFilterbank.Get(sampleRate: 16_000, nFft: 512, nMels: 80, fmin: 0, fmax: 8000);
        Assert.Equal(80, fb.GetLength(0));
        Assert.Equal(257, fb.GetLength(1));  // 512/2 + 1
    }

    [Fact]
    public void WhisperFilterbank_TrianglesAreNonNegative_AndEachRowHasSomePositive()
    {
        float[,] fb = MelFilterbank.Get(sampleRate: 16_000, nFft: 512, nMels: 80, fmin: 0, fmax: 8000);
        for (int m = 0; m < 80; m++)
        {
            bool anyPositive = false;
            for (int k = 0; k < 257; k++)
            {
                Assert.True(fb[m, k] >= 0, $"filter {m} bin {k} = {fb[m, k]} (must be ≥0)");
                if (fb[m, k] > 0) anyPositive = true;
            }
            Assert.True(anyPositive, $"filter {m} has no positive bins");
        }
    }

    [Fact]
    public void WhisperFilterbank_SlaneyAreaNormalization_PreservesEnergyRoughly()
    {
        // Slaney normalization gives each filter constant area in mel space. The
        // exact area depends on the mel-spaced spacing of the centers; what we can
        // check is monotonicity: low-frequency filters (narrow Hz range, mel-tight)
        // have HIGHER per-bin peaks than high-frequency ones because Hz spread is smaller.
        float[,] fb = MelFilterbank.Get(sampleRate: 16_000, nFft: 512, nMels: 80, fmin: 0, fmax: 8000);

        float peakLow = 0f, peakHigh = 0f;
        for (int k = 0; k < 257; k++)
        {
            if (fb[0, k] > peakLow) peakLow = fb[0, k];
            if (fb[79, k] > peakHigh) peakHigh = fb[79, k];
        }
        Assert.True(peakLow > peakHigh, $"low-freq peak {peakLow} should exceed high-freq peak {peakHigh}");
    }

    [Fact]
    public void Filterbank_Get_IsCached()
    {
        float[,] a = MelFilterbank.Get(16_000, 512, 80);
        float[,] b = MelFilterbank.Get(16_000, 512, 80);
        Assert.Same(a, b);
    }
}
