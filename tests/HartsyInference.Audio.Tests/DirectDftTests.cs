using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The small-size direct DFT reads its twiddles from a cached table; this pins it bit-for-bit to the
/// inline <c>Math.Cos/Sin</c> evaluation it replaced, so the iSTFT vocoders that call it per frame (Kokoro at
/// n_fft = 20) reproduce their previous waveforms exactly.</summary>
public sealed class DirectDftTests
{
    [Theory]
    [InlineData(16)]
    [InlineData(20)]
    [InlineData(24)]
    [InlineData(40)]
    public void CachedTwiddles_BitIdenticalToInlineTrig(int n)
    {
        Random rng = new(n * 31);
        for (int trial = 0; trial < 50; trial++)
        {
            float[] re = new float[n], im = new float[n];
            for (int i = 0; i < n; i++)
            {
                re[i] = (float)(rng.NextDouble() * 2 - 1);
                // Half the trials are real-input frames (what the STFT feeds), half complex (the iSTFT inverse trick).
                im[i] = trial % 2 == 0 ? 0f : (float)(rng.NextDouble() * 2 - 1);
            }
            float[] expectedRe = new float[n], expectedIm = new float[n];
            InlineTrigReference(re, im, expectedRe, expectedIm, n);

            Fft.DirectDft(re, im, n);
            for (int k = 0; k < n; k++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(expectedRe[k]) == BitConverter.SingleToInt32Bits(re[k]),
                    $"n={n} trial={trial} re[{k}]: {expectedRe[k]:R} vs {re[k]:R}");
                Assert.True(BitConverter.SingleToInt32Bits(expectedIm[k]) == BitConverter.SingleToInt32Bits(im[k]),
                    $"n={n} trial={trial} im[{k}]: {expectedIm[k]:R} vs {im[k]:R}");
            }
        }
    }

    /// <summary>The previous implementation, verbatim: per-element double trig, double accumulation, float store.</summary>
    private static void InlineTrigReference(float[] re, float[] im, float[] outRe, float[] outIm, int n)
    {
        double twoPiOverN = 2.0 * Math.PI / n;
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0, sumIm = 0;
            for (int t = 0; t < n; t++)
            {
                double ang = twoPiOverN * k * t;
                double c = Math.Cos(ang), s = Math.Sin(ang);
                sumRe += re[t] * c + im[t] * s;
                sumIm += im[t] * c - re[t] * s;
            }
            outRe[k] = (float)sumRe;
            outIm[k] = (float)sumIm;
        }
    }
}
