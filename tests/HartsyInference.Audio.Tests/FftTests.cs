using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the radix-2 FFT against analytically known results.
/// The mel spectrogram tests cover the end-to-end-vs-numpy comparison; these
/// component tests catch regressions in the FFT itself (e.g. forgetting bit
/// reversal, swapping forward/inverse sign convention, wrong twiddle indexing).</summary>
public sealed class FftTests
{
    [Fact]
    public void SingleSinusoid_ProducesDeltaAtCorrectBin()
    {
        // x[n] = cos(2*pi*k0*n/N). The FFT magnitude should be a delta at bins k0 and N-k0
        // (the conjugate). For real-input FFT we only see the first n/2+1 bins, so bin k0 should
        // have magnitude N/2, all others zero.
        int n = 256;
        int k0 = 13;
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = MathF.Cos(2f * MathF.PI * k0 * i / n);

        float[] re = new float[n / 2 + 1];
        float[] im = new float[n / 2 + 1];
        Fft.RealTransform(x, re, im, n);

        for (int k = 0; k < n / 2 + 1; k++)
        {
            float mag = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
            if (k == k0)
                Assert.Equal(n / 2f, mag, precision: 1);
            else
                Assert.True(mag < 0.01f, $"bin {k}: magnitude {mag} should be ~0");
        }
    }

    [Fact]
    public void FftSize_512_RoundTripsConsistently()
    {
        // The legacy Whisper layout (still the S3 tokenizer's) zero-pads n_fft=400 to 512. Sanity-check that
        // the 512-point FFT produces consistent results across calls.
        int n = 512;
        Random rng = new(42);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] re1 = new float[n / 2 + 1];
        float[] im1 = new float[n / 2 + 1];
        float[] re2 = new float[n / 2 + 1];
        float[] im2 = new float[n / 2 + 1];
        Fft.RealTransform(x, re1, im1, n);
        Fft.RealTransform(x, re2, im2, n);

        for (int k = 0; k < n / 2 + 1; k++)
        {
            Assert.Equal(re1[k], re2[k]);
            Assert.Equal(im1[k], im2[k]);
        }
    }
}
