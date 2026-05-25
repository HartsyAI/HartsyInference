using SharpInference.Audio.Preprocessing;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Verifies the SIMD path of <see cref="Fft.Transform"/> agrees with a fresh
/// scalar reference at every FFT size in scope for audio models (Whisper n=512,
/// Kokoro n=2048, EnCodec / DAC don't directly use the FFT but vocoder iSTFT does
/// at n=1024 / 1280).
///
/// <para>Strategy: a scalar reference is implemented inline below (the existing
/// <c>Fft.Transform</c> has a SIMD-vs-scalar branch — this test calls the published
/// API and compares against an independent reference, so a regression in either
/// branch surfaces). Identical inputs go to both; outputs must match within
/// <c>1e-4</c> absolute (typical accumulated FP error for these sizes).</para></summary>
public sealed class FftSimdCorrectnessTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2048)]
    public void Transform_AgreesWithScalarReference_ForRealInput(int n)
    {
        float[] re = new float[n];
        float[] im = new float[n];
        Random rng = new(42);
        for (int i = 0; i < n; i++)
        {
            re[i] = (float)(rng.NextDouble() * 2 - 1);
            im[i] = 0f;
        }

        // Fft.Transform routes through the SIMD path for any stage where step==1 and
        // half >= Vector<float>.Count; otherwise falls back to scalar. We compare the
        // resulting transform against a fully-scalar reference.
        float[] reSimd = (float[])re.Clone();
        float[] imSimd = (float[])im.Clone();
        Fft.Transform(reSimd, imSimd, n);

        (float[] reRef, float[] imRef) = ScalarReferenceForwardFft(re, im, n);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(reRef[i], reSimd[i], precision: 4);
            Assert.Equal(imRef[i], imSimd[i], precision: 4);
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(512)]
    [InlineData(2048)]
    public void Transform_AgreesWithScalarReference_ForSineInput(int n)
    {
        // Sine input has a known FFT structure (two delta peaks at +/- frequency bin)
        // so this both validates the SIMD/scalar agreement AND that the FFT produces
        // correct magnitude peaks at expected frequency bins.
        float[] re = new float[n];
        float[] im = new float[n];
        int freqBin = 5;     // arbitrary low frequency
        for (int i = 0; i < n; i++) re[i] = MathF.Sin(2f * MathF.PI * freqBin * i / n);

        float[] reSimd = (float[])re.Clone();
        float[] imSimd = (float[])im.Clone();
        Fft.Transform(reSimd, imSimd, n);

        (float[] reRef, float[] imRef) = ScalarReferenceForwardFft(re, im, n);

        for (int i = 0; i < n; i++)
        {
            Assert.Equal(reRef[i], reSimd[i], precision: 3);
            Assert.Equal(imRef[i], imSimd[i], precision: 3);
        }

        // Spectral check: a real sine should put magnitude at bins [freqBin, n-freqBin]
        // with all other bins near zero. Imaginary part: anti-symmetric peak pair.
        float[] mag = new float[n];
        for (int k = 0; k < n; k++) mag[k] = MathF.Sqrt(reSimd[k] * reSimd[k] + imSimd[k] * imSimd[k]);
        Assert.True(mag[freqBin] > n * 0.45f, $"expected magnitude peak at bin {freqBin}, got {mag[freqBin]}");
        Assert.True(mag[n - freqBin] > n * 0.45f, $"expected magnitude peak at bin {n - freqBin}, got {mag[n - freqBin]}");
        // Bins between the peaks (away from DC) should be small.
        Assert.True(mag[freqBin + 3] < n * 0.05f, $"unexpected energy at bin {freqBin + 3}: {mag[freqBin + 3]}");
    }

    [Fact]
    public void RealTransform_MatchesFullTransformLeadingHalf()
    {
        int n = 512;
        float[] input = new float[n];
        Random rng = new(7);
        for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // Full path.
        float[] full = (float[])input.Clone();
        float[] fullIm = new float[n];
        Fft.Transform(full, fullIm, n);

        // Real-input path.
        int half = n / 2 + 1;
        float[] outRe = new float[half];
        float[] outIm = new float[half];
        Fft.RealTransform(input, outRe, outIm, n);

        for (int k = 0; k < half; k++)
        {
            Assert.Equal(full[k], outRe[k], precision: 4);
            Assert.Equal(fullIm[k], outIm[k], precision: 4);
        }
    }

    /// <summary>Independent scalar reference Cooley-Tukey radix-2 FFT. Identical math
    /// to <see cref="Fft.Transform"/>'s scalar branch but written from scratch here so
    /// the comparison surfaces any regression in the SIMD path (the SIMD path is
    /// exercised internally for stages where <c>step==1</c>).</summary>
    private static (float[] Re, float[] Im) ScalarReferenceForwardFft(float[] reIn, float[] imIn, int n)
    {
        float[] re = (float[])reIn.Clone();
        float[] im = (float[])imIn.Clone();

        // Bit-reverse.
        int logN = 0;
        while ((1 << logN) < n) logN++;
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, logN);
            if (j > i)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        // Cooley-Tukey butterflies.
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1;
            double thetaBase = -2.0 * Math.PI / size;     // forward FFT
            for (int i = 0; i < n; i += size)
            {
                for (int k = 0; k < half; k++)
                {
                    double angle = thetaBase * k;
                    float wr = (float)Math.Cos(angle);
                    float wi = (float)Math.Sin(angle);
                    int idxJ = i + k;
                    int idxK = idxJ + half;
                    float tr = wr * re[idxK] - wi * im[idxK];
                    float ti = wr * im[idxK] + wi * re[idxK];
                    re[idxK] = re[idxJ] - tr;
                    im[idxK] = im[idxJ] - ti;
                    re[idxJ] += tr;
                    im[idxJ] += ti;
                }
            }
        }
        return (re, im);
    }

    private static int BitReverse(int x, int logN)
    {
        int r = 0;
        for (int i = 0; i < logN; i++)
        {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        return r;
    }
}
