using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The planned mixed-radix FFT against a double-precision DFT. Every butterfly runs at some size here —
/// radix 5 and 3 only in the non-power-of-two sizes, radix 2 in both of the positions kiss_fft's factoring can put
/// it — and a wrong twiddle stride or permutation produces a plausible spectrum that is simply wrong, which is
/// exactly what a spectral denoiser would then apply gains to.</summary>
public sealed class FftPlanTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(32)]
    [InlineData(60)]
    [InlineData(240)]
    [InlineData(480)]
    [InlineData(512)]
    [InlineData(960)]
    [InlineData(1920)]
    public void Forward_MatchesADoublePrecisionDft(int n)
    {
        Random rng = new(n);
        float[] re = new float[n], im = new float[n];
        for (int i = 0; i < n; i++)
        {
            re[i] = (float)(rng.NextDouble() * 2 - 1);
            im[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float[] outRe = new float[n], outIm = new float[n];
        new FftPlan(n).Forward(re, im, outRe, outIm);

        (double[] refRe, double[] refIm) = Dft(re, im);
        double error = 0, energy = 0;
        for (int k = 0; k < n; k++)
        {
            error += Math.Pow(outRe[k] - refRe[k], 2) + Math.Pow(outIm[k] - refIm[k], 2);
            energy += refRe[k] * refRe[k] + refIm[k] * refIm[k];
        }
        Assert.True(Math.Sqrt(error / energy) < 1e-6, $"n={n}: relative error {Math.Sqrt(error / energy):E2}");
    }

    /// <summary>Pins the port to upstream rather than just to the maths: a 960-point transform of a fixed input,
    /// scaled by 1/N first as upstream's <c>rnn_fft_c</c> does, reproduces upstream's output bit for bit at bins spread
    /// across the spectrum. Expected values dumped from xiph/rnnoise's <c>kiss_fft.c</c> with its static 960-point
    /// tables (gcc -O2, x86-64); the whole 1920-value output matched when this was written.</summary>
    [Fact]
    public void Forward_At960_ReproducesUpstreamKissFft_BitForBit()
    {
        // The twiddles come from the platform's cos and sin, and the expected bits are glibc's; another libm may
        // round a twiddle differently in its last place.
        if (!OperatingSystem.IsLinux()) return;
        const int N = 960;
        float[] re = new float[N], im = new float[N];
        uint state = 12345u;
        for (int i = 0; i < N; i++)
        {
            state = unchecked(state * 1103515245u + 12345u);
            re[i] = ((int)(state >> 8) % 20000 - 10000) * (1f / N);
            state = unchecked(state * 1103515245u + 12345u);
            im[i] = ((int)(state >> 8) % 20000 - 10000) * (1f / N);
        }
        float[] outRe = new float[N], outIm = new float[N];
        new FftPlan(N).Forward(re, im, outRe, outIm);

        (int Bin, uint Re, uint Im)[] upstream =
        [
            (0, 0xc3061266, 0xc2fff99e), (1, 0x428413cf, 0xc2dece32), (7, 0x438b16c8, 0x41f3de60),
            (100, 0x424c648b, 0x422a6368), (479, 0xc2ae8fb0, 0x41d75c00), (480, 0x41e00ef2, 0xc3449222),
            (481, 0xc2e45ecb, 0xc2af462d), (959, 0xc3ac79da, 0x43039a6c),
        ];
        foreach ((int bin, uint expectedRe, uint expectedIm) in upstream)
        {
            Assert.Equal(expectedRe, BitConverter.SingleToUInt32Bits(outRe[bin]));
            Assert.Equal(expectedIm, BitConverter.SingleToUInt32Bits(outIm[bin]));
        }
    }

    /// <summary>Against the path it replaced, at the sizes RNNoise uses: <see cref="Fft.Transform"/> (Bluestein at
    /// both, since neither is a power of two) and the planned transform agree to float rounding.</summary>
    [Theory]
    [InlineData(960)]
    [InlineData(480)]
    public void Forward_MatchesTheFftItReplaces_ToRounding(int n)
    {
        Random rng = new(n + 1);
        float[] re = new float[n], im = new float[n];
        for (int i = 0; i < n; i++)
        {
            re[i] = (float)(rng.NextDouble() * 2 - 1);
            im[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float[] oldRe = (float[])re.Clone(), oldIm = (float[])im.Clone();
        Fft.Transform(oldRe, oldIm, n);
        float[] newRe = new float[n], newIm = new float[n];
        new FftPlan(n).Forward(re, im, newRe, newIm);

        double error = 0, energy = 0, worst = 0;
        for (int k = 0; k < n; k++)
        {
            double d = Math.Sqrt(Math.Pow(newRe[k] - oldRe[k], 2) + Math.Pow(newIm[k] - oldIm[k], 2));
            error += d * d;
            energy += (double)oldRe[k] * oldRe[k] + (double)oldIm[k] * oldIm[k];
            worst = Math.Max(worst, d);
        }
        double rms = Math.Sqrt(energy / n);
        Assert.True(Math.Sqrt(error / energy) < 1e-6, $"n={n}: relative error {Math.Sqrt(error / energy):E2}");
        Assert.True(worst < 1e-5 * rms, $"n={n}: worst bin differs by {worst / rms:E2} of the spectrum's RMS");
    }

    [Fact]
    public void ForwardReal_EqualsTheComplexTransformOfARealSignal()
    {
        const int N = 960;
        Random rng = new(7);
        float[] x = new float[N];
        for (int i = 0; i < N; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        FftPlan plan = new(N);
        float[] re = new float[N / 2 + 1], im = new float[N / 2 + 1];
        plan.ForwardReal(x, re, im);
        float[] fullRe = new float[N], fullIm = new float[N];
        plan.Forward(x, new float[N], fullRe, fullIm);
        Assert.Equal(fullRe[..(N / 2 + 1)], re);
        Assert.Equal(fullIm[..(N / 2 + 1)], im);
    }

    /// <summary>Same input, same answer: the work buffer is fully overwritten each call, so nothing leaks from
    /// the previous transform.</summary>
    [Fact]
    public void Forward_IsRepeatable_AndMayRunInPlace()
    {
        const int N = 480;
        Random rng = new(3);
        float[] re = new float[N], im = new float[N];
        for (int i = 0; i < N; i++) re[i] = (float)rng.NextDouble();
        FftPlan plan = new(N);
        float[] aRe = new float[N], aIm = new float[N];
        plan.Forward(re, im, aRe, aIm);
        plan.Forward(re, im, re, im);
        Assert.Equal(aRe, re);
        Assert.Equal(aIm, im);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(14)]
    [InlineData(3528)]
    public void Sizes_WithAFactorAboveFive_AreRefused(int n)
    {
        Assert.False(FftPlan.IsSupported(n));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FftPlan(n));
    }

    private static (double[] Re, double[] Im) Dft(float[] re, float[] im)
    {
        int n = re.Length;
        double[] outRe = new double[n], outIm = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sr = 0, si = 0;
            for (int t = 0; t < n; t++)
            {
                double angle = -2 * Math.PI * ((long)k * t % n) / n;
                double c = Math.Cos(angle), s = Math.Sin(angle);
                sr += re[t] * c - im[t] * s;
                si += re[t] * s + im[t] * c;
            }
            outRe[k] = sr;
            outIm[k] = si;
        }
        return (outRe, outIm);
    }
}
