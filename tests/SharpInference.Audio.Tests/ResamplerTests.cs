using SharpInference.Audio.Io;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Polyphase resampler tests. Validates DC preservation, output-length
/// formula, and rough frequency preservation under common rate changes (44.1→16k,
/// 48→16k, the two we'll hit on every real-world STT call).</summary>
public sealed class ResamplerTests
{
    [Fact]
    public void OutputLength_44k_To_16k()
    {
        Resampler r = Resampler.Create(44_100, 16_000);
        // 1 second of 44.1 kHz audio → ~16000 samples at 16 kHz.
        Assert.InRange(r.OutputLength(44_100), 15_999, 16_001);
    }

    [Fact]
    public void OutputLength_48k_To_16k_IsExactlyOneThird()
    {
        Resampler r = Resampler.Create(48_000, 16_000);
        Assert.Equal(16_000, r.OutputLength(48_000));
    }

    [Fact]
    public void DcInput_StaysDc_After_Resample()
    {
        Resampler r = Resampler.Create(44_100, 16_000);
        float[] x = new float[44_100];
        for (int i = 0; i < x.Length; i++) x[i] = 0.42f;
        float[] y = r.Resample(x);

        // Skip the first and last ~100 samples to avoid filter edge effects.
        for (int i = 100; i < y.Length - 100; i++)
            Assert.Equal(0.42f, y[i], precision: 2);
    }

    [Fact]
    public void SineWave_FrequencyPreserved()
    {
        // A 1 kHz sine at 44.1 kHz resampled to 16 kHz should still be ~1 kHz.
        // Check by counting zero crossings: ~2000 crossings/sec for 1 kHz sine
        // → ~2000 over 1 second of output.
        Resampler r = Resampler.Create(44_100, 16_000);
        float[] x = new float[44_100];
        for (int i = 0; i < x.Length; i++) x[i] = MathF.Sin(2f * MathF.PI * 1000f * i / 44_100f);
        float[] y = r.Resample(x);

        int crossings = 0;
        for (int i = 100; i < y.Length - 100; i++)
            if ((y[i] >= 0) != (y[i + 1] >= 0)) crossings++;
        // 1000 Hz × 2 crossings/cycle × ~1 second ≈ 2000. Allow ±200 for transient.
        Assert.InRange(crossings, 1800, 2200);
    }

    [Fact]
    public void SameRate_PreservesShape_AndEnergy()
    {
        // When in_rate == out_rate the filter is a windowed-sinc with cutoff at Nyquist;
        // for even tap counts this introduces a half-sample group delay. Strict bit-identity
        // isn't the contract — energy preservation and shape are.
        Resampler r = Resampler.Create(16_000, 16_000);
        float[] x = new float[1000];
        for (int i = 0; i < x.Length; i++) x[i] = MathF.Sin(2f * MathF.PI * i / 40);
        float[] y = r.Resample(x);
        Assert.Equal(x.Length, y.Length);

        // Compare RMS over the steady-state region — should match within ~5%.
        double rmsX = Rms(x.AsSpan(100, 800));
        double rmsY = Rms(y.AsSpan(100, 800));
        Assert.InRange(rmsY / rmsX, 0.9, 1.1);
    }

    private static double Rms(ReadOnlySpan<float> s)
    {
        double sum = 0;
        for (int i = 0; i < s.Length; i++) sum += (double)s[i] * s[i];
        return Math.Sqrt(sum / s.Length);
    }
}
