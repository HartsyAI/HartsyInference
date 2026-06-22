using System.Linq;
using HartsyInference.Audio.Dsp;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the YIN F0 estimator against synthetic signals — pitch accuracy on a known sine and
/// unvoiced detection on silence. Asset-free.</summary>
public sealed class F0EstimatorTests
{
    private const int SampleRate = 16_000;
    private const int Hop = 320; // 50 Hz, RVC/HuBERT alignment

    private static float[] Sine(float hz, double seconds)
    {
        int n = (int)(SampleRate * seconds);
        float[] s = new float[n];
        for (int i = 0; i < n; i++)
        {
            s[i] = 0.6f * MathF.Sin(2f * MathF.PI * hz * i / SampleRate);
        }
        return s;
    }

    [Theory]
    [InlineData(120f)]
    [InlineData(220f)]
    [InlineData(440f)]
    public void EstimateYin_RecoversSinePitch_WithinTolerance(float hz)
    {
        float[] f0 = F0Estimator.EstimateYin(Sine(hz, 0.6), SampleRate, Hop);
        float[] voiced = [.. f0.Where(f => f > 0f)];

        Assert.True(voiced.Length > f0.Length / 2, "most frames of a steady tone should be voiced");
        Array.Sort(voiced);
        float median = voiced[voiced.Length / 2];
        Assert.True(MathF.Abs(median - hz) < 5f, $"median F0 {median:0.0} Hz should be within 5 Hz of {hz}");
    }

    [Fact]
    public void EstimateYin_Silence_IsUnvoiced()
    {
        float[] f0 = F0Estimator.EstimateYin(new float[SampleRate / 2], SampleRate, Hop);
        Assert.All(f0, f => Assert.Equal(0f, f));
    }

    [Fact]
    public void EstimateYin_FrameCount_MatchesHopGrid()
    {
        float[] f0 = F0Estimator.EstimateYin(Sine(200f, 1.0), SampleRate, Hop);
        Assert.Equal(SampleRate / Hop, f0.Length); // 1 s @ 16 kHz / hop 320 = 50 frames
    }
}
