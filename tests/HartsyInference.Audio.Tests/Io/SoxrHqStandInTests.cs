using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests.Io;

/// <summary>Holds <see cref="SincResampler.ResampleSoxrHqStandIn"/> to soxr_hq's response. The expected gains were measured with python-soxr
/// 1.1.0: <c>soxr.resample(0.5 * sine, inRate, 16000, quality="soxr_hq")</c>, one second, amplitude from the RMS of samples 4000 to 12000.</summary>
public sealed class SoxrHqStandInTests
{
    [Theory]
    [InlineData(0.50, 0.9999999)]
    [InlineData(0.90, 0.9999998)]
    [InlineData(0.93, 0.9912219)]
    [InlineData(0.94, 0.9245255)]
    [InlineData(0.95, 0.7173528)]
    [InlineData(0.96, 0.3970150)]
    [InlineData(0.97, 0.1338824)]
    [InlineData(0.98, 0.0222576)]
    public void TransitionBandGainFollowsSoxrHq(double ofOutputNyquist, double soxrGain) =>
        Assert.InRange(Gain(ofOutputNyquist), soxrGain - 5e-4, soxrGain + 5e-4);

    [Theory]
    [InlineData(1.1)]
    [InlineData(1.3)]
    public void ContentAboveTheOutputNyquistIsRemoved(double ofOutputNyquist)
    {
        double gain = Gain(ofOutputNyquist);
        Assert.True(gain < 1e-5, $"gain {gain:E2}");
    }

    /// <summary>The separator's ratio (160 / 441). soxr's response here differs from the shipped filter by about 1e-3 below the transition band and up to
    /// 6e-3 inside it; the first stand-in was up to 6e-2 off.</summary>
    [Theory]
    [InlineData(0.80, 0.9996296)]
    [InlineData(0.90, 0.9990704)]
    [InlineData(0.93, 0.9913726)]
    [InlineData(0.94, 0.9283818)]
    [InlineData(0.95, 0.7212647)]
    [InlineData(0.96, 0.3938563)]
    [InlineData(0.97, 0.1276808)]
    [InlineData(0.98, 0.0196506)]
    public void SeparatorRatioFollowsSoxrHqWithinItsDroop(double ofOutputNyquist, double soxrGain) =>
        Assert.InRange(Gain(ofOutputNyquist, 44_100), soxrGain - 1e-2, soxrGain + 1e-2);

    /// <summary>Output amplitude over input amplitude for a one-second sine at <paramref name="ofOutputNyquist"/> × 8 kHz, resampled to 16 kHz.</summary>
    private static double Gain(double ofOutputNyquist, int inRate = 24_000)
    {
        float[] input = new float[inRate];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(0.5 * Math.Sin(2 * Math.PI * ofOutputNyquist * 8000 * i / inRate));
        float[] output = SincResampler.ResampleSoxrHqStandIn(input, inRate, 16_000);
        double energy = 0;
        for (int i = 4000; i < 12_000; i++) energy += (double)output[i] * output[i];
        return Math.Sqrt(2 * energy / 8000) / 0.5;
    }
}
