using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests.Io;

/// <summary>Holds <see cref="SincResampler.ResampleSoxrHqStandIn"/> to soxr_hq's response. The expected gains were measured with python-soxr
/// 1.1.0: <c>soxr.resample(0.5 * sine, 24000, 16000, quality="soxr_hq")</c>, one second, amplitude from the RMS of samples 4000 to 12000.</summary>
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
    public void ContentAboveTheOutputNyquistIsRemoved(double ofOutputNyquist) => Assert.True(Gain(ofOutputNyquist) < 1e-5, $"gain {Gain(ofOutputNyquist):E2}");

    private static double Gain(double ofOutputNyquist)
    {
        float[] input = new float[24_000];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(0.5 * Math.Sin(2 * Math.PI * ofOutputNyquist * 8000 * i / 24_000));
        float[] output = SincResampler.ResampleSoxrHqStandIn(input, 24_000, 16_000);
        double energy = 0;
        for (int i = 4000; i < 12_000; i++) energy += (double)output[i] * output[i];
        return Math.Sqrt(2 * energy / 8000) / 0.5;
    }
}
