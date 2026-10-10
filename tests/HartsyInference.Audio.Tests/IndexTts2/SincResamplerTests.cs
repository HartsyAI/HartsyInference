using HartsyInference.Audio.Io;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The torchaudio-compatible sinc resampler IndexTTS-2 uses for its reference clip.</summary>
public sealed class SincResamplerTests
{
    private static float[] Sine(double hz, int rate, int samples) =>
        [.. Enumerable.Range(0, samples).Select(i => (float)Math.Sin(2 * Math.PI * hz * i / rate))];

    [Theory]
    [InlineData(24_000, 16_000)]
    public void Resample_ReproducesAToneAtTheNewRate(int inRate, int outRate)
    {
        float[] input = Sine(440, inRate, inRate);              // one second
        float[] output = SincResampler.Resample(input, inRate, outRate);
        Assert.Equal((int)Math.Ceiling((double)outRate * input.Length / inRate), output.Length);

        // Compare the interior (the filter's edge effect fades within a few ms) with the analytic tone.
        int margin = outRate / 50;
        double maxError = 0;
        for (int i = margin; i < output.Length - margin; i++)
            maxError = Math.Max(maxError, Math.Abs(output[i] - Math.Sin(2 * Math.PI * 440 * i / outRate)));
        Assert.True(maxError < 0.02, $"max error {maxError:F4}");
    }

    [Fact]
    public void Resample_AttenuatesContentAboveTheNewNyquist()
    {
        float[] input = Sine(10_000, 24_000, 24_000);           // above the 8 kHz Nyquist of the 16 kHz output
        float[] output = SincResampler.Resample(input, 24_000, 16_000);
        double peak = output.Skip(400).Take(output.Length - 800).Max(static v => Math.Abs(v));
        Assert.True(peak < 0.05, $"aliased energy {peak:F4}");
    }
}
