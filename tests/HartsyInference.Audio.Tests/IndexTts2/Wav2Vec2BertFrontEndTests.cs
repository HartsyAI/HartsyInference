using HartsyInference.Audio.Models.Wav2Vec2Bert;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The w2v-bert front end must see 16-bit-scaled samples, like the reference's feature extractor
/// (<c>waveform * 2**15</c>). Without the scale, quiet audio falls under the fbank's log floor and its detail is
/// flattened; after the scale the (per-bin normalized) features of a signal and of the same signal at 1000x gain agree.
/// Regression test for the 16% hidden-state error this caused against the PyTorch reference.</summary>
public sealed class Wav2Vec2BertFrontEndTests
{
    [Fact]
    public void StackedFeatures_AreGainInvariant_EvenForVeryQuietAudio()
    {
        Random rng = new(5);
        float[] quiet = new float[16_000];
        double phase = 0;
        for (int i = 0; i < quiet.Length; i++)
        {
            phase += 2 * Math.PI * (200 + 150 * Math.Sin(i / 4000.0)) / 16_000;
            quiet[i] = (float)(3e-6 * (Math.Sin(phase) + 0.3 * (rng.NextDouble() - 0.5)));
        }
        float[] loud = [.. quiet.Select(static v => v * 1000f)];

        using Wav2Vec2BertExtractor extractor = new(Wav2Vec2BertConfig.V2(17));
        using Tensor a = extractor.ExtractStackedFeatures(quiet);
        using Tensor b = extractor.ExtractStackedFeatures(loud);
        Assert.Equal(a.Shape[1], b.Shape[1]);

        ReadOnlySpan<float> fa = a.AsSpan<float>(), fb = b.AsSpan<float>();
        double num = 0, den = 0;
        for (int i = 0; i < fa.Length; i++)
        {
            double d = fa[i] - fb[i];
            num += d * d;
            den += (double)fb[i] * fb[i];
        }
        Assert.True(Math.Sqrt(num / den) < 0.02, $"features depend on the input gain: relative RMS {Math.Sqrt(num / den):E2}");
    }
}
