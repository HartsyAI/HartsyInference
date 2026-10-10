using HartsyInference.Audio.Models.Auk;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>AuK VAE encoder and latent statistics on a tiny synthetic checkpoint: frame-count formula, injectable and seeded noise, normalization.</summary>
public sealed class AukVaeEncoderTests
{
    private static int Cascade(AukVaeConfig c, int samples)
    {
        int t = samples;
        foreach (int f in c.DownsampleRates) t = (int)Math.Floor((t - 2) / (double)f) + 1;
        return t;
    }

    [Theory]
    [InlineData(481, 1)]
    [InlineData(480 * 5 + 3, 5)]
    public void OutputFrames_FollowsCascadeForRemainders(int samples, int expected)
    {
        AukVaeConfig c = new();
        Assert.Equal(expected, AukVaeEncoder.OutputFrames(c, samples));
        Assert.Equal(Cascade(c, samples), AukVaeEncoder.OutputFrames(c, samples));
    }

    [Fact]
    public void Encode_MatchesReference_WithInjectedNoise()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> w = AukVaeTestData.BuildEncoder(c, 31);
        Dictionary<string, Tensor> stats = AukVaeTestData.BuildStats(c, 32);
        using AukVaeStats s = AukVaeStats.Load(stats);
        using AukVaeEncoder enc = new(c, s);
        enc.LoadWeights(w);

        const int Frames = 5;
        float[] pcm = AukVaeTestData.Random(new Random(1), Frames * c.Hop, -1, 1);
        int d = c.LatentDim;
        float[] noiseData = AukVaeTestData.Random(new Random(2), d * Frames, -1, 1);
        using Tensor x = AukVaeTestData.Make(pcm, 1, 1, pcm.Length);
        using Tensor noise = AukVaeTestData.Make(noiseData, 1, d, Frames);
        using Tensor z = enc.Encode(new CpuBackend(), x, noise);

        Assert.Equal(Frames, (int)z.Shape[1]);
        Assert.Equal(d, (int)z.Shape[2]);
        double[][] st = AukVaeTestData.EncodeStats(c, w, Array.ConvertAll(pcm, v => (double)v));
        float[] gm = AukVaeTestData.Values(stats["global_mean"]), gv = AukVaeTestData.Values(stats["global_log_std"]);
        double[] expected = new double[Frames * d];
        for (int t = 0; t < Frames; t++)
            for (int k = 0; k < d; k++)
                expected[t * d + k] = (st[k][t] + noiseData[k * Frames + t] * Math.Exp(st[d + k][t]) - gm[k]) / Math.Sqrt(gv[k]);
        Assert.True(AukVaeTestData.MaxAbsDiff(expected, AukVaeTestData.Values(z)) < 2e-4);
    }

    [Fact]
    public void Encode_SeededNoise_IsDeterministicAndSeedSensitive()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        using AukVaeStats s = AukVaeStats.Load(AukVaeTestData.BuildStats(c, 3));
        using AukVaeEncoder enc = new(c, s);
        enc.LoadWeights(AukVaeTestData.BuildEncoder(c, 4));
        using Tensor x = AukVaeTestData.Make(AukVaeTestData.Random(new Random(5), 4 * c.Hop, -1, 1), 1, 1, 4 * c.Hop);
        CpuBackend backend = new();
        using Tensor a = enc.Encode(backend, x, 77);
        using Tensor b = enc.Encode(backend, x, 77);
        using Tensor other = enc.Encode(backend, x, 78);
        Assert.Equal(AukVaeTestData.Values(a), AukVaeTestData.Values(b));
        Assert.NotEqual(AukVaeTestData.Values(a), AukVaeTestData.Values(other));
    }

    [Theory]
    [InlineData("audio_encoder.generator.0.layer.weight_v")]
    public void LoadWeights_MissingKey_Throws(string missing)
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> w = AukVaeTestData.BuildEncoder(c, 6);
        Assert.True(w.Remove(missing), missing);
        using AukVaeStats s = AukVaeStats.Load(AukVaeTestData.BuildStats(c, 3));
        using AukVaeEncoder enc = new(c, s);
        Assert.Throws<KeyNotFoundException>(() => enc.LoadWeights(w));
    }

}
