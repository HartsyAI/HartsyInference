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
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    public void OutputFrames_IsExactForHopMultiples_Default(int n)
    {
        AukVaeConfig c = new();
        Assert.Equal(n, AukVaeEncoder.OutputFrames(c, 480 * n));
    }

    [Theory]
    [InlineData(481, 1)]
    [InlineData(480 * 3 + 7, 3)]
    [InlineData(480 * 3 + 479, 4)]
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

    [Fact]
    public void Encode_NoiseShapeMismatch_Throws()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        using AukVaeStats s = AukVaeStats.Load(AukVaeTestData.BuildStats(c, 3));
        using AukVaeEncoder enc = new(c, s);
        enc.LoadWeights(AukVaeTestData.BuildEncoder(c, 4));
        using Tensor x = AukVaeTestData.Make(new float[3 * c.Hop], 1, 1, 3 * c.Hop);
        using Tensor noise = AukVaeTestData.Make(new float[c.LatentDim * 2], 1, c.LatentDim, 2);
        Assert.Throws<ArgumentException>(() => enc.Encode(new CpuBackend(), x, noise));
    }

    [Theory]
    [InlineData("audio_encoder.generator.0.layer.weight_v")]
    [InlineData("audio_encoder.generator.6.layers.1.3.bias")]
    [InlineData("audio_encoder.generator.8.layer.weight_g")]
    [InlineData("audio_encoder.generator.6.layers.0.1.weight_v")]
    [InlineData("audio_encoder.generator.8.layer.bias")]
    public void LoadWeights_MissingKey_Throws(string missing)
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> w = AukVaeTestData.BuildEncoder(c, 6);
        Assert.True(w.Remove(missing), missing);
        using AukVaeStats s = AukVaeStats.Load(AukVaeTestData.BuildStats(c, 3));
        using AukVaeEncoder enc = new(c, s);
        Assert.Throws<KeyNotFoundException>(() => enc.LoadWeights(w));
    }

    [Fact]
    public void Stats_NormalizeDenormalize_RoundTrip_AndFormula()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> raw = AukVaeTestData.BuildStats(c, 9);
        using AukVaeStats s = AukVaeStats.Load(raw);
        const int T = 4;
        float[] data = AukVaeTestData.Random(new Random(1), T * c.LatentDim, -2, 2);
        using Tensor x = AukVaeTestData.Make(data, 1, T, c.LatentDim);
        CpuBackend backend = new();
        using Tensor n = s.Normalize(backend, x);
        using Tensor back = s.Denormalize(backend, n);
        float[] gm = AukVaeTestData.Values(raw["global_mean"]), gv = AukVaeTestData.Values(raw["global_log_std"]);
        float[] nv = AukVaeTestData.Values(n);
        for (int i = 0; i < data.Length; i++)
        {
            int k = i % c.LatentDim;
            Assert.Equal((data[i] - gm[k]) / Math.Sqrt(gv[k]), nv[i], 1e-5);
        }
        Assert.True(AukVaeTestData.MaxAbsDiff(Array.ConvertAll(data, v => (double)v), AukVaeTestData.Values(back)) < 1e-5);
    }

    [Theory]
    [InlineData("global_mean")]
    [InlineData("global_log_std")]
    public void Stats_MissingKey_Throws(string missing)
    {
        Dictionary<string, Tensor> raw = AukVaeTestData.BuildStats(AukVaeTestData.Tiny(), 9);
        raw.Remove(missing);
        Assert.Throws<KeyNotFoundException>(() => AukVaeStats.Load(raw));
    }
}
