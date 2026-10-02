using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>AuK VAE decoder on a tiny synthetic checkpoint: structure, causal/non-causal semantics, weight norm and key handling against a naive reference.</summary>
public sealed class AukVaeDecoderTests
{
    private const int Frames = 6;

    private static double[][] Latents(AukVaeConfig c, float[] data) =>
        Enumerable.Range(0, c.LatentDim).Select(d => Enumerable.Range(0, Frames).Select(t => (double)data[d * Frames + t]).ToArray()).ToArray();

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Decode_MatchesReference_AndLengthIsFramesTimesHop(bool causal)
    {
        AukVaeConfig c = AukVaeTestData.Tiny(causal);
        Dictionary<string, Tensor> w = AukVaeTestData.BuildDecoder(c, 11);
        using AukVae vae = new(c);
        vae.LoadWeights(w);

        float[] data = AukVaeTestData.Random(new Random(3), c.LatentDim * Frames, -1, 1);
        using Tensor z = AukVaeTestData.Make(data, 1, c.LatentDim, Frames);
        using Tensor pcm = vae.Decode(new CpuBackend(), z);

        // Non-causal transposed convs pad (k - s) / 2 per side, which is exact only for even strides.
        if (causal) Assert.Equal(Frames * c.Hop, (int)pcm.Shape[2]);
        Assert.Equal(1, (int)pcm.Shape[1]);
        double[] expected = AukVaeTestData.Decode(c, w, Latents(c, data))[0];
        Assert.True(AukVaeTestData.MaxAbsDiff(expected, AukVaeTestData.Values(pcm)) < 2e-4);
    }

    [Fact]
    public void DefaultConfig_IsConsistent()
    {
        AukVaeConfig c = new();
        c.Validate();
        Assert.Equal(480, c.Hop);
        Assert.Equal(24, c.FinalChannels);
    }

    [Fact]
    public void DecodeTimeMajor_EqualsChannelsFirst()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        using AukVae vae = new(c);
        vae.LoadWeights(AukVaeTestData.BuildDecoder(c, 12));
        float[] data = AukVaeTestData.Random(new Random(8), c.LatentDim * Frames, -1, 1);
        float[] timeMajor = new float[data.Length];
        for (int d = 0; d < c.LatentDim; d++)
            for (int t = 0; t < Frames; t++) timeMajor[t * c.LatentDim + d] = data[d * Frames + t];
        using Tensor a = AukVaeTestData.Make(data, 1, c.LatentDim, Frames);
        using Tensor b = AukVaeTestData.Make(timeMajor, 1, Frames, c.LatentDim);
        using Tensor ya = vae.Decode(new CpuBackend(), a);
        using Tensor yb = vae.DecodeTimeMajor(new CpuBackend(), b);
        Assert.Equal(AukVaeTestData.Values(ya), AukVaeTestData.Values(yb));
    }

    [Fact]
    public void Decode_ClampsOutputToUnitRange()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> w = AukVaeTestData.BuildDecoder(c, 13, postGain: 500);
        using AukVae vae = new(c);
        vae.LoadWeights(w);
        using Tensor z = AukVaeTestData.Make(AukVaeTestData.Random(new Random(1), c.LatentDim * Frames, -1, 1), 1, c.LatentDim, Frames);
        using Tensor pcm = vae.Decode(new CpuBackend(), z);
        float[] y = AukVaeTestData.Values(pcm);
        Assert.All(y, v => Assert.InRange(v, -1f, 1f));
        Assert.Contains(y, v => Math.Abs(v) == 1f);
    }

    [Fact]
    public void FlowAndEncoderKeys_AreIgnored_AndPrefixIsHonoured()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> plain = AukVaeTestData.BuildDecoder(c, 14);
        Dictionary<string, Tensor> noisy = AukVaeTestData.BuildDecoder(c, 14, "vae.");
        noisy["vae.flow.flows.0.pre.weight_v"] = AukVaeTestData.Make([1f, 2f], 2);
        foreach (KeyValuePair<string, Tensor> kv in AukVaeTestData.BuildEncoder(c, 5, "vae.")) noisy[kv.Key] = kv.Value;

        float[] data = AukVaeTestData.Random(new Random(2), c.LatentDim * Frames, -1, 1);
        using Tensor z = AukVaeTestData.Make(data, 1, c.LatentDim, Frames);
        using AukVae a = new(c);
        using AukVae b = new(c);
        a.LoadWeights(plain);
        b.LoadWeights(noisy, "vae.");
        using Tensor ya = a.Decode(new CpuBackend(), z);
        using Tensor yb = b.Decode(new CpuBackend(), z);
        Assert.Equal(AukVaeTestData.Values(ya), AukVaeTestData.Values(yb));
    }

    [Theory]
    [InlineData("conv_post.weight_v")]
    [InlineData("conv_pre.bias")]
    [InlineData("ups.1.0.weight_g")]
    [InlineData("resblocks.3.convs2.1.weight_v")]
    [InlineData("resblocks.2.activations.3.upsample.filter")]
    [InlineData("activation_post.act.alpha")]
    public void LoadWeights_MissingKey_Throws(string missing)
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        Dictionary<string, Tensor> w = AukVaeTestData.BuildDecoder(c, 15);
        Assert.True(w.Remove(missing), missing);
        using AukVae vae = new(c);
        Assert.Throws<KeyNotFoundException>(() => vae.LoadWeights(w));
    }

    [Fact]
    public void Decode_BeforeLoad_Throws()
    {
        using AukVae vae = new(AukVaeTestData.Tiny());
        using Tensor z = AukVaeTestData.Make(new float[4 * 3], 1, 4, 3);
        Assert.Throws<InvalidOperationException>(() => vae.Decode(new CpuBackend(), z));
    }

    /// <summary>The transposed conv stores g as <c>[in, 1, 1]</c> over v <c>[in, out, k]</c>; the norm runs over everything but axis 0, and conv_post's g is <c>[1, 1, 1]</c>.</summary>
    [Fact]
    public void WeightNorm_OnTransposedShapeAndSingleOutputConv()
    {
        Random rng = new(21);
        using Tensor v = AukVaeTestData.Make(AukVaeTestData.Random(rng, 3 * 2 * 4, -1, 1), 3, 2, 4);
        using Tensor g = AukVaeTestData.Make([0.5f, 1f, 2f], 3, 1, 1);
        using Tensor fused = WeightNormFusion.Fuse(g, v);
        float[] f = AukVaeTestData.Values(fused);
        float[] gg = AukVaeTestData.Values(g);
        for (int i = 0; i < 3; i++)
        {
            double norm = Math.Sqrt(f.Skip(i * 8).Take(8).Sum(x => (double)x * x));
            Assert.Equal(gg[i], norm, 1e-5);
        }

        using Tensor v1 = AukVaeTestData.Make(AukVaeTestData.Random(rng, 24 * 7, -1, 1), 1, 24, 7);
        using Tensor g1 = AukVaeTestData.Make([0.7f], 1, 1, 1);
        using Tensor fused1 = WeightNormFusion.Fuse(g1, v1);
        Assert.Equal(0.7, Math.Sqrt(AukVaeTestData.Values(fused1).Sum(x => (double)x * x)), 1e-5);
    }

    [Fact]
    public void Decode_RejectsWrongLatentShape()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        using AukVae vae = new(c);
        vae.LoadWeights(AukVaeTestData.BuildDecoder(c, 16));
        using Tensor wrong = AukVaeTestData.Make(new float[3 * 5], 1, 3, 5);
        Assert.Throws<ArgumentException>(() => vae.Decode(new CpuBackend(), wrong));
    }
}
