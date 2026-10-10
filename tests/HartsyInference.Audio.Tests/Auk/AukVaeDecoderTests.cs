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
    public void Decode_RejectsWrongLatentShape()
    {
        AukVaeConfig c = AukVaeTestData.Tiny();
        using AukVae vae = new(c);
        vae.LoadWeights(AukVaeTestData.BuildDecoder(c, 16));
        using Tensor wrong = AukVaeTestData.Make(new float[3 * 5], 1, 3, 5);
        Assert.Throws<ArgumentException>(() => vae.Decode(new CpuBackend(), wrong));
    }
}
