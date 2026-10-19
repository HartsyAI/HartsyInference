using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A FLUX.2 VAE whose decoder is narrower than its encoder (OpenMOSS's MOSS-Image folder: 96-384 against 128-512)
/// must load at its own widths and decode at them. A wrong plan loads silently and decodes garbage, so these run on a
/// deterministic synthetic checkpoint rather than real weights.</summary>
public sealed class Flux2VaeNarrowDecoderTests
{
    /// <summary>A tiny stand-in with the FLUX.2 decoder's proportions (decoder = 3/4 of the encoder's widths).</summary>
    private static VaeConfig TinyNarrow() => new()
    {
        LatentChannels = 4,
        BlockOutChannels = [8, 16, 32, 32],
        DecoderBlockOutChannels = [6, 12, 24, 24],
        LayersPerBlock = 1,
        NormNumGroups = 2,
        UsePostQuantConv = false,
        UseQuantConv = false,
        SampleSize = 64,
    };

    [Fact]
    public void WithCheckpointDecoder_NarrowFlux2Decoder_InfersTheCheckpointWidths()
    {
        // Only conv2's out-channel count is read, so each up-block weight is shaped [out, 1, 1, 1] to keep the test light.
        // Up-block i outputs the decoder's stage (3 - i) reversed: [384, 384, 192, 96] gives decoder widths [96, 192, 384, 384].
        int[] upOut = [384, 384, 192, 96];
        Dictionary<string, Tensor> weights = [];
        for (int up = 0; up < upOut.Length; up++)
            weights[$"decoder.up_blocks.{up}.resnets.0.conv2.weight"] = new Tensor(new TensorShape(upOut[up], 1, 1, 1), DType.F32);

        VaeConfig config = Flux2VaeComponents.WithCheckpointDecoder(VaeConfig.Flux2, weights);

        Assert.Equal(new[] { 96, 192, 384, 384 }, config.DecoderBlockOutChannels);
        Assert.Equal(VaeConfig.Flux2.BlockOutChannels, config.BlockOutChannels);
    }

    [Fact]
    public void WithCheckpointDecoder_PresetWidths_KeepsTheEncoderMirroringConfig()
    {
        // The full FLUX.2 decoder mirrors the encoder: up-block outputs [512, 512, 256, 128]. FLUX.2, FLUX.3 and Lens rely on this.
        int[] upOut = [512, 512, 256, 128];
        Dictionary<string, Tensor> weights = [];
        for (int up = 0; up < upOut.Length; up++)
            weights[$"decoder.up_blocks.{up}.resnets.0.conv2.weight"] = new Tensor(new TensorShape(upOut[up], 1, 1, 1), DType.F32);

        VaeConfig config = Flux2VaeComponents.WithCheckpointDecoder(VaeConfig.Flux2, weights);

        Assert.Null(config.DecoderBlockOutChannels);
    }

    [Fact]
    public void WithCheckpointDecoder_MissingUpBlockKeys_KeepsTheConfig()
    {
        VaeConfig config = Flux2VaeComponents.WithCheckpointDecoder(VaeConfig.Flux2, []);

        Assert.Null(config.DecoderBlockOutChannels);
    }

    [Fact]
    public void NarrowDecoder_LoadsAndDecodesAtItsOwnWidths()
    {
        VaeConfig config = TinyNarrow();
        Random rng = new(7);
        VaeDecoder decoder = new(config);
        decoder.LoadWeights(SyntheticDecoder(config, rng));

        using IBackend cpu = new CpuBackend();
        using Tensor latent = new(new TensorShape(1, config.LatentChannels, 2, 2), DType.F32);
        Fill(latent, rng, 1f);
        using Tensor image = decoder.Decode(cpu, latent);

        // Three upsamplers: 2x2 latent -> 16x16 RGB.
        Assert.Equal(new[] { 1, 3, 16, 16 }, new[] { (int)image.Shape[0], (int)image.Shape[1], (int)image.Shape[2], (int)image.Shape[3] });
        float[] pixels = image.AsSpan<float>().ToArray();
        Assert.All(pixels, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(pixels, v => v != 0f);
    }

    [Fact]
    public void Decoder_ConvInWiderThanItsPlan_RejectsTheCheckpoint()
    {
        VaeConfig config = TinyNarrow();
        Dictionary<string, Tensor> weights = SyntheticDecoder(config, new Random(7));
        // The encoder-width conv_in (32 channels, the old preset) where this decoder's plan is 24 wide.
        weights["decoder.conv_in.weight"] = new Tensor(new TensorShape(32, config.LatentChannels, 3, 3), DType.F32);

        VaeDecoder decoder = new(config);
        Assert.Throws<InvalidDataException>(() => decoder.LoadWeights(weights));
    }

    /// <summary>Every tensor a decoder built from <paramref name="config"/> reads, at its own widths.</summary>
    private static Dictionary<string, Tensor> SyntheticDecoder(VaeConfig config, Random rng)
    {
        int[] stages = config.DecoderBlockOutChannels ?? config.BlockOutChannels;
        int mid = stages[^1];
        Dictionary<string, Tensor> w = [];

        w["decoder.conv_in.weight"] = Gauss(new TensorShape(mid, config.LatentChannels, 3, 3), rng);
        w["decoder.conv_in.bias"] = Zeros(mid);
        AddResnet(w, "decoder.mid_block.resnets.0", mid, mid, rng);
        AddAttention(w, "decoder.mid_block.attentions.0", mid, rng);
        AddResnet(w, "decoder.mid_block.resnets.1", mid, mid, rng);

        int[] reversed = [.. stages.Reverse()];
        for (int up = 0; up < stages.Length; up++)
        {
            int outCh = reversed[up];
            int inCh = up == 0 ? mid : reversed[up - 1];
            for (int res = 0; res <= config.LayersPerBlock; res++)
                AddResnet(w, $"decoder.up_blocks.{up}.resnets.{res}", res == 0 ? inCh : outCh, outCh, rng);
            if (up < stages.Length - 1)
            {
                w[$"decoder.up_blocks.{up}.upsamplers.0.conv.weight"] = Gauss(new TensorShape(outCh, outCh, 3, 3), rng);
                w[$"decoder.up_blocks.{up}.upsamplers.0.conv.bias"] = Zeros(outCh);
            }
        }

        w["decoder.conv_norm_out.weight"] = Ones(stages[0]);
        w["decoder.conv_norm_out.bias"] = Zeros(stages[0]);
        w["decoder.conv_out.weight"] = Gauss(new TensorShape(3, stages[0], 3, 3), rng);
        w["decoder.conv_out.bias"] = Zeros(3);
        return w;
    }

    private static void AddResnet(Dictionary<string, Tensor> w, string prefix, int inCh, int outCh, Random rng)
    {
        w[$"{prefix}.norm1.weight"] = Ones(inCh);
        w[$"{prefix}.norm1.bias"] = Zeros(inCh);
        w[$"{prefix}.conv1.weight"] = Gauss(new TensorShape(outCh, inCh, 3, 3), rng);
        w[$"{prefix}.conv1.bias"] = Zeros(outCh);
        w[$"{prefix}.norm2.weight"] = Ones(outCh);
        w[$"{prefix}.norm2.bias"] = Zeros(outCh);
        w[$"{prefix}.conv2.weight"] = Gauss(new TensorShape(outCh, outCh, 3, 3), rng);
        w[$"{prefix}.conv2.bias"] = Zeros(outCh);
        if (inCh != outCh)
        {
            w[$"{prefix}.conv_shortcut.weight"] = Gauss(new TensorShape(outCh, inCh, 1, 1), rng);
            w[$"{prefix}.conv_shortcut.bias"] = Zeros(outCh);
        }
    }

    private static void AddAttention(Dictionary<string, Tensor> w, string prefix, int ch, Random rng)
    {
        w[$"{prefix}.group_norm.weight"] = Ones(ch);
        w[$"{prefix}.group_norm.bias"] = Zeros(ch);
        foreach (string proj in new[] { "to_q", "to_k", "to_v", "to_out.0" })
        {
            w[$"{prefix}.{proj}.weight"] = Gauss(new TensorShape(ch, ch), rng, 0.2f);
            w[$"{prefix}.{proj}.bias"] = Zeros(ch);
        }
    }

    private static Tensor Gauss(TensorShape shape, Random rng, float scale = 0.2f)
    {
        Tensor t = new(shape, DType.F32);
        Fill(t, rng, scale);
        return t;
    }

    private static Tensor Zeros(int n) => new(new TensorShape(n), DType.F32);

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    private static void Fill(Tensor t, Random rng, float scale)
    {
        foreach (ref float v in t.AsSpan<float>())
            v = (float)(NextGaussian(rng) * scale);
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
