using Xunit;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Audio.Tests;

/// <summary>Structural tests for the Oobleck waveform VAE on a tiny synthetic config: key naming,
/// weight-norm fusion, logscale snake handling, stride math (encode T/hop, decode T·hop), the
/// mean-half latent slice, and decode-only checkpoint handling. Numerics vs the diffusers reference
/// are validation-pending (no weights in this environment).</summary>
public sealed unsafe class OobleckVaeTests
{
    /// <summary>Tiny config: hop = 2·4 = 8, latent dim 2, stereo.</summary>
    private static OobleckConfig TinyConfig => new()
    {
        EncoderHiddenSize = 4,
        DownsamplingRatios = [2, 4],
        ChannelMultiples = [1, 2],
        DecoderChannels = 4,
        DecoderInputChannels = 2,
        AudioChannels = 2,
        SamplingRate = 48,
    };

    [Fact]
    public void Decode_ExpandsByHop()
    {
        CpuBackend backend = new();
        OobleckVae vae = new(TinyConfig);
        vae.LoadWeights(BuildWeights(TinyConfig, includeEncoder: false));

        Tensor latent = Rand([1, 2, 6], seed: 5);
        Tensor pcm = vae.Decode(backend, latent);

        Assert.Equal(3, pcm.Shape.Rank);
        Assert.Equal(2, (int)pcm.Shape[1]);
        Assert.Equal(6 * 8, (int)pcm.Shape[2]);
        float* p = (float*)pcm.DataPointer;
        for (long i = 0; i < pcm.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
        latent.Dispose();
        pcm.Dispose();
    }

    [Fact]
    public void EncodeMode_CompressesByHop_AndDecodeRoundTripsShape()
    {
        CpuBackend backend = new();
        OobleckVae vae = new(TinyConfig);
        vae.LoadWeights(BuildWeights(TinyConfig, includeEncoder: true));
        Assert.True(vae.HasEncoder);

        Tensor pcm = Rand([1, 2, 48], seed: 11);
        Tensor latent = vae.EncodeMode(backend, pcm);
        Assert.Equal(2, (int)latent.Shape[1]);
        Assert.Equal(6, (int)latent.Shape[2]);

        Tensor decoded = vae.Decode(backend, latent);
        Assert.Equal(48, (int)decoded.Shape[2]);
        pcm.Dispose();
        latent.Dispose();
        decoded.Dispose();
    }

    [Fact]
    public void DecodeOnlyCheckpoint_EncodeThrows()
    {
        CpuBackend backend = new();
        OobleckVae vae = new(TinyConfig);
        vae.LoadWeights(BuildWeights(TinyConfig, includeEncoder: false));
        Assert.False(vae.HasEncoder);

        Tensor pcm = Rand([1, 2, 48], seed: 3);
        Assert.Throws<InvalidOperationException>(() => vae.EncodeMode(backend, pcm));
        pcm.Dispose();
    }

    /// <summary>Builds a synthetic diffusers-layout weight dict mirroring the real checkpoint's key
    /// structure (verified against the ACE-Step 1.5 vae safetensors header).</summary>
    private static Dictionary<string, Tensor> BuildWeights(OobleckConfig cfg, bool includeEncoder)
    {
        Dictionary<string, Tensor> w = new();
        Random rng = new(42);
        int n = cfg.DownsamplingRatios.Length;
        int[] mults = new int[cfg.ChannelMultiples.Length + 1];
        mults[0] = 1;
        cfg.ChannelMultiples.CopyTo(mults, 1);

        // ── Decoder ──
        int[] decDims = new int[n + 1];
        for (int i = 0; i <= n; i++) decDims[i] = cfg.DecoderChannels * mults[n - i];
        AddConv(w, rng, "decoder.conv1", decDims[0], cfg.DecoderInputChannels, 7, bias: true);
        for (int i = 0; i < n; i++)
        {
            string blk = $"decoder.block.{i}";
            int stride = cfg.DownsamplingRatios[n - 1 - i];
            AddSnake(w, $"{blk}.snake1", decDims[i]);
            AddConvT(w, rng, $"{blk}.conv_t1", decDims[i], decDims[i + 1], 2 * stride);
            AddResUnits(w, rng, blk, decDims[i + 1]);
        }
        AddSnake(w, "decoder.snake1", decDims[^1]);
        AddConv(w, rng, "decoder.conv2", cfg.AudioChannels, decDims[^1], 7, bias: false);

        if (!includeEncoder) return w;

        // ── Encoder ──
        int[] encDims = new int[n + 1];
        for (int i = 0; i <= n; i++) encDims[i] = cfg.EncoderHiddenSize * mults[i];
        AddConv(w, rng, "encoder.conv1", encDims[0], cfg.AudioChannels, 7, bias: true);
        for (int i = 0; i < n; i++)
        {
            string blk = $"encoder.block.{i}";
            AddResUnits(w, rng, blk, encDims[i]);
            AddSnake(w, $"{blk}.snake1", encDims[i]);
            AddConv(w, rng, $"{blk}.conv1", encDims[i + 1], encDims[i], 2 * cfg.DownsamplingRatios[i], bias: true);
        }
        AddSnake(w, "encoder.snake1", encDims[^1]);
        AddConv(w, rng, "encoder.conv2", cfg.EncoderHiddenSize, encDims[^1], 3, bias: true);
        return w;
    }

    private static void AddResUnits(Dictionary<string, Tensor> w, Random rng, string blk, int dim)
    {
        for (int j = 0; j < 3; j++)
        {
            string unit = $"{blk}.res_unit{j + 1}";
            AddSnake(w, $"{unit}.snake1", dim);
            AddConv(w, rng, $"{unit}.conv1", dim, dim, 7, bias: true);
            AddSnake(w, $"{unit}.snake2", dim);
            AddConv(w, rng, $"{unit}.conv2", dim, dim, 1, bias: true);
        }
    }

    private static void AddConv(Dictionary<string, Tensor> w, Random rng, string prefix, int oc, int ic, int k, bool bias)
    {
        w[$"{prefix}.weight_g"] = Rand([oc, 1, 1], rng);
        w[$"{prefix}.weight_v"] = Rand([oc, ic, k], rng);
        if (bias) w[$"{prefix}.bias"] = Rand([oc], rng);
    }

    private static void AddConvT(Dictionary<string, Tensor> w, Random rng, string prefix, int ic, int oc, int k)
    {
        // ConvTranspose1d weight layout is [in, out, k]; weight_norm is per input channel (dim 0).
        w[$"{prefix}.weight_g"] = Rand([ic, 1, 1], rng);
        w[$"{prefix}.weight_v"] = Rand([ic, oc, k], rng);
        w[$"{prefix}.bias"] = Rand([oc], rng);
    }

    private static void AddSnake(Dictionary<string, Tensor> w, string prefix, int dim)
    {
        // Logscale params: zeros → exp(0) = 1, the well-conditioned default.
        w[$"{prefix}.alpha"] = Zeros([1, dim, 1]);
        w[$"{prefix}.beta"] = Zeros([1, dim, 1]);
    }

    private static Tensor Rand(long[] shape, Random rng)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static Tensor Rand(long[] shape, int seed) => Rand(shape, new Random(seed));

    private static Tensor Zeros(long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = 0f;
        return t;
    }
}
