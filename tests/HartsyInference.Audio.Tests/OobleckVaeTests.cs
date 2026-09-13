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
    /// <summary>Same shape as <see cref="TinyConfig"/> but with an ODD stride, as YuE2 has. That stage emits
    /// <c>5L − 1</c> instead of <c>5L</c>, so a decode comes out short of <c>frames × hop</c> — the case an
    /// all-even config cannot exercise, and the one that broke tiling.</summary>
    private static OobleckConfig OddStrideConfig => TinyConfig with { DownsamplingRatios = [2, 5] };

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

    /// <summary>Tiling must be a memory decision only: every core sample carries its whole input support inside its
    /// own tile, so a tiled decode has to agree with a whole-song decode sample for sample — no crossfade, no seam.
    /// <para>This is what validates the default halo. The decoder's support in latent frames is the stem's ±3 plus
    /// each block's dilated residual units divided by the strides above them, so a halo that is too small shows up
    /// here as a mismatch at the tile boundaries rather than as an audible artefact months later.</para></summary>
    [Theory]
    [InlineData(200, 16, OobleckVae.DefaultHaloFrames, false)]   // many short cores: every boundary is interior
    [InlineData(200, 7, OobleckVae.DefaultHaloFrames, false)]    // core that does not divide the length
    [InlineData(2100, OobleckVae.DefaultCoreFrames, OobleckVae.DefaultHaloFrames, false)]   // the production tiling
    [InlineData(200, 16, OobleckVae.DefaultHaloFrames, true)]    // odd stride: the last core is short of end·hop
    [InlineData(200, 7, OobleckVae.DefaultHaloFrames, true)]
    [InlineData(2100, OobleckVae.DefaultCoreFrames, OobleckVae.DefaultHaloFrames, true)]
    public void DecodeTiled_MatchesWholeSongDecode(int frames, int coreFrames, int haloFrames, bool oddStride)
    {
        OobleckConfig config = oddStride ? OddStrideConfig : TinyConfig;
        CpuBackend backend = new();
        OobleckVae vae = new(config);
        vae.LoadWeights(BuildWeights(config, includeEncoder: false));

        using Tensor latent = Rand([1, 2, frames], seed: 17);
        using Tensor whole = vae.Decode(backend, latent);
        using Tensor tiled = vae.DecodeTiled(backend, latent, coreFrames, haloFrames);

        Assert.Equal(whole.Shape.ToString(), tiled.Shape.ToString());
        float* a = (float*)whole.DataPointer;
        float* b = (float*)tiled.DataPointer;
        float scale = 0f, worst = 0f;
        long worstAt = -1;
        for (long i = 0; i < whole.Shape.ElementCount; i++)
        {
            scale = MathF.Max(scale, MathF.Abs(a[i]));
            float diff = MathF.Abs(a[i] - b[i]);
            if (diff > worst) { worst = diff; worstAt = i; }
        }
        Assert.True(worst <= 1e-4f * MathF.Max(1f, scale),
            $"tiled decode diverged at sample {worstAt}: {worst:E3} against a peak of {scale:E3} "
            + $"(frames={frames}, core={coreFrames}, halo={haloFrames}, odd={oddStride}) — the halo is too small.");
    }

    /// <summary>A decode is <c>frames × hop</c> only when every stride is even. YuE2's stride of 5 sits under a
    /// further 64× of upsampling and costs exactly 64 samples, which is what tiling has to budget for.</summary>
    [Fact]
    public void DecodedLength_AccountsForOddStrides()
    {
        Assert.Equal(15_296L, OobleckConfig.Yue2.DecodedLength(8));        // not 8 × 1920 = 15,360
        Assert.Equal(800L * 1920 - 64, OobleckConfig.Yue2.DecodedLength(800));   // the loss is constant in length

        // Every all-even config stays exact, including the two older presets.
        Assert.Equal(200L * TinyConfig.HopLength, TinyConfig.DecodedLength(200));
        Assert.Equal(200L * OobleckConfig.StableAudioOpen.HopLength, OobleckConfig.StableAudioOpen.DecodedLength(200));
        Assert.Equal(200L * OobleckConfig.AceStep15.HopLength, OobleckConfig.AceStep15.DecodedLength(200));
        Assert.Equal(200L * OddStrideConfig.HopLength - 2, OddStrideConfig.DecodedLength(200));
    }

    /// <summary>A song that fits one core takes the straight-through path, so short clips pay nothing for tiling.</summary>
    [Fact]
    public void DecodeTiled_ShorterThanOneCore_MatchesDecode()
    {
        CpuBackend backend = new();
        OobleckVae vae = new(TinyConfig);
        vae.LoadWeights(BuildWeights(TinyConfig, includeEncoder: false));

        using Tensor latent = Rand([1, 2, 12], seed: 23);
        using Tensor whole = vae.Decode(backend, latent);
        using Tensor tiled = vae.DecodeTiled(backend, latent, coreFrames: 64, haloFrames: OobleckVae.DefaultHaloFrames);

        Assert.Equal(12 * 8, (int)tiled.Shape[2]);
        float* a = (float*)whole.DataPointer;
        float* b = (float*)tiled.DataPointer;
        for (long i = 0; i < whole.Shape.ElementCount; i++) Assert.Equal(a[i], b[i]);
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

        // ── Decoder: flat nn.Sequential — layers.0 stem WNConv1d, layers.{1..n} DecoderBlock
        //    ([snake, convT, resUnit×3] at .layers.{0,1,2,3,4}), layers.{n+1} final snake, layers.{n+2} out conv. ──
        int[] decDims = new int[n + 1];
        for (int i = 0; i <= n; i++) decDims[i] = cfg.DecoderChannels * mults[n - i];
        AddConv(w, rng, "decoder.layers.0", decDims[0], cfg.DecoderInputChannels, 7, bias: true);
        for (int i = 0; i < n; i++)
        {
            string blk = $"decoder.layers.{i + 1}";
            int stride = cfg.DownsamplingRatios[n - 1 - i];
            AddSnake(w, $"{blk}.layers.0", decDims[i]);
            AddConvT(w, rng, $"{blk}.layers.1", decDims[i], decDims[i + 1], 2 * stride);
            for (int j = 0; j < 3; j++) AddResUnit(w, rng, $"{blk}.layers.{j + 2}", decDims[i + 1]);
        }
        AddSnake(w, $"decoder.layers.{n + 1}", decDims[^1]);
        AddConv(w, rng, $"decoder.layers.{n + 2}", cfg.AudioChannels, decDims[^1], 7, bias: false);   // out conv, bias=False

        if (!includeEncoder) return w;

        // ── Encoder: layers.0 stem, layers.{1..n} EncoderBlock ([resUnit×3, snake, downConv] at .layers.{0..4}),
        //    layers.{n+1} final snake, layers.{n+2} out conv (→ 2·latent Gaussian params, k=3). ──
        int[] encDims = new int[n + 1];
        for (int i = 0; i <= n; i++) encDims[i] = cfg.EncoderHiddenSize * mults[i];
        AddConv(w, rng, "encoder.layers.0", encDims[0], cfg.AudioChannels, 7, bias: true);
        for (int i = 0; i < n; i++)
        {
            string blk = $"encoder.layers.{i + 1}";
            for (int j = 0; j < 3; j++) AddResUnit(w, rng, $"{blk}.layers.{j}", encDims[i]);
            AddSnake(w, $"{blk}.layers.3", encDims[i]);
            AddConv(w, rng, $"{blk}.layers.4", encDims[i + 1], encDims[i], 2 * cfg.DownsamplingRatios[i], bias: true);
        }
        AddSnake(w, $"encoder.layers.{n + 1}", encDims[^1]);
        AddConv(w, rng, $"encoder.layers.{n + 2}", 2 * cfg.DecoderInputChannels, encDims[^1], 3, bias: true);
        return w;
    }

    // ResidualUnit = Sequential[snake, WNConv1d(k7,dilated), snake, WNConv1d(k1)] at .layers.{0,1,2,3}.
    private static void AddResUnit(Dictionary<string, Tensor> w, Random rng, string unit, int dim)
    {
        AddSnake(w, $"{unit}.layers.0", dim);
        AddConv(w, rng, $"{unit}.layers.1", dim, dim, 7, bias: true);
        AddSnake(w, $"{unit}.layers.2", dim);
        AddConv(w, rng, $"{unit}.layers.3", dim, dim, 1, bias: true);
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
