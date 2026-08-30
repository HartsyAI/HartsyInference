using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs the MiniMax-H3 ViT3D video VAE decoder end to end on a tiny synthetic config: latent denormalization,
/// spatial tiling with ramp blending, temporal chunking with the overlap carry, the transformer stack and the folded
/// pixel denormalization. Gates wiring and shapes, not numerics — real-weight parity comes later.</summary>
[Trait("Category", "SyntheticSmoke")]
public unsafe class MiniMaxH3VideoVaeTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3VideoVaeTests(ITestOutputHelper output) => _output = output;

    private static int _seed = 23;

    private static Tensor Rand(params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            _seed = _seed * 1103515245 + 12345;
            p[i] = ((_seed >> 16) & 0x7fff) / 32768f * 0.08f - 0.04f;
        }
        return t;
    }

    private static Tensor Ones(params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    /// <summary>A miniature H3 VAE that keeps every structural relationship — 3-axis partial rope (6 of 8 head dims),
    /// gated FFN, register+cls suffix, 4× spatial / 2× temporal patch — while staying CPU-sized. Tile size, overlap
    /// and latent extent are all multiples of the spatial ratio, which the tile arithmetic assumes.</summary>
    private static MiniMaxH3VideoVaeConfig TinyConfig() => new MiniMaxH3VideoVaeConfig
    {
        Heads = 2,
        DimHead = 8,
        NumLayers = 2,
        NumRegisterTokens = 2,
        FfnMult = 2,
        ZChannels = 4,
        LatentChannels = 4,
        OutChannels = 3,
        VaeRatio = 4,
        VaeRatioT = 2,
        ClipLength = 5,
        TokenDrop = 1,
        TileSize = 16,
        TileOverlapMin = 8,
        LatentsMean = [0.1f, -0.2f, 0.3f, -0.4f],
        LatentsStd = [1.1f, 0.9f, 1.3f, 0.7f],
    };

    private static Dictionary<string, Tensor> BuildWeights(MiniMaxH3VideoVaeConfig c)
    {
        int dim = c.Dim, inner = c.FfnInner;
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>
        {
            ["post_quant_conv.weight"] = Rand(c.ZChannels, c.LatentChannels, 1, 1, 1),
            ["post_quant_conv.bias"] = Rand(c.ZChannels),
            ["decoder.x_embedder.weight"] = Rand(dim, c.ZChannels),
            ["decoder.x_embedder.bias"] = Rand(dim),
            ["decoder.register_tokens"] = Rand(1, c.NumRegisterTokens, dim),
            ["decoder.norm_out.weight"] = Ones(dim),
            ["decoder.norm_out.bias"] = Rand(dim),
            ["decoder.proj_out.weight"] = Rand(c.PatchDim, dim),
            ["decoder.proj_out.bias"] = Rand(c.PatchDim),
            ["latents_mean"] = Rand(c.LatentChannels),
            ["latents_std"] = Ones(c.LatentChannels),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"decoder.transformer_blocks.{i}";
            w[$"{p}.norm1.weight"] = Ones(dim);
            w[$"{p}.norm2.weight"] = Ones(dim);
            // scale1/scale2 initialize to zero but are learned; zeros here would make every block the identity.
            w[$"{p}.scale1"] = Rand(dim);
            w[$"{p}.scale2"] = Rand(dim);
            w[$"{p}.attn.to_qkv.weight"] = Rand(3L * dim, dim);
            w[$"{p}.attn.to_qkv.bias"] = Rand(3L * dim);
            w[$"{p}.attn.to_out.weight"] = Rand(dim, dim);
            w[$"{p}.attn.to_out.bias"] = Rand(dim);
            w[$"{p}.ff.w1.weight"] = Rand(2L * inner, dim);
            w[$"{p}.ff.w1.bias"] = Rand(2L * inner);
            w[$"{p}.ff.w2.weight"] = Rand(dim, inner);
            w[$"{p}.ff.w2.bias"] = Rand(dim);
        }
        return w;
    }

    [Theory]
    [InlineData(true, 7, 6, 6, 11)]    // two spatial tiles per axis: ramp blend + canvas crop, two temporal chunks
    [InlineData(false, 7, 6, 6, 11)]
    [InlineData(true, 1, 6, 6, 1)]     // single latent token: the non-chunked path that keeps only the last frame
    [InlineData(false, 1, 6, 6, 1)]
    [InlineData(true, 7, 6, 4, 11)]    // tiles on H only, so the W overlap list is empty
    public void DecodeProducesFiniteRgbOfTheExpectedShape(bool tiling, int latentT, int latentH, int latentW, int frames)
    {
        MiniMaxH3VideoVaeConfig c = TinyConfig() with { DecoderTiling = tiling };
        IBackend backend = new CpuBackend();
        using MiniMaxH3VideoVaeDecoder vae = new MiniMaxH3VideoVaeDecoder(c);
        Dictionary<string, Tensor> weights = BuildWeights(c);
        vae.LoadWeights(weights);

        int expectedFrames = vae.OutputFrames(latentT);
        _output.WriteLine($"tokens_chunk={c.TokensChunkSize} token_overlap={c.TokenOverlap} "
            + $"frame_pre_pad={c.FramePrePadding} frame_overlap={c.FrameOverlap} frames={expectedFrames}");
        Assert.Equal(frames, expectedFrames);

        using Tensor latent = Rand(1, c.LatentChannels, latentT, latentH, latentW);
        using Tensor rgb = vae.Decode(backend, latent);

        Assert.Equal(5, rgb.Shape.Rank);
        Assert.Equal(1, (int)rgb.Shape[0]);
        Assert.Equal(c.OutChannels, (int)rgb.Shape[1]);
        Assert.Equal(expectedFrames, (int)rgb.Shape[2]);
        Assert.Equal(latentH * c.VaeRatio, (int)rgb.Shape[3]);
        Assert.Equal(latentW * c.VaeRatio, (int)rgb.Shape[4]);

        float* p = (float*)rgb.DataPointer;
        float min = p[0], max = p[0];
        for (long i = 0; i < rgb.ElementCount; i++)
        {
            Assert.True(float.IsFinite(p[i]), $"non-finite pixel at {i}");
            if (p[i] < min) min = p[i];
            if (p[i] > max) max = p[i];
        }
        _output.WriteLine($"tiling={tiling} pixel range [{min:F5}, {max:F5}]");
        Assert.True(min >= -1f && max <= 1f, $"pixels escaped [-1, 1]: [{min}, {max}]");
        Assert.True(max > min, "decoder produced a constant image");

        foreach (Tensor t in weights.Values) t.Dispose();
    }

    [Fact]
    public void PartialLoadFailure_CanBeDisposedWithoutTakingSourceWeights()
    {
        MiniMaxH3VideoVaeConfig config = TinyConfig();
        Dictionary<string, Tensor> weights = BuildWeights(config);
        Tensor[] sourceTensors = weights.Values.Distinct().ToArray();
        weights.Remove("decoder.transformer_blocks.1.ff.w2.weight");
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        try
        {
            Assert.Throws<KeyNotFoundException>(() => decoder.LoadWeights(weights));

            decoder.Dispose();
            decoder.Dispose();
            foreach (Tensor source in sourceTensors)
            {
                Assert.NotEqual(0, (nint)source.DataPointer);
            }
        }
        finally
        {
            decoder.Dispose();
            foreach (Tensor source in sourceTensors)
            {
                source.Dispose();
            }
        }
    }

    /// <summary>The shipped FL2VA configs, verbatim. <c>time_up</c> is JSON null, so the temporal ratio has to fall
    /// through to the encoder's <c>time_down</c> list — the branch that would silently yield ratio 1.</summary>
    [Fact]
    public void FromJsonReadsTheShippedConfiguration()
    {
        const string wrapper = """
        {
          "vae_clip_length": 17, "vae_token_drop": 3, "vae_decoder_tiling": 1,
          "vae_tile_size": 256, "vae_tile_overlap_min": 64, "latent_channels": 24,
          "latents_mean": [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
            -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
            -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
            -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
            -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
            -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264],
          "latents_std": [1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
            1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
            0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
            0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
            2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
            3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525]
        }
        """;
        const string source = """
        {
          "embed_dim": 24, "z_channels": 24, "out_ch": 3,
          "space_down": [2, 2, 2, 2, 1, 1], "space_up": [1, 2, 2, 2, 2, 1],
          "time_down": [1, 2, 2, 1, 1, 1], "time_up": null,
          "vit_decoder_kwargs": {
            "dim_head": 64, "heads": 32, "num_layers": 36,
            "rope_dim_ratio": 0.75, "rope_theta": 100.0
          }
        }
        """;
        MiniMaxH3VideoVaeConfig c = MiniMaxH3VideoVaeConfig.FromJson(wrapper, source);
        Assert.Equal(32, c.Heads);
        Assert.Equal(64, c.DimHead);
        Assert.Equal(2048, c.Dim);
        Assert.Equal(36, c.NumLayers);
        Assert.Equal(16, c.VaeRatio);
        Assert.Equal(4, c.VaeRatioT);
        Assert.Equal(24, c.LatentChannels);
        Assert.Equal(17, c.ClipLength);
        Assert.Equal(3, c.TokenDrop);
        Assert.Equal(256, c.TileSize);
        Assert.Equal(64, c.TileOverlapMin);
        Assert.True(c.DecoderTiling);
        Assert.Equal(24, c.LatentsMean.Length);
        Assert.Equal(24, c.LatentsStd.Length);
        Assert.Equal(48, c.RopeDim);
        Assert.Equal(8, c.RopeInvFreqLen);
        Assert.Equal(3072, c.PatchDim);
        // clip 17 / ratio_t 4 -> chunk 5 tokens, overlap 2, 3 leading frames dropped, 5 blended.
        Assert.Equal(5, c.TokensChunkSize);
        Assert.Equal(2, c.TokenOverlap);
        Assert.Equal(3, c.FramePrePadding);
        Assert.Equal(5, c.FrameOverlap);
    }

    [Fact]
    public void DetectRecoversTheStructureFromWeightShapes()
    {
        MiniMaxH3VideoVaeConfig c = TinyConfig();
        Dictionary<string, Tensor> weights = BuildWeights(c);
        try
        {
            Assert.True(MiniMaxH3VideoVaeConfig.Matches(weights));
            MiniMaxH3VideoVaeConfig detected = MiniMaxH3VideoVaeConfig.Detect(weights, c);
            Assert.Equal(c.Dim, detected.Dim);
            Assert.Equal(c.NumLayers, detected.NumLayers);
            Assert.Equal(c.NumRegisterTokens, detected.NumRegisterTokens);
            Assert.Equal(c.FfnMult, detected.FfnMult);
            Assert.Equal(c.ZChannels, detected.ZChannels);
            Assert.Equal(c.LatentChannels, detected.LatentChannels);
            // latents_std ships in the checkpoint and must win over the config defaults.
            Assert.Equal(1f, detected.LatentsStd[0]);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
