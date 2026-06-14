using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.Gguf.KeyMappers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU structural tests for Zeta-Chroma (pixel-space Z-Image S3-DiT): decoder head shapes and
/// defensive loading, patch-size inference, the full transformer forward, the txt2img pipeline loop, and the
/// detection/converter plumbing. The real model is mid-pretraining — numerics are validation-pending.</summary>
public unsafe class ZetaChromaTests
{
    /// <summary>Tiny Zeta config: hidden 32 (2 heads × 16), 2 layers, 1 refiner pair, 4-px pixel patches,
    /// 24-wide decoder with 2 res blocks.</summary>
    private static ZetaChromaConfig TinyConfig => new()
    {
        Backbone = new ZImageConfig
        {
            HiddenSize = 32,
            NumHeads = 2,
            HeadDim = 16,
            NumLayers = 2,
            NumRefinerLayers = 1,
            FfnDim = 48,
            InChannels = 3,
            PatchSize = 4,
            CapFeatDim = 8,
            AdaLNEmbedDim = 16,
            AxesDims = [4, 6, 6],
        },
        PatchSize = 4,
        DecoderHidden = 24,
        DecoderResBlocks = 2,
        DecoderMaxFreqs = 2,
    };

    [Fact]
    public void DecoderHead_Forward_RoundTripsPatchShape()
    {
        CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        Dictionary<string, Tensor> weights = ZetaChromaSyntheticWeights.Build(cfg);

        using ZetaChromaDecoderHead head = new(patchDim: 48, maxFreqs: 2, resBlockCount: 2);
        head.LoadWeights(weights);
        Assert.Equal(24, head.Channels);

        Tensor patches = Rand3d(1, 4, 48, seed: 51);
        Tensor cond = Rand3d(1, 4, 32, seed: 52);
        Tensor output = head.Forward(backend, patches, cond);

        Assert.Equal(1, (int)output.Shape[0]);
        Assert.Equal(4, (int)output.Shape[1]);
        Assert.Equal(48, (int)output.Shape[2]);
        float* p = (float*)output.DataPointer;
        for (long i = 0; i < output.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));

        output.Dispose();
        patches.Dispose();
        cond.Dispose();
    }

    [Fact]
    public void DecoderHead_ConstantDctFeatures_AreCoefficientDamped()
    {
        // Internal patch size 1 → cosines collapse to 1, leaving f[u·F+v] = 1/(1+u·v).
        float[] features = ZetaChromaDecoderHead.BuildConstantDctFeatures(maxFreqs: 3);
        Assert.Equal(9, features.Length);
        Assert.Equal(1.0f, features[0], 5);             // u=0, v=0
        Assert.Equal(1.0f, features[2], 5);             // u=0, v=2
        Assert.Equal(0.5f, features[4], 5);             // u=1, v=1
        Assert.Equal(0.2f, features[8], 5);             // u=2, v=2 → 1/5
    }

    [Fact]
    public void DecoderHead_LoadWeights_ThrowsListingUnmatchedKeys()
    {
        ZetaChromaConfig cfg = TinyConfig;
        Dictionary<string, Tensor> weights = ZetaChromaSyntheticWeights.Build(cfg);
        weights["dec_net.bogus.weight"] = Rand3d(1, 1, 4, seed: 61);

        using ZetaChromaDecoderHead head = new(patchDim: 48, maxFreqs: 2, resBlockCount: 2);
        UnsupportedModelException ex = Assert.Throws<UnsupportedModelException>(() => head.LoadWeights(weights));
        Assert.Contains("dec_net.bogus.weight", ex.Message);
    }

    [Fact]
    public void DecoderHead_LoadWeights_ThrowsListingMissingKeys()
    {
        ZetaChromaConfig cfg = TinyConfig;
        Dictionary<string, Tensor> weights = ZetaChromaSyntheticWeights.Build(cfg);
        weights.Remove("dec_net.cond_embed.weight");

        using ZetaChromaDecoderHead head = new(patchDim: 48, maxFreqs: 2, resBlockCount: 2);
        UnsupportedModelException ex = Assert.Throws<UnsupportedModelException>(() => head.LoadWeights(weights));
        Assert.Contains("dec_net.cond_embed.weight", ex.Message);
    }

    [Fact]
    public void Config_InfersPatchSizeFromXEmbedderInDim()
    {
        // Reported release shape: in-dim 3072 = 3·32² → patch 32. Hidden 256 keeps NumHeads sane (256/128 = 2).
        Dictionary<string, Tensor> weights = new()
        {
            ["x_embedder.weight"] = Zeros(256, 3072),
            ["dec_net.cond_embed.weight"] = Zeros(64, 256),
        };
        ZetaChromaConfig config = ZetaChromaConfig.FromWeights(weights);
        Assert.Equal(32, config.PatchSize);
        Assert.Equal(32, config.Backbone.PatchSize);
        Assert.Equal(3, config.Backbone.InChannels);
        Assert.Equal(64, config.DecoderHidden);
        Assert.True(ZetaChromaConfig.IsZetaChroma(weights));

        Dictionary<string, Tensor> classic = new() { ["x_embedder.weight"] = Zeros(3840, 64) };
        Assert.False(ZetaChromaConfig.IsZetaChroma(classic));
    }

    [Fact]
    public void Transformer_Forward_ProducesX0InPixelShape()
    {
        CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        using ZetaChromaTransformer transformer = new(cfg);
        transformer.LoadWeights(ZetaChromaSyntheticWeights.Build(cfg));

        Tensor pixels = Rand4d(1, 3, 16, 16, seed: 71);
        Tensor caption = Rand3d(1, 4, 8, seed: 72);

        Tensor x0 = transformer.Forward(backend, pixels, caption, sigma: 0.75f);
        Assert.Equal(1, (int)x0.Shape[0]);
        Assert.Equal(3, (int)x0.Shape[1]);
        Assert.Equal(16, (int)x0.Shape[2]);
        Assert.Equal(16, (int)x0.Shape[3]);
        float* p = (float*)x0.DataPointer;
        for (long i = 0; i < x0.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));

        x0.Dispose();
        pixels.Dispose();
        caption.Dispose();
    }

    [Fact]
    public void Transformer_LoadWeights_RejectsPlainZImageDicts()
    {
        ZetaChromaConfig cfg = TinyConfig;
        Dictionary<string, Tensor> weights = ZetaChromaSyntheticWeights.Build(cfg);
        foreach (string key in weights.Keys.Where(k => k.StartsWith("dec_net.", StringComparison.Ordinal)).ToList())
            weights.Remove(key);

        using ZetaChromaTransformer transformer = new(cfg);
        Assert.Throws<UnsupportedModelException>(() => transformer.LoadWeights(weights));
    }

    [Fact]
    public void Pipeline_GeneratesRgb_AndValidatesResolution()
    {
        CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        using ZetaChromaTransformer transformer = new(cfg);
        transformer.LoadWeights(ZetaChromaSyntheticWeights.Build(cfg));
        using ZetaChromaPipeline pipeline = new(backend, transformer, cfg);

        Tensor caption = Rand3d(1, 4, 8, seed: 81);
        TextToImageRequest request = new()
        {
            Prompt = "test",
            Width = 16,
            Height = 16,
            Steps = 2,
            Seed = 7,
        };

        int progressCount = 0;
        (byte[] rgb, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(
            caption, request, cfgScale: 1.0f, onProgress: _ => progressCount++);
        Assert.Equal(16 * 16 * 3, rgb.Length);
        Assert.Equal(16, w);
        Assert.Equal(16, h);
        Assert.Equal(7, seed);
        Assert.Equal(2, progressCount);

        // Resolution not divisible by the pixel patch must fail fast.
        TextToImageRequest badRequest = new() { Prompt = "test", Width = 18, Height = 16, Steps = 2 };
        Assert.Throws<ArgumentException>(() => pipeline.GenerateFromEmbeddings(caption, badRequest, cfgScale: 1.0f));

        caption.Dispose();
    }

    [Fact]
    public void Converter_BucketsDecNetIntoTransformer_AndStripsOrigMod()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["_orig_mod.dec_net.cond_embed.weight"] = Zeros(8, 16),
            ["_orig_mod.x_embedder.weight"] = Zeros(16, 48),
            ["_orig_mod.layers.0.attention.qkv.weight"] = Zeros(48, 16),
        };

        ZImageCheckpointConverter.ConvertedWeights converted = ZetaChromaCheckpointConverter.Convert(raw);
        Assert.True(converted.Transformer.ContainsKey("dec_net.cond_embed.weight"));
        Assert.True(converted.Transformer.ContainsKey("x_embedder.weight"));
        Assert.True(converted.Transformer.ContainsKey("layers.0.attention.qkv.weight"));
        Assert.True(ZetaChromaCheckpointConverter.IsZetaChroma(converted.Transformer));
    }

    [Fact]
    public void KeyMapper_DetectsZetaBeforeZImage()
    {
        string[] zetaNames =
        [
            "noise_refiner.0.attention.qkv.weight",
            "context_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
            "dec_net.cond_embed.weight",
        ];
        IGgufKeyMapper mapper = GgufKeyMapperRegistry.DetectByKeys(zetaNames);
        Assert.Equal("zeta-chroma", mapper.Architecture);

        string[] zImageNames =
        [
            "noise_refiner.0.attention.qkv.weight",
            "context_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
            "final_layer.linear.weight",
        ];
        IGgufKeyMapper zMapper = GgufKeyMapperRegistry.DetectByKeys(zImageNames);
        Assert.Equal("zimage", zMapper.Architecture);
    }

    private static Tensor Rand4d(int b, int c, int h, int w, int seed)
    {
        Tensor t = new Tensor(new TensorShape(b, c, h, w), DType.F32);
        FillRandom(t, seed);
        return t;
    }

    private static Tensor Rand3d(int b, int s, int d, int seed)
    {
        Tensor t = new Tensor(new TensorShape(b, s, d), DType.F32);
        FillRandom(t, seed);
        return t;
    }

    private static Tensor Zeros(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = 0f;
        return t;
    }

    private static void FillRandom(Tensor t, int seed)
    {
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }
}
