using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Zeta-Chroma img2img / inpaint on <see cref="ZetaChromaPipeline.GenerateFromEmbeddings"/>. Zeta-Chroma is pixel-space (no VAE): the source image IS the clean sample and is noised directly at sigma[startStep], so there is no missing-encoder failure mode — only shape validation and the strength=0 pass-through. Uses the tiny synthetic config from <see cref="ZetaChromaTests"/> (4-px patch).</summary>
public sealed class ZetaChromaImg2ImgTests
{
    /// <summary>Tiny Zeta config mirroring ZetaChromaTests: hidden 32 (2 heads × 16), 2 layers, 4-px pixel patches.</summary>
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

    private static Tensor MakeCaptionEmbeddings(int seqLen, int hidden)
    {
        return new Tensor(new TensorShape(1, seqLen, hidden), DType.F32);
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        ZetaChromaTransformer transformer = new(cfg);
        using ZetaChromaPipeline pipeline = new(backend, transformer, cfg);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 8);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(cap, request, cfgScale: 1.0f));

        source.Dispose();
        cap.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        ZetaChromaTransformer transformer = new(cfg);
        using ZetaChromaPipeline pipeline = new(backend, transformer, cfg);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, 32, 32), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 8);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
            Mask = mask,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(cap, request, cfgScale: 1.0f));

        source.Dispose();
        mask.Dispose();
        cap.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        ZetaChromaConfig cfg = TinyConfig;
        ZetaChromaTransformer transformer = new(cfg);
        using ZetaChromaPipeline pipeline = new(backend, transformer, cfg);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 23) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);
        Tensor cap = MakeCaptionEmbeddings(8, 8);

        ImageToImageRequest request = new()
        {
            Prompt = "ignored",
            Width = w,
            Height = h,
            Steps = 4,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(cap, request, cfgScale: 1.0f);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
        cap.Dispose();
    }
}
