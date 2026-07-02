using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Chroma Radiance img2img / inpaint on <see cref="ChromaRadiancePipeline.GenerateFromTokens"/>. Radiance is pixel-space (no VAE): the source image IS the clean sample and is noised directly at sigma[startStep], so there is no missing-encoder failure mode — only shape validation and the strength=0 pass-through.</summary>
public sealed class ChromaRadianceImg2ImgTests
{
    private static ChromaRadiancePipeline MakePipeline(CpuBackend backend)
    {
        T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
        ChromaRadianceTransformer transformer = new(ChromaRadianceConfig.V1);
        return new ChromaRadiancePipeline(backend, t5, transformer, ChromaRadianceConfig.V1);
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using ChromaRadiancePipeline pipeline = MakePipeline(backend);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(
                promptTokenIdsT5: [0], negativePromptTokenIdsT5: [0],
                promptAttentionMaskT5: [1], negativeAttentionMaskT5: [1], request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        using ChromaRadiancePipeline pipeline = MakePipeline(backend);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
            Mask = mask,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(
                promptTokenIdsT5: [0], negativePromptTokenIdsT5: [0],
                promptAttentionMaskT5: [1], negativeAttentionMaskT5: [1], request));

        source.Dispose();
        mask.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        using ChromaRadiancePipeline pipeline = MakePipeline(backend);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 19) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);

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

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
            promptTokenIdsT5: [0], negativePromptTokenIdsT5: [0],
            promptAttentionMaskT5: [1], negativeAttentionMaskT5: [1], request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
    }
}
