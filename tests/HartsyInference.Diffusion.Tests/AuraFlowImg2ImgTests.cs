using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for AuraFlow img2img / inpaint on <see cref="AuraFlowPipeline.GenerateFromTokens"/> (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). AuraFlow denoises an unpacked 4-channel SDXL-family latent under a static-shift flow-match schedule; img2img is the Z-Image pattern with a <see cref="VaeEncoder"/> configured <see cref="VaeConfig.AuraFlow"/>. End-to-end nonzero-strength runs require a real AuraFlow checkpoint.</summary>
public sealed class AuraFlowImg2ImgTests
{
    private static AuraFlowPipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
        AuraFlowTransformer transformer = new(AuraFlowConfig.V03);
        VaeDecoder vaeDecoder = new(VaeConfig.AuraFlow);
        return withEncoder
            ? new AuraFlowPipeline(backend, t5, transformer, vaeDecoder, new VaeEncoder(VaeConfig.AuraFlow), AuraFlowConfig.V03)
            : new AuraFlowPipeline(backend, t5, transformer, vaeDecoder, AuraFlowConfig.V03);
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using AuraFlowPipeline pipeline = MakePipeline(backend, withEncoder: false);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(
                promptTokenIdsT5: [0], negativePromptTokenIdsT5: [0],
                promptAttentionMaskT5: [1], negativeAttentionMaskT5: [1], request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using AuraFlowPipeline pipeline = MakePipeline(backend, withEncoder: true);

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
        using AuraFlowPipeline pipeline = MakePipeline(backend, withEncoder: true);

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
        using AuraFlowPipeline pipeline = MakePipeline(backend, withEncoder: true);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 17) & 0xFF);
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
