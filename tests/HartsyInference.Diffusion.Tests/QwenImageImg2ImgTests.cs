using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Qwen-Image img2img / inpaint on the unified <see cref="QwenImagePipeline.GenerateFromTokens"/> API (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). The img2img path uses <see cref="QwenImageVaeEncoder"/> (same encoder AnimaPipeline uses) + packed flow-matching AddNoise mirroring Flux. End-to-end nonzero-strength runs require a real Qwen-Image checkpoint.</summary>
public sealed class QwenImageImg2ImgTests
{
    private static QwenImagePipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
        QwenImageTransformer transformer = new(QwenImageConfig.V1);
        QwenImageVaeDecoder vaeDecoder = new(VaeConfig.QwenImage);
        return withEncoder
            ? new QwenImagePipeline(backend, textEncoder, transformer, vaeDecoder, new QwenImageVaeEncoder(VaeConfig.QwenImage), QwenImageConfig.V1)
            : new QwenImagePipeline(backend, textEncoder, transformer, vaeDecoder, QwenImageConfig.V1);
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: false);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

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
            pipeline.GenerateFromTokens(promptTokenIds: [0], negativeTokenIds: [0], request));

        source.Dispose();
        mask.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        using QwenImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 29) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);

        ImageToImageRequest request = new()
        {
            Prompt = "ignored",
            Width = w,
            Height = h,
            Steps = 4,
            CfgScale = 1.0f,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
            promptTokenIds: [0], negativeTokenIds: [0], request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
    }
}
