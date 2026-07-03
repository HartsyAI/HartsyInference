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

/// <summary>Wiring tests for ERNIE-Image img2img / inpaint on <see cref="ErnieImagePipeline.GenerateFromTokens"/> (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). The img2img path goes VAE-encode → 2×2 patchify (32 → 128 ch) → optional BN-normalize → flow-match AddNoise, mirroring Flux.2's round-1 implementation. End-to-end nonzero-strength runs require the real ERNIE-Image checkpoint.</summary>
public sealed class ErnieImageImg2ImgTests
{
    private static ErnieImagePipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        ErnieImageLlamaTextEncoder textEncoder = new(new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B));
        ErnieImageTransformer transformer = new(ErnieImageConfig.V1);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux2);
        return withEncoder
            ? new ErnieImagePipeline(backend, textEncoder, transformer, vaeDecoder, ErnieImageConfig.V1,
                vaeEncoder: new VaeEncoder(VaeConfig.Flux2))
            : new ErnieImagePipeline(backend, textEncoder, transformer, vaeDecoder, ErnieImageConfig.V1);
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using ErnieImagePipeline pipeline = MakePipeline(backend, withEncoder: false);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens([0], [0], promptRealLen: 1, negativeRealLen: 1, request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using ErnieImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens([0], [0], promptRealLen: 1, negativeRealLen: 1, request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        using ErnieImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

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
            pipeline.GenerateFromTokens([0], [0], promptRealLen: 1, negativeRealLen: 1, request));

        source.Dispose();
        mask.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        using ErnieImagePipeline pipeline = MakePipeline(backend, withEncoder: true);

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
            [0], [0], promptRealLen: 1, negativeRealLen: 1, request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
    }
}
