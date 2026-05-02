using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// Wiring tests for SDXL img2img: validates the new <see cref="SdxlPipeline.GenerateImg2ImgFromTokens"/>
/// method's shape checks, missing-VaeEncoder error path, and strength=0 byte-identical pass-through.
///
/// SDXL img2img's primary use case is as a refiner step over arbitrary base output (the cross-model pattern):
/// any pipeline's RGB → <see cref="ImagePostProcessor.RgbBytesToTensor"/> → SDXL img2img with low Strength.
/// End-to-end real-checkpoint runs require a full SDXL checkpoint (skipped here; see SdxlGenerationTests).
/// </summary>
public sealed class SdxlImg2ImgTests
{
    private readonly ITestOutputHelper _output;

    public SdxlImg2ImgTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateImg2Img_WithoutVaeEncoder_Throws()
    {
        // Legacy ctor (no VaeEncoder) → img2img must fail loudly.
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet unet = new(UNetConfig.SdxlBase);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);

        using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vaeDecoder);

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
                promptTokenIdsL: [],
                negativePromptTokenIdsL: [],
                promptTokenIdsG: [],
                negativePromptTokenIdsG: [],
                promptEosPositionG: 0,
                negativeEosPositionG: 0,
                request));

        source.Dispose();
    }

    [Fact]
    public void GenerateImg2Img_WrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet unet = new(UNetConfig.SdxlBase);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);

        using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder);

        // Source shape doesn't match the request resolution → ArgumentException.
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
                promptTokenIdsL: [],
                negativePromptTokenIdsL: [],
                promptTokenIdsG: [],
                negativePromptTokenIdsG: [],
                promptEosPositionG: 0,
                negativeEosPositionG: 0,
                request));

        source.Dispose();
    }

    [Fact]
    public void GenerateImg2Img_Strength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so empty token arrays + uninitialized weights are fine.
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet unet = new(UNetConfig.SdxlBase);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);

        using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 3;
                sourceBytes[o + 0] = (byte)(x * 4);
                sourceBytes[o + 1] = (byte)(y * 4);
                sourceBytes[o + 2] = (byte)((x + y) * 2);
            }

        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);

        ImageToImageRequest request = new()
        {
            Prompt = "ignored",
            Width = w,
            Height = h,
            Steps = 10,
            CfgScale = 7.5f,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
            promptTokenIdsL: [],
            negativePromptTokenIdsL: [],
            promptTokenIdsG: [],
            negativePromptTokenIdsG: [],
            promptEosPositionG: 0,
            negativeEosPositionG: 0,
            request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
    }
}
