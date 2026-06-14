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

/// <summary>Wiring tests for Flux img2img: validates the unified-API <see cref="FluxPipeline.GenerateFromTokens"/> handles ImageToImageRequest correctly (no-VaeEncoder throws, wrong-shape throws, strength=0 byte-identical pass-through). End-to-end nonzero-strength img2img requires a real Flux Schnell/Dev checkpoint and is exercised separately when one is available.</summary>
public sealed class FluxImg2ImgTests
{
    [Fact]
    public void GenerateFromTokens_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
        FluxTransformer transformer = new(FluxConfig.Schnell);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);

        using FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, FluxConfig.Schnell);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(promptTokenIdsL: [], promptEosPositionL: 0,
                promptTokenIdsT5: [], promptAttentionMaskT5: null, request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
        FluxTransformer transformer = new(FluxConfig.Schnell);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);

        using FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder, FluxConfig.Schnell);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIdsL: [], promptEosPositionL: 0,
                promptTokenIdsT5: [], promptAttentionMaskT5: null, request));

        source.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so empty token arrays + uninitialized weights are fine.
        using CpuBackend backend = new();
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
        T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
        FluxTransformer transformer = new(FluxConfig.Schnell);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);

        using FluxPipeline pipeline = new(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder, FluxConfig.Schnell);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 7) & 0xFF);
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
            promptTokenIdsL: [], promptEosPositionL: 0,
            promptTokenIdsT5: [], promptAttentionMaskT5: null, request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
    }
}
