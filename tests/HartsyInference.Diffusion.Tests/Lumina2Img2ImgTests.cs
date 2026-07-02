using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Lumina-Image-2.0 img2img / inpaint on <see cref="Lumina2Pipeline.GenerateFromEmbeddings"/> (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). Lumina 2 uses the 16-channel Flux VAE, so the standard <see cref="VaeEncoder"/> with <c>VaeConfig.Flux</c> enables img2img. End-to-end nonzero-strength runs require a real checkpoint + Gemma-2 embeddings.</summary>
public sealed class Lumina2Img2ImgTests
{
    private static Tensor MakeCaptionEmbeddings(int seqLen, int hidden)
    {
        return new Tensor(new TensorShape(1, seqLen, hidden), DType.F32);
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        Lumina2Transformer transformer = new(Lumina2Config.V2_0);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);

        using Lumina2Pipeline pipeline = new(backend, transformer, vaeDecoder, Lumina2Config.V2_0);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 2304);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromEmbeddings(cap, request, cfgScale: 1.0f));

        source.Dispose();
        cap.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        Lumina2Transformer transformer = new(Lumina2Config.V2_0);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);

        using Lumina2Pipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, Lumina2Config.V2_0);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 2304);
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
        Lumina2Transformer transformer = new(Lumina2Config.V2_0);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);

        using Lumina2Pipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, Lumina2Config.V2_0);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, 32, 32), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 2304);
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
        Lumina2Transformer transformer = new(Lumina2Config.V2_0);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux);

        using Lumina2Pipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, Lumina2Config.V2_0);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 31) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);
        Tensor cap = MakeCaptionEmbeddings(8, 2304);

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
