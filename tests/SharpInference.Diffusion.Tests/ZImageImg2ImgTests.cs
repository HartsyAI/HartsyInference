using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Wiring tests for Z-Image img2img on the unified <see cref="ZImagePipeline.GenerateFromEmbeddings"/> API. End-to-end nonzero-strength img2img on Z-Image needs validation against a reference (the latent-normalization split between pipeline and VAE decoder makes the math non-obvious); for production refining, prefer the cross-model pixel-space pattern with a different pipeline as the refiner.</summary>
public sealed class ZImageImg2ImgTests
{
    private static Tensor MakeCaptionEmbeddings(int seqLen, int hidden)
    {
        Tensor t = new Tensor(new TensorShape(1, seqLen, hidden), DType.F32);
        return t;
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        ZImageTransformer transformer = new(ZImageConfig.Turbo);
        VaeDecoder vaeDecoder = new(VaeConfig.ZImage);

        using ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, ZImageConfig.Turbo);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 2560);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromEmbeddings(cap, request));

        source.Dispose();
        cap.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ZImageTransformer transformer = new(ZImageConfig.Turbo);
        VaeDecoder vaeDecoder = new(VaeConfig.ZImage);
        VaeEncoder vaeEncoder = new(VaeConfig.ZImage);

        using ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, ZImageConfig.Turbo);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        Tensor cap = MakeCaptionEmbeddings(8, 2560);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(cap, request));

        source.Dispose();
        cap.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgStrength0_PassesSourceThrough()
    {
        using CpuBackend backend = new();
        ZImageTransformer transformer = new(ZImageConfig.Turbo);
        VaeDecoder vaeDecoder = new(VaeConfig.ZImage);
        VaeEncoder vaeEncoder = new(VaeConfig.ZImage);

        using ZImagePipeline pipeline = new(backend, transformer, vaeDecoder, vaeEncoder, ZImageConfig.Turbo);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 11) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);
        Tensor cap = MakeCaptionEmbeddings(8, 2560);

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

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(cap, request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
        cap.Dispose();
    }
}
