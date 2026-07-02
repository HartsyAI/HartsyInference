using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Anima img2img / inpaint on <see cref="AnimaPipeline.GenerateFromEmbeddings"/> (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). Anima uses the Qwen-Image VAE, so <see cref="QwenImageVaeEncoder"/> enables img2img; the masked-inpaint blend runs on the unpacked <c>[1, 16, H, W]</c> latent.</summary>
public sealed class AnimaImg2ImgTests
{
    private static AnimaPipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        AnimaConfig config = AnimaConfig.AnimaPreview3;
        AnimaTransformer transformer = new(config);
        AnimaLlmAdapter llmAdapter = new(config.LlmAdapter);
        QwenImageVaeDecoder vaeDecoder = new(VaeConfig.QwenImage);
        return withEncoder
            ? new AnimaPipeline(backend, transformer, llmAdapter, vaeDecoder, new QwenImageVaeEncoder(VaeConfig.QwenImage), config)
            : new AnimaPipeline(backend, transformer, llmAdapter, vaeDecoder, config);
    }

    private static Tensor MakeTextEmbeddings(int seqLen, int hidden)
    {
        return new Tensor(new TensorShape(1, seqLen, hidden), DType.F32);
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using AnimaPipeline pipeline = MakePipeline(backend, withEncoder: false);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor emb = MakeTextEmbeddings(8, 1024);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromEmbeddings(emb, t5TokenIds: [0], request));

        source.Dispose();
        emb.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using AnimaPipeline pipeline = MakePipeline(backend, withEncoder: true);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        Tensor emb = MakeTextEmbeddings(8, 1024);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(emb, t5TokenIds: [0], request));

        source.Dispose();
        emb.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        using AnimaPipeline pipeline = MakePipeline(backend, withEncoder: true);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, 32, 32), DType.F32);
        Tensor emb = MakeTextEmbeddings(8, 1024);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
            Mask = mask,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(emb, t5TokenIds: [0], request));

        source.Dispose();
        mask.Dispose();
        emb.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        using AnimaPipeline pipeline = MakePipeline(backend, withEncoder: true);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 37) & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);
        Tensor emb = MakeTextEmbeddings(8, 1024);

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

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
            emb, t5TokenIds: [0], request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
        emb.Dispose();
    }
}
