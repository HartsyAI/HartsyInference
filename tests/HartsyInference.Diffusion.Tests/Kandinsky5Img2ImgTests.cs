using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wiring tests for Kandinsky 5 img2img / inpaint on <see cref="Kandinsky5Pipeline.GenerateFromEmbeddings"/> (no-encoder throws, wrong-shape throws, wrong-mask-shape throws, strength=0 byte-identical pass-through). Kandinsky 5 Lite denoises an unpacked 16-channel Flux-VAE latent under a shift-5 flow-match schedule; img2img is the Z-Image pattern with a <see cref="VaeEncoder"/> configured <see cref="VaeConfig.Flux"/>. Requests use CfgScale=1 so no negative embeddings are needed. End-to-end nonzero-strength runs require a real Kandinsky 5 checkpoint.</summary>
public sealed class Kandinsky5Img2ImgTests
{
    private static Kandinsky5Pipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        Kandinsky5Transformer transformer = new(Kandinsky5Config.Lite);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux);
        return withEncoder
            ? new Kandinsky5Pipeline(backend, transformer, vaeDecoder, new VaeEncoder(VaeConfig.Flux), Kandinsky5Config.Lite)
            : new Kandinsky5Pipeline(backend, transformer, vaeDecoder, Kandinsky5Config.Lite);
    }

    private static (Tensor qwen, Tensor clip) MakeDummyEmbeddings()
    {
        return (new Tensor(new TensorShape(1, 4, 8), DType.F32), new Tensor(new TensorShape(1, 8), DType.F32));
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        using Kandinsky5Pipeline pipeline = MakePipeline(backend, withEncoder: false);
        (Tensor qwen, Tensor clip) = MakeDummyEmbeddings();

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            CfgScale = 1.0f,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromEmbeddings(qwen, clip, null, null, request));

        source.Dispose();
        qwen.Dispose();
        clip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        using Kandinsky5Pipeline pipeline = MakePipeline(backend, withEncoder: true);
        (Tensor qwen, Tensor clip) = MakeDummyEmbeddings();

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            CfgScale = 1.0f,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(qwen, clip, null, null, request));

        source.Dispose();
        qwen.Dispose();
        clip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_InpaintWrongMaskShape_Throws()
    {
        using CpuBackend backend = new();
        using Kandinsky5Pipeline pipeline = MakePipeline(backend, withEncoder: true);
        (Tensor qwen, Tensor clip) = MakeDummyEmbeddings();

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            CfgScale = 1.0f,
            SourceImage = source,
            Mask = mask,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(qwen, clip, null, null, request));

        source.Dispose();
        mask.Dispose();
        qwen.Dispose();
        clip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_Img2ImgStrength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so uninitialized weights are fine.
        using CpuBackend backend = new();
        using Kandinsky5Pipeline pipeline = MakePipeline(backend, withEncoder: true);
        (Tensor qwen, Tensor clip) = MakeDummyEmbeddings();

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
            CfgScale = 1.0f,
            SourceImage = source,
            Strength = 0.0f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
            qwen, clip, null, null, request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
        qwen.Dispose();
        clip.Dispose();
    }
}
