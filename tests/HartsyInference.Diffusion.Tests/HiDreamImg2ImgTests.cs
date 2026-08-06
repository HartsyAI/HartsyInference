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

/// <summary>Wiring tests for the HiDream img2img path built in Phase 4 — HiDream previously had no img2img at any
/// layer. Same three facts the other families' wiring tests pin: the missing-encoder failure is loud, a mismatched
/// source shape is rejected before any model work, and strength=0 passes the source through byte-identically.
/// No trained weights are needed because all three short-circuit ahead of the denoise loop.</summary>
public sealed class HiDreamImg2ImgTests
{
    private const int Width = 64;
    private const int Height = 64;

    private static HiDreamPipeline MakePipeline(CpuBackend backend, bool withEncoder)
    {
        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B);
        HiDreamTransformer transformer = new HiDreamTransformer(HiDreamConfig.Full);
        VaeDecoder decoder = new VaeDecoder(VaeConfig.Flux);
        VaeEncoder? encoder = withEncoder ? new VaeEncoder(VaeConfig.Flux) : null;
        return new HiDreamPipeline(backend, clipL, clipG, t5, llama, transformer, decoder, encoder, HiDreamConfig.Full);
    }

    private static (byte[] rgbData, int width, int height, int seed) Run(HiDreamPipeline pipeline, TextToImageRequest request) =>
        pipeline.GenerateFromTokens(
            promptTokenIdsL: [0], negativePromptTokenIdsL: [0],
            promptTokenIdsG: [0], negativePromptTokenIdsG: [0],
            promptEosPositionL: 0, negativeEosPositionL: 0,
            promptEosPositionG: 0, negativeEosPositionG: 0,
            promptTokenIdsT5: [0], negativePromptTokenIdsT5: [0],
            promptAttentionMaskT5: [1], negativeAttentionMaskT5: [1],
            promptTokenIdsLlama: [0], negativePromptTokenIdsLlama: [0],
            request: request);

    [Fact]
    public void Img2Img_WithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new CpuBackend();
        using HiDreamPipeline pipeline = MakePipeline(backend, withEncoder: false);
        using Tensor source = new Tensor(new TensorShape(1, 3, Height, Width), DType.F32);

        Assert.Throws<InvalidOperationException>(() => Run(pipeline, new ImageToImageRequest
        {
            Prompt = "test",
            Width = Width,
            Height = Height,
            SourceImage = source,
        }));
    }

    [Fact]
    public void Img2Img_WrongSourceShape_Throws()
    {
        using CpuBackend backend = new CpuBackend();
        using HiDreamPipeline pipeline = MakePipeline(backend, withEncoder: true);
        using Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);

        Assert.Throws<ArgumentException>(() => Run(pipeline, new ImageToImageRequest
        {
            Prompt = "test",
            Width = Width,
            Height = Height,
            SourceImage = source,
        }));
    }

    [Fact]
    public void Img2Img_Strength0_PassesSourceThrough()
    {
        using CpuBackend backend = new CpuBackend();
        using HiDreamPipeline pipeline = MakePipeline(backend, withEncoder: true);

        byte[] sourceBytes = new byte[Width * Height * 3];
        for (int i = 0; i < sourceBytes.Length; i++)
        {
            sourceBytes[i] = (byte)((i * 19) & 0xFF);
        }
        using Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, Width, Height);

        (byte[] outBytes, int outW, int outH, _) = Run(pipeline, new ImageToImageRequest
        {
            Prompt = "ignored",
            Width = Width,
            Height = Height,
            Steps = 4,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
        });

        Assert.Equal(Width, outW);
        Assert.Equal(Height, outH);
        Assert.Equal(sourceBytes, outBytes);
    }
}
