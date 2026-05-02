using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Wiring tests for Flux.2 img2img on the unified <see cref="Flux2Pipeline.GenerateFromTokens"/> API. Img2img source goes through 2×2 patchify + BN-normalize before AddNoise; the inverse helpers (<c>PatchifyLatent</c>, <c>ApplyBnNormalize</c>) are exercised end-to-end only with a real Flux.2 checkpoint (Klein 4B / Dev).</summary>
public sealed class Flux2Img2ImgTests
{
    private static (Tensor mean, Tensor var_) MakeBnTensors(int channels)
    {
        Tensor mean = new Tensor(new TensorShape(channels), DType.F32);
        Tensor var_ = new Tensor(new TensorShape(channels), DType.F32);
        // Initialize var to 1.0 so BN normalize doesn't divide by zero in any code path that touches it.
        unsafe
        {
            float* vp = (float*)var_.DataPointer;
            for (int i = 0; i < channels; i++) vp[i] = 1.0f;
        }
        return (mean, var_);
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWithoutVaeEncoder_Throws()
    {
        using CpuBackend backend = new();
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
        Flux2Transformer transformer = new(Flux2Config.Klein4B);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux2);
        (Tensor bnMean, Tensor bnVar) = MakeBnTensors(128);

        using Flux2Pipeline pipeline = new(backend, textEncoder, transformer, vaeDecoder,
            bnMean, bnVar, Flux2Config.Klein4B);

        Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], request));

        source.Dispose();
        bnMean.Dispose();
        bnVar.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgWrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
        Flux2Transformer transformer = new(Flux2Config.Klein4B);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux2);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux2);
        (Tensor bnMean, Tensor bnVar) = MakeBnTensors(128);

        using Flux2Pipeline pipeline = new(backend, textEncoder, transformer, vaeDecoder, vaeEncoder,
            bnMean, bnVar, Flux2Config.Klein4B);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        ImageToImageRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens(promptTokenIds: [0], request));

        source.Dispose();
        bnMean.Dispose();
        bnVar.Dispose();
    }

    [Fact]
    public void GenerateFromTokens_Img2ImgStrength0_PassesSourceThrough()
    {
        using CpuBackend backend = new();
        LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_4B);
        Flux2Transformer transformer = new(Flux2Config.Klein4B);
        VaeDecoder vaeDecoder = new(VaeConfig.Flux2);
        VaeEncoder vaeEncoder = new(VaeConfig.Flux2);
        (Tensor bnMean, Tensor bnVar) = MakeBnTensors(128);

        using Flux2Pipeline pipeline = new(backend, textEncoder, transformer, vaeDecoder, vaeEncoder,
            bnMean, bnVar, Flux2Config.Klein4B);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)((i * 13) & 0xFF);
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
            promptTokenIds: [0], request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);

        source.Dispose();
        bnMean.Dispose();
        bnVar.Dispose();
    }
}
