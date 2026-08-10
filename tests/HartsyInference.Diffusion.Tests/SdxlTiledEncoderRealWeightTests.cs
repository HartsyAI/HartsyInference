using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight coverage for Tier 1.1/1.2: <see cref="SdxlPipeline"/> now routes its img2img source
/// through <see cref="VaeTiledEncoder"/> instead of the plain <see cref="VaeEncoder"/> directly, and
/// <see cref="VaeTiledEncoder"/> casts each tile to the VAE's own weight dtype before encoding (matching
/// <c>DecodeTiled</c>'s existing per-tile cast). This forces a genuinely tiled encode (image larger than SDXL's
/// 1024px single-tile threshold) and confirms it completes and produces a sane image, rather than the dtype
/// mismatch this test exists to catch (garbage output or a hard failure from feeding a BF16-weighted conv an
/// F32 tile without a matching cuDNN path).</summary>
[Trait("Category", "Integration")]
public sealed class SdxlTiledEncoderRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public SdxlTiledEncoderRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlPipeline_LargeImg2Img_UsesTiledEncoderAndGenerates()
    {
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        // SDXL's VaeConfig.SampleSize=1024 with 4 down-block stages -> single-tile threshold is 1024px.
        // 1536x1536 forces a real 2x2-ish tiled encode, not the single-shot short-circuit.
        const int width = 1536;
        const int height = 1536;
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = SdxlCheckpointConverter.LoadAndConvert(TestPaths.Sdxl.SingleFile);
        _output.WriteLine($"Loaded {converted.UNet.Count} UNet keys.");

        using (loader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            UNet unet = new UNet(UNetConfig.SdxlBase);
            unet.LoadWeights(converted.UNet);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(converted.ClipG, "text_model");

            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(converted.Vae);
            VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sdxl);
            vaeEncoder.LoadWeights(converted.Vae);
            _output.WriteLine($"VAE weight dtype: {converted.Vae.Values.First().DType.Name} (SDXL's own precision policy — BF16 on Ampere+).");

            using ClipTokenizer tokenizer = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            SdxlPipeline pipeline = new SdxlPipeline(backend, clipL, clipG, unet, vaeDecoder, vaeEncoder);

            int[] tokens = tokenizer.Encode("a snowy mountain landscape at sunrise");
            int[] neg = tokenizer.Encode("blurry, low quality");
            int posEosG = ClipTokenizer.FindEosPosition(tokens);
            int negEosG = ClipTokenizer.FindEosPosition(neg);

            ImageToImageRequest req = new ImageToImageRequest(
                new TextToImageRequest { Prompt = "a snowy mountain landscape at sunrise", NegativePrompt = "blurry, low quality", Width = width, Height = height, Steps = 8, CfgScale = 6.0f, Seed = 777 },
                RgbToTensor(HorizontalGradient(width, height), width, height))
            {
                Strength = 0.6f,
            };

            (byte[] rgbData, int outW, int outH, int seed) = pipeline.GenerateFromTokens(tokens, neg, tokens, neg, posEosG, negEosG, req);

            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            Assert.Equal(width * height * 3, rgbData.Length);
            Assert.False(rgbData.All(b => b == rgbData[0]), "Output is a flat/constant image — the encode or decode likely produced garbage.");

            string outPath = Path.Combine(RepoRoot.Path, $"sdxl_tiled_img2img_output_{width}x{height}.rgb");
            File.WriteAllBytes(outPath, rgbData);
            _output.WriteLine($"Wrote raw RGB24 output to {outPath} ({rgbData.Length} bytes, {width}x{height}).");

            pipeline.Dispose();
        }
    }

    private static byte[] HorizontalGradient(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte v = (byte)(x * 255 / width);
                int i = (y * width + x) * 3;
                rgb[i] = v;
                rgb[i + 1] = (byte)(255 - v);
                rgb[i + 2] = 128;
            }
        }
        return rgb;
    }

    private static unsafe Tensor RgbToTensor(byte[] rgb, int width, int height)
    {
        Tensor t = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* dp = (float*)t.DataPointer;
        int spatial = width * height;
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < spatial; i++)
            {
                dp[c * spatial + i] = (rgb[i * 3 + c] / 127.5f) - 1f;
            }
        }
        return t;
    }
}
