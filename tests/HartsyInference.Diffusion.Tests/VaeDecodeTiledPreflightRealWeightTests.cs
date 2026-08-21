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

/// <summary>Real-weight coverage for Tier 1.3: <see cref="VaeDecoder.DecodeTiled"/>'s tile loop now runs a
/// per-tile pre-flight workspace estimate + pool trim before each tile's <c>Decode</c> call. Plain text-to-image
/// at a resolution above SDXL's single-tile threshold (forces the tile loop, not the full-res fast path) — the
/// most heavily-proven code path in this whole session (every earlier real-weight test in this backlog already
/// exercises the decode side successfully), so this specifically isolates whether the NEW per-tile check itself
/// regresses anything, deliberately avoiding the still-unresolved encoder-side cuDNN crash (see
/// the removed SDXL tiled-encoder test used to cover — not applicable here since this is pure T2I, no img2img encode).</summary>
[Trait("Category", "Integration")]
public sealed class VaeDecodeTiledPreflightRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public VaeDecodeTiledPreflightRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SdxlPipeline_LargeText2Image_TiledDecodeWithPreflightCheck_Generates()
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

        // Above SDXL's 1024px single-tile threshold - forces DecodeTiled's actual tile loop, not the
        // full-res-direct-decode fast path this method also contains.
        const int width = 1536;
        const int height = 1536;
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = SdxlCheckpointConverter.LoadAndConvert(TestPaths.Sdxl.SingleFile);
        _output.WriteLine($"Loaded {converted.UNet.Count} UNet keys. VAE weight dtype: {converted.Vae.Values.First().DType.Name}.");

        using (loader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            UNet unet = new UNet(UNetConfig.SdxlBase);
            unet.LoadWeights(converted.UNet);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(converted.ClipG, "text_model");

            // Force F32: the checkpoint's native F16 VAE weights hit a pre-existing black-output bug at this
            // resolution (confirmed independent of this test's own changes — reproduces identically with the
            // new pre-flight check disabled; same bug class as doc 14's "F16 VAE black output" entry, now shown
            // to affect SDXL too, not just Flux Schnell). F32 sidesteps it so this test can actually verify the
            // tile loop + pre-flight check succeed, rather than conflating two unrelated bugs.
            Dictionary<string, Tensor> vaeF32 = new(converted.Vae.Count);
            foreach (KeyValuePair<string, Tensor> kv in converted.Vae)
            {
                vaeF32[kv.Key] = kv.Value.DType == DType.F32 ? kv.Value : kv.Value.CastTo(DType.F32);
            }
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeF32);

            using ClipTokenizer tokenizer = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            SdxlPipeline pipeline = new SdxlPipeline(backend, clipL, clipG, unet, vaeDecoder);

            int[] tokens = tokenizer.Encode("a lighthouse on a rocky cliff, dramatic sky");
            int[] neg = tokenizer.Encode("blurry, low quality");
            int posEosG = ClipTokenizer.FindEosPosition(tokens);
            int negEosG = ClipTokenizer.FindEosPosition(neg);

            TextToImageRequest req = new TextToImageRequest
            {
                Prompt = "a lighthouse on a rocky cliff, dramatic sky",
                NegativePrompt = "blurry, low quality",
                Width = width,
                Height = height,
                Steps = 8,
                CfgScale = 6.0f,
                Seed = 999,
            };

            (byte[] rgbData, int outW, int outH, int seed) = pipeline.GenerateFromTokens(tokens, neg, tokens, neg, posEosG, negEosG, req);

            string outPath = Path.Combine(RepoRoot.Path, $"sdxl_tiled_decode_preflight_output_{width}x{height}.rgb");
            File.WriteAllBytes(outPath, rgbData);
            _output.WriteLine($"Wrote raw RGB24 output to {outPath} ({rgbData.Length} bytes, {width}x{height}). First byte: {rgbData[0]}.");

            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            Assert.Equal(width * height * 3, rgbData.Length);
            Assert.False(rgbData.All(b => b == rgbData[0]), "Output is a flat/constant image.");

            pipeline.Dispose();
        }
    }
}
