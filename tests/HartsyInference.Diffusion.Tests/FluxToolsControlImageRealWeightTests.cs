using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight regression test for the 0.1 fix: <c>Flux1RecipePipeline.Generate</c> used to never pass
/// <c>controlImage</c> to <see cref="FluxPipeline.GenerateFromTokens"/> at all, so every FLUX.1 Tools (Canny/Depth)
/// checkpoint refused unconditionally regardless of what the host supplied. This drives the ACTUAL recipe-layer
/// method through a real Canny checkpoint with a control image carried in <see cref="ImageRequest.Extra"/> under
/// <see cref="RequestExtras.FluxToolsControlImage"/> — the exact path the SwarmUI extension's <c>BuildImageExtra</c>
/// populates — and asserts it completes rather than throwing "requires a control image". The control image itself is
/// a synthetic geometric pattern (this project can't reference the extension's ImageSharp-based CannyPreprocessor),
/// so this proves the engine-side plumbing, not the extension's own annotation quality.
/// <para>Writes raw RGB output to <c>HARTSYINFERENCE_FLUX_CANNY_OUTPUT_DIR</c> (or the repo root) for visual
/// inspection — this is a "does it run and respond to the control input" check, not an SSIM gate (no Python
/// reference exists for FLUX.1 Tools).</para></summary>
[Trait("Category", "Integration")]
public sealed class FluxToolsControlImageRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public FluxToolsControlImageRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Flux1RecipePipeline_CannyCheckpoint_AcceptsControlImageAndGenerates()
    {
        string checkpointPath = TestPaths.Flux.Canny;
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Flux Canny checkpoint not found at {checkpointPath}.");
            return;
        }
        // This community repack is DiT-only (confirmed: every key is under model.diffusion_model.*, no CLIP-L/T5/VAE
        // bundled) — BFL ships identical text encoders/VAE across every Flux.1 Dev-family variant, so borrow them
        // from the vanilla Dev checkpoint rather than requiring a full bundle just for this plumbing test.
        string encoderDonorPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors";
        if (!File.Exists(encoderDonorPath))
        {
            _output.WriteLine($"SKIPPED: CLIP-L/T5/VAE donor checkpoint not found at {encoderDonorPath}.");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges) || !File.Exists(TestPaths.Tokenizers.T5Spiece))
        {
            _output.WriteLine("SKIPPED: tokenizer files not found.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        // The MSBuild content copy lands PTX in this test assembly's own output dir, not a separate native build
        // tree — FluxSsimTests' "native/cuda/build" convention is stale and never actually resolves here.
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        const int width = 512;
        const int height = 512;
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) = FluxCheckpointConverter.LoadAndConvert(checkpointPath);
        (FluxCheckpointConverter.ConvertedWeights donor, SafeTensorsLoader donorLoader) = FluxCheckpointConverter.LoadAndConvert(encoderDonorPath);
        _output.WriteLine($"Loaded {converted.Transformer.Count} transformer keys from the Canny checkpoint; "
            + $"{donor.ClipL.Count} CLIP-L + {donor.T5.Count} T5 + {donor.Vae.Count} VAE keys from the Dev donor.");

        using (loader)
        using (donorLoader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            FluxTransformer transformer = new FluxTransformer(FluxConfig.Dev);
            transformer.LoadWeights(converted.Transformer);
            _output.WriteLine($"XEmbedInputDim={transformer.XEmbedInputDim} (128 expected for a Canny/Depth Tools checkpoint).");
            Assert.True(transformer.XEmbedInputDim > 64, $"Expected a Tools-wide x_embedder, got {transformer.XEmbedInputDim} — is this really the Canny checkpoint?");

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(donor.ClipL, "text_model");
            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(donor.T5);
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
            vaeDecoder.LoadWeights(donor.Vae);
            VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Flux);
            vaeEncoder.LoadWeights(donor.Vae);

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, vaeEncoder, FluxConfig.Dev);
            ClipTokenizer clipTok = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            T5Tokenizer t5Tok = new T5Tokenizer(TestPaths.Tokenizers.T5Spiece, maxLength: 256);
            Flux1RecipePipeline recipePipeline = new Flux1RecipePipeline(pipeline, clipTok, t5Tok, isDev: true, loaders: [], loraStack: null, backend: backend);

            ImageData controlImage = new ImageData { Rgb = SyntheticEdgeMap(width, height), Width = width, Height = height };
            ImageRequest request = new ImageRequest
            {
                Prompt = "a photo of a mountain landscape",
                Width = width,
                Height = height,
                Steps = 8,
                Seed = 12345,
                Extra = new Dictionary<string, object>(StringComparer.Ordinal)
                {
                    [RequestExtras.FluxToolsControlImage] = controlImage,
                },
            };

            ImageResult result = recipePipeline.Generate(request, progress: null, cancel: default);

            Assert.Equal(width, result.Width);
            Assert.Equal(height, result.Height);
            Assert.Equal(width * height * 3, result.Rgb.Length);

            string outDir = Environment.GetEnvironmentVariable("HARTSYINFERENCE_FLUX_CANNY_OUTPUT_DIR") ?? RepoRoot.Path;
            string outPath = Path.Combine(outDir, $"flux_canny_test_output_{width}x{height}.rgb");
            File.WriteAllBytes(outPath, result.Rgb);
            _output.WriteLine($"Wrote raw RGB24 output to {outPath} ({result.Rgb.Length} bytes, {width}x{height}).");

            recipePipeline.Dispose();
        }
    }

    /// <summary>A white rectangular ring on black — a stand-in "edge map" shaped enough that the DiT should visibly
    /// follow its geometry, without depending on the extension's own (unreferenceable here) Canny implementation.</summary>
    private static byte[] SyntheticEdgeMap(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        int margin = width / 4;
        int thickness = 6;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                bool onRing =
                    (x >= margin && x < margin + thickness && y >= margin && y < height - margin) ||
                    (x >= width - margin - thickness && x < width - margin && y >= margin && y < height - margin) ||
                    (y >= margin && y < margin + thickness && x >= margin && x < width - margin) ||
                    (y >= height - margin - thickness && y < height - margin && x >= margin && x < width - margin);
                byte v = (byte)(onRing ? 255 : 0);
                int i = (y * width + x) * 3;
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        return rgb;
    }
}
