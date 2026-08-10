using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight coverage for FLUX.1 Fill via the codebase's EXISTING generic Img2Img/Inpaint path — unlike
/// Canny/Depth (which needed the new <c>Extra[FluxToolsControlImage]</c> plumbing, see
/// <see cref="FluxToolsControlImageRealWeightTests"/>), Fill was suspected to already work through
/// <see cref="RecipeImg2ImgBinder"/> once a checkpoint is loaded: <c>ImageRequest.Img2Img</c> + <c>Inpaint</c> already
/// produce a Mask-bearing <see cref="Diffusion.Requests.ImageToImageRequest"/>, and <see cref="FluxPipeline"/>'s own
/// <c>isFillModel</c> detection (x_embedder input dim ≥ 384) only requires that shape — this test confirms that
/// hypothesis on real weights rather than assuming it.</summary>
[Trait("Category", "Integration")]
public sealed class FluxFillRealWeightTests
{
    private readonly ITestOutputHelper _output;
    public FluxFillRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Flux1RecipePipeline_FillCheckpoint_AcceptsImg2ImgMaskAndGenerates()
    {
        string checkpointPath = "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Flux/flux1-fill-dev.safetensors";
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Flux Fill checkpoint not found at {checkpointPath}.");
            return;
        }
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
        _output.WriteLine($"Loaded {converted.Transformer.Count} transformer keys from the Fill checkpoint; "
            + $"{donor.ClipL.Count} CLIP-L + {donor.T5.Count} T5 + {donor.Vae.Count} VAE keys from the Dev donor.");

        using (loader)
        using (donorLoader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            FluxTransformer transformer = new FluxTransformer(FluxConfig.Dev);
            transformer.LoadWeights(converted.Transformer);
            _output.WriteLine($"XEmbedInputDim={transformer.XEmbedInputDim} (384 expected for Fill).");
            Assert.True(transformer.XEmbedInputDim >= 384, $"Expected Fill's wide x_embedder, got {transformer.XEmbedInputDim} — is this really the Fill checkpoint?");

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

            ImageData sourceImage = new ImageData { Rgb = SolidWithGradient(width, height), Width = width, Height = height };
            ImageData mask = new ImageData { Rgb = CenterSquareMask(width, height), Width = width, Height = height };
            ImageRequest request = new ImageRequest
            {
                Prompt = "a red apple",
                Width = width,
                Height = height,
                Steps = 8,
                Seed = 54321,
                Img2Img = new Img2Img { InitImage = sourceImage, Creativity = 1.0, Mode = Img2ImgMode.Auto },
                Inpaint = new Inpaint { Mask = mask },
            };

            ImageResult result = recipePipeline.Generate(request, progress: null, cancel: default);

            Assert.Equal(width, result.Width);
            Assert.Equal(height, result.Height);
            Assert.Equal(width * height * 3, result.Rgb.Length);

            string outDir = Environment.GetEnvironmentVariable("HARTSYINFERENCE_FLUX_CANNY_OUTPUT_DIR") ?? RepoRoot.Path;
            string outPath = Path.Combine(outDir, $"flux_fill_test_output_{width}x{height}.rgb");
            File.WriteAllBytes(outPath, result.Rgb);
            _output.WriteLine($"Wrote raw RGB24 output to {outPath} ({result.Rgb.Length} bytes, {width}x{height}).");

            recipePipeline.Dispose();
        }
    }

    /// <summary>A plain blue-to-gray vertical gradient — a stand-in "photo" outside the masked region, so a
    /// successful fill is visually obvious against it (unmasked pixels should survive close to verbatim).</summary>
    private static byte[] SolidWithGradient(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            byte shade = (byte)(60 + (y * 120 / height));
            for (int x = 0; x < width; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = (byte)(shade / 2);
                rgb[i + 1] = (byte)(shade / 2);
                rgb[i + 2] = shade;
            }
        }
        return rgb;
    }

    /// <summary>White (regenerate) square in the center, black (keep) elsewhere.</summary>
    private static byte[] CenterSquareMask(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        int margin = width / 4;
        for (int y = margin; y < height - margin; y++)
        {
            for (int x = margin; x < width - margin; x++)
            {
                int i = (y * width + x) * 3;
                rgb[i] = 255;
                rgb[i + 1] = 255;
                rgb[i + 2] = 255;
            }
        }
        return rgb;
    }
}
