using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Capability facts for the MiniMax-H3-Image recipe that need no weights: which builds resolve to it, what it
/// declares, the settings it refuses by name, the size and cancellation checks in <c>Generate</c>, and the component
/// lookup in <c>Construct</c>. Each fact names the behaviour the user sees, so a failure reads as a capability lost.</summary>
public sealed class MiniMaxH3ImageCapabilityTests
{
    private static readonly double[] SixStepGrid =
    [
        1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0,
    ];

    // ---- What the recipe declares --------------------------------------------------------------------------------

    [Theory]
    [InlineData("minimax-h3-image")]
    [InlineData("minimax-h3-image-bf16")]
    [InlineData("minimax-h3-image-w4a8")]
    public void EveryImageBuildResolvesToTheImageRecipe(string familyId)
    {
        Assert.IsType<MiniMaxH3ImageRecipe>(RecipeRegistry.Resolve(familyId));
    }

    [Fact]
    public void TheVideoFamilyDoesNotResolveToTheImageRecipe()
    {
        Assert.IsNotType<MiniMaxH3ImageRecipe>(RecipeRegistry.Resolve("minimax-h3"));
    }

    [Fact]
    public void TheRecipeDeclaresNoOptionalFeaturesYet()
    {
        Assert.Equal(ImageFeatures.None, new MiniMaxH3ImageRecipe().Supports);
    }

    [Fact]
    public void TheRecipeDeclaresNoPromptWeightingYet()
    {
        Assert.Equal(PromptWeightingMode.None, new MiniMaxH3ImageRecipe().PromptWeighting);
    }

    [Fact]
    public void SamplingIsOwnedBySolverSoNoSamplerOrScheduleIsOffered()
    {
        SamplingCapabilities.SamplingSupport support = SamplingCapabilities.ForImage("minimax-h3-image");

        Assert.Empty(support.Samplers);
        Assert.Empty(support.Schedules);
    }

    // ---- Settings the recipe refuses by name --------------------------------------------------------------------

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(20)]
    public void StepCountsOtherThanSixAreRefused(int steps)
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3ImageRecipePipeline.RefuseUnsupported(Request(), Settings(1024, 1024, steps, 1f)));

        Assert.Contains("exactly 6 steps", ex.Message);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1.5f)]
    [InlineData(3.5f)]
    [InlineData(7f)]
    public void GuidanceOtherThanOneIsRefused(float cfg)
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3ImageRecipePipeline.RefuseUnsupported(Request(), Settings(1024, 1024, 6, cfg)));

        Assert.Contains("CFG must be 1", ex.Message);
    }

    [Fact]
    public void ANegativePromptIsRefused()
    {
        ImageRequest request = Request() with { NegativePrompt = "blurry" };

        Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3ImageRecipePipeline.RefuseUnsupported(request, Settings(1024, 1024, 6, 1f)));
    }

    [Fact]
    public void AnExplicitSchedulerIsRefused()
    {
        ImageRequest request = Request() with { Scheduler = "karras" };

        Assert.Throws<NotSupportedException>(() =>
            MiniMaxH3ImageRecipePipeline.RefuseUnsupported(request, Settings(1024, 1024, 6, 1f)));
    }

    [Fact]
    public void SixStepsAtGuidanceOneWithNoNegativeAndNoSchedulerIsAccepted()
    {
        MiniMaxH3ImageRecipePipeline.RefuseUnsupported(Request(), Settings(2048, 2048, 6, 1f));
    }

    [Theory]
    [InlineData(1000, 1024)]
    [InlineData(1024, 1000)]
    [InlineData(1050, 1050)]
    public void GenerateRefusesSidesThatAreNotMultiplesOfThirtyTwo(int width, int height)
    {
        using MiniMaxH3ImageRecipePipeline pipeline = Pipeline();

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() =>
            pipeline.Generate(Request(), Settings(width, height, 6, 1f), null, CancellationToken.None));

        Assert.Contains("multiples of 32", ex.Message);
    }

    [Fact]
    public void GenerateStopsBeforeAnyWorkWhenAlreadyCancelled()
    {
        using MiniMaxH3ImageRecipePipeline pipeline = Pipeline();

        Assert.Throws<OperationCanceledException>(() =>
            pipeline.Generate(Request(), Settings(1024, 1024, 6, 1f), null, new CancellationToken(canceled: true)));
    }

    [Fact]
    public void PipelineRefusesAConfigWithoutAnOutputHeadGrid()
    {
        MiniMaxH3Config config = new();

        Assert.Throws<ArgumentException>(() => new MiniMaxH3ImageRecipePipeline(
            new MiniMaxH3Transformer(config), config, Decoder(), new MiniMaxH3TextEncoder(), new Qwen2Tokenizer(),
            new CpuBackend(), new CpuBackend(), new CpuBackend(), Array.Empty<IDisposable>()));
    }

    [Fact]
    public void PipelineDisposeIsIdempotent()
    {
        MiniMaxH3ImageRecipePipeline pipeline = Pipeline();

        pipeline.Dispose();
        pipeline.Dispose();
    }

    // ---- Component lookup in Construct -------------------------------------------------------------------------

    [Fact]
    public void ConstructNamesTheImageVaeWhenItIsMissing()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "dit.safetensors"));

            FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() =>
                new MiniMaxH3ImageRecipe().Construct(Context(dit)));

            Assert.Contains("minimax_h3_image_vae_fp16.safetensors", ex.Message);
        });
    }

    [Fact]
    public void ConstructNamesTheTextEncoderWhenItIsMissing()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "dit.safetensors"));
            WriteFile(Path.Combine(root, "vae", "minimax_h3_image_vae_fp16.safetensors"));

            FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() =>
                new MiniMaxH3ImageRecipe().Construct(Context(dit)));

            Assert.Contains("text encoder", ex.Message);
        });
    }

    [Fact]
    public void ConstructFindsTheImageVaeAndTextEncoderInTheLayoutThisMachineUses()
    {
        // Comfy-Org DiT under diffusion_models/MiniMaxH3Image, the image VAE under VAE/MiniMaxH3 and the Qwen encoder
        // under text_encoders: the layout of the models root, not the catalog's flat/ folder.
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "MiniMaxH3Image",
                "minimax_h3_image_pruned_int8_convrot.safetensors"));
            WriteFile(Path.Combine(root, "VAE", "MiniMaxH3", "minimax_h3_image_vae_fp16.safetensors"));
            WriteFile(Path.Combine(root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"));

            // The DiT is an empty placeholder, so the load after the lookups is expected to fail. The lookups must not.
            Exception? ex = Record.Exception(() => new MiniMaxH3ImageRecipe().Construct(Context(dit)));

            Assert.False(ex is FileNotFoundException fnf
                && (fnf.Message.Contains("VAE", StringComparison.OrdinalIgnoreCase)
                    || fnf.Message.Contains("text encoder", StringComparison.OrdinalIgnoreCase)),
                $"The components were not found: {ex?.Message}");
        });
    }

    // ---- Helpers ------------------------------------------------------------------------------------------------

    private static ImageRequest Request() => new() { Prompt = "a red fox in snow" };

    private static ImageEffectiveSettings Settings(int width, int height, int steps, float cfg) => new()
    {
        Width = width,
        Height = height,
        Steps = steps,
        CfgScale = cfg,
    };

    private static MiniMaxH3VideoVaeDecoder Decoder() => new(MiniMaxH3VideoVaeConfig.ImageBaseline);

    private static MiniMaxH3ImageRecipePipeline Pipeline()
    {
        MiniMaxH3Config config = new() { OutputHeadSigmas = SixStepGrid, OutputHeadCount = 6, HasAudio = false };
        CpuBackend backend = new();
        return new MiniMaxH3ImageRecipePipeline(new MiniMaxH3Transformer(config) { BodyDType = DType.F32 }, config,
            Decoder(), new MiniMaxH3TextEncoder(), new Qwen2Tokenizer(), backend, backend, backend,
            Array.Empty<IDisposable>());
    }

    private static RecipeContext Context(string checkpointPath) => new()
    {
        CheckpointPath = checkpointPath,
        Backend = new CpuBackend(),
    };

    private static void WithTempRoot(Action<string> test)
    {
        string root = Path.Combine(Path.GetTempPath(), "h3-image-caps-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            test(root);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string WriteFile(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, Array.Empty<byte>());
        return path;
    }
}
