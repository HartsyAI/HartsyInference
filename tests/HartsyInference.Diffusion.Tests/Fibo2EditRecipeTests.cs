using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 edit routing and refusals. Each refusal is reached before any encoder or denoiser runs, so the components are
/// unloaded.</summary>
public sealed class Fibo2EditRecipeTests
{
    private static readonly ImageData Source = new() { Rgb = new byte[16 * 16 * 3], Width = 16, Height = 16 };

    [Fact]
    public void EditTurbo_RefusesTextToImageWithAClearError()
    {
        using Harness harness = new(Fibo2Variants.EditTurbo);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            harness.Recipe.Generate(new ImageRequest { Prompt = "a cat" }, Settings(), null, CancellationToken.None));

        Assert.Contains("only edits images", error.Message, StringComparison.Ordinal);
        Assert.Contains("no text-to-image", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Dev_TextToImageStillRefusesNegativePrompts()
    {
        using Harness harness = new(Fibo2Variants.Dev);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            harness.Recipe.Generate(new ImageRequest { Prompt = "a cat", NegativePrompt = "blurry" }, Settings(), null, CancellationToken.None));

        Assert.Contains("negative prompts", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Edit_RefusesMoreThanFourImages()
    {
        using Harness harness = new(Fibo2Variants.Dev);
        ImageRequest request = new() { Prompt = "edit", ReferenceImages = [Source, Source, Source, Source, Source] };

        ArgumentException error = Assert.Throws<ArgumentException>(() =>
            harness.Recipe.Generate(request, Settings(), null, CancellationToken.None));

        Assert.Contains("at most 4", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MaskedEdit_NeedsExactlyOneImage()
    {
        using Harness harness = new(Fibo2Variants.Dev);
        ImageRequest request = new()
        {
            Prompt = "edit",
            ReferenceImages = [Source, Source],
            Inpaint = new Inpaint { Mask = Source },
        };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            harness.Recipe.Generate(request, Settings(), null, CancellationToken.None));

        Assert.Contains("exactly one image", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Edit_RefusesReferencesMixedWithADenoiseModeInitImage()
    {
        using Harness harness = new(Fibo2Variants.Dev);
        ImageRequest request = new()
        {
            Prompt = "edit", ReferenceImages = [Source], Img2Img = new Img2Img { InitImage = Source, Mode = Img2ImgMode.Denoise },
        };

        Assert.Throws<NotSupportedException>(() => harness.Recipe.Generate(request, Settings(), null, CancellationToken.None));
    }

    [Fact]
    public void Edit_WithoutEditingComponents_FailsClearly()
    {
        using Harness harness = new(Fibo2Variants.EditTurbo);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            harness.Recipe.Generate(new ImageRequest { Prompt = "edit", ReferenceImages = [Source] }, Settings(), null, CancellationToken.None));

        Assert.Contains("editing components", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EditTurboDefaults_RunWithoutGuidance()
    {
        Assert.False(CfgHelper.IsGuidanceActive(Fibo2Recipe.EditTurboDefaults.CfgScale));
        Assert.True(CfgHelper.IsGuidanceActive(Fibo2Recipe.FamilyDefaults.CfgScale));
    }

    [Fact]
    public void GenerateEdit_RejectsMismatchedVisionAndPixelCopies()
    {
        CpuBackend backend = new();
        using Fibo2Transformer transformer = new(TinyConfig());
        using LlamaStyleEncoder encoder = new(TinyEncoderConfig());
        VaeDecoder decoder = new(VaeConfig.Flux2);
        using Tensor mean = new(new TensorShape(128), DType.F32);
        using Tensor variance = new(new TensorShape(128), DType.F32);
        using Fibo2Pipeline pipeline = new(backend, encoder, transformer, decoder, mean, variance, TinyConfig());
        Qwen3VlVisionConfig vision = Qwen3VlVisionConfig.Qwen3Vl8B;
        using Qwen3VlVisionEncoder tower = new(vision);
        Qwen3VlMultimodalEncoder multimodal = new(encoder, tower, new Qwen3VlImageProcessor(vision), vision);
        using Tensor pixels = new(new TensorShape(1, 3, 16, 16), DType.F32);
        Fibo2EditInputs inputs = new()
        {
            ConditionalTokens = [1],
            VisionImages = [],
            ReferencePixels = [pixels],
            VisionEncoder = multimodal,
            VaeEncoder = new VaeEncoder(VaeConfig.Flux2),
        };

        Assert.Throws<ArgumentException>(() => pipeline.GenerateEdit(inputs, [1], 64, 64, 4, 1f, 1.15f, 7));
    }

    private static ImageEffectiveSettings Settings() => new() { Width = 1024, Height = 704, Steps = 4, CfgScale = 1f };

    private static Fibo2Config TinyConfig() => Fibo2Config.Checkpoint with
    {
        Dim = 64, NumHeads = 2, HeadDim = 32, NumLayers = 2, NumRefinerLayers = 1, InjectionLayers = [1], CapFeatDim = 96,
        PerceiverQueries = 8, PerceiverLayers = 1, FfnDim = 96, AxesDims = [8, 12, 12], MaxGist = 8, MinGist = 4, GistStep = 2,
        GistMinTextLen = 16, GistMaxTextLen = 64,
    };

    private static LlamaStyleEncoderConfig TinyEncoderConfig() => LlamaStyleEncoderConfig.Qwen3_VL_4B with
    {
        HiddenSize = 32, NumLayers = 36, NumQueryHeads = 2, NumKvHeads = 1, HeadDim = 16, IntermediateSize = 64, VocabSize = 64,
    };

    /// <summary>A recipe pipeline over unloaded components, for the argument rules that run before any of them is used.</summary>
    private sealed class Harness : IDisposable
    {
        private readonly Qwen3Tokenizer _tokenizer = new();
        private readonly Fibo2Transformer _transformer = new(TinyConfig());
        private readonly LlamaStyleEncoder _encoder = new(TinyEncoderConfig());
        private readonly Tensor _mean = new(new TensorShape(128), DType.F32);
        private readonly Tensor _variance = new(new TensorShape(128), DType.F32);
        private readonly Fibo2Pipeline _pipeline;

        public Harness(ModelVariant variant)
        {
            _pipeline = new Fibo2Pipeline(new CpuBackend(), _encoder, _transformer, new VaeDecoder(VaeConfig.Flux2), _mean, _variance, TinyConfig());
            Recipe = new Fibo2RecipePipeline(_pipeline, _tokenizer, Fibo2Recipe.FamilyDefaults,
                new ResolvedModelVariant("fibo2", variant, ModelVariantSource.CallerHint, "test"), []);
        }

        public Fibo2RecipePipeline Recipe { get; }

        public void Dispose()
        {
            Recipe.Dispose();
            _pipeline.Dispose();
            _mean.Dispose();
            _variance.Dispose();
            _encoder.Dispose();
            _transformer.Dispose();
            _tokenizer.Dispose();
        }
    }

    [Theory]
    [InlineData(Img2ImgMode.Auto, false)]
    [InlineData(Img2ImgMode.Denoise, false)]
    [InlineData(Img2ImgMode.Reference, true)]
    public void InitImage_RoutesToEditOnlyInReferenceMode_ForTheDevBuild(Img2ImgMode mode, bool expected)
    {
        ImageRequest request = new()
        {
            Prompt = "a cat",
            Img2Img = new Img2Img { InitImage = new ImageData { Rgb = new byte[12], Width = 2, Height = 2 }, Mode = mode },
        };

        Assert.Equal(expected, Fibo2RecipePipeline.RoutesToEdit(request, editOnly: false));
        Assert.True(Fibo2RecipePipeline.RoutesToEdit(request, editOnly: true));
        Assert.False(Fibo2RecipePipeline.RoutesToEdit(new ImageRequest { Prompt = "a cat" }, editOnly: false));
    }
}
