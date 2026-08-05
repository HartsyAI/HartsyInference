using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Contract tests for <see cref="RecipeImg2ImgBinder"/> — the single mapping every architecture recipe routes
/// its init image through — plus the <see cref="ImageFeatures.Img2Img"/> gate that decides whether a family accepts an
/// init image at all. Weight-free and CPU-only: the binder and the gate are both resolved before any checkpoint is
/// touched, which is exactly why this layer had no coverage before.</summary>
public sealed class RecipeImg2ImgBinderTests
{
    private const string BareFamilyId = "test-binder-bare";
    private const string Img2ImgFamilyId = "test-binder-img2img";

    private static ImageData SolidImage(int width, int height, byte value) =>
        new ImageData { Rgb = Enumerable.Repeat(value, width * height * 3).ToArray(), Width = width, Height = height };

    private static ImageRequest RequestWithInit(int width, int height, double creativity = 0.6) =>
        new ImageRequest
        {
            Prompt = "test",
            Width = width,
            Height = height,
            Img2Img = new Img2Img { InitImage = SolidImage(8, 8, 128), Creativity = creativity },
        };

    [Fact]
    public void Apply_WithNullSpec_ReturnsInnerUnchanged()
    {
        TextToImageRequest inner = new TextToImageRequest { Prompt = "a fox" };

        TextToImageRequest result = RecipeImg2ImgBinder.Apply(inner, null);

        Assert.Same(inner, result);
        Assert.IsNotType<ImageToImageRequest>(result);
    }

    /// <summary>The regression this binder exists to prevent: the hand-written per-family mapping it replaced rebuilt
    /// the request field by field, so any field the family had already resolved was silently dropped on the img2img
    /// path (and any newly added base field would be dropped in every family at once).</summary>
    [Fact]
    public void Apply_CarriesEveryBaseFieldOntoTheImg2ImgRequest()
    {
        using Tensor source = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = "a fox in snow",
            NegativePrompt = "blurry",
            Width = 64,
            Height = 64,
            Steps = 17,
            CfgScale = 4.5f,
            Seed = 1234,
            Scheduler = "dpmpp_2m",
            ClipSkip = 2,
        };
        Img2ImgResolver.Img2ImgSpec spec = new Img2ImgResolver.Img2ImgSpec { SourceTensor = source, Strength = 0.42f };

        ImageToImageRequest result = Assert.IsType<ImageToImageRequest>(RecipeImg2ImgBinder.Apply(inner, spec));

        Assert.Equal("a fox in snow", result.Prompt);
        Assert.Equal("blurry", result.NegativePrompt);
        Assert.Equal(64, result.Width);
        Assert.Equal(64, result.Height);
        Assert.Equal(17, result.Steps);
        Assert.Equal(4.5f, result.CfgScale);
        Assert.Equal(1234, result.Seed);
        Assert.Equal("dpmpp_2m", result.Scheduler);
        Assert.Equal(2, result.ClipSkip);
        Assert.Same(source, result.SourceImage);
        Assert.Equal(0.42f, result.Strength);
        Assert.Null(result.Mask);
        // Derived-record defaults must survive the promotion too, not just the copied base fields.
        Assert.True(result.RecompositeAtEnd);
    }

    [Fact]
    public void Resolve_WithoutInitImage_ReturnsNull()
    {
        Assert.Null(RecipeImg2ImgBinder.Resolve(new ImageRequest { Prompt = "test" }, 64, 64));
    }

    [Fact]
    public void Resolve_InpaintMaskWithoutInitImage_Throws()
    {
        ImageRequest request = new ImageRequest
        {
            Prompt = "test",
            Inpaint = new Inpaint { Mask = SolidImage(8, 8, 255) },
        };

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => RecipeImg2ImgBinder.Resolve(request, 64, 64));
        Assert.Contains("init image", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The resolver resizes to the size it is handed, not the size of the supplied image — this is what lets a
    /// family that snaps its dimensions stay consistent with the exact-shape check in <c>Img2ImgSetup.Prepare</c>.</summary>
    [Fact]
    public void Resolve_ResizesInitImageToTheSuppliedSize()
    {
        using Img2ImgResolver.Img2ImgSpec? spec = RecipeImg2ImgBinder.Resolve(RequestWithInit(48, 32), 48, 32);

        Assert.NotNull(spec);
        Assert.Equal(new TensorShape(1, 3, 32, 48), spec!.SourceTensor.Shape);
        Assert.Null(spec.MaskTensor);
    }

    [Fact]
    public void Resolve_WithInpaintMask_ResolvesMaskAtTheSameSize()
    {
        ImageRequest request = RequestWithInit(48, 32) with { Inpaint = new Inpaint { Mask = SolidImage(8, 8, 255) } };

        using Img2ImgResolver.Img2ImgSpec? spec = RecipeImg2ImgBinder.Resolve(request, 48, 32);

        Assert.NotNull(spec);
        Assert.NotNull(spec!.MaskTensor);
        Assert.Equal(new TensorShape(1, 1, 32, 48), spec.MaskTensor!.Shape);
    }

    [Theory]
    [InlineData(0.0, 0.0f)]
    [InlineData(0.6, 0.6f)]
    [InlineData(1.0, 1.0f)]
    [InlineData(2.5, 1.0f)]
    [InlineData(-1.0, 0.0f)]
    public void Resolve_ClampsCreativityIntoStrengthRange(double creativity, float expected)
    {
        using Img2ImgResolver.Img2ImgSpec? spec = RecipeImg2ImgBinder.Resolve(RequestWithInit(16, 16, creativity), 16, 16);

        Assert.NotNull(spec);
        Assert.Equal(expected, spec!.Strength);
    }

    /// <summary>The user-visible refusal: a family whose recipe does not declare <see cref="ImageFeatures.Img2Img"/>
    /// rejects an init image by name rather than silently generating text-to-image.</summary>
    [Fact]
    public async Task Generate_InitImageOnFamilyThatDoesNotDeclareImg2Img_ThrowsNamingTheFeature()
    {
        RecipeRegistry.Register(new FakeRecipe(BareFamilyId, ImageFeatures.None));
        using InferenceEngine engine = new InferenceEngine("cpu");

        NotSupportedException ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Images.GenerateAsync(SpecFor(BareFamilyId), RequestWithInit(64, 64)));

        Assert.Contains(BareFamilyId, ex.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(ImageFeatures.Img2Img), ex.Message, StringComparison.Ordinal);
    }

    /// <summary>The converse: once the recipe declares the bit, the gate lets the request through with its init image
    /// intact, so the recipe pipeline is free to bind it.</summary>
    [Fact]
    public async Task Generate_InitImageOnDeclaringFamily_ReachesThePipelineWithInitIntact()
    {
        FakeRecipe recipe = new FakeRecipe(Img2ImgFamilyId, ImageFeatures.Img2Img);
        RecipeRegistry.Register(recipe);
        using InferenceEngine engine = new InferenceEngine("cpu");

        await engine.Images.GenerateAsync(SpecFor(Img2ImgFamilyId), RequestWithInit(64, 64));

        Assert.NotNull(recipe.Pipeline.Received);
        Assert.NotNull(recipe.Pipeline.Received!.Img2Img);
        Assert.Equal(0.6, recipe.Pipeline.Received.Img2Img!.Creativity);
    }

    private static ModelSpec SpecFor(string familyId) => new ModelSpec
    {
        Requested = familyId,
        Modality = Modality.Image,
        LocalPath = $"/nonexistent/{familyId}.safetensors",
        Catalog = new CatalogEntry
        {
            Id = familyId,
            Modality = Modality.Image,
            DisplayName = familyId,
            Architecture = familyId,
            Status = ModelStatus.Verified,
        },
    };

    /// <summary>A recipe that loads nothing — it exists to exercise the feature gate and request plumbing that run
    /// before any checkpoint is opened.</summary>
    private sealed class FakeRecipe(string familyId, ImageFeatures supports) : IArchitectureRecipe
    {
        public string Name => familyId;

        public ImageFeatures Supports => supports;

        public FakeRecipePipeline Pipeline { get; } = new FakeRecipePipeline();

        public bool Matches(string candidate) => string.Equals(candidate, familyId, StringComparison.OrdinalIgnoreCase);

        public IRecipePipeline Construct(RecipeContext context) => Pipeline;
    }

    /// <summary>Records the request it was handed instead of generating.</summary>
    private sealed class FakeRecipePipeline : IRecipePipeline
    {
        public ImageRequest? Received { get; private set; }

        public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
        {
            Received = request;
            return new ImageResult
            {
                Rgb = new byte[3],
                Width = 1,
                Height = 1,
                Seed = 0,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
            };
        }

        public void Dispose()
        {
        }
    }
}
