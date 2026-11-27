using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Iris-Z recipe pipeline against the native <see cref="ImageRequest"/>: what it refuses by name, and the
/// <c>depth</c> and <c>rgbd</c> tasks end to end on the tiny random model of <see cref="IrisZPipelineFixture"/>.</summary>
public sealed class IrisZRecipePipelineTests
{
    private const float SkyCeiling = 0.95f;

    [Fact]
    public void ReferenceImagesAreRefused()
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);
        ImageRequest request = Request() with { ReferenceImages = [Photo(32, 32)] };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Generate(fixture, request));

        Assert.Contains("reference images", error.Message);
    }

    /// <summary>A refiner hand-off arrives as a denoise-mode init image, which Iris-Z would otherwise read as a photo.</summary>
    [Fact]
    public void ADenoiseModeInitImageIsRefused()
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);
        ImageRequest request = Request() with { Img2Img = new Img2Img { InitImage = Photo(32, 32), Mode = Img2ImgMode.Denoise } };

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Generate(fixture, request));

        Assert.Contains("refiner", error.Message);
    }

    [Theory]
    [InlineData("image")]
    [InlineData("rgbd")]
    public void AnInitImageOnAPromptTaskIsRefused(string task)
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);
        ImageRequest request = Request(task) with { Img2Img = new Img2Img { InitImage = Photo(32, 32) } };

        Assert.Throws<NotSupportedException>(() => Generate(fixture, request));
    }

    [Theory]
    [InlineData("depth")]
    [InlineData("paint")]
    [InlineData("segment")]
    public void ATaskWithoutItsInputOrAnUnknownTaskIsRefused(string task)
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);

        Assert.Throws<NotSupportedException>(() => Generate(fixture, Request(task)));
    }

    [Theory]
    [InlineData(RequestExtras.IrisZDepthSeeds, 0d)]
    [InlineData(RequestExtras.IrisZDepthSeeds, 1e10)]
    [InlineData(RequestExtras.IrisZDepthSteps, 2.5)]
    [InlineData(RequestExtras.IrisZResolution, 768d)]
    public void OutOfRangeDepthOptionsAreRefusedWithTheValueAsSent(string key, double value)
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);
        ImageRequest request = Request("depth", (key, value)) with { Img2Img = new Img2Img { InitImage = Photo(64, 48) } };

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => Generate(fixture, request));

        Assert.Equal(value, error.ActualValue);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void DepthReturnsARenderingAndAPlaneThatNameTheSameSky()
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);
        ImageRequest request = Request() with { Img2Img = new Img2Img { InitImage = Photo(64, 48) } };

        ImageResult result = Generate(fixture, request);

        Assert.Equal((64, 48), (result.Width, result.Height));
        Assert.Equal("depth", result.Meta[DepthImage.TaskMeta]);
        Assert.NotNull(result.Depth);
        Assert.Equal(64 * 48, result.Depth.Length);
        // Every pixel is sky at exactly +1 or real depth below the sky value, so the plane alone rebuilds the rendering.
        Assert.All(result.Depth, value => Assert.True(value == 1f || value < IrisZDepth.SkyValue));
        Assert.Equal(result.Rgb, DepthImage.Colourize(result));
        Assert.Null(DepthImage.Companion(result));
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void RgbdReturnsTheImageWithItsDepthAsACompanion()
    {
        using IrisZPipelineFixture fixture = new(SkyCeiling);

        ImageResult result = Generate(fixture, Request("rgbd"));

        Assert.Equal(32 * 32 * 3, result.Rgb.Length);
        Assert.NotNull(result.Depth);
        Assert.Equal(32 * 32, result.Depth.Length);
        Assert.NotNull(DepthImage.Companion(result));
    }

    private static ImageResult Generate(IrisZPipelineFixture fixture, ImageRequest request)
    {
        IrisZRecipePipeline pipeline = new(fixture.Pipeline, new Flux2PromptTokenizer(fixture.Config, null, null, ""), SkyCeiling,
            "iris-z-big", null, null, fixture.TextEncoder, []);
        return pipeline.Generate(request, ImageEffectiveSettings.Of(request), null, CancellationToken.None);
    }

    private static ImageRequest Request(string? task = null, (string Key, double Value)? option = null)
    {
        Dictionary<string, object> extra = new(StringComparer.Ordinal);
        if (task is not null) extra[RequestExtras.IrisZTask] = task;
        if (option is { } set) extra[set.Key] = set.Value;
        return new ImageRequest { Prompt = "", Width = 32, Height = 32, Steps = 2, CfgScale = 4f, Seed = 7, Extra = extra };
    }

    private static ImageData Photo(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        new Random(13).NextBytes(rgb);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
