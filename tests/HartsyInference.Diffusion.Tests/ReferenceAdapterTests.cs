using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The reference adapter's engine surface: its metadata as the released file writes it, the refusals of the layouts it does not
/// run, the request settings and their defaults, the adapter's place in a pipeline's identity, and the horizontal stitch. The generation
/// itself is a GPU check.</summary>
public sealed class ReferenceAdapterTests
{
    /// <summary>The <c>anima_ref_adapter</c> metadata of the released v2 file, verbatim.</summary>
    private const string ReleasedMetadata =
        """{"rank": 64, "alpha": 64, "targets": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj", "mlp.layer1", "mlp.layer2"], "slots": {"character": {"t": 1, "offset_h": 1.0, "offset_w": 0}}, "ref_sigma": 0.0, "base": "anima_baseV10.safetensors", "version": 3, "task": "copy", "resolutions": "0.56-1 MP", "arch": "asym", "ref_lora": true, "ref_cross_attn": false, "step": 1000}""";

    [Fact]
    public void Spec_ReadsTheReleasedLayout()
    {
        ReferenceAdapterSpec spec = ReferenceAdapterSpec.FromMetadata("/models/Lora/adapter.safetensors", ReleasedMetadata);

        Assert.Equal(new ReferencePlacement(1, 1.0f, 0), spec.Placement);
    }

    [Theory]
    [InlineData("""{"arch": "asym", "ref_lora": false, "slots": {"character": {"t": 1}}}""", "ref_lora false")]
    [InlineData("""{"arch": "asym", "slots": {"character": {"t": "one"}}}""", "malformed")]
    public void Spec_NamesTheFileForAMetadataItCannotRun(string metadata, string reason)
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => ReferenceAdapterSpec.FromMetadata("/models/Lora/adapter.safetensors", metadata));
        Assert.Contains("adapter.safetensors", error.Message);
        Assert.Contains(reason, error.Message);
    }

    [Theory]
    [InlineData("""{"arch": "joint", "slots": {"character": {"t": 1}}}""", "joint-frame pilot")]
    [InlineData("""{"slots": {"character": {"t": 1}}}""", "arch")]
    [InlineData("""{"arch": "asym", "slots": {"character": {"t": 1}, "style": {"t": 2}}}""", "two-slot")]
    [InlineData("""{"arch": "asym"}""", "reference slots")]
    public void Spec_RefusesTheLayoutsItDoesNotRun(string metadata, string reason)
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => ReferenceAdapterSpec.FromMetadata("/models/Lora/adapter.safetensors", metadata));
        Assert.Contains(reason, error.Message);
    }

    [Fact]
    public void Spec_RefusesAFileWithNoMetadata()
    {
        Assert.Throws<NotSupportedException>(() => ReferenceAdapterSpec.FromMetadata("/models/Lora/adapter.safetensors", null));
    }

    [Fact]
    public void Settings_DefaultToTheReleasedAdapterRecommendation()
    {
        ReferenceAdapterSettings settings = ReferenceAdapterSettings.From(null);

        Assert.Null(settings.Adapter);
        Assert.Equal(1.0f, settings.Strength);
        Assert.Equal(ReferenceAdapterSettings.DefaultMaxPixels, settings.MaxPixels);
        Assert.Equal(4.5f, settings.Guidance.Scale);
        Assert.Equal(ReferenceCorrection.Apg, settings.Guidance.Correction);
        Assert.Equal(ReferenceOutside.KeepReference, settings.Guidance.Outside);
        Assert.Equal(0f, settings.Guidance.Start);
        Assert.Equal(1f, settings.Guidance.End);
    }

    [Fact]
    public void Settings_ReadTheRequestExtras()
    {
        Dictionary<string, object> extra = new()
        {
            [RequestExtras.ReferenceAdapter] = "custom.safetensors",
            [RequestExtras.ReferenceStrength] = 0.5,
            [RequestExtras.ReferenceGuidance] = 3.0,
            [RequestExtras.ReferenceCorrection] = "rescale",
            [RequestExtras.ReferenceRescalePhi] = 0.25,
            [RequestExtras.ReferenceEnd] = 0.6,
            [RequestExtras.ReferenceOutside] = "drop",
            [RequestExtras.ReferenceMaxPixels] = 786432,
        };

        ReferenceAdapterSettings settings = ReferenceAdapterSettings.From(extra);

        Assert.Equal("custom.safetensors", settings.Adapter);
        Assert.Equal(0.5f, settings.Strength);
        Assert.Equal(786432L, settings.MaxPixels);
        Assert.Equal(3.0f, settings.Guidance.Scale);
        Assert.Equal(ReferenceCorrection.StdRescale, settings.Guidance.Correction);
        Assert.Equal(0.25f, settings.Guidance.RescalePhi);
        Assert.Equal(0.6f, settings.Guidance.End);
        Assert.Equal(ReferenceOutside.DropReference, settings.Guidance.Outside);
    }

    [Theory]
    [InlineData(RequestExtras.ReferenceCorrection, "sharpen")]
    [InlineData(RequestExtras.ReferenceOutside, "fade")]
    [InlineData(RequestExtras.ReferenceStrength, -1.0)]
    [InlineData(RequestExtras.ReferenceEnd, 1.5)]
    [InlineData(RequestExtras.ReferenceMaxPixels, 4194304)]
    public void Settings_RefuseOutOfRangeValuesByName(string key, object value)
    {
        Dictionary<string, object> extra = new() { [key] = value };

        Assert.ThrowsAny<ArgumentException>(() => ReferenceAdapterSettings.From(extra));
    }

    [Fact]
    public void RequestedAdapter_IsNullWithoutAReference_AndEmptyForTheDefault()
    {
        Assert.Null(ReferenceAdapterSettings.RequestedAdapter(new ImageRequest { Prompt = "a cat" }));

        ImageRequest withReference = new()
        {
            Prompt = "a cat",
            ReferenceImages = [Blank(2, 2)],
        };
        Assert.Equal("", ReferenceAdapterSettings.RequestedAdapter(withReference));

        ImageRequest named = new()
        {
            Prompt = "a cat",
            ReferenceImages = [Blank(2, 2)],
            Extra = new Dictionary<string, object> { [RequestExtras.ReferenceAdapter] = "custom.safetensors" },
        };
        Assert.Equal("custom.safetensors", ReferenceAdapterSettings.RequestedAdapter(named));
    }

    [Fact]
    public void RequestedAdapter_NamedWithoutAReference_IsRefused()
    {
        // The adapter would load and then change nothing, so the request is refused before the pipeline is built.
        ImageRequest named = new()
        {
            Prompt = "a cat",
            Extra = new Dictionary<string, object> { [RequestExtras.ReferenceAdapter] = "custom.safetensors" },
        };

        ArgumentException error = Assert.Throws<ArgumentException>(() => ReferenceAdapterSettings.RequestedAdapter(named));
        Assert.Equal(RequestExtras.ReferenceAdapter, error.ParamName);
    }

    [Fact]
    public void Anima_DeclaresTheReferencePath_AndItsLimits()
    {
        AnimaRecipe recipe = new();

        Assert.IsAssignableFrom<IReferenceAdapterRecipe>(recipe);
        Assert.True(recipe.Supports.HasFlag(ImageFeatures.RefEdit));
        Assert.Equal(2, recipe.InputLimits.MaxReferences);
        Assert.Equal(3, recipe.InputLimits.MaxImages);
    }

    [Fact]
    public void StitchHorizontal_JoinsAtTheLeftHeightWithNoGap()
    {
        ImageData left = Solid(2, 2, 255, 0, 0);
        ImageData right = Solid(4, 4, 0, 0, 255);

        ImageData stitched = FeatureImaging.StitchHorizontal(left, right, long.MaxValue);

        Assert.Equal(2, stitched.Height);
        Assert.Equal(4, stitched.Width);
        Assert.Equal(new byte[] { 255, 0, 0 }, Pixel(stitched, 0, 0));
        Assert.Equal(new byte[] { 255, 0, 0 }, Pixel(stitched, 1, 1));
        Assert.Equal(new byte[] { 0, 0, 255 }, Pixel(stitched, 2, 0));
        Assert.Equal(new byte[] { 0, 0, 255 }, Pixel(stitched, 3, 1));
    }

    [Fact]
    public void StitchHorizontal_ShrinksBothImagesToTheAreaCap()
    {
        // The joined image would be 8×4 = 32 pixels; capped at 8 it must shrink to about that, not allocate the full join.
        ImageData left = Solid(4, 4, 255, 0, 0);
        ImageData right = Solid(4, 4, 0, 0, 255);

        ImageData stitched = FeatureImaging.StitchHorizontal(left, right, 8);

        Assert.True((long)stitched.Width * stitched.Height <= 12, $"stitched {stitched.Width}x{stitched.Height} exceeds the cap");
        Assert.Equal(stitched.Width * stitched.Height * 3, stitched.Rgb.Length);
    }

    private static ImageData Blank(int width, int height) => Solid(width, height, 0, 0, 0);

    private static ImageData Solid(int width, int height, byte r, byte g, byte b)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    private static byte[] Pixel(ImageData image, int x, int y)
    {
        int offset = (y * image.Width + x) * 3;
        return [image.Rgb[offset], image.Rgb[offset + 1], image.Rgb[offset + 2]];
    }
}
