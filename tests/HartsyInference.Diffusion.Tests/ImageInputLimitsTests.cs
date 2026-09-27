using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the per-family input-image limits and the gate that refuses past them. Before these existed, a fourth
/// Qwen-Image-Edit reference was dropped with a log line, and a reference handed to Boogu / OmniGen2 / Mage-Flow was
/// never read at all — both came back as a normal-looking image, which is the failure a refusal exists to prevent.</summary>
public sealed class ImageInputLimitsTests
{
    private static ImageData Image() => new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 };

    private static ImageRequest Request(bool init, int references) => new ImageRequest
    {
        Prompt = "limits test",
        Img2Img = init ? new Img2Img { InitImage = Image() } : null,
        ReferenceImages = references == 0 ? null : [.. Enumerable.Range(0, references).Select(_ => Image())],
    };

    /// <summary>A catalog-backed spec so family resolution never touches disk-based architecture detection.</summary>
    private static ModelSpec Spec(string family) => new ModelSpec
    {
        Requested = family,
        Modality = Modality.Image,
        Catalog = new CatalogEntry
        {
            Id = family,
            Modality = Modality.Image,
            DisplayName = family,
            Architecture = family,
            Status = ModelStatus.Verified,
        },
    };

    private static ImageInputLimits LimitsOf(string family) =>
        (RecipeRegistry.Resolve(family) ?? throw new InvalidOperationException($"No recipe for '{family}'.")).InputLimits;

    [Fact]
    public void QwenImage_TakesTheEditTemplatesSlots_WithoutNeedingAnInit()
    {
        ImageInputLimits limits = LimitsOf("qwen-image");
        Assert.Equal(QwenImageEditConditioning.MaxReferences, limits.MaxImages);
        Assert.False(limits.ReferencesRequireInitImage);
    }

    [Theory]
    [InlineData("boogu")]
    [InlineData("omnigen2")]
    [InlineData("mage-flow")]
    public void ReferenceOnlyFamilies_ReadExactlyTheInitImage(string family)
    {
        Assert.Equal(ImageInputLimits.SingleInitImage, LimitsOf(family));
    }

    [Theory]
    [InlineData("sdxl")]
    [InlineData("flux1")]
    [InlineData("zimage")]
    public void DenoiseFamilies_DeriveOneImage(string family)
    {
        Assert.Equal(ImageInputLimits.SingleInitImage, LimitsOf(family));
    }

    [Fact]
    public void DerivedFrom_NoInitFeature_IsTextOnly()
    {
        Assert.Equal(ImageInputLimits.TextOnly, ImageInputLimits.DerivedFrom(ImageFeatures.Lora | ImageFeatures.ControlNet));
    }

    [Theory]
    [InlineData(true, 0)]
    [InlineData(true, 2)]
    [InlineData(false, 3)]
    [InlineData(false, 1)]
    public void Qwen_WithinThreeImages_Passes(bool init, int references)
    {
        Assert.Null(LimitsOf("qwen-image").Violation("qwen-image", Request(init, references)));
    }

    [Theory]
    [InlineData(true, 3, 4)]
    [InlineData(false, 4, 4)]
    public void Qwen_PastThreeImages_IsRefusedWithTheCount(bool init, int references, int total)
    {
        string? violation = LimitsOf("qwen-image").Violation("qwen-image", Request(init, references));
        Assert.Equal($"Model family 'qwen-image' takes at most 3 input images; {total} were supplied.", violation);
    }

    [Fact]
    public void ReferenceOnly_InitPlusReference_IsRefused()
    {
        string? violation = LimitsOf("boogu").Violation("boogu", Request(init: true, references: 1));
        Assert.Equal("Model family 'boogu' takes at most 1 input image; 2 were supplied.", violation);
    }

    [Fact]
    public void ReferenceOnly_ReferenceWithoutInit_IsRefused()
    {
        string? violation = LimitsOf("omnigen2").Violation("omnigen2", Request(init: false, references: 1));
        Assert.Equal("Model family 'omnigen2' needs an init image to edit; reference images alone are not used.", violation);
    }

    /// <summary>Several references and no init image: the missing init image is the actionable problem, so it is the
    /// one reported, not the count.</summary>
    [Fact]
    public void ReferenceOnly_SeveralReferencesWithoutInit_AsksForTheInitImage()
    {
        string? violation = LimitsOf("omnigen2").Violation("omnigen2", Request(init: false, references: 2));
        Assert.Equal("Model family 'omnigen2' needs an init image to edit; reference images alone are not used.", violation);
    }

    [Fact]
    public void ReferenceOnly_InitAlone_Passes()
    {
        Assert.Null(LimitsOf("mage-flow").Violation("mage-flow", Request(init: true, references: 0)));
    }

    /// <summary>End to end through the service gate: the refusal is thrown before any weights are touched.</summary>
    [Fact]
    public async Task ImagesService_RefusesExtraReferences_BeforeLoading()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Images.GenerateAsync(Spec("qwen-image"), Request(init: true, references: 3)));
        Assert.Equal("Model family 'qwen-image' takes at most 3 input images; 4 were supplied.", error.Message);
    }

    [Fact]
    public async Task ImagesService_RefusesReferenceWithoutInit_OnReferenceOnlyFamily()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Images.GenerateAsync(Spec("boogu"), Request(init: false, references: 1)));
        Assert.Equal("Model family 'boogu' needs an init image to edit; reference images alone are not used.", error.Message);
    }

    /// <summary>A family with no reference path keeps the existing feature refusal, which names the missing bit.</summary>
    [Fact]
    public async Task ImagesService_DenoiseFamilyWithReferences_KeepsTheFeatureRefusal()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Images.GenerateAsync(Spec("sdxl"), Request(init: true, references: 1)));
        Assert.Equal("Model family 'sdxl' does not support: RefEdit.", error.Message);
    }
}
