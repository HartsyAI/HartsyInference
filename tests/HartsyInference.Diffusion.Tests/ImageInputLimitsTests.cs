using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;
using HartsyInference.Engine.Catalog;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the per-family input-image limits and the gate that refuses past them. Before these existed, a fourth
/// Qwen-Image-Edit reference was dropped with a log line, and a reference handed to Boogu / OmniGen2 / Mage-Flow was
/// never read at all — both came back as a normal-looking image, which is the failure a refusal exists to prevent.</summary>
public sealed class ImageInputLimitsTests
{
    private static ImageData Image() => new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 };

    private static ImageRequest Request(bool init, int references, Img2ImgMode mode = Img2ImgMode.Auto) => new ImageRequest
    {
        Prompt = "limits test",
        Img2Img = init ? new Img2Img { InitImage = Image(), Mode = mode } : null,
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

    /// <summary>The family ceiling is the Plus template's; each variant narrows it to what its template addresses, and
    /// the text-to-image base reads its init image only.</summary>
    [Theory]
    [InlineData("qwen-image-edit-plus", 3)]
    [InlineData("qwen-image", 1)]
    public void QwenImage_LimitsFollowTheResolvedVariant(string swarmClass, int maxImages)
    {
        ModelSpec spec = new ModelSpec { Requested = "qwen-image", Modality = Modality.Image, Variant = swarmClass };
        ImageInputLimits limits = ((ImageModelDescriptor)new RecipeCatalog().Describe(spec)).InputLimits;
        Assert.Equal(maxImages, limits.MaxImages);
        Assert.Equal(swarmClass == "qwen-image", limits.ReferencesRequireInitImage);
    }

    [Theory]
    [InlineData("boogu")]
    public void ReferenceOnlyFamilies_ReadExactlyTheInitImage(string family)
    {
        Assert.Equal(ImageInputLimits.SingleInitImage, LimitsOf(family));
    }

    [Theory]
    [InlineData("sdxl")]
    public void DenoiseFamilies_DeriveOneImage(string family)
    {
        Assert.Equal(ImageInputLimits.SingleInitImage, LimitsOf(family));
    }

    [Theory]
    [InlineData(true, 0)]
    [InlineData(false, 3)]
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

    /// <summary>FLUX.3 conditions on ten references and can also denoise an init image, so the image count alone cannot
    /// say which eleven-image requests it reads: an init image in Reference mode is one of the ten.</summary>
    [Theory]
    [InlineData(Img2ImgMode.Denoise, true, 10)]
    [InlineData(Img2ImgMode.Auto, true, 10)]
    [InlineData(Img2ImgMode.Reference, true, 9)]
    [InlineData(Img2ImgMode.Reference, false, 10)]
    public void Flux3_TenReferencesPlusADenoisedInit_Passes(Img2ImgMode mode, bool init, int references)
    {
        Assert.Null(LimitsOf("flux3").Violation("flux3", Request(init, references, mode)));
    }

    [Theory]
    [InlineData(Img2ImgMode.Reference, true, 10, 11)]
    [InlineData(Img2ImgMode.Auto, false, 11, 11)]
    public void Flux3_PastTenReferences_IsRefusedWithTheCount(Img2ImgMode mode, bool init, int references, int total)
    {
        string? violation = LimitsOf("flux3").Violation("flux3", Request(init, references, mode));
        Assert.Equal($"Model family 'flux3' conditions on at most 10 reference images; {total} were supplied "
            + "(an init image in Reference mode counts as one).", violation);
    }

    /// <summary>MOSS-Image edits take the Ostris node's three references, plus an init image that is denoised.</summary>
    [Theory]
    [InlineData(Img2ImgMode.Denoise, true, 3)]
    [InlineData(Img2ImgMode.Reference, true, 2)]
    [InlineData(Img2ImgMode.Reference, false, 3)]
    public void MossImage_ThreeReferencesPlusADenoisedInit_Passes(Img2ImgMode mode, bool init, int references)
    {
        Assert.Null(LimitsOf("moss-image").Violation("moss-image", Request(init, references, mode)));
    }

    [Theory]
    [InlineData(Img2ImgMode.Reference, true, 3, 4)]
    [InlineData(Img2ImgMode.Auto, false, 4, 4)]
    public void MossImage_PastThreeReferences_IsRefusedWithTheCount(Img2ImgMode mode, bool init, int references, int total)
    {
        string? violation = LimitsOf("moss-image").Violation("moss-image", Request(init, references, mode));
        Assert.Equal($"Model family 'moss-image' conditions on at most 3 reference images; {total} were supplied "
            + "(an init image in Reference mode counts as one).", violation);
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

    /// <summary>End to end through the service gate: the refusal is thrown before any weights are touched. The Plus
    /// build is named explicitly, since the three-image ceiling is its template's.</summary>
    [Fact]
    public async Task ImagesService_RefusesExtraReferences_BeforeLoading()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        ModelSpec spec = Spec("qwen-image") with { Variant = "qwen-image-edit-plus" };
        NotSupportedException error = await Assert.ThrowsAsync<NotSupportedException>(
            () => engine.Images.GenerateAsync(spec, Request(init: true, references: 3)));
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
