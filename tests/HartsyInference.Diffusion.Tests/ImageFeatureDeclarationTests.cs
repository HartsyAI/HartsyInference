using HartsyInference.Engine.Recipes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins which families declare <see cref="ImageFeatures.Img2Img"/>, because that single bit is what decides
/// whether SwarmUI routes an init-image request to this backend or refuses it. A family that gains the bit without its
/// recipe pipeline actually binding the init image would silently generate text-to-image — the exact failure mode
/// Mage-Flow shipped with — so this list is meant to be edited deliberately, one family at a time, as each is wired
/// and verified.</summary>
public sealed class ImageFeatureDeclarationTests
{
    /// <summary>Families whose recipe pipeline binds <c>request.Img2Img</c> through <c>RecipeImg2ImgBinder</c>.</summary>
    private static readonly string[] ExpectedImg2Img =
    [
        "sd15", "sdxl",                                             // Phase 0 (pre-existing)
        "sd3", "zimage", "qwen-image", "zeta-chroma", "chroma-radiance", // Phase 1
        "mage-flow",                                                // reference-edit conditioning, not strength-based
    ];

    /// <summary>Inpaint additionally requires a masked path in the diffusion pipeline; Mage-Flow has none.</summary>
    private static readonly string[] ExpectedInpaint =
    [
        "sd15", "sdxl",
        "sd3", "zimage", "qwen-image", "zeta-chroma", "chroma-radiance",
    ];

    private readonly ITestOutputHelper _output;

    public ImageFeatureDeclarationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Img2ImgIsDeclaredByExactlyTheWiredFamilies()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.Img2Img);
        _output.WriteLine($"img2img: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedImg2Img.Order(StringComparer.Ordinal)], actual);
    }

    [Fact]
    public void InpaintIsDeclaredByExactlyTheFamiliesWithAMaskedPath()
    {
        string[] actual = DeclaringFamilies(ImageFeatures.Inpaint);
        _output.WriteLine($"inpaint: {string.Join(", ", actual)}");
        Assert.Equal([.. ExpectedInpaint.Order(StringComparer.Ordinal)], actual);
    }

    /// <summary>Inpaint without img2img is incoherent — a mask re-paints an existing image, so the init image is
    /// mandatory. The composition plan enforces this per request; this catches a recipe that declares the pair wrong.</summary>
    [Fact]
    public void NoFamilyDeclaresInpaintWithoutImg2Img()
    {
        foreach (string family in DeclaringFamilies(ImageFeatures.Inpaint))
        {
            ImageFeatures supports = RecipeRegistry.Resolve(family)!.Supports;
            Assert.True((supports & ImageFeatures.Img2Img) != 0,
                $"{family} declares Inpaint without Img2Img; an inpaint mask requires an init image.");
        }
    }

    private static string[] DeclaringFamilies(ImageFeatures feature) =>
        [.. RecipeRegistry.RegisteredNames
            .Distinct(StringComparer.Ordinal)
            .Where(name => (Supports(name) & feature) != ImageFeatures.None)
            .Order(StringComparer.Ordinal)];

    private static ImageFeatures Supports(string family) =>
        RecipeRegistry.Resolve(family)?.Supports ?? ImageFeatures.None;
}
