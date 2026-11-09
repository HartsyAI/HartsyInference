using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The people-removal recipe: the request it builds and the catalog id it picks for each precision.</summary>
public sealed class PeopleRemovalTests
{
    [Fact]
    public void Build_CarriesTheRecipeAndTheCallersPhotoAndMask()
    {
        ImageData image = new() { Rgb = new byte[4 * 6 * 3], Width = 4, Height = 6 };
        ImageData mask = new() { Rgb = new byte[4 * 6 * 3], Width = 4, Height = 6 };

        ImageRequest request = PeopleRemoval.Build(image, mask, seed: 7);

        Assert.Equal("remove", request.Prompt);
        Assert.Equal(PeopleRemoval.NegativePrompt, request.NegativePrompt);
        Assert.Equal(4f, request.TrueCfgScale);
        Assert.Equal(1f, request.FluxGuidance);
        Assert.Equal(50, request.Steps);
        Assert.Equal(4, request.Width);
        Assert.Equal(6, request.Height);
        Assert.Equal(7, request.Seed);
        LoraEntry lora = Assert.Single(request.Loras!.Entries);
        Assert.Equal(PeopleRemoval.LoraId, lora.Model);
        Assert.Equal(1.0, lora.Weight);
        Assert.Same(image, request.Img2Img!.InitImage);
        Assert.Equal(1.0, request.Img2Img!.Creativity);
        Assert.Same(mask, request.Inpaint!.Mask);
    }

    [Theory]
    [InlineData(null, "onereward-fill-fp8")]
    [InlineData("", "onereward-fill-fp8")]
    [InlineData("fp8", "onereward-fill-fp8")]
    [InlineData("bf16", "onereward-fill")]
    public void ModelId_MapsThePrecisionToItsCatalogEntry(string? precision, string expected) =>
        Assert.Equal(expected, PeopleRemoval.ModelId(precision));

    [Fact]
    public void ModelId_RefusesAnUnknownPrecision() =>
        Assert.Throws<ArgumentException>(() => PeopleRemoval.ModelId("fp16"));
}
