using HartsyInference.Cli.Commands;
using HartsyInference.Engine.Recipes.Image;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>`hartsy remove-people` runs the people-removal recipe through the `image` command.</summary>
public sealed class RemovePeopleCommandTests
{
    [Fact]
    public void ToImageSettings_RunsTheRecipeOnTheCallersPhotoAndMask()
    {
        RemovePeopleCommand.Settings settings = new()
        {
            Image = "photo.jpg",
            Mask = "people.png",
            Precision = "bf16",
            Seed = 3,
            Output = "out",
        };

        ImageCommand.Settings image = RemovePeopleCommand.ToImageSettings(settings, 768, 1024);

        Assert.Equal("onereward-fill", image.Model);
        Assert.Equal(768, image.Width);
        Assert.Equal(1024, image.Height);
        Assert.Equal("remove", image.Prompt);
        Assert.Equal(PeopleRemoval.NegativePrompt, image.Negative);
        Assert.Equal(PeopleRemoval.TrueCfgScale, image.TrueCfg);
        Assert.Equal(PeopleRemoval.FluxGuidance, image.FluxGuidance);
        Assert.Equal(PeopleRemoval.Steps, image.Steps);
        Assert.Equal(PeopleRemoval.FillCreativity, image.Creativity);
        Assert.Equal(new[] { PeopleRemoval.LoraId }, image.Loras);
        Assert.Equal(new[] { PeopleRemoval.LoraWeight }, image.LoraWeights);
        Assert.Equal("photo.jpg", image.InitImage);
        Assert.Equal("people.png", image.Mask);
        Assert.Equal(3, image.Seed);
        Assert.Equal("out", image.Output);
    }

    [Fact]
    public void ToImageSettings_DefaultsToTheFp8Checkpoint()
    {
        RemovePeopleCommand.Settings settings = new() { Image = "photo.jpg", Mask = "people.png" };

        Assert.Equal("onereward-fill-fp8", RemovePeopleCommand.ToImageSettings(settings, 768, 1024).Model);
    }
}
