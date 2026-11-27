using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Tests.Features;

public sealed class BackgroundRemovalModelTests
{
    private static ImageRequest Request(bool? remove, string? model) =>
        new() { Prompt = "p", RemoveBackground = remove, RemoveBackgroundModel = model };

    [Fact]
    public void NoCutoutRequested_NamesNoMatter()
    {
        Assert.Null(BackgroundRemovalStage.ModelIdFor(Request(null, null)));
        Assert.False(BackgroundRemovalStage.NeedsGeneratorReleased(Request(null, null)));
    }

    [Fact]
    public void CutoutWithoutAModel_KeepsRmbg14AsTheDefault()
    {
        Assert.Equal("rmbg", BackgroundRemovalStage.ModelIdFor(Request(true, null)));
        Assert.Equal("rmbg", BackgroundRemovalStage.ModelIdFor(Request(true, "  ")));
        Assert.False(BackgroundRemovalStage.NeedsGeneratorReleased(Request(true, null)));
    }

    [Theory]
    [InlineData("rmbg3")]
    [InlineData("RMBG3-INT8")]
    public void Rmbg3_NeedsTheGeneratorReleased(string model)
    {
        Assert.True(BackgroundRemovalStage.NeedsGeneratorReleased(Request(true, model)));
    }

    [Fact]
    public void AnUnknownMatter_OrAModelWithoutTheFlag_IsRefused()
    {
        Assert.Throws<NotSupportedException>(() => BackgroundRemovalStage.ModelIdFor(Request(true, "birefnet-matting")));
        Assert.Throws<NotSupportedException>(() => BackgroundRemovalStage.ModelIdFor(Request(null, "rmbg3")));
        Assert.Throws<NotSupportedException>(() => BackgroundRemovalStage.ModelIdFor(Request(false, "rmbg3")));
    }
}
