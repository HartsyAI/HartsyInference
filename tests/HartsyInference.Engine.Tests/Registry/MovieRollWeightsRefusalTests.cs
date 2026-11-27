using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The plan-time refusal of the released MovieRoll DiT on a backend without weight streaming: the DiT is held at its FP32 load dtype off
/// CUDA, so a 33.3 GB file cannot run on a 12 GB card there. A backend that streams is not refused.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollWeightsRefusalTests
{
    private const long ReleasedDitBytes = 33_291_800_480;
    private const long Gb = 1024L * 1024 * 1024;

    [Fact]
    public void A_NonStreamingBackend_SmallerThanTheDit_IsRefused()
    {
        string? refusal = MovieRollRecipe.WeightsRefusal(streams: false, totalBytes: 12 * Gb, weightBytes: ReleasedDitBytes, backendName: "Vulkan");
        Assert.NotNull(refusal);
        Assert.Contains("cannot run on Vulkan", refusal);
    }

    [Fact]
    public void A_StreamingBackend_IsNotRefused()
    {
        Assert.Null(MovieRollRecipe.WeightsRefusal(streams: true, totalBytes: 12 * Gb, weightBytes: ReleasedDitBytes, backendName: "CUDA"));
    }

    [Fact]
    public void A_NonStreamingBackend_LargerThanTheDit_IsNotRefused()
    {
        Assert.Null(MovieRollRecipe.WeightsRefusal(streams: false, totalBytes: 48 * Gb, weightBytes: ReleasedDitBytes, backendName: "Vulkan"));
    }
}
