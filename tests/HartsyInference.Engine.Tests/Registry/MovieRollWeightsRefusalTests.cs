using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The MovieRoll refusals and the shared device fit they use: the DiT is held at its FP32 load dtype off CUDA, so a 33.3 GB file cannot run
/// on a 12 GB non-streaming device; quantized checkpoints and LoRA requests have no path and are refused.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollWeightsRefusalTests
{
    private const long ReleasedDitBytes = 33_291_800_480;
    private const long Gb = 1024L * 1024 * 1024;

    [Fact]
    public void NonStreamingDevice_SmallerThanTheDit_IsRefused()
    {
        string? refusal = VideoWeightFit.Decide("Vulkan", 12 * Gb, ReleasedDitBytes, "MovieRoll", MovieRollRecipe.DitHoldDetail);
        Assert.NotNull(refusal);
        Assert.Contains("MovieRoll cannot run on Vulkan", refusal);
        Assert.Contains("FP32 load dtype", refusal);
    }

    [Fact]
    public void NonStreamingDevice_LargerThanTheDit_IsNotRefused()
    {
        Assert.Null(VideoWeightFit.Decide("Vulkan", 48 * Gb, ReleasedDitBytes, "MovieRoll", MovieRollRecipe.DitHoldDetail));
    }

    [Fact]
    public void LongCatMessage_IsUnchangedThroughTheSharedFit()
    {
        string? refusal = VideoWeightFit.Decide("Vulkan", 12 * Gb, 20 * Gb, "LongCat", " (its INT8 weights widen to F32 on this backend)");
        Assert.StartsWith("LongCat cannot run on Vulkan: this backend has no weight streaming", refusal);
        Assert.Contains("(its INT8 weights widen to F32 on this backend)", refusal);
    }

    [Fact]
    public void QuantizedDtypes_AreRefused_AndFloatDtypesAreNot()
    {
        Assert.NotNull(MovieRollRecipe.QuantRefusal([DType.F32, DType.I8]));
        Assert.Null(MovieRollRecipe.QuantRefusal([DType.F32, DType.BF16, DType.F16]));
    }

    [Fact]
    public void LoraAdapters_AreRefused_AndNoAdaptersAreNot()
    {
        Assert.NotNull(MovieRollRecipe.LoraRefusal(hasAdapters: true));
        Assert.Null(MovieRollRecipe.LoraRefusal(hasAdapters: false));
    }
}
