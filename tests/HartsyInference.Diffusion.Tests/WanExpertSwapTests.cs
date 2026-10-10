using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the Wan 2.2 dual-expert boundary math and the swap-aware pipeline cache key — both silent-failure
/// numerics: a wrong boundary quietly runs the wrong expert for part of the schedule, and a missing cache suffix
/// silently reuses a single-expert pipeline for a swap request.</summary>
public sealed class WanExpertSwapTests
{
    [Fact]
    public void ResolveBoundary_NullPercent_UsesPresetBoundary()
    {
        Assert.Equal(0.875f, WanVideoRecipe.ResolveBoundary(null, WanVideoConfig.T2V_A14B, isConcatI2V: false));
        Assert.Equal(0.9f, WanVideoRecipe.ResolveBoundary(null, WanVideoConfig.I2V_A14B, isConcatI2V: true));
    }

    /// <summary>p=0.5 at shift 8 → 8·0.5/(1+7·0.5) = 0.888…, the recovered WanVideoLoader warp.</summary>
    [Fact]
    public void ResolveBoundary_ExplicitFraction_WarpsThroughFlowShift()
    {
        WanVideoConfig shift8 = WanVideoConfig.T2V_A14B with { FlowShift = 8f };
        Assert.Equal(8f * 0.5f / (1f + 7f * 0.5f), WanVideoRecipe.ResolveBoundary(0.5, shift8, isConcatI2V: false), 5);
        // The A14B preset's own shift 5: 5·0.5/(1+4·0.5) = 0.833…
        Assert.Equal(5f * 0.5f / (1f + 4f * 0.5f), WanVideoRecipe.ResolveBoundary(0.5, WanVideoConfig.T2V_A14B, isConcatI2V: false), 5);
    }

    [Fact]
    public void CacheKey_DivergesOnSwapModelAndPercent()
    {
        VideoRequest plain = new VideoRequest { Prompt = "p" };
        VideoRequest swapped = plain with { VideoSwapModel = "wan22-low.safetensors" };
        VideoRequest swappedAt06 = swapped with { VideoSwapPercent = 0.6 };
        Assert.NotEqual(RecipeCacheKey.Describe(plain), RecipeCacheKey.Describe(swapped));
        Assert.NotEqual(RecipeCacheKey.Describe(swapped), RecipeCacheKey.Describe(swappedAt06));
        Assert.Equal(RecipeCacheKey.Describe(swapped), RecipeCacheKey.Describe(plain with { VideoSwapModel = "wan22-low.safetensors" }));
    }
}
