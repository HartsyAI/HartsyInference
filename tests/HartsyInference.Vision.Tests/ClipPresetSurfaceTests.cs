using HartsyInference.Vision.Clip;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Sanity checks on the bundled preset records — these are pure data so the tests
/// are mostly guarding against accidental edits that would break checkpoint loading.</summary>
public sealed class ClipPresetSurfaceTests
{
    [Fact]
    public void OpenAiClipLarge_HasMatchingProjectionDims()
    {
        ClipPreset p = ClipPreset.OpenAiClipLarge;
        Assert.Equal(768, p.TextConfig.ProjectionDim);
        Assert.Equal(768, p.VisionConfig.ProjectionDim);
        Assert.Equal(p.TextConfig.ProjectionDim, p.VisionConfig.ProjectionDim);
        Assert.Equal(p.EmbeddingDim, p.VisionConfig.ProjectionDim);
    }

    [Fact]
    public void OpenClipHuge_HasMatchingProjectionDims()
    {
        ClipPreset p = ClipPreset.OpenClipHuge;
        Assert.Equal(1024, p.TextConfig.ProjectionDim);
        Assert.Equal(1024, p.VisionConfig.ProjectionDim);
    }

    [Fact]
    public void OpenClipBigG_HasMatchingProjectionDims()
    {
        ClipPreset p = ClipPreset.OpenClipBigG;
        Assert.Equal(1280, p.TextConfig.ProjectionDim);
        Assert.Equal(1280, p.VisionConfig.ProjectionDim);
        // CLIP-bigG has the non-standard head_dim = 1664 / 16 = 104.
        Assert.Equal(1664, p.VisionConfig.HiddenSize);
        Assert.Equal(16, p.VisionConfig.NumHeads);
    }

    [Fact]
    public void OpenAiClipLarge_UsesQuickGelu_InVisionTower()
    {
        // OpenAI weights expect Quick-GELU. Mixing this up with standard GELU silently
        // degrades cosine similarity by a few percent — exactly the kind of bug Phase 6
        // validation needs to catch.
        Assert.True(ClipPreset.OpenAiClipLarge.VisionConfig.UseQuickGelu);
    }

    [Fact]
    public void OpenClipHugeAndBigG_UseStandardGelu_InVisionTower()
    {
        Assert.False(ClipPreset.OpenClipHuge.VisionConfig.UseQuickGelu);
        Assert.False(ClipPreset.OpenClipBigG.VisionConfig.UseQuickGelu);
    }
}
