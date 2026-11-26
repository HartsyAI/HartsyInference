using HartsyInference.Engine.Dispatch;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The 1.6 row routing in engine resolution, and the later-run figures of a mixed (uncached) 1.6 run against a cached 1.5 run.</summary>
public sealed class LongCatResolutionTests
{
    private const long Mb = 1L << 20;

    [Theory]
    [InlineData("longcat-avatar:1.6")]
    [InlineData("longcat-avatar:avatar-1.6")]
    [InlineData("longcat-avatar-1.6")]
    public void A16RequestResolvesToTheInt8Row(string model)
    {
        ModelSpec spec = ModelResolver.Resolve(model, modelPathArg: null, Modality.Video);
        Assert.Equal(LongCatAvatarVariants.Avatar16FamilyId, spec.Catalog?.Id);
        Assert.Null(spec.Variant);
    }

    [Fact]
    public void A15RequestStaysOnTheFamilyRow()
    {
        ModelSpec spec = ModelResolver.Resolve("longcat-avatar:1.5", modelPathArg: null, Modality.Video);
        Assert.Equal("longcat-avatar", spec.Catalog?.Id);
        Assert.Equal("1.5", spec.Variant);
    }

    [Fact]
    public void AMixedRunIsUncachedUnmeasuredAndNotOffloaded()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        LongCatRecipeSupport.LongCatLaterRunFigures figures = LongCatRecipeSupport.LaterRunFigures(mixed: true, plan, 46, 80, LongCatTestConfig(),
            640, 368, distilled: true);
        Assert.Equal(LongCatRecipeSupport.TokenCount(plan.ContextLatents + 1 + plan.NewLatentsPerSegment, 46, 80), figures.Tokens);
        Assert.Equal(0, figures.CacheBytes);
        Assert.Null(figures.MeasuredPeak);
        Assert.False(figures.OffloadKv);
    }

    [Fact]
    public void ACachedRunKeepsItsCacheMeasurementAndOffload()
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        LongCatRecipeSupport.LongCatLaterRunFigures figures = LongCatRecipeSupport.LaterRunFigures(mixed: false, plan, 46, 80, LongCatTestConfig(),
            640, 368, distilled: true);
        Assert.Equal(LongCatRecipeSupport.TokenCount(plan.NewLatentsPerSegment, 46, 80), figures.Tokens);
        Assert.True(figures.CacheBytes > 0);
        Assert.Equal(7808 * Mb, figures.MeasuredPeak);
        Assert.True(figures.OffloadKv);
    }

    [Fact]
    public void AMissingAvatar16LoraIsRefusedNamingTheRow()
    {
        string root = Path.Combine(Path.GetTempPath(), "longcat-16-missing-" + Guid.NewGuid().ToString("N"));
        string baseFolder = Path.Combine(root, "base_model_int8");
        Directory.CreateDirectory(baseFolder);
        try
        {
            NotSupportedException refused = Assert.Throws<NotSupportedException>(() => LongCatRecipeSupport.RequireAvatar16Loras(baseFolder));
            Assert.Contains("longcat-avatar-1.6", refused.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static HartsyInference.Diffusion.Models.Denoisers.LongCatConfig LongCatTestConfig() =>
        new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
}
