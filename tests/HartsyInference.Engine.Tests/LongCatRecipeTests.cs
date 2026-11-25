using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The LongCat recipes' declarations: their names resolve, their defaults sit on their own frame grids, the avatar refuses a clip past one
/// segment, and both families refuse a sampler selection. The real construction and generation need the weights and are not run here.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatRecipeTests
{
    [Theory]
    [InlineData("longcat-video", typeof(LongCatVideoRecipe))]
    [InlineData("LongCat-Avatar", typeof(LongCatAvatarRecipe))]
    public void TheFamilyResolvesToItsRecipe(string familyId, Type recipeType)
    {
        IVideoRecipe? recipe = VideoRecipeRegistry.Resolve(familyId);

        Assert.NotNull(recipe);
        Assert.IsType(recipeType, recipe);
    }

    [Fact]
    public void BothRecipesPassTheRegistryValidator()
    {
        Assert.Empty(RecipeRegistryValidator.Problems(new LongCatVideoRecipe()));
        Assert.Empty(RecipeRegistryValidator.Problems(new LongCatAvatarRecipe()));
    }

    [Fact]
    public void TheBaseDefaultsSitOnTheClipGridAndTheAvatarSegmentLimit()
    {
        LongCatVideoRecipe baseRecipe = new();
        LongCatAvatarRecipe avatar = new();

        Assert.True(baseRecipe.Constraints.Frames.Contains(baseRecipe.Defaults.Frames));
        Assert.True(avatar.Constraints.Frames.Contains(avatar.Defaults.Frames));
        Assert.Equal(93, avatar.Defaults.Frames);
        Assert.False(avatar.Constraints.Frames.Contains(97));
        Assert.Equal(25, avatar.Defaults.Fps);
        Assert.Equal(7f, avatar.Defaults.FlowShift);
    }

    [Fact]
    public void BothFamiliesRefuseAnySamplerSelection()
    {
        Assert.Empty(SamplingCapabilities.ForVideo("longcat-video").Samplers);
        Assert.Empty(SamplingCapabilities.ForVideo("longcat-video").Schedules);
        Assert.Empty(SamplingCapabilities.ForVideo("longcat-avatar").Samplers);
        Assert.Empty(SamplingCapabilities.ForVideo("longcat-avatar").Schedules);
    }

    [Fact]
    public void TheAvatarDeclaresTheInitImageAndDrivingAudio()
    {
        LongCatAvatarRecipe avatar = new();

        Assert.Equal(VideoFeatures.InitImage | VideoFeatures.DrivingAudio, avatar.Supports);
        Assert.False(avatar.Supports.HasFlag(VideoFeatures.LongFormChain));
    }

    [Fact]
    public void AnUnnamedAvatarRequestGetsThe15SettingsNotThe16()
    {
        LongCatAvatarRecipe avatar = new();
        ResolvedModelVariant unnamed = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), []);
        ResolvedModelVariant named15 = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.5"]);

        Assert.Equal("1.5", unnamed.Id);
        foreach (ResolvedModelVariant variant in new[] { unnamed, named15 })
        {
            VideoDefaults settings = avatar.DefaultsFor(variant);
            Assert.Equal(8, settings.Steps);
            Assert.Equal(1f, settings.CfgScale);
            Assert.Equal(7f, settings.FlowShift);
        }
        Assert.Equal(8, avatar.Defaults.Steps);
        Assert.Equal(50, avatar.DefaultsFor(ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.6"])).Steps);
    }

    [Fact]
    public void AUnnamedOr15RequestPassesTheRefusalAndA16RequestIsRefusedBeforeAnyWeightIsRead()
    {
        using CpuBackend backend = new();
        ResolvedModelVariant unnamed = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), []);
        ResolvedModelVariant named16 = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.6"]);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => new LongCatAvatarRecipe().Construct(new RecipeContext
        {
            CheckpointPath = "no-such-folder", Backend = backend, Variant = named16,
        }));
        Assert.Contains("1.6 INT8 catalog row", refused.Message, StringComparison.Ordinal);

        Exception error = Assert.ThrowsAny<Exception>(() => new LongCatAvatarRecipe().Construct(new RecipeContext
        {
            CheckpointPath = "no-such-folder", Backend = backend, Variant = unnamed,
        }));
        Assert.IsNotType<NotSupportedException>(error);
    }

    [Fact]
    public void TheStreamingHeadroomCoversTheFullAvatarRunsActivationsWithAMargin()
    {
        // 832 x 480 x 93: 24 latent frames on a 60 x 104 grid, so 24 x 30 x 52 = 37,440 tokens; hidden 4096, FFN 11,008.
        long tokens = LongCatRecipeSupport.TokenCount(latentFrames: 24, latentHeight: 60, latentWidth: 104);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(tokens, hidden: 4096, ffnHidden: 11008);

        Assert.Equal(37_440, tokens);
        Assert.Equal(37_440L * (10L * 4096 + 3L * 11008) * 4 * 3 / 2, headroom);
        Assert.InRange(headroom, 15L << 30, 16L << 30);
    }

    [Fact]
    public void TheSmallerFirstRunNeedsAboutAThirdOfTheFullHeadroom()
    {
        // 640 x 368 x 49: 13 latent frames on a 46 x 80 grid, so 13 x 23 x 40 = 11,960 tokens.
        long small = LongCatRecipeSupport.ActivationHeadroomBytes(
            LongCatRecipeSupport.TokenCount(latentFrames: 13, latentHeight: 46, latentWidth: 80), hidden: 4096, ffnHidden: 11008);
        long full = LongCatRecipeSupport.ActivationHeadroomBytes(
            LongCatRecipeSupport.TokenCount(latentFrames: 24, latentHeight: 60, latentWidth: 104), hidden: 4096, ffnHidden: 11008);

        Assert.Equal(11_960, LongCatRecipeSupport.TokenCount(latentFrames: 13, latentHeight: 46, latentWidth: 80));
        Assert.InRange((double)small / full, 0.30, 0.34);
    }

    private static CheckpointProbe EmptyProbe() =>
        new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), []);
}
