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
    public void ANamed15BuildGetsTheDistilledDefaultsAndAnUnnamedRequestKeepsThe16()
    {
        LongCatAvatarRecipe avatar = new();
        ResolvedModelVariant named = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.5"]);
        ResolvedModelVariant unnamed = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), []);

        VideoDefaults distilled = avatar.DefaultsFor(named);

        Assert.Equal(8, distilled.Steps);
        Assert.Equal(1f, distilled.CfgScale);
        Assert.Equal(7f, distilled.FlowShift);
        Assert.Equal(50, avatar.DefaultsFor(unnamed).Steps);
        Assert.Equal(50, avatar.DefaultsFor(null).Steps);
        Assert.Equal("1.6", unnamed.Id);
    }

    private static CheckpointProbe EmptyProbe() =>
        new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), []);
}
