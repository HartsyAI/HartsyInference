using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram.Streaming;
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
    public void A16RequestIsRefusedBeforeAnyWeightIsRead()
    {
        using CpuBackend backend = new();
        ResolvedModelVariant named16 = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.6"]);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => new LongCatAvatarRecipe().Construct(new RecipeContext
        {
            CheckpointPath = "no-such-folder", Backend = backend, Variant = named16,
        }));
        Assert.Contains("1.6 INT8 catalog row", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADistilledBuildWithoutItsLoraIsRefusedBeforeAnyWeightIsRead()
    {
        using CpuBackend backend = new();
        ResolvedModelVariant unnamed = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), []);
        ResolvedModelVariant named15 = ModelVariantResolver.Classify(LongCatAvatarVariants.Catalog, EmptyProbe(), ["1.5"]);

        foreach (ResolvedModelVariant variant in new[] { unnamed, named15 })
        {
            NotSupportedException refused = Assert.Throws<NotSupportedException>(() => new LongCatAvatarRecipe().Construct(new RecipeContext
            {
                CheckpointPath = "no-such-folder/base_model_int8", Backend = backend, Variant = variant,
            }));
            Assert.Contains("dmd_lora.safetensors", refused.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TheStreamingHeadroomCoversTheFullAvatarRunsActivationsWithAMargin()
    {
        // 832 x 480 x 93: 24 latent frames on a 60 x 104 grid, so 24 x 30 x 52 = 37,440 tokens; hidden 4096, FFN 11,008.
        long tokens = LongCatRecipeSupport.TokenCount(latentFrames: 24, latentHeight: 60, latentWidth: 104);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(tokens, hidden: 4096, ffnHidden: 11008);

        Assert.Equal(37_440, tokens);
        Assert.Equal(LongCatRecipeSupport.ContextBytes + LongCatRecipeSupport.FixedPoolBytes + LongCatRecipeSupport.SafetyBytes
            + 37_440L * (17L * 4096 + 3L * 11008) * 4, headroom);
        Assert.InRange(headroom, 22L << 30, 23L << 30);
    }

    [Theory]
    // 640 x 368 x 49: 13 latent frames on a 46 x 80 grid, so 13 x 23 x 40 = 11,960 tokens.
    // The 22.8 GB card: 22,849 MiB free, the shared weights (1 GiB) preloaded before the free-memory read.
    [InlineData(13, 46, 80, 22_849, 27)]
    // 832 x 480 x 93: 24 latent frames on a 60 x 104 grid, so 37,440 tokens.
    [InlineData(24, 60, 104, 22_849, 0)]
    // The 4090 small-config Auto run: 11,960 tokens, headroom 13,182 MB, prefix 11 of 48: about 16.8 GB free once the shared weights are in.
    [InlineData(13, 46, 80, 17_800, 11)]
    public void TheResidentPrefixMatchesTheCalibrationForTheFreeMemoryAtPlanTime(int latentFrames, int latentHeight, int latentWidth,
        int freeMib, int expectedPrefix)
    {
        // The shared weights are preloaded before the free-memory read, so the planner sees free minus shared; blocks are 313 MiB across 48.
        const long sharedBytes = 1L << 30, blockBytes = 313L << 20;
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(
            LongCatRecipeSupport.TokenCount(latentFrames, latentHeight, latentWidth), hidden: 4096, ffnHidden: 11008);

        int prefix = ResidentPrefixSizing.Size(((long)freeMib << 20) - sharedBytes, headroom, blockBytes, blockCount: 48);

        Assert.Equal(expectedPrefix, prefix);
    }

    private static CheckpointProbe EmptyProbe() =>
        new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), []);
}
