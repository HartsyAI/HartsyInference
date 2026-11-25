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

        Assert.Equal(VideoFeatures.InitImage | VideoFeatures.DrivingAudio | VideoFeatures.LongFormChain, avatar.Supports);
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
        // The whole-token term (14 × hidden per token) and one 4096-row tile of the QKV and FFN intermediates (3 × hidden + 3 × FFN per row).
        Assert.Equal(LongCatRecipeSupport.FixedPoolBytes + LongCatRecipeSupport.SafetyBytes
            + 37_440L * 14 * 4096 * 4 + 4096L * (3 * 4096 + 3 * 11008) * 4, headroom);
        Assert.InRange(headroom, 15L << 30, 16L << 30);
    }

    [Fact]
    public void TheTiledIntermediatesStopGrowingPastOneTile()
    {
        // Past one tile, adding tokens adds only the whole-token term: the FFN and QKV intermediates are bounded by the tile.
        long at50k = LongCatRecipeSupport.ActivationHeadroomBytes(50_000, hidden: 4096, ffnHidden: 11008);
        long at100k = LongCatRecipeSupport.ActivationHeadroomBytes(100_000, hidden: 4096, ffnHidden: 11008);

        Assert.Equal(50_000L * 14 * 4096 * 4, at100k - at50k);
    }

    [Theory]
    // 640 x 368 x 49: 13 latent frames on a 46 x 80 grid, so 13 x 23 x 40 = 11,960 tokens. The 4090 small-config Auto run logged 22,847 MB free before
    // the shared preload; 16,700 MiB free after it is assumed here (the planner does not log that figure yet). Headroom 9,980 MiB gives 21 blocks.
    [InlineData(13, 46, 80, 16_700, 21)]
    // 832 x 480 x 93: 24 latent frames on a 60 x 104 grid, so 37,440 tokens. Headroom 15,769 MiB: the prefix is 0 only below about 16.1 GiB free after
    // the shared weights. The full run logged 22,852 MB free before the shared preload; the prefix-0 result pins the after-shared figure only by this bound.
    [InlineData(24, 60, 104, 14_000, 0)]
    public void TheResidentPrefixMatchesTheCalibrationForTheFreeMemoryAfterTheSharedWeights(int latentFrames, int latentHeight, int latentWidth,
        int freeAfterSharedMib, int expectedPrefix)
    {
        // The planner reads free memory after the shared weights are preloaded; blocks are 313 MiB across 48.
        const long blockBytes = 313L << 20;
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(
            LongCatRecipeSupport.TokenCount(latentFrames, latentHeight, latentWidth), hidden: 4096, ffnHidden: 11008);

        int prefix = ResidentPrefixSizing.Size((long)freeAfterSharedMib << 20, headroom, blockBytes, blockCount: 48);

        Assert.Equal(expectedPrefix, prefix);
    }

    private static CheckpointProbe EmptyProbe() =>
        new(new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, TensorShape>(StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal), []);
}
