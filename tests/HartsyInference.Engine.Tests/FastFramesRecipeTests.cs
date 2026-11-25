using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The FastFrames recipe's declarations, its refusals and the catalog pins, checked without weights. The real-weight
/// run is a separate lane (see PARITY_VERIFICATION.md).</summary>
public sealed class FastFramesRecipeTests
{
    [Theory]
    [InlineData("fastframes")]
    [InlineData("fastframes-s")]
    [InlineData("FastFrames-L")]
    public void ItMatchesEveryCatalogIdItShips(string familyId) =>
        Assert.True(new FastFramesRecipe().Matches(familyId));

    [Fact]
    public void ItDoesNotClaimAWanFamily() =>
        Assert.False(new FastFramesRecipe().Matches("wan-21-14b"));

    [Fact]
    public void TheRegistryResolvesFastFramesToItsRecipe()
    {
        Assert.IsType<FastFramesRecipe>(VideoRecipeRegistry.Resolve("fastframes"));
        Assert.IsType<FastFramesRecipe>(VideoRecipeRegistry.Resolve("fastframes-l"));
    }

    [Fact]
    public void TheDefaultsAreUpstreamsAndLieOnTheFrameGrid()
    {
        FastFramesRecipe recipe = new();
        Assert.Equal(832, recipe.Defaults.Width);
        Assert.Equal(480, recipe.Defaults.Height);
        Assert.Equal(81, recipe.Defaults.Frames);
        Assert.Equal(16, recipe.Defaults.Fps);
        Assert.Equal(FastFramesRecipe.NoGuidance, recipe.Defaults.CfgScale);
        Assert.True(recipe.Constraints.Frames.Contains(recipe.Defaults.Frames));
        Assert.True(recipe.Constraints.Frames.Contains(1));
        Assert.False(recipe.Constraints.Frames.Contains(80));
        Assert.False(recipe.Constraints.Frames.Contains(85));
    }

    [Fact]
    public void ItDeclaresTextToVideoOnlyAndNoNegativePrompt()
    {
        FastFramesRecipe recipe = new();
        Assert.Equal(VideoFeatures.None, ((IVideoRecipe)recipe).Supports);
        Assert.Equal(MemoryCapabilities.None, ((IVideoRecipe)recipe).MemorySupports);
        Assert.Equal("none", recipe.Prompt!.Negative);
        Assert.Equal(["t5"], recipe.Prompt.Encoders);
    }

    [Fact]
    public void ConfigForWidthPicksTheReleasedConfiguration()
    {
        Assert.Same(HartsyInference.Video.Pipelines.FastFramesConfig.S, FastFramesRecipe.ConfigFor(1536));
        Assert.Same(HartsyInference.Video.Pipelines.FastFramesConfig.L, FastFramesRecipe.ConfigFor(5120));
        Assert.Throws<NotSupportedException>(() => FastFramesRecipe.ConfigFor(2048));
    }

    [Fact]
    public void TheSamplerIsSolverOwnedUnderEveryCatalogId()
    {
        FastFramesRecipe recipe = new();
        Assert.Empty(recipe.SamplingSupportFor("fastframes-s", null).Samplers);
        Assert.Empty(recipe.SamplingSupportFor("fastframes-l", null).Samplers);
    }

    [Fact]
    public void ASizeOrLengthTheModelsCannotProduceIsRefused()
    {
        FastFramesRecipePipeline.CheckSettings(Settings(832, 480, 81));
        FastFramesRecipePipeline.CheckSettings(Settings(832, 480, 1));
        Assert.Throws<NotSupportedException>(() => FastFramesRecipePipeline.CheckSettings(Settings(640, 480, 81)));
        Assert.Throws<NotSupportedException>(() => FastFramesRecipePipeline.CheckSettings(Settings(832, 480, 80)));
        Assert.Throws<NotSupportedException>(() => FastFramesRecipePipeline.CheckSettings(Settings(832, 480, 85)));
    }

    [Fact]
    public void AFolderWithoutItsCompanionsFailsBeforeAnyDownload()
    {
        string dir = Path.Combine(Path.GetTempPath(), "fastframes-recipe-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            using CpuBackend backend = new();
            RecipeContext context = new()
            {
                CheckpointPath = Path.Combine(dir, "framewise-2step.pt"),
                Backend = backend,
            };
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => new FastFramesRecipe().Construct(context));
            Assert.Contains("kv_refiner.safetensors", error.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Theory]
    [InlineData("fastframes-s", "framewise-2step.pt", "kv_refiner.safetensors", "pixel_decoder.safetensors")]
    [InlineData("fastframes-l", "dit.safetensors", "kv_refiner.safetensors", "pixel_decoder.safetensors")]
    public void TheCatalogPinsEveryFileWithItsUpstreamHash(string id, string generator, string refiner, string decoder)
    {
        CatalogEntry? entry = ModelCatalog.Find(id);
        Assert.NotNull(entry);
        Assert.True(entry.CliDrivable);
        Assert.Equal(3, entry.Assets.Count);
        Assert.Equal(new[] { generator, refiner, decoder }, entry.Assets.Select(a => a.TargetName).ToArray());
        foreach (ModelAsset asset in entry.Assets)
        {
            Assert.NotNull(asset.Sha256);
            Assert.Matches("^[0-9a-f]{64}$", asset.Sha256);
            Assert.Equal("FastFrames/" + id, asset.TargetSubdir);
        }
    }

    private static VideoEffectiveSettings Settings(int width, int height, int frames) => new()
    {
        Width = width, Height = height, Frames = frames, Fps = 16, Steps = 4, CfgScale = 1f, Seed = 1,
        ReferenceSizing = VideoReferenceSizing.Native, LockedFields = default,
    };
}
