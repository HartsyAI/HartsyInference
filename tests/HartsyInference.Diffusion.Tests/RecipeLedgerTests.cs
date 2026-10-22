using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using Xunit;
using HartsyInference.Engine.Catalog;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The recipe registries are the single source of model defaults: every built-in recipe and variant must
/// declare a complete, valid set, filling only what a caller left unset, and planning must run exactly those values.
/// Each of these fails silently in production: a broken default just generates the wrong clip.</summary>
public sealed class RecipeLedgerTests
{
    public static TheoryData<string> ImageFamilies => [.. RecipeRegistry.DefaultNames];

    public static TheoryData<string> VideoFamilies => [.. VideoRecipeRegistry.DefaultNames];

    [Theory]
    [MemberData(nameof(ImageFamilies))]
    public void ImageRecipe_DeclarationsAreValid(string family) =>
        Assert.Empty(RecipeRegistryValidator.Problems(RecipeRegistry.Resolve(family)!));

    [Theory]
    [MemberData(nameof(VideoFamilies))]
    public void VideoRecipe_DeclarationsAreValid(string family) =>
        Assert.Empty(RecipeRegistryValidator.Problems(VideoRecipeRegistry.Resolve(family)!));

    [Theory]
    [MemberData(nameof(ImageFamilies))]
    public void ImageDefaults_FillOnlyUnsetValues(string family)
    {
        IArchitectureRecipe recipe = RecipeRegistry.Resolve(family)!;
        foreach (ImageDefaults defaults in DefaultsOf(recipe.Variants, recipe.DefaultsFor))
        {
            ImageRequest filled = defaults.Apply(new ImageRequest { Prompt = "p" });
            Assert.Equal((defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height),
                (filled.Steps, filled.CfgScale, filled.Width, filled.Height));
            Assert.Equal((defaults.Sampler, defaults.Scheduler, defaults.SigmaShift), (filled.Sampler, filled.Scheduler, filled.SigmaShift));

            ImageRequest explicitRequest = new ImageRequest
            {
                Prompt = "p", Steps = defaults.Steps + 3, CfgScale = defaults.CfgScale + 0.5f,
                Width = defaults.Width + 64, Height = defaults.Height + 64, Sampler = "user-sampler",
                Scheduler = "user-scheduler", SigmaShift = 9.25,
            };
            Assert.Equal(explicitRequest, defaults.Apply(explicitRequest));
        }
    }

    [Theory]
    [MemberData(nameof(VideoFamilies))]
    public void VideoDefaults_FillOnlyUnsetValues(string family)
    {
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(family)!;
        foreach (VideoDefaults defaults in DefaultsOf(recipe.Variants, recipe.DefaultsFor))
        {
            VideoRequest filled = defaults.Apply(new VideoRequest { Prompt = "p" });
            Assert.Equal((defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height, defaults.Frames, defaults.Fps),
                (filled.Steps, filled.CfgScale, filled.Width, filled.Height, filled.Frames, filled.Fps));

            VideoRequest explicitRequest = new VideoRequest
            {
                Prompt = "p", Steps = defaults.Steps + 3, CfgScale = defaults.CfgScale + 0.5f,
                Width = defaults.Width + 64, Height = defaults.Height + 64, Frames = defaults.Frames + 8,
                Fps = defaults.Fps + 6, FlowShift = 2.5f, AudioFlowShift = 1.5f, Sampler = "user-sampler",
                Scheduler = "user-scheduler", ReferenceSizing = VideoReferenceSizing.Native,
            };
            Assert.Equal(explicitRequest, defaults.Apply(explicitRequest));
        }
    }

    /// <summary>Every family planned without a checkpoint profile runs exactly its recipe's defaults; H3 is excluded
    /// because its exact-hash profiles legitimately overlay them.</summary>
    [Theory]
    [MemberData(nameof(VideoFamilies))]
    public async Task GenericVideoPlan_RunsRecipeDefaults(string family)
    {
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(family)!;
        if (recipe is Engine.Recipes.Video.MiniMaxH3Recipe)
        {
            return;
        }
        ModelSpec spec = new ModelSpec { Requested = family, Modality = Modality.Video };
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(spec, new VideoRequest { Prompt = "p" },
            VideoRecipeDeclaration.For(recipe, family, null, recipe.Supports), CancellationToken.None);

        VideoEffectiveSettings effective = plan.EffectiveSettings;
        VideoDefaults defaults = recipe.Defaults;
        Assert.Equal((defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height, defaults.Frames, defaults.Fps),
            (effective.Steps, effective.CfgScale, effective.Width, effective.Height, effective.Frames, effective.Fps));
    }

    [Fact]
    public void UnregisteredVideoFamily_HasNoDefaults()
    {
        // SDXL resolves to a real family that deliberately has no video recipe.
        ModelSpec spec = new ModelSpec { Requested = "sdxl", Modality = Modality.Video };

        RecipeNotRegisteredException error = Assert.Throws<RecipeNotRegisteredException>(() => new RecipeCatalog().Describe(spec));
        Assert.Equal("sdxl", error.FamilyId);
        Assert.Contains("wan", error.Drivable);
    }

    [Fact]
    public void Validator_RejectsValuesTheTypesAllow()
    {
        StubVideoRecipe recipe = new StubVideoRecipe("stub", new VideoDefaults
        {
            Steps = 0, CfgScale = float.NaN, Width = 700, Height = 480, Frames = 26, Fps = 0, Sampler = " ",
        });

        IReadOnlyList<string> problems = RecipeRegistryValidator.Problems(recipe);

        Assert.Contains(problems, problem => problem.Contains("0 steps", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("guidance scale", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("700x480", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("frame count 26", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("frame rate 0", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("sampler is blank", StringComparison.Ordinal));
    }

    [Fact]
    public void Validator_ReportsEveryProblemWhenTheGridIsMissing()
    {
        StubVideoRecipe recipe = new StubVideoRecipe("stub-no-grid", new VideoDefaults
        {
            Steps = 20, CfgScale = 5f, Width = 832, Height = 480, Frames = 33, Fps = 0,
        }, hasGrid: false);

        IReadOnlyList<string> problems = RecipeRegistryValidator.Problems(recipe);

        Assert.Contains(problems, problem => problem.Contains("declares no frame grid", StringComparison.Ordinal));
        Assert.Contains(problems, problem => problem.Contains("frame rate 0", StringComparison.Ordinal));
    }

    [Fact]
    public void Validator_RejectsAShadowedName()
    {
        VideoDefaults valid = new VideoDefaults { Steps = 20, CfgScale = 5f, Width = 832, Height = 480, Frames = 33, Fps = 16 };
        StubVideoRecipe greedy = new StubVideoRecipe("wan-x", valid, matchesPrefix: true);
        StubVideoRecipe shadowed = new StubVideoRecipe("wan-x-2", valid);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => RecipeRegistryValidator.ThrowIfInvalid([greedy, shadowed]));
        Assert.Contains("wan-x-2: shadowed by the earlier video recipe 'wan-x'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Register_RefusesAnInvalidRecipe() =>
        Assert.Throws<ArgumentException>(() => VideoRecipeRegistry.Register(new StubVideoRecipe("stub-invalid", new VideoDefaults
        {
            Steps = 20, CfgScale = 5f, Width = 832, Height = 480, Frames = 34, Fps = 16,
        })));

    [Theory]
    [InlineData(1, true)]
    [InlineData(33, true)]
    [InlineData(34, false)]
    [InlineData(0, false)]
    [InlineData(125, false)]
    public void FrameGrid_ContainsOnlyCountsOnTheGridUpToItsCap(int frames, bool expected) =>
        Assert.Equal(expected, FrameGrid.StepPlusOne(4, FrameNormalization.Reject, max: 121).Contains(frames));

    [Fact]
    public void FrameGrid_RefusesACapOffItsOwnGrid() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => FrameGrid.StepPlusOne(4, FrameNormalization.Reject, max: 120));

    /// <summary>The family-level defaults, then each declared variant's, resolved as if the caller had named it.</summary>
    private static IEnumerable<T> DefaultsOf<T>(ModelVariantCatalog? catalog, Func<ResolvedModelVariant?, T> defaultsFor)
    {
        yield return defaultsFor(null);
        foreach (ModelVariant variant in catalog?.Variants ?? [])
        {
            yield return defaultsFor(new ResolvedModelVariant(catalog!.FamilyId, variant, ModelVariantSource.CallerHint, "test"));
        }
    }

    private static readonly FrameGrid StubGrid = FrameGrid.StepPlusOne(4, FrameNormalization.Reject);

    private sealed class StubVideoRecipe(string name, VideoDefaults defaults, bool matchesPrefix = false,
        FrameGrid? grid = null, bool hasGrid = true) : IVideoRecipe
    {
        public string Name => name;

        public VideoDefaults Defaults => defaults;

        // A missing grid can only reach the validator from a recipe that bypasses the required member.
        public VideoConstraints Constraints { get; } = hasGrid ? new VideoConstraints { Frames = grid ?? StubGrid } : null!;

        public bool Matches(string familyId) => matchesPrefix
            ? familyId.StartsWith(name, StringComparison.Ordinal)
            : string.Equals(familyId, name, StringComparison.Ordinal);

        public IVideoRecipePipeline Construct(RecipeContext context) =>
            throw new NotSupportedException("Declaration-only stub.");
    }
}
