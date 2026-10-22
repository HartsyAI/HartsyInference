using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Planning owns frame normalization and records where every effective value came from. Moving the snap out
/// of the pipelines must not move a single frame count, and a wrong source label would mislead every host that shows
/// it, so both are pinned here.</summary>
public sealed class PlanProvenanceTests
{
    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void SnapNearest_MatchesThePipelinesFormerRounding(int step)
    {
        FrameGrid grid = FrameGrid.StepPlusOne(step, FrameNormalization.SnapNearest);
        for (int requested = 1; requested <= 400; requested++)
        {
            // VideoRecipeUtils.ResolveFrames, which every step·n+1 pipeline applied before planning did.
            int former = Math.Max(1, 1 + (int)Math.Round((requested - 1) / (double)step) * step);
            Assert.Equal(former, grid.Normalize(requested));
        }
    }

    [Fact]
    public void SnapUp_MatchesMiniMaxH3Alignment()
    {
        FrameGrid grid = new MiniMaxH3Recipe().Constraints.Frames;
        for (int requested = 1; requested <= 400; requested++)
        {
            Assert.Equal(MiniMaxH3Geometry.AlignFrameCount(requested), grid.Normalize(requested));
        }
    }

    [Fact]
    public void Snapping_PastTheCap_LandsOnTheCap()
    {
        FrameGrid lance = new LanceVideoRecipe().Constraints.Frames;
        for (int requested = 1; requested <= 400; requested++)
        {
            // LanceVideoRecipePipeline's former Math.Min(MaxFrames, ResolveFrames(...)).
            int former = Math.Min(121, Math.Max(1, 1 + (int)Math.Round((requested - 1) / 4.0) * 4));
            Assert.Equal(former, lance.Normalize(requested));
        }
    }

    [Fact]
    public void Reject_RefusesOffGridCountsOnly()
    {
        FrameGrid grid = FrameGrid.StepPlusOne(4, FrameNormalization.Reject);
        Assert.Equal(49, grid.Normalize(49));
        Assert.Null(grid.Normalize(50));
    }

    [Fact]
    public async Task UnsetValues_AreRecipeDefaults_AndTheSeedIsDerived()
    {
        WanVideoRecipe recipe = new WanVideoRecipe();
        VideoPlan plan = await Plan(recipe, new VideoRequest { Prompt = "p" });

        Assert.Equal(new RecipeIdentity { FamilyId = "wan", Revision = recipe.Defaults.Revision }, plan.Recipe);
        foreach (string name in new[] { "Width", "Height", "Frames", "Fps", "Steps", "CfgScale" })
        {
            Assert.Equal(ParameterSource.RecipeDefault, plan.Provenance[name].Source);
        }
        Assert.Equal("33", plan.Provenance["Frames"].Value);
        Assert.Equal(ParameterSource.Derived, plan.Provenance["Seed"].Source);
        Assert.Null(plan.Provenance["Sampler"].Value);
    }

    [Fact]
    public async Task ExplicitValues_AreUserOverrides_AndOffGridFramesAreNormalized()
    {
        VideoPlan plan = await Plan(new WanVideoRecipe(), new VideoRequest
        {
            Prompt = "p", Steps = 12, Fps = 16, Frames = 50, Seed = 7,
        });

        Assert.Equal(ParameterSource.UserOverride, plan.Provenance["Steps"].Source);
        Assert.Equal(ParameterSource.UserOverride, plan.Provenance["Fps"].Source);
        Assert.Equal(ParameterSource.UserOverride, plan.Provenance["Seed"].Source);
        Assert.Equal(49, plan.EffectiveSettings.Frames);
        Assert.Equal(new ResolvedParameter("49", ParameterSource.Normalized, "requested 50"), plan.Provenance["Frames"]);
        Assert.True(plan.IsValid);
    }

    [Fact]
    public async Task AFamilyThatRejectsOffGridFrames_RefusesThePlan()
    {
        VideoRecipeDeclaration declaration = new VideoRecipeDeclaration
        {
            FamilyId = "strict-test",
            Defaults = new VideoDefaults { Steps = 20, CfgScale = 5f, Width = 832, Height = 480, Frames = 33, Fps = 16 },
            Features = VideoFeatures.None,
            Constraints = new VideoConstraints { Frames = FrameGrid.StepPlusOne(4, FrameNormalization.Reject) },
        };

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(Spec("strict-test"),
            new VideoRequest { Prompt = "p", Frames = 50 }, declaration, CancellationToken.None);

        VideoPlanIssue issue = Assert.Single(plan.Issues, item => item.Code == "video.frames.off_grid");
        Assert.Equal(VideoPlanIssueSeverity.Error, issue.Severity);
        Assert.Contains("49 and 53", issue.Message, StringComparison.Ordinal);
        Assert.False(plan.IsValid);
    }

    [Fact]
    public void Describe_NamesTheRecipeAndEveryParameterInOrder()
    {
        RecipeIdentity recipe = new RecipeIdentity { FamilyId = "wan", Revision = 2 };
        Dictionary<string, ResolvedParameter> provenance = new Dictionary<string, ResolvedParameter>(StringComparer.Ordinal)
        {
            ["Steps"] = new ResolvedParameter("30", ParameterSource.UserOverride),
            ["Frames"] = new ResolvedParameter("81", ParameterSource.RecipeDefault),
        };

        Assert.Equal("wan r2: Frames=81 (RecipeDefault), Steps=30 (UserOverride)",
            ParameterProvenance.Describe(recipe, provenance));
    }

    private static Task<VideoPlan> Plan(IVideoRecipe recipe, VideoRequest request) =>
        VideoProfileResolver.ResolveAsync(Spec(recipe.Name), request,
            VideoRecipeDeclaration.For(recipe, recipe.Name, null, recipe.Supports), CancellationToken.None);

    private static ModelSpec Spec(string family) => new ModelSpec { Requested = family, Modality = Modality.Video };
}
