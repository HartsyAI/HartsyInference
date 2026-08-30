using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class GenericVideoPlanningCompatibilityTests
{
    [Fact]
    public async Task OmittedSolverAndShiftFields_RemainNullForLegacyFamilies()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", Seed = (1L << 31) + 17 };
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(
            Spec(), request, "legacy-test", LegacyDefaults(), VideoFeatures.None, CancellationToken.None);

        Assert.Null(plan.EffectiveSettings.FlowShift);
        Assert.Null(plan.EffectiveSettings.AudioFlowShift);
        Assert.Null(plan.EffectiveSettings.Sampler);
        Assert.Null(plan.EffectiveSettings.Scheduler);
        Assert.Equal(17, plan.EffectiveSettings.Seed);

        VideoRequest effective = plan.EffectiveSettings.Apply(request);
        Assert.Null(effective.FlowShift);
        Assert.Null(effective.AudioFlowShift);
        Assert.Null(effective.Sampler);
        Assert.Null(effective.Scheduler);
        Assert.Equal(17, effective.Seed);
        Assert.Null(VideoService.BuildExecutionSummary(plan));
    }

    [Fact]
    public async Task ExplicitLegacySolverAndShiftFields_ArePreserved()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            FlowShift = 5f,
            AudioFlowShift = 2f,
            Sampler = "unipc",
            Scheduler = "custom",
            Seed = 123,
        };
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(
            Spec(), request, "legacy-test", LegacyDefaults(), VideoFeatures.None, CancellationToken.None);

        Assert.Equal(5f, plan.EffectiveSettings.FlowShift);
        Assert.Equal(2f, plan.EffectiveSettings.AudioFlowShift);
        Assert.Equal("unipc", plan.EffectiveSettings.Sampler);
        Assert.Equal("custom", plan.EffectiveSettings.Scheduler);
        Assert.Equal(123, plan.EffectiveSettings.Seed);
    }

    [Fact]
    public void StandardDefaults_DoNotInventCrossFamilySamplingSemantics()
    {
        VideoRequest effective = VideoDefaults.Standard.Apply(new VideoRequest { Prompt = "test" });

        Assert.Null(effective.FlowShift);
        Assert.Null(effective.AudioFlowShift);
        Assert.Null(effective.Sampler);
        Assert.Null(effective.Scheduler);
    }

    [Theory]
    [InlineData(SparseAttentionPolicy.Auto)]
    [InlineData(SparseAttentionPolicy.Disable)]
    public async Task DenseLegacyFamilies_PreserveAutoAndDisableSparsePolicies(
        SparseAttentionPolicy policy)
    {
        VideoRequest request = new VideoRequest { Prompt = "test", SparseAttentionPolicy = policy };

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(
            Spec(), request, "legacy-test", LegacyDefaults(), VideoFeatures.None, CancellationToken.None);

        Assert.True(plan.IsValid);
        Assert.DoesNotContain(plan.Issues,
            issue => issue.Field == nameof(VideoRequest.SparseAttentionPolicy));
    }

    [Fact]
    public async Task DenseLegacyFamilies_RejectExplicitSparseRequirementDuringPlanning()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            SparseAttentionPolicy = SparseAttentionPolicy.Require,
        };

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(
            Spec(), request, "legacy-test", LegacyDefaults(), VideoFeatures.None, CancellationToken.None);

        VideoPlanIssue issue = Assert.Single(plan.Issues,
            issue => issue.Code == "video.vsa.profile_required");
        Assert.Equal(VideoPlanIssueSeverity.Error, issue.Severity);
        Assert.Equal(nameof(VideoRequest.SparseAttentionPolicy), issue.Field);
        Assert.False(plan.IsValid);
    }

    private static ModelSpec Spec() => new ModelSpec
    {
        Requested = "legacy-test",
        LocalPath = "legacy-test.safetensors",
        Modality = Modality.Video,
    };

    private static VideoDefaults LegacyDefaults() => new VideoDefaults
    {
        Steps = 20,
        CfgScale = 1f,
        Width = 512,
        Height = 288,
        Frames = 25,
        Fps = 24,
    };
}
