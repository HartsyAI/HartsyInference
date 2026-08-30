using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class H3ExpansionReleaseGateTests
{
    [Theory]
    [InlineData(VideoAccelerationKind.Turbo, VideoTaskFamily.Fl2Va, null)]
    [InlineData(VideoAccelerationKind.Pdd, VideoTaskFamily.Fl2Va, null)]
    [InlineData(VideoAccelerationKind.None, VideoTaskFamily.Hybrid, null)]
    [InlineData(VideoAccelerationKind.None, VideoTaskFamily.Fl2Va, "control")]
    [InlineData(VideoAccelerationKind.None, VideoTaskFamily.Fl2Va, "int8-vae")]
    public void ValidationPendingExpansionPaths_AreBlockedByDefault(
        VideoAccelerationKind acceleration,
        VideoTaskFamily task,
        string? component)
    {
        VideoRequest request = Request(component);
        VideoPlan plan = Plan(acceleration, task, component, request);

        using (KnobProfile.Create("h3-expansion-default-gate")
                   .With(EngineKnobs.H3ExpansionExperimental, false)
                   .Push())
        {
            VideoPlan gated = VideoService.ApplyExperimentalH3ExpansionGate(plan, request);

            VideoPlanIssue issue = Assert.Single(gated.Issues,
                candidate => candidate.Code == "video.h3_expansion.experimental_disabled");
            Assert.Equal(VideoPlanIssueSeverity.Error, issue.Severity);
            Assert.False(gated.IsValid);
        }
    }

    [Fact]
    public void DenseGuidesAndMasks_RemainAvailableWithoutExperimentalOptIn()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            Guides =
            [
                new VideoGuide
                {
                    FrameIndex = 0,
                    Image = new ImageData { Width = 1, Height = 1, Rgb = [0, 0, 0] },
                },
            ],
            VideoDenoiseMask = new VideoDenoiseMask
            {
                MaskImage = new ImageData { Width = 1, Height = 1, Rgb = [255, 255, 255] },
            },
            AudioDenoiseMask = new AudioDenoiseMask
            {
                Values = [1f],
            },
        };
        VideoPlan plan = Plan(VideoAccelerationKind.None, VideoTaskFamily.Fl2Va, null, request);

        using (KnobProfile.Create("h3-expansion-dense-gate")
                   .With(EngineKnobs.H3ExpansionExperimental, false)
                   .Push())
        {
            VideoPlan gated = VideoService.ApplyExperimentalH3ExpansionGate(plan, request);
            Assert.DoesNotContain(gated.Issues,
                candidate => candidate.Code == "video.h3_expansion.experimental_disabled");
            Assert.True(gated.IsValid);
        }
    }

    [Fact]
    public void ScopedExperimentalOptIn_OpensOnlyTheTemporaryExpansionGate()
    {
        VideoRequest request = Request("control");
        VideoPlan plan = Plan(VideoAccelerationKind.Pdd, VideoTaskFamily.Hybrid, "int8-vae", request);

        using (KnobProfile.Create("h3-expansion-controlled-validation")
                   .With(EngineKnobs.H3ExpansionExperimental, true)
                   .Push())
        {
            VideoPlan enabled = VideoService.ApplyExperimentalH3ExpansionGate(plan, request);
            Assert.DoesNotContain(enabled.Issues,
                candidate => candidate.Code == "video.h3_expansion.experimental_disabled");
            Assert.True(enabled.IsValid);
        }
    }

    [Fact]
    public void H3Plan_ProducesAnExactExecutionSummary()
    {
        VideoRequest request = new VideoRequest { Prompt = "test" };
        VideoPlan plan = Plan(VideoAccelerationKind.None, VideoTaskFamily.Fl2Va, null, request);

        VideoExecutionSummary summary = Assert.IsType<VideoExecutionSummary>(
            VideoService.BuildExecutionSummary(plan));
        Assert.Equal("test-h3-profile", summary.ProfileId);
        Assert.Equal(12f, summary.FlowShift);
        Assert.Equal(3f, summary.AudioFlowShift);
        Assert.Equal("euler", summary.Sampler);
        Assert.Equal("normal", summary.Scheduler);
    }

    private static VideoRequest Request(string? component) => new VideoRequest
    {
        Prompt = "test",
        Controls = component == "control"
            ?
            [
                new VideoControl
                {
                    Model = "control.safetensors",
                    Video = new VideoClip { Data = [1] },
                },
            ]
            : null,
    };

    private static VideoPlan Plan(
        VideoAccelerationKind acceleration,
        VideoTaskFamily task,
        string? component,
        VideoRequest request)
    {
        VideoDefaults defaults = new VideoDefaults
        {
            Steps = 30,
            CfgScale = 1f,
            Width = 512,
            Height = 288,
            Frames = 39,
            Fps = 24,
            FlowShift = 12f,
            AudioFlowShift = 3f,
            Sampler = "euler",
            Scheduler = "normal",
        };
        return new VideoPlan
        {
            SourceRequest = request,
            Model = new ModelSpec
            {
                Requested = "model.safetensors",
                LocalPath = "model.safetensors",
                Modality = Modality.Video,
            },
            Profile = new VideoModelProfile
            {
                Id = "test-h3-profile",
                DisplayName = "Test H3 profile",
                FamilyId = "minimax-h3",
                Task = task,
                Acceleration = acceleration,
                Attention = VideoAttentionKind.Dense,
                Defaults = defaults,
                Features = VideoFeatures.Guides
                    | VideoFeatures.VideoDenoiseMask
                    | VideoFeatures.AudioDenoiseMask
                    | VideoFeatures.VideoControlNet,
            },
            EffectiveSettings = new VideoEffectiveSettings
            {
                Width = 512,
                Height = 288,
                Frames = 39,
                Fps = 24,
                Steps = 30,
                CfgScale = 1f,
                FlowShift = 12f,
                AudioFlowShift = 3f,
                Sampler = "euler",
                Scheduler = "normal",
                Seed = 1,
                ReferenceSizing = VideoReferenceSizing.Native,
                LockedFields = VideoLockedFields.None,
            },
            Issues = [],
            CacheIdentity = "test",
            ComponentFormats = component == "int8-vae"
                ? new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["videoVae"] = "h3-video-vae-int8-convrot",
                }
                : new Dictionary<string, string>(StringComparer.Ordinal),
        };
    }
}
