using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Request-surface gates for long-form chaining. It drives the denoise masks from inside the pipeline
/// rather than through the request, so it inherits neither their feature gate nor their release gate.</summary>
public class MiniMaxH3ChainRequestTests
{
    [Fact]
    public void ChainTotalFramesIsARequestedFeature()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", ChainTotalFrames = 900 };
        Assert.True(VideoService.RequestedFeatures(request).HasFlag(VideoFeatures.LongFormChain));
    }

    [Fact]
    public void AnOrdinaryRequestDoesNotAskForChaining()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", Frames = 141 };
        Assert.False(VideoService.RequestedFeatures(request).HasFlag(VideoFeatures.LongFormChain));
    }

    /// <summary>The recipe is the release authority the profile's features are intersected against.</summary>
    [Fact]
    public void TheH3RecipeDeclaresChaining()
        => Assert.True(new MiniMaxH3Recipe().Supports.HasFlag(VideoFeatures.LongFormChain));

    /// <summary>Chaining cleared its own gate, and must not re-acquire one from the masks it drives internally.</summary>
    [Fact]
    public void ChainingIsReleased()
    {
        VideoRequest request = new VideoRequest { Prompt = "test", ChainTotalFrames = 900 };
        Assert.Null(request.VideoDenoiseMask);
        Assert.Null(request.AudioDenoiseMask);
        Assert.Null(request.Guides);

        Assert.DoesNotContain(
            VideoService.ApplyH3ExpansionReleaseGate(H3PlanFixture.Plan(request), request).Issues,
            candidate => candidate.Code == "video.h3_expansion.release_blocked");
    }

    /// <summary>Clearing chaining must not have widened the gate for the surfaces still pending.</summary>
    [Fact]
    public void GuidesAndMasksStayBlocked()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            VideoDenoiseMask = new VideoDenoiseMask { MaskFrameValues = [1f] },
        };
        VideoPlanIssue issue = Assert.Single(
            VideoService.ApplyH3ExpansionReleaseGate(H3PlanFixture.Plan(request), request).Issues,
            candidate => candidate.Code == "video.h3_expansion.release_blocked");
        Assert.Contains("AV denoise masks", issue.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("long-form chaining", issue.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnUnchainedRequestIsNotBlockedByTheChainEntry()
    {
        VideoRequest request = new VideoRequest { Prompt = "test" };
        Assert.DoesNotContain(
            VideoService.ApplyH3ExpansionReleaseGate(H3PlanFixture.Plan(request), request).Issues,
            candidate => candidate.Code == "video.h3_expansion.release_blocked");
    }

    /// <summary>An off-grid context must fail at plan time rather than inside the sampler.</summary>
    [Theory]
    [InlineData(40)]
    [InlineData(38)]
    [InlineData(4)]
    public void OffGridContextIsRefusedByThePlanner(int contextFrames)
        => Assert.Throws<ArgumentOutOfRangeException>(
            () => MiniMaxH3ChainPlanner.Plan(900, contextFrames));

    /// <summary>Per-segment length and total together decide the segment count.</summary>
    [Fact]
    public void SegmentCountFollowsPerSegmentLength()
    {
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> longSegments =
            MiniMaxH3ChainPlanner.Plan(900, MiniMaxH3ChainPlanner.DefaultContextFrames, 362);
        IReadOnlyList<MiniMaxH3ChainPlanner.Segment> shortSegments =
            MiniMaxH3ChainPlanner.Plan(900, MiniMaxH3ChainPlanner.DefaultContextFrames, 141);
        Assert.True(shortSegments.Count > longSegments.Count,
            "Shorter segments must need more of them to cover the same total.");
        Assert.All(shortSegments, segment => Assert.True(segment.FrameCount <= 141));
    }
}

/// <summary>A minimal planned H3 <see cref="VideoPlan"/>, enough for the release gate to inspect.</summary>
internal static class H3PlanFixture
{
    internal static VideoPlan Plan(VideoRequest request) => new VideoPlan
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
            Task = VideoTaskFamily.Fl2Va,
            Acceleration = VideoAccelerationKind.None,
            Attention = VideoAttentionKind.Dense,
            Defaults = new VideoDefaults
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
            },
            Features = VideoFeatures.LongFormChain,
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
        ComponentFormats = new Dictionary<string, string>(StringComparer.Ordinal),
    };
}
