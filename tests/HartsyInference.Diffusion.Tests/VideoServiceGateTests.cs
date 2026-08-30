using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the video feature gate itself: a request carrying conditioning a family never consumes must throw,
/// not silently drop it. The declaration side is pinned by <see cref="VideoFeatureDeclarationTests"/>.</summary>
public sealed class VideoServiceGateTests
{
    private static readonly byte[] TinyClip = [0x00];

    private static VideoRequest Request() => new VideoRequest { Prompt = "gate test" };

    /// <summary>A catalog-backed spec so family resolution never touches disk-based architecture detection.</summary>
    private static ModelSpec Spec(string family) => new ModelSpec
    {
        Requested = family,
        Modality = Modality.Video,
        Catalog = new CatalogEntry
        {
            Id = family,
            Modality = Modality.Video,
            DisplayName = family,
            Architecture = family,
            Status = ModelStatus.Verified,
        },
    };

    [Fact]
    public void RequestedFeatures_MapsEachConditioningObjectToItsBit()
    {
        Assert.Equal(VideoFeatures.None, VideoService.RequestedFeatures(Request()));
        Assert.Equal(VideoFeatures.ReferenceImages, VideoService.RequestedFeatures(Request() with
        {
            ReferenceImages = [new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 }],
        }));
        Assert.Equal(VideoFeatures.ReferenceVideos, VideoService.RequestedFeatures(Request() with
        {
            ReferenceVideos = [new ReferenceVideo { Video = new VideoClip { Data = TinyClip } }],
        }));
        Assert.Equal(VideoFeatures.ReferenceAudios, VideoService.RequestedFeatures(Request() with
        {
            ReferenceAudios = [new AudioClip { Data = TinyClip }],
        }));
        Assert.Equal(VideoFeatures.DrivingVideo, VideoService.RequestedFeatures(Request() with
        {
            DrivingVideo = new VideoClip { Data = TinyClip },
        }));
        Assert.Equal(VideoFeatures.DrivingVideo, VideoService.RequestedFeatures(Request() with
        {
            DrivingPoseVideo = new VideoClip { Data = TinyClip },
        }));
    }

    /// <summary>Empty reference lists are "not requested" — an empty list must not trip the gate.</summary>
    [Fact]
    public void RequestedFeatures_IgnoresEmptyReferenceLists()
    {
        VideoRequest request = Request() with { ReferenceImages = [], ReferenceVideos = [], ReferenceAudios = [] };
        Assert.Equal(VideoFeatures.None, VideoService.RequestedFeatures(request));
    }

    [Fact]
    public async Task PlanningRejectsReferenceImagesOnNonDeclaringFamily()
    {
        VideoRequest request = Request() with
        {
            ReferenceImages = [new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 }],
        };
        VideoPlan plan = await PlanGenericAsync("wan-22-5b", request);
        VideoPlanIssue issue = Assert.Single(plan.Issues, item => item.Code == "video.feature.unsupported");
        Assert.Contains("ReferenceImages", issue.Message, StringComparison.Ordinal);
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task PlanningRejectsDrivingVideoOnNonDeclaringFamily()
    {
        VideoRequest request = Request() with { DrivingVideo = new VideoClip { Data = TinyClip } };
        VideoPlan plan = await PlanGenericAsync("wan-22-5b", request);
        VideoPlanIssue issue = Assert.Single(plan.Issues, item => item.Code == "video.feature.unsupported");
        Assert.Contains("DrivingVideo", issue.Message, StringComparison.Ordinal);
        Assert.False(plan.IsValid);
    }

    [Fact]
    public void RequestedFeaturesAcceptsReferencesDeclaredByMiniMaxH3()
    {
        VideoRequest request = Request() with
        {
            ReferenceImages = [new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 }],
            ReferenceVideos = [new ReferenceVideo { Video = new VideoClip { Data = TinyClip } }],
            ReferenceAudios = [new AudioClip { Data = TinyClip }],
        };
        VideoFeatures missing = VideoService.RequestedFeatures(request) & ~new MiniMaxH3Recipe().Supports;
        Assert.Equal(VideoFeatures.None, missing);
    }

    private static Task<VideoPlan> PlanGenericAsync(string family, VideoRequest request)
    {
        ModelSpec spec = Spec(family);
        VideoDefaults defaults = InferenceEngine.VideoDefaultsFor(spec);
        VideoFeatures features = InferenceEngine.SupportedVideoFeatures(spec);
        return VideoProfileResolver.ResolveAsync(spec, request, family, defaults, features, CancellationToken.None);
    }
}
