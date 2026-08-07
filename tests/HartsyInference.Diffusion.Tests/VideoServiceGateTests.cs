using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
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
    public void RejectUnsupported_ThrowsForReferenceImagesOnNonDeclaringFamily()
    {
        VideoRequest request = Request() with
        {
            ReferenceImages = [new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 }],
        };
        NotSupportedException ex = Assert.Throws<NotSupportedException>(
            () => VideoService.RejectUnsupported(Spec("wan-22-5b"), request));
        Assert.Contains("ReferenceImages", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectUnsupported_ThrowsForDrivingVideoOnNonDeclaringFamily()
    {
        VideoRequest request = Request() with { DrivingVideo = new VideoClip { Data = TinyClip } };
        Assert.Throws<NotSupportedException>(() => VideoService.RejectUnsupported(Spec("wan-22-5b"), request));
    }

    [Fact]
    public void RejectUnsupported_AcceptsReferencesOnMiniMaxH3()
    {
        VideoRequest request = Request() with
        {
            ReferenceImages = [new ImageData { Rgb = [0, 0, 0], Width = 1, Height = 1 }],
            ReferenceVideos = [new ReferenceVideo { Video = new VideoClip { Data = TinyClip } }],
            ReferenceAudios = [new AudioClip { Data = TinyClip }],
        };
        VideoService.RejectUnsupported(Spec("minimax-h3"), request);
    }
}
