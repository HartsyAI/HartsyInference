using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Video.Pipelines;
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

    [Fact]
    public void MiniMaxH3DoesNotAdvertiseValidationPendingControlFeatures()
    {
        VideoFeatures released = new MiniMaxH3Recipe().Supports;

        Assert.False(released.HasFlag(VideoFeatures.VideoControlNet));
        Assert.False(released.HasFlag(VideoFeatures.VideoInpaint));
        Assert.False(released.HasFlag(VideoFeatures.Guides));
        Assert.False(released.HasFlag(VideoFeatures.VideoDenoiseMask));
        Assert.False(released.HasFlag(VideoFeatures.AudioDenoiseMask));
    }

    [Fact]
    public void ValidationPendingH3ExecutionIngressIsNotPublic()
    {
        string[] forbiddenParameters =
        [
            "pddAdapter", "pddHeads", "pddLoraStack", "sparseAttention", "sparseAttentionA",
            "sparseAttentionB", "sparseAttentionProfile", "controls", "funControlModelIndices",
            "supportsHybridConditioning",
        ];

        Assert.All(typeof(MiniMaxH3Pipeline).GetConstructors(), constructor =>
            Assert.DoesNotContain(constructor.GetParameters(), parameter =>
                forbiddenParameters.Contains(parameter.Name, StringComparer.Ordinal)));
        Assert.All(typeof(MiniMaxH3RecipePipeline).GetConstructors(), constructor =>
            Assert.DoesNotContain(constructor.GetParameters(), parameter =>
                forbiddenParameters.Contains(parameter.Name, StringComparer.Ordinal)));

        Assert.Null(typeof(MiniMaxH3GenerationRequest).GetProperty("HybridProfile"));
        Assert.Null(typeof(MiniMaxH3GenerationRequest).GetProperty("Controls"));
        Assert.DoesNotContain(typeof(MiniMaxH3Transformer).GetMethods(), method =>
            method.DeclaringType == typeof(MiniMaxH3Transformer)
            && method.Name is "ForwardPlanned" or "ForwardShardedPlanned"
                or "ValidateVideoSparseAttentionWeights" or "CreateVideoSparseAttentionPlan"
                or "RegisterFunControlNet");
        Assert.All(typeof(MiniMaxH3Transformer).GetMethods()
                .Where(method => method.DeclaringType == typeof(MiniMaxH3Transformer)
                    && method.Name is "Forward" or "ForwardSharded"),
            method => Assert.DoesNotContain(method.GetParameters(), parameter =>
                forbiddenParameters.Contains(parameter.Name, StringComparer.Ordinal)));
    }

    [Fact]
    public void MiniMaxH3RecipeRejectsMissingPlanBeforeTouchingCheckpoint()
    {
        using CpuBackend backend = new CpuBackend();
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = Path.Combine(Path.GetTempPath(), $"missing-h3-{Guid.NewGuid():N}.safetensors"),
            Backend = backend,
        };

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => new MiniMaxH3Recipe().Construct(context));

        Assert.Contains("service-bound VideoPlan", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MiniMaxH3RecipeRejectsCallerFabricatedPlanBeforeTouchingCheckpoint()
    {
        using CpuBackend backend = new CpuBackend();
        VideoPlan fabricated = H3Plan();
        RecipeContext context = new RecipeContext
        {
            CheckpointPath = fabricated.ComponentPaths["transformer"],
            Backend = backend,
            VideoPlan = fabricated,
        };

        ArgumentException error = Assert.Throws<ArgumentException>(
            () => new MiniMaxH3Recipe().Construct(context));

        Assert.Contains("immutable execution binding", error.Message, StringComparison.Ordinal);
    }

    private static VideoPlan H3Plan()
    {
        string checkpoint = Path.Combine(Path.GetTempPath(), $"fabricated-h3-{Guid.NewGuid():N}.safetensors");
        return new VideoPlan
        {
            Model = new ModelSpec { Requested = checkpoint, LocalPath = checkpoint, Modality = Modality.Video },
            Profile = new VideoModelProfile
            {
                Id = "fabricated-h3",
                DisplayName = "Fabricated H3",
                FamilyId = "minimax-h3",
                Task = VideoTaskFamily.Fl2Va,
                Acceleration = VideoAccelerationKind.None,
                Attention = VideoAttentionKind.Dense,
                Defaults = new MiniMaxH3Recipe().Defaults,
                Features = new MiniMaxH3Recipe().Supports,
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
            CacheIdentity = "fabricated",
            ComponentPaths = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["transformer"] = checkpoint,
            },
        };
    }

    private static Task<VideoPlan> PlanGenericAsync(string family, VideoRequest request)
    {
        ModelSpec spec = Spec(family);
        VideoDefaults defaults = InferenceEngine.VideoDefaultsFor(spec);
        VideoFeatures features = InferenceEngine.SupportedVideoFeatures(spec);
        return VideoProfileResolver.ResolveAsync(spec, request, family, defaults, features, CancellationToken.None);
    }
}
