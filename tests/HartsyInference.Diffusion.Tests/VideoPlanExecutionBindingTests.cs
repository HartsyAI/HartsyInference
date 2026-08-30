using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

[CollectionDefinition(CollectionName, DisableParallelization = true)]
public sealed class VideoPlanExecutionBindingCollection
{
    public const string CollectionName = "Video plan execution binding registry";
}

[Collection(VideoPlanExecutionBindingCollection.CollectionName)]
public sealed class VideoPlanExecutionBindingTests
{
    [Fact]
    public async Task MediaBuffersRemainZeroCopyInExecutionSnapshot()
    {
        TrackingRecipe recipe = RegisterRecipe();
        ImageData extraImage = new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 };
        VideoRequest request = Request(extraImage);
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);
        Assert.True(plan.IsValid);

        await engine.Video.GenerateAsync(plan, request);

        ImageData executedImage = Assert.IsType<ImageData>(recipe.Pipeline.LastRequest!.Extra["image"]);
        Assert.Same(extraImage, executedImage);
        Assert.Same(extraImage.Rgb, executedImage.Rgb);
        Assert.Equal([1, 2, 3], executedImage.Rgb);
        Assert.Equal(1, recipe.ConstructCalls);
        Assert.Equal(1, recipe.Pipeline.GenerateCalls);
    }

    [Fact]
    public void CreateSharesTypedMediaBuffersWhileSnapshottingCollections()
    {
        ImageData image = new ImageData { Rgb = new byte[12], Width = 2, Height = 2 };
        VideoClip video = new VideoClip { Data = new byte[32], Format = "mp4" };
        AudioClip audio = new AudioClip { Data = new byte[16], Format = "wav" };
        List<ImageData> references = [image];
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            InitImage = image,
            VideoAudioInput = audio,
            ReferenceImages = references,
            DrivingVideo = video,
        };
        ModelSpec model = new ModelSpec { Requested = "test", Modality = Modality.Video };

        VideoRequestExecutionBinding binding = VideoRequestExecutionBinding.Create(model, request);

        Assert.NotSame(request, binding.Request);
        Assert.Same(image.Rgb, binding.Request.InitImage!.Rgb);
        Assert.Same(audio.Data, binding.Request.VideoAudioInput!.Data);
        Assert.Same(video.Data, binding.Request.DrivingVideo!.Data);
        Assert.NotSame(references, binding.Request.ReferenceImages);
        Assert.Same(image, Assert.Single(binding.Request.ReferenceImages!));
    }

    [Fact]
    public async Task NestedListMutationAfterPlanningDoesNotChangeExecutionSnapshot()
    {
        TrackingRecipe recipe = RegisterRecipe();
        List<ImageData> references = [];
        VideoRequest request = Request(new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 }) with
        {
            ReferenceImages = references,
        };
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);
        Assert.True(plan.IsValid);

        references.Add(new ImageData { Rgb = [4, 5, 6], Width = 1, Height = 1 });

        await engine.Video.GenerateAsync(plan, request);

        Assert.Empty(recipe.Pipeline.LastRequest!.ReferenceImages!);
        Assert.Equal(1, recipe.ConstructCalls);
        Assert.Equal(1, recipe.Pipeline.GenerateCalls);
    }

    [Fact]
    public async Task UntouchedRequestExecutesStructuralSnapshotWithSharedMedia()
    {
        TrackingRecipe recipe = RegisterRecipe();
        ImageData extraImage = new ImageData { Rgb = [7, 8, 9], Width = 1, Height = 1 };
        Dictionary<string, string> aux = new Dictionary<string, string> { ["tokenizer"] = "original" };
        VideoRequest request = Request(extraImage);
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(
            Spec(recipe.Name, checkpoint.Path) with { Aux = aux }, request);

        VideoGenerationResult result = await engine.Video.GenerateAsync(plan, request);

        Assert.Empty(result.Frames);
        Assert.Equal(1, recipe.ConstructCalls);
        Assert.Equal(1, recipe.Pipeline.GenerateCalls);
        VideoRequest executed = Assert.IsType<VideoRequest>(recipe.Pipeline.LastRequest);
        Assert.NotSame(request, executed);
        Assert.NotSame(request.Extra, executed.Extra);
        ImageData executedImage = Assert.IsType<ImageData>(executed.Extra["image"]);
        Assert.Same(extraImage, executedImage);
        Assert.Same(extraImage.Rgb, executedImage.Rgb);
        Assert.Equal([7, 8, 9], executedImage.Rgb);

        aux["tokenizer"] = "changed-after-plan";
        Assert.Equal("original", plan.Model.Aux["tokenizer"]);
    }

    [Fact]
    public async Task ArbitraryMutableExtraIsPreserved()
    {
        TrackingRecipe recipe = RegisterRecipe();
        CustomExtra custom = new CustomExtra();
        Dictionary<string, object> extra = new Dictionary<string, object> { ["mutable"] = custom };
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            Extra = extra,
        };
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");

        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);
        extra["mutable"] = new object();
        await engine.Video.GenerateAsync(plan, request);

        object executed = recipe.Pipeline.LastRequest!.Extra["mutable"];
        Assert.Same(custom, executed);
        Assert.Equal(1, recipe.ConstructCalls);
        Assert.Equal(1, recipe.Pipeline.GenerateCalls);
    }

    [Fact]
    public async Task NullNestedListEntryFailsAsTypedPlanningInput()
    {
        TrackingRecipe recipe = RegisterRecipe();
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            Guides = new List<VideoGuide> { null! },
        };
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");

        ArgumentException error = await Assert.ThrowsAsync<ArgumentException>(
            () => engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request));
        Assert.Contains(nameof(VideoRequest.Guides), error.Message, StringComparison.Ordinal);
        Assert.Contains("index 0", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, recipe.ConstructCalls);
    }

    [Fact]
    public async Task CheckpointReplacementAfterPlanningRejectsBeforeConstruction()
    {
        TrackingRecipe recipe = RegisterRecipe();
        VideoRequest request = Request(new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 });
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);

        File.WriteAllBytes(checkpoint.Path, [9]);
        File.SetLastWriteTimeUtc(checkpoint.Path, DateTime.UtcNow.AddSeconds(2));

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => engine.Video.GenerateAsync(plan, request));
        Assert.Contains("changed after planning", error.Message, StringComparison.Ordinal);
        Assert.Contains("Re-plan", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, recipe.ConstructCalls);
        Assert.Equal(0, recipe.Pipeline.GenerateCalls);
    }

    [Fact]
    public async Task AlteredPublicPlanRejectsBeforeConstruction()
    {
        TrackingRecipe recipe = RegisterRecipe();
        VideoRequest request = Request(new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 });
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);
        VideoPlan altered = plan with { CacheIdentity = plan.CacheIdentity + "altered;" };

        ArgumentException error = await Assert.ThrowsAsync<ArgumentException>(
            () => engine.Video.GenerateAsync(altered, request));
        Assert.Contains("VideoPlan was altered", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, recipe.ConstructCalls);
        Assert.Equal(0, recipe.Pipeline.GenerateCalls);
    }

    [Fact]
    public async Task VerifiedPlanCanBeReverifiedAtRecipeConstructionBoundary()
    {
        TrackingRecipe recipe = RegisterRecipe();
        VideoRequest request = Request(new ImageData { Rgb = [1, 2, 3], Width = 1, Height = 1 });
        using TempCheckpoint checkpoint = new TempCheckpoint();
        using InferenceEngine engine = new InferenceEngine("cpu");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(Spec(recipe.Name, checkpoint.Path), request);

        VideoPlan serviceVerified = VideoRequestExecutionBinding.RequirePlannedState(plan);
        VideoPlan recipeVerified = VideoRequestExecutionBinding.RequirePlannedState(serviceVerified);

        Assert.Same(plan, serviceVerified);
        Assert.Same(plan, recipeVerified);
    }

    private static TrackingRecipe RegisterRecipe()
    {
        TrackingRecipe recipe = new TrackingRecipe("binding-test-" + Guid.NewGuid().ToString("N"));
        VideoRecipeRegistry.Register(recipe);
        return recipe;
    }

    private static ModelSpec Spec(string family, string checkpoint) => new ModelSpec
    {
        Requested = family,
        LocalPath = checkpoint,
        Modality = Modality.Video,
        Catalog = new CatalogEntry
        {
            Id = family,
            Modality = Modality.Video,
            DisplayName = family,
            Architecture = "test",
            Status = ModelStatus.Structural,
        },
    };

    private static VideoRequest Request(ImageData image) => new VideoRequest
    {
        Prompt = "test",
        Seed = 123,
        Extra = new Dictionary<string, object> { ["image"] = image },
    };

    private sealed class TrackingRecipe(string name) : IVideoRecipe
    {
        public string Name { get; } = name;

        public int ConstructCalls { get; private set; }

        public TrackingPipeline Pipeline { get; } = new TrackingPipeline();

        public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.Ordinal);

        public IVideoRecipePipeline Construct(RecipeContext context)
        {
            ConstructCalls++;
            return Pipeline;
        }
    }

    private sealed class CustomExtra
    {
        public List<int> Values { get; } = [1, 2, 3];
    }

    private sealed class TrackingPipeline : IVideoRecipePipeline
    {
        public int GenerateCalls { get; private set; }

        public VideoRequest? LastRequest { get; private set; }

        public VideoGenerationResult Generate(
            VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
        {
            GenerateCalls++;
            LastRequest = request;
            return VideoGenerationResult.FromFrames([]);
        }

        public void Dispose()
        {
        }
    }

    private sealed class TempCheckpoint : IDisposable
    {
        public TempCheckpoint()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"hartsy-video-binding-{Guid.NewGuid():N}.bin");
            File.WriteAllBytes(Path, [0]);
        }

        public string Path { get; }

        public void Dispose() => File.Delete(Path);
    }
}
