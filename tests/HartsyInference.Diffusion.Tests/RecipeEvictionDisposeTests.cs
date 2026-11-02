using HartsyInference.Core.Exceptions;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A cached recipe pipeline whose Dispose throws must not wedge the engine: model-switch eviction, a
/// between-jobs free and teardown each drop it and carry on releasing the rest.</summary>
public sealed class RecipeEvictionDisposeTests
{
    [Fact]
    public void ModelSwitch_PastAPipelineThatThrowsOnDispose_DropsItAndConstructsTheNewModel()
    {
        FakeRecipe faulty = Register("test-evict-faulty", throwOnDispose: true);
        FakeRecipe healthy = Register("test-evict-healthy", throwOnDispose: false);
        using InferenceEngine engine = new InferenceEngine("cpu");

        engine.GetOrConstructRecipe(SpecFor(faulty.Name));
        IRecipePipeline next = engine.GetOrConstructRecipe(SpecFor(healthy.Name));

        Assert.Same(healthy.Built[0], next);
        Assert.Equal(1, faulty.Built[0].DisposeCalls);
        engine.GetOrConstructRecipe(SpecFor(faulty.Name));
        Assert.Equal(2, faulty.Built.Count);
    }

    [Fact]
    public void FreeMemory_WithAPipelineThatThrowsOnDispose_EmptiesTheCacheWithoutThrowing()
    {
        FakeRecipe faulty = Register("test-free-faulty", throwOnDispose: true);
        using InferenceEngine engine = new InferenceEngine("cpu");
        engine.GetOrConstructRecipe(SpecFor(faulty.Name));

        engine.FreeMemory();

        Assert.Equal(1, faulty.Built[0].DisposeCalls);
        engine.GetOrConstructRecipe(SpecFor(faulty.Name));
        Assert.Equal(2, faulty.Built.Count);
    }

    [Fact]
    public void Dispose_WithAPipelineThatThrows_StillDisposesTheRestAndReportsTheFailure()
    {
        FakeRecipe faulty = Register("test-teardown-faulty", throwOnDispose: true);
        FakeRecipe healthy = Register("test-teardown-healthy", throwOnDispose: false);
        InferenceEngine engine = new InferenceEngine("cpu");
        engine.GetOrConstructRecipe(SpecFor(faulty.Name));
        engine.GetOrConstructRecipe(SpecFor(healthy.Name), alsoKeepPath: SpecFor(faulty.Name).LocalPath);

        Assert.Throws<AggregateException>(engine.Dispose);

        Assert.Equal(1, faulty.Built[0].DisposeCalls);
        Assert.Equal(1, healthy.Built[0].DisposeCalls);
    }

    [Fact]
    public void ModelSwitch_WhileAnotherRequestHoldsThePipeline_FailsByNameAndDisposesNothing()
    {
        FakeRecipe held = Register("test-lease-held", throwOnDispose: false);
        FakeRecipe next = Register("test-lease-next", throwOnDispose: false);
        using InferenceEngine engine = new InferenceEngine("cpu");
        engine.PipelineLeaseWait = TimeSpan.FromMilliseconds(200);
        LeasedRecipe lease = engine.GetOrConstructRecipeLeased(SpecFor(held.Name));

        HartsyInferenceException busy = Assert.Throws<HartsyInferenceException>(
            () => engine.GetOrConstructRecipe(SpecFor(next.Name)));

        Assert.Contains("still serving a request", busy.Message, StringComparison.Ordinal);
        Assert.Equal(0, held.Built[0].DisposeCalls);
        Assert.Empty(next.Built);

        lease.Lease.Dispose();
        IRecipePipeline built = engine.GetOrConstructRecipe(SpecFor(next.Name));

        Assert.Same(next.Built[0], built);
        Assert.Equal(1, held.Built[0].DisposeCalls);
    }

    [Fact]
    public void FreeMemory_WhileAPipelineIsLeased_SkipsTheFreeAndKeepsThePipeline()
    {
        FakeRecipe held = Register("test-lease-free", throwOnDispose: false);
        using InferenceEngine engine = new InferenceEngine("cpu");
        LeasedRecipe lease = engine.GetOrConstructRecipeLeased(SpecFor(held.Name));

        // Returns at once: a free between jobs never waits on a request, and never disposes a pipeline one holds.
        engine.FreeMemory();
        Assert.Equal(0, held.Built[0].DisposeCalls);

        lease.Lease.Dispose();
        engine.FreeMemory();
        Assert.Equal(1, held.Built[0].DisposeCalls);
    }

    [Fact]
    public void ModelSwitch_CancelledWhileWaitingForALease_StopsWaitingPromptly()
    {
        FakeRecipe held = Register("test-lease-cancel", throwOnDispose: false);
        FakeRecipe next = Register("test-lease-cancel-next", throwOnDispose: false);
        using InferenceEngine engine = new InferenceEngine("cpu");
        LeasedRecipe lease = engine.GetOrConstructRecipeLeased(SpecFor(held.Name));
        using CancellationTokenSource cancel = new(TimeSpan.FromMilliseconds(200));
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

        // The default lease wait is ten minutes. A cancelled request must not hold the switch behind that lease for the whole wait.
        Assert.ThrowsAny<OperationCanceledException>(() => engine.GetOrConstructRecipeLeased(SpecFor(next.Name), cancel: cancel.Token));

        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(10));
        Assert.Equal(0, held.Built[0].DisposeCalls);
        Assert.Empty(next.Built);
        lease.Lease.Dispose();
    }

    private static FakeRecipe Register(string familyId, bool throwOnDispose)
    {
        FakeRecipe recipe = new FakeRecipe(familyId, throwOnDispose);
        RecipeRegistry.Register(recipe);
        return recipe;
    }

    private static ModelSpec SpecFor(string familyId) => new ModelSpec
    {
        Requested = familyId,
        Modality = Modality.Image,
        LocalPath = $"/nonexistent/{familyId}.safetensors",
        Catalog = new CatalogEntry
        {
            Id = familyId,
            Modality = Modality.Image,
            DisplayName = familyId,
            Architecture = familyId,
            Status = ModelStatus.Verified,
        },
    };

    /// <summary>Builds a fresh pipeline per construction and keeps every one; only the first can be made to fail disposal.</summary>
    private sealed class FakeRecipe(string familyId, bool throwOnDispose) : IArchitectureRecipe
    {
        public string Name => familyId;

        public ImageDefaults Defaults { get; } = new ImageDefaults { Steps = 20, CfgScale = 7.5f, Width = 1024, Height = 1024 };

        public List<FakeRecipePipeline> Built { get; } = new List<FakeRecipePipeline>();

        public bool Matches(string candidate) => string.Equals(candidate, familyId, StringComparison.OrdinalIgnoreCase);

        public IRecipePipeline Construct(RecipeContext context)
        {
            FakeRecipePipeline pipeline = new FakeRecipePipeline(throwOnDispose && Built.Count == 0);
            Built.Add(pipeline);
            return pipeline;
        }
    }

    /// <summary>Counts its disposals and, when asked, throws from each one.</summary>
    private sealed class FakeRecipePipeline(bool throwOnDispose) : IRecipePipeline
    {
        public int DisposeCalls { get; private set; }

        public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings,
            IProgress<StepPreview>? progress, CancellationToken cancel) =>
            throw new NotSupportedException("Eviction tests never generate.");

        public void Dispose()
        {
            DisposeCalls++;
            if (throwOnDispose)
                throw new InvalidOperationException("Simulated dispose failure.");
        }
    }
}
