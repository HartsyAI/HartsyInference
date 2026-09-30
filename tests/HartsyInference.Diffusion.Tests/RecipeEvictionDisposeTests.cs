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

        public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel) =>
            throw new NotSupportedException("Eviction tests never generate.");

        public void Dispose()
        {
            DisposeCalls++;
            if (throwOnDispose)
                throw new InvalidOperationException("Simulated dispose failure.");
        }
    }
}
