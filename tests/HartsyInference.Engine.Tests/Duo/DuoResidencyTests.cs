using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Duo;

/// <summary>What a cached Duo pipeline leaves on the device between runs. A weight it did not preload can still be resident afterwards, since
/// CUDA makes a weight resident the second time an op uploads it, so the pipeline has to free it by name. The backend here is the CPU one
/// behind a recorder that logs every preload and free and reports whatever free memory a test sets. A run is stopped as it builds its video
/// decoder, after the residency decisions and before any forward: the recorder cannot forward the ops that take spans.</summary>
public sealed class DuoResidencyTests : IDisposable
{
    private const string Stopped = "Stopped";
    private const long Ample = 1L << 40, Tight = 1;

    private readonly DuoReleaseFixture _release = new();
    private readonly CpuBackend _cpu = new();
    private readonly Tensor _codecWeight = new(new TensorShape(4L), DType.F32);
    private readonly IBackend _backend;
    private readonly Recorder _device;
    private Func<Exception> _stop = static () => new StopRun();

    public DuoResidencyTests() => _backend = Recorder.Create(_cpu, out _device);

    public void Dispose()
    {
        _codecWeight.Dispose();
        _cpu.Dispose();
        _release.Dispose();
    }

    [Fact]
    public async Task AStreamedDitIsFreedAfterTheRun()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = await Settings(request);
        using IDisposable unloading = KeepModels(false);

        HashSet<Tensor> dit = Preloaded(() => Run(pipeline, request, settings));
        _device.FreeBytes = Tight;
        _device.Calls.Clear();
        Run(pipeline, request, settings);

        Assert.DoesNotContain(_device.Calls, c => c.Kind == nameof(IBackend.PreloadWeights) && c.Tensors.Overlaps(dit));
        Assert.Subset(FreedAfterTheStop(), dit);
    }

    [Fact]
    public async Task CodecsWhosePreloadDidNotFitAreFreedAfterTheRun()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = await Settings(request);
        _device.RefusePreloadOf = _codecWeight;

        Run(pipeline, request, settings);

        Assert.Contains(_codecWeight, FreedAfterTheStop());
    }

    /// <summary>A kept DiT takes its own share of the free memory, so the next run's fit test reads as if it did not fit.</summary>
    [Fact]
    public async Task AKeptDitSurvivesARunThatFindsLittleMemoryFree()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = await Settings(request);
        using IDisposable keeping = KeepModels(true);

        HashSet<Tensor> dit = Preloaded(() => Run(pipeline, request, settings));
        _device.FreeBytes = Tight;
        Run(pipeline, request, settings);

        Assert.DoesNotContain(_device.Calls, c => c.Kind == nameof(IBackend.FreeWeights) && c.Tensors.Overlaps(dit));
        // Asked for again, so a copy something else evicted comes back before the run.
        Assert.Equal(2, _device.Calls.Count(c => c.Kind == nameof(IBackend.PreloadWeights) && c.Tensors.SetEquals(dit)));
    }

    /// <summary>The engine frees all device memory after a run that runs out, so the next run has to decide about the DiT afresh.</summary>
    [Fact]
    public async Task ARunThatRunsOutOfMemoryGivesUpItsKeptDit()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = await Settings(request);
        using IDisposable keeping = KeepModels(true);

        HashSet<Tensor> dit = Preloaded(() => Run(pipeline, request, settings));
        _stop = static () => new OutOfVramException("out of memory in the run");
        _device.Calls.Clear();
        Assert.Throws<OutOfVramException>(() => pipeline.Generate(request, settings, null, default));
        Assert.Subset(FreedAfterTheStop(), dit);

        _stop = static () => new StopRun();
        _device.FreeBytes = Tight;
        _device.Calls.Clear();
        Run(pipeline, request, settings);
        Assert.DoesNotContain(_device.Calls, c => c.Kind == nameof(IBackend.PreloadWeights) && c.Tensors.Overlaps(dit));
    }

    /// <summary>An engine that keeps other pipelines cached sweeps nothing after disposing this one.</summary>
    [Fact]
    public async Task DisposingThePipelineFreesWhatItKept()
    {
        using DuoRecipePipeline pipeline = Build();
        VideoRequest request = Request();
        VideoEffectiveSettings settings = await Settings(request);
        using IDisposable keeping = KeepModels(true);

        HashSet<Tensor> dit = Preloaded(() => Run(pipeline, request, settings));
        _device.Calls.Clear();
        pipeline.Dispose();

        HashSet<Tensor> freed = [.. _device.Calls.Where(c => c.Kind == nameof(IBackend.FreeWeights)).SelectMany(c => c.Tensors)];
        Assert.Subset(freed, dit);
        Assert.Contains(_codecWeight, freed);
    }

    private static IDisposable KeepModels(bool keep) =>
        KnobProfile.Create("duo-residency").With(EngineKnobs.KeepModels, keep).Push();

    /// <summary>The DiT weights a run with ample free memory preloads: the largest preload it makes.</summary>
    private HashSet<Tensor> Preloaded(Action run)
    {
        _device.FreeBytes = Ample;
        run();
        HashSet<Tensor> dit = _device.Calls.Where(c => c.Kind == nameof(IBackend.PreloadWeights))
            .Select(c => c.Tensors).MaxBy(t => t.Count)!;
        Assert.Contains(dit, t => t.Shape.Rank == 2 && t.Shape[0] == 3 * DuoReleaseFixture.Inner);
        return dit;
    }

    private HashSet<Tensor> FreedAfterTheStop()
    {
        int stop = _device.Calls.FindLastIndex(c => c.Kind == Stopped);
        Assert.True(stop >= 0, "The run did not reach its decoders.");
        return _device.Calls.Skip(stop + 1).Where(c => c.Kind == nameof(IBackend.FreeWeights)).SelectMany(c => c.Tensors).ToHashSet();
    }

    private static void Run(DuoRecipePipeline pipeline, VideoRequest request, VideoEffectiveSettings settings) =>
        Assert.Throws<StopRun>(() => pipeline.Generate(request, settings, null, default));

    private DuoRecipePipeline Build() => new DuoRecipe().Build(
        new RecipeContext { CheckpointPath = _release.Release, Backend = _backend },
        (_, _) => DuoReleaseFixture.Codecs() with
        {
            Video = _ =>
            {
                _device.Calls.Add((Stopped, []));
                throw _stop();
            },
            Weights = () => [_codecWeight],
        });

    private VideoRequest Request() => new()
    {
        Prompt = "", Width = DuoReleaseFixture.Size, Height = DuoReleaseFixture.Size, Frames = DuoReleaseFixture.Frames,
        Steps = DuoReleaseFixture.Steps, Seed = 1,
        Extra = new Dictionary<string, object> { [DuoRecipePipeline.RunFolderKey] = _release.RunFolder },
    };

    private static async Task<VideoEffectiveSettings> Settings(VideoRequest request)
    {
        DuoRecipe recipe = new();
        ModelSpec spec = new() { Requested = "duo-v1-causal", Modality = Modality.Video };
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(spec, request,
            VideoRecipeDeclaration.For(recipe, "duo-v1-causal", null, recipe.Supports), default);
        return plan.EffectiveSettings;
    }

    private sealed class StopRun : Exception;

    /// <summary>Forwards every call to the CPU backend, logging each preload and free.</summary>
    public class Recorder : DispatchProxy
    {
        private IBackend _inner = null!;

        public List<(string Kind, HashSet<Tensor> Tensors)> Calls { get; } = [];

        /// <summary>The free memory reported; 0, the CPU backend's own answer, reads as unknown.</summary>
        public long FreeBytes { get; set; }

        /// <summary>A preload that includes this tensor runs out of memory.</summary>
        public Tensor? RefusePreloadOf { get; set; }

        public static IBackend Create(IBackend inner, out Recorder recorder)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, Recorder>();
            recorder = (Recorder)(object)proxy;
            recorder._inner = inner;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case nameof(IBackend.GetVramInfo):
                    return (FreeBytes, FreeBytes);
                case nameof(IBackend.PreloadWeights):
                case nameof(IBackend.FreeWeights):
                    HashSet<Tensor> tensors = ((IEnumerable<Tensor>)args![0]!).ToHashSet();
                    Calls.Add((targetMethod.Name, tensors));
                    if (targetMethod.Name == nameof(IBackend.PreloadWeights) && RefusePreloadOf is { } refused && tensors.Contains(refused))
                    {
                        throw new OutOfVramException("refused by the test");
                    }
                    break;
            }
            return targetMethod.Invoke(_inner, args);
        }
    }
}
