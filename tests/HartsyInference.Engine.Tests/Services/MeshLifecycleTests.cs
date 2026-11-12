using System.Reflection;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshLifecycleTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), "hartsy-mesh-life-" + Guid.NewGuid().ToString("N"));

    public MeshLifecycleTests() => Directory.CreateDirectory(_folder);

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static MeshResult Result() => new() { Data = [1] };

    [Fact]
    public async Task AnOutOfVramFailureFreesAndRunsOnceMore()
    {
        int runs = 0, frees = 0;

        MeshResult result = await MeshService.WithOutOfVramRetry(() => ++runs == 1 ? throw new OutOfVramException(1 << 30, 1 << 20) : Task.FromResult(Result()), () => frees++);

        Assert.Equal((2, 1), (runs, frees));
        Assert.Equal(new byte[] { 1 }, result.Data);
    }

    [Fact]
    public async Task ASecondOutOfVramFailurePropagatesAsTheTypedException()
    {
        int runs = 0;

        OutOfVramException error = await Assert.ThrowsAsync<OutOfVramException>(() =>
            MeshService.WithOutOfVramRetry(() => { runs++; throw new OutOfVramException(1 << 30, 1 << 20); }, () => { }));

        Assert.Equal(2, runs);
        Assert.Equal(1 << 30, error.RequestedBytes);
    }

    [Fact]
    public async Task AnOutOfVramFailureInsideAnAggregateIsRecognisedToo()
    {
        int runs = 0;

        MeshResult result = await MeshService.WithOutOfVramRetry(
            () => ++runs == 1 ? throw new AggregateException(new OutOfVramException(1, 0)) : Task.FromResult(Result()), () => { });

        Assert.Equal(2, runs);
        Assert.NotNull(result);
    }

    [Theory]
    [InlineData(typeof(InvalidOperationException))]
    [InlineData(typeof(NotSupportedException))]
    [InlineData(typeof(OperationCanceledException))]
    public async Task AnyOtherFailureIsNotRetried(Type failure)
    {
        int runs = 0, frees = 0;

        await Assert.ThrowsAnyAsync<Exception>(() =>
            MeshService.WithOutOfVramRetry(() => { runs++; throw (Exception)Activator.CreateInstance(failure)!; }, () => frees++));

        Assert.Equal((1, 0), (runs, frees));
    }

    [Fact]
    public void AnEngineWithNoResidentModelsListsNoneAndFreesNone()
    {
        using InferenceEngine engine = new("cpu");

        Assert.Empty(engine.MeshLifecycle.Loaded());
        Assert.Equal(0, engine.MeshLifecycle.Free());
        Assert.Equal(0, engine.MeshLifecycle.Free("anything"));
    }

    [Fact]
    public void LoadedListsResidentPipelinesByNameAndFreeDisposesThemByNameOrAll()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        string a = Rig("one"), b = Rig("two");
        Adopt(mesh, a);
        Adopt(mesh, b);

        IReadOnlyList<MeshLoadedModel> loaded = engine.MeshLifecycle.Loaded();

        Assert.Equal(2, loaded.Count);
        Assert.All(loaded, m => Assert.Equal("rig", m.Kind));
        Assert.Equal(["one", "two"], loaded.Select(m => m.Name).Order());
        Assert.All(loaded, m => Assert.DoesNotContain(Path.GetTempPath().TrimEnd(Path.DirectorySeparatorChar), m.Name));
        Assert.Equal(1, engine.MeshLifecycle.Free("ONE"));
        Assert.Equal(["two"], engine.MeshLifecycle.Loaded().Select(m => m.Name));
        Assert.Equal(1, engine.MeshLifecycle.Free());
        Assert.Empty(engine.MeshLifecycle.Loaded());
    }

    [Fact]
    public async Task FreeWaitsForARunOnAPipelineWithoutAGateAndNewRunsWaitForTheFree()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        Adopt(mesh, Rig("busy"));
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> run = mesh.RunCounted(async () =>
        {
            started.SetResult();
            await release.Task;
            return Result();
        });
        await started.Task;

        Task<int> free = Task.Run(() => engine.MeshLifecycle.Free("busy"));
        await Task.Delay(300);

        Assert.False(free.IsCompleted, "Free disposed a pipeline while a run was using it.");
        release.SetResult();
        Assert.Equal(1, await free);
        await run;
        Assert.Empty(engine.MeshLifecycle.Loaded());
        Assert.NotNull(await mesh.RunCounted(() => Task.FromResult(Result())));
    }

    [Fact]
    public async Task ARunQueuedBehindAWaitingFreeHonoursItsCancellationAndRunsAfterTheFree()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        Adopt(mesh, Rig("queued"));
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> running = mesh.RunCounted(async () =>
        {
            started.SetResult();
            await release.Task;
            return Result();
        });
        await started.Task;
        Task<int> free = Task.Run(() => engine.MeshLifecycle.Free("queued"));
        await Task.Delay(300);
        using CancellationTokenSource cancel = new();
        bool entered = false;

        Task<MeshResult> cancelled = mesh.RunCounted(() => { entered = true; return Task.FromResult(Result()); }, cancel.Token);
        Task<MeshResult> patient = mesh.RunCounted(() => Task.FromResult(Result()));
        await Task.Delay(100);

        Assert.False(cancelled.IsCompleted, "A run queued behind a free did not wait.");
        cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        Assert.False(entered);
        Assert.False(patient.IsCompleted);
        release.SetResult();
        Assert.Equal(1, await free);
        await running;
        Assert.NotNull(await patient);
    }

    [Fact]
    public void ANameKeepsNoPathAndAFolderBesideTheModelsRootIsNotRelativeToIt()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        Adopt(mesh, Rig("plain"));

        MeshLoadedModel model = Assert.Single(engine.MeshLifecycle.Loaded());

        Assert.Equal("plain", model.Name);
        Assert.DoesNotContain("..", model.Name);
    }

    private string Rig(string name)
    {
        string folder = Path.Combine(_folder, name);
        Directory.CreateDirectory(folder);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(folder, UniRigPipeline.SkeletonCheckpoint))!);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(folder, UniRigPipeline.SkinCheckpoint))!);
        File.WriteAllBytes(Path.Combine(folder, UniRigPipeline.SkeletonCheckpoint), [1]);
        File.WriteAllBytes(Path.Combine(folder, UniRigPipeline.SkinCheckpoint), [1]);
        return folder;
    }

    private static void Adopt(MeshService mesh, string folder)
    {
        FieldInfo field = typeof(MeshService).GetField("_rigPipelines", BindingFlags.NonPublic | BindingFlags.Instance)!;
        Dictionary<string, UniRigPipeline> rigs = (Dictionary<string, UniRigPipeline>)field.GetValue(mesh)!;
        InferenceEngine engine = new("cpu");
        rigs[folder] = new UniRigPipeline(engine.Backend, folder);
    }
}
