using System.Reflection;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Models.CoLa3D;
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
        string busy = Rig("busy");
        Adopt(mesh, busy);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> run = mesh.RunCounted(RigKey(busy), async () =>
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
        Assert.NotNull(await mesh.RunCounted(RigKey(busy), () => Task.FromResult(Result())));
    }

    [Fact]
    public async Task ARunQueuedBehindAWaitingFreeHonoursItsCancellationAndRunsAfterTheFree()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        string queued = Rig("queued");
        Adopt(mesh, queued);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> running = mesh.RunCounted(RigKey(queued), async () =>
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

        Task<MeshResult> cancelled = mesh.RunCounted(RigKey(queued), () => { entered = true; return Task.FromResult(Result()); }, cancel.Token);
        Task<MeshResult> patient = mesh.RunCounted(RigKey(queued), () => Task.FromResult(Result()));
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
    public async Task AFreeWaitsOnlyForTheRunsOnThePipelineItFrees()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        string busy = Rig("busy"), idle = Rig("idle");
        Adopt(mesh, busy);
        Adopt(mesh, idle);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> run = mesh.RunCounted(RigKey(busy), async () =>
        {
            started.SetResult();
            await release.Task;
            return Result();
        });
        await started.Task;

        Task<int> freeIdle = Task.Run(() => engine.MeshLifecycle.Free("idle"));

        Assert.Equal(1, await freeIdle.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(["busy"], engine.MeshLifecycle.Loaded().Select(m => m.Name));
        release.SetResult();
        await run;
    }

    [Fact]
    public async Task ARunOnAnotherPipelineIsNotHeldOffByAFreeThatWaits()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        string busy = Rig("busy"), other = Rig("other");
        Adopt(mesh, busy);
        Adopt(mesh, other);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> run = mesh.RunCounted(RigKey(busy), async () =>
        {
            started.SetResult();
            await release.Task;
            return Result();
        });
        await started.Task;
        Task<int> free = Task.Run(() => engine.MeshLifecycle.Free("busy"));
        await Task.Delay(300);
        Assert.False(free.IsCompleted);

        Task<MeshResult> onOther = mesh.RunCounted(RigKey(other), () => Task.FromResult(Result()));
        Task<MeshResult> onBusy = mesh.RunCounted(RigKey(busy), () => Task.FromResult(Result()));

        Assert.NotNull(await onOther.WaitAsync(TimeSpan.FromSeconds(10)));
        await Task.Delay(100);
        Assert.False(onBusy.IsCompleted, "A run on the pipeline being freed did not wait.");
        release.SetResult();
        Assert.Equal(1, await free);
        await run;
        Assert.NotNull(await onBusy);
    }

    [Fact]
    public async Task AFreeOfEverythingWaitsForARunOnAnyPipelineAndTwoFreesDoNotDeadlock()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;
        string one = Rig("one"), two = Rig("two");
        Adopt(mesh, one);
        Adopt(mesh, two);
        TaskCompletionSource release = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<MeshResult> run = mesh.RunCounted(RigKey(two), async () =>
        {
            started.SetResult();
            await release.Task;
            return Result();
        });
        await started.Task;

        Task<int> freeAll = Task.Run(() => engine.MeshLifecycle.Free());
        await Task.Delay(300);
        Task<int> freeOne = Task.Run(() => engine.MeshLifecycle.Free("one"));
        Task<int> freeTwo = Task.Run(() => engine.MeshLifecycle.Free("two"));
        await Task.Delay(100);

        Assert.False(freeAll.IsCompleted);
        release.SetResult();
        Assert.Equal(2, await freeAll.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, await freeOne.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, await freeTwo.WaitAsync(TimeSpan.FromSeconds(10)));
        await run;
    }

    [Fact]
    public async Task ARunWithNoKeyIsNotCountedAndNeverWaits()
    {
        using InferenceEngine engine = new("cpu");
        MeshService mesh = (MeshService)engine.Mesh;

        Assert.NotNull(await mesh.RunCounted(null, () => Task.FromResult(Result())));
    }

    [Fact]
    public void TheRunKeyOfARequestIsTheKeyItsPipelineIsFreedUnder()
    {
        string rig = Rig("keyed");
        string part = Path.Combine(_folder, "part", P3SamPipeline.Checkpoint);
        Directory.CreateDirectory(Path.GetDirectoryName(part)!);
        File.WriteAllBytes(part, [1]);
        string paint = Path.Combine(_folder, "paint");
        Directory.CreateDirectory(paint);
        MeshRequest request = new();

        Assert.Equal(RigKey(rig), MeshService.RunKeyOf("unirig", Spec("unirig", rig), request));
        Assert.Equal(MeshRunCounter.KeyOf("part", Path.GetFullPath(part)), MeshService.RunKeyOf("hunyuan3d-part", Spec("hunyuan3d-part", part), request));
        Assert.Equal(MeshRunCounter.KeyOf("paint", paint), MeshService.RunKeyOf("hunyuan3d-paint21", Spec("hunyuan3d-paint21", paint), request));
        Assert.Equal(MeshRunCounter.KeyOf("image", rig), MeshService.RunKeyOf("hunyuan3d-turbo", Spec("hunyuan3d-turbo", rig), request));
        Assert.Equal(MeshRunCounter.KeyOf("scene", $"{rig}|Fixed"), MeshService.RunKeyOf("cola3d", Spec("cola3d", rig), request));
    }

    [Fact]
    public void TheSceneVariantOfAPromptIsTheOneTheLoaderSelects()
    {
        SceneMeshPrompt masks = new() { Masks = [new ScenePromptMask { Coverage = new byte[4], Width = 2, Height = 2 }] };
        SceneMeshPrompt points = new() { Points = [new ScenePoint(0, 0, 0)] };

        Assert.Equal(MeshService.ToCoLa3DPrompt(masks).Kind, MeshService.SceneSource(masks));
        Assert.Equal(MeshService.ToCoLa3DPrompt(points).Kind, MeshService.SceneSource(points));
        Assert.Equal(CoLa3DConfig.QuerySourceKind.Fixed, MeshService.SceneSource(null));
    }

    [Fact]
    public void ARunOnACheckpointTheLoaderWillRefuseIsNotCounted()
    {
        MeshRequest request = new();

        Assert.Null(MeshService.RunKeyOf("hunyuan3d-part", Spec("hunyuan3d-part", Path.Combine(_folder, "missing")), request));
        Assert.Null(MeshService.RunKeyOf("hunyuan3d-turbo", new ModelSpec { Requested = "hunyuan3d-turbo", Modality = Modality.Mesh }, request));
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

    private static string RigKey(string folder) => MeshRunCounter.KeyOf("rig", folder);

    private static ModelSpec Spec(string id, string path) => new() { Requested = id, Modality = Modality.Mesh, LocalPath = path };

    private static void Adopt(MeshService mesh, string folder)
    {
        FieldInfo field = typeof(MeshService).GetField("_rigPipelines", BindingFlags.NonPublic | BindingFlags.Instance)!;
        Dictionary<string, UniRigPipeline> rigs = (Dictionary<string, UniRigPipeline>)field.GetValue(mesh)!;
        InferenceEngine engine = new("cpu");
        rigs[folder] = new UniRigPipeline(engine.Backend, folder);
    }
}
