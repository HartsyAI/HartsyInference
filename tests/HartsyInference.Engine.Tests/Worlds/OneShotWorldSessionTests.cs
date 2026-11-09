using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>A one-shot clip pipeline whose run each test scripts. Runs are counted, and the in-flight peak shows whether two runs overlapped.</summary>
internal sealed class ScriptedClipPipeline : IOneShotClipPipeline
{
    private readonly Func<OneShotClipRequest, CancellationToken, IReadOnlyList<VideoFrame>> _script;
    private readonly List<OneShotClipRequest> _requests = [];
    private readonly object _gate = new();
    private int _inFlight;
    private int _maxInFlight;

    /// <summary>Creates a pipeline whose every run is <paramref name="script"/>.</summary>
    public ScriptedClipPipeline(Func<OneShotClipRequest, CancellationToken, IReadOnlyList<VideoFrame>> script)
    {
        ArgumentNullException.ThrowIfNull(script);
        _script = script;
    }

    /// <summary>Released once for each run as it begins, so a test can wait for a run to start.</summary>
    public SemaphoreSlim Started { get; } = new(0);

    /// <summary>Runs that have begun.</summary>
    public int StartCount
    {
        get
        {
            lock (_gate)
            {
                return _requests.Count;
            }
        }
    }

    /// <summary>The requests the runs were given, in the order they began.</summary>
    public IReadOnlyList<OneShotClipRequest> Requests
    {
        get
        {
            lock (_gate)
            {
                return _requests.ToArray();
            }
        }
    }

    /// <summary>The most runs that were in progress at the same time.</summary>
    public int MaxInFlight
    {
        get
        {
            lock (_gate)
            {
                return _maxInFlight;
            }
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<VideoFrame> GenerateClip(OneShotClipRequest request, CancellationToken cancel)
    {
        lock (_gate)
        {
            _requests.Add(request);
            _inFlight++;
            _maxInFlight = Math.Max(_maxInFlight, _inFlight);
        }
        Started.Release();
        try
        {
            return _script(request, cancel);
        }
        finally
        {
            lock (_gate)
            {
                _inFlight--;
            }
        }
    }
}

public sealed class OneShotWorldSessionTests
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(5);

    [Fact]
    public async Task Stream_DeliversTheWholeClipInOrder_OnceTheRunCompletes()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) => ClipFrames(request.Plan.Frames));
        IWorldSession session = service.OpenOneShot(pipeline, ClipPlan(4));
        session.SendAction("forward");
        session.SendAction("forward");
        session.SendAction("left");

        IAsyncEnumerable<VideoFrame> stream = session.StreamAsync(CancellationToken.None);
        Assert.Equal(0, pipeline.StartCount);
        List<VideoFrame> frames = await CollectAsync(stream);

        Assert.Equal(new[] { 0, 1, 2, 3 }, frames.Select(frame => frame.Index).ToArray());
        Assert.Equal(new[] { "forward", "forward", "left" }, pipeline.Requests.Single().Actions);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
    }

    [Fact]
    public async Task SendAction_AfterTheStreamStarts_IsRejectedWithAClearError()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) => ClipFrames(request.Plan.Frames));
        IWorldSession session = service.OpenOneShot(pipeline, ClipPlan(4));
        session.SendAction("forward");

        await using IAsyncEnumerator<VideoFrame> frames = session.StreamAsync(CancellationToken.None).GetAsyncEnumerator();
        Assert.True(await frames.MoveNextAsync());

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() => session.SendAction("forward"));
        Assert.Contains("batch model", ex.Message);
        Assert.Contains("before StreamAsync starts", ex.Message);
    }

    [Fact]
    public async Task SecondEnumeration_ThrowsAClearError_EvenAfterTheFirstFinished()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) => ClipFrames(request.Plan.Frames));
        IWorldSession session = service.OpenOneShot(pipeline, ClipPlan(4));
        await CollectAsync(session.StreamAsync(CancellationToken.None));

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CollectAsync(session.StreamAsync(CancellationToken.None)));

        Assert.Contains("already streamed", ex.Message);
        Assert.Equal(1, pipeline.StartCount);
    }

    [Fact]
    public async Task Cancel_DuringTheRun_YieldsNothingAndIsReportedAsCancelled()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        SemaphoreSlim gate = new SemaphoreSlim(0);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            gate.Wait(cancel);
            return ClipFrames(request.Plan.Frames);
        });
        using CancellationTokenSource stop = new CancellationTokenSource();
        Task<List<VideoFrame>> stream = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(stop.Token));
        await pipeline.Started.WaitAsync(Patience);

        stop.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stream);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
    }

    [Fact]
    public async Task Cancel_AfterTheRunCompleted_IsStillCancelledNotSuccess()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        using CancellationTokenSource stop = new CancellationTokenSource();
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            // The clip finishes, but the cancel arrives before the engine takes delivery of it.
            stop.Cancel();
            return ClipFrames(request.Plan.Frames);
        });

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(stop.Token)));
        Assert.Equal(0, service.RunLedger.Outstanding);
    }

    [Fact]
    public async Task Cancel_DuringDelivery_DropsTheRestOfTheClip()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) => ClipFrames(request.Plan.Frames));
        using CancellationTokenSource stop = new CancellationTokenSource();
        List<VideoFrame> delivered = [];

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (VideoFrame frame in service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(stop.Token))
            {
                delivered.Add(frame);
                stop.Cancel();
            }
        });

        Assert.Single(delivered);
    }

    [Fact]
    public async Task RunComponents_AreReleasedOnceAfterSuccess()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ResourceTally tally = new ResourceTally();
        List<CountingComponent> components = [];
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            for (int i = 0; i < 3; i++)
            {
                components.Add(request.Phase.Own(new CountingComponent(tally, $"intermediate{i}")));
            }
            return ClipFrames(request.Plan.Frames);
        });

        await CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None));

        AssertReleasedExactlyOnce(tally, service.RunLedger, components);
    }

    [Fact]
    public async Task RunComponents_AreReleasedOnceWhenTheRunThrows()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ResourceTally tally = new ResourceTally();
        List<CountingComponent> components = [];
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            for (int i = 0; i < 3; i++)
            {
                components.Add(request.Phase.Own(new CountingComponent(tally, $"intermediate{i}")));
            }
            throw new InvalidOperationException("the sampler failed");
        });

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None)));

        Assert.Equal("the sampler failed", ex.Message);
        AssertReleasedExactlyOnce(tally, service.RunLedger, components);
    }

    [Fact]
    public async Task RunComponents_AreReleasedOnceWhenTheRunIsCancelled()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ResourceTally tally = new ResourceTally();
        List<CountingComponent> components = [];
        SemaphoreSlim gate = new SemaphoreSlim(0);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            for (int i = 0; i < 3; i++)
            {
                components.Add(request.Phase.Own(new CountingComponent(tally, $"intermediate{i}")));
            }
            gate.Wait(cancel);
            return ClipFrames(request.Plan.Frames);
        });
        using CancellationTokenSource stop = new CancellationTokenSource();
        Task<List<VideoFrame>> stream = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(stop.Token));
        await pipeline.Started.WaitAsync(Patience);

        stop.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stream);
        AssertReleasedExactlyOnce(tally, service.RunLedger, components);
    }

    [Fact]
    public async Task MalformedClip_IsRejected_AndItsComponentsAreStillReleased()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        ResourceTally tally = new ResourceTally();
        List<CountingComponent> components = [];
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            components.Add(request.Phase.Own(new CountingComponent(tally, "intermediate")));
            return ClipFrames(3);
        });

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None)));

        Assert.Contains("returned 3 frames", ex.Message);
        AssertReleasedExactlyOnce(tally, service.RunLedger, components, expectedCount: 1);
    }

    [Fact]
    public async Task TwoRuns_OnOneService_SerializeThroughTheLease()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        SemaphoreSlim gate = new SemaphoreSlim(0);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            gate.Wait(cancel);
            return ClipFrames(request.Plan.Frames);
        });
        Task<List<VideoFrame>> first = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None));
        await pipeline.Started.WaitAsync(Patience);
        Task<List<VideoFrame>> second = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None));

        await Task.Delay(TimeSpan.FromMilliseconds(100));
        Assert.Equal(1, pipeline.StartCount);
        Assert.Equal(1, service.QueuedRuns);

        gate.Release();
        await pipeline.Started.WaitAsync(Patience);
        Assert.Equal(2, pipeline.StartCount);
        gate.Release();
        await Task.WhenAll(first, second).WaitAsync(Patience);

        Assert.Equal(1, pipeline.MaxInFlight);
        Assert.Equal(0, service.QueuedRuns);
        Assert.Equal(0, service.RunLedger.Outstanding);
    }

    [Fact]
    public async Task CancelledWaitingRun_LeavesTheQueueCleanly()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        SemaphoreSlim gate = new SemaphoreSlim(0);
        ScriptedClipPipeline pipeline = new ScriptedClipPipeline((request, cancel) =>
        {
            gate.Wait(cancel);
            return ClipFrames(request.Plan.Frames);
        });
        Task<List<VideoFrame>> first = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None));
        await pipeline.Started.WaitAsync(Patience);

        using CancellationTokenSource stop = new CancellationTokenSource();
        Task<List<VideoFrame>> waiting = CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(stop.Token));
        Assert.Equal(1, service.QueuedRuns);
        stop.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        Assert.Equal(0, service.QueuedRuns);
        Assert.Equal(1, pipeline.StartCount);

        gate.Release();
        await first.WaitAsync(Patience);
        gate.Release();
        List<VideoFrame> later = await CollectAsync(service.OpenOneShot(pipeline, ClipPlan(4)).StreamAsync(CancellationToken.None)).WaitAsync(Patience);

        Assert.Equal(2, pipeline.StartCount);
        Assert.Equal(4, later.Count);
        Assert.False(service.RunInFlight);
        Assert.Equal(0, service.RunLedger.Outstanding);
    }

    [Fact]
    public void Dispose_ThenSendAction_IsRejected()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        IWorldSession session = service.OpenOneShot(new ScriptedClipPipeline((request, cancel) => ClipFrames(request.Plan.Frames)), ClipPlan(4));

        session.Dispose();

        Assert.Throws<ObjectDisposedException>(() => session.SendAction("forward"));
    }

    /// <summary>A plan for a catalog-less batch model: any backend, default caps, and a seed frame of 2 by 2 pixels.</summary>
    private static WorldPlan ClipPlan(int frames) => WorldPlanResolver.Resolve(
        WorldTestFixtures.Spec(null, "fixture-batch"),
        new WorldRequest { InitImage = WorldTestFixtures.Image(2, 2), Frames = frames, Seed = 1 },
        "cpu");

    /// <summary><paramref name="count"/> indexed 2 by 2 frames.</summary>
    private static IReadOnlyList<VideoFrame> ClipFrames(int count)
    {
        List<VideoFrame> frames = [];
        for (int i = 0; i < count; i++)
        {
            frames.Add(new VideoFrame { Rgb = new byte[2 * 2 * 3], Width = 2, Height = 2, Index = i });
        }
        return frames;
    }

    private static async Task<List<VideoFrame>> CollectAsync(IAsyncEnumerable<VideoFrame> stream)
    {
        List<VideoFrame> frames = [];
        await foreach (VideoFrame frame in stream)
        {
            frames.Add(frame);
        }
        return frames;
    }

    private static void AssertReleasedExactlyOnce(
        ResourceTally tally, WorldPhaseLedger ledger, List<CountingComponent> components, int expectedCount = 3)
    {
        Assert.Equal(expectedCount, components.Count);
        Assert.Equal(components.Count, tally.Acquired);
        Assert.Equal(tally.Acquired, tally.Released);
        Assert.All(components, component => Assert.Equal(1, component.DisposeCalls));
        Assert.Equal(0, ledger.Outstanding);
    }
}
