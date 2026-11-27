using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;
using HartsyInference.World.Sessions;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary><see cref="ChunkedWorldSession"/> over a scripted stream: frames arrive in order chunk by chunk, live actions reach
/// the stream, the compute thread that opened the stream also disposes it, a slow consumer holds the producer back, and
/// failure, cancellation and disposal all release the stream and the execution lease.</summary>
public sealed class ChunkedWorldSessionTests
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(10);

    [Fact]
    public void AChunkedSession_DefersItsDeviceWorkToTheStream()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        ScriptedPipeline pipeline = new(_ => new ScriptedStream(25, 25, 24));
        using IWorldSession session = service.OpenChunked(pipeline, Plan(25));

        Assert.True(session.DefersDeviceWork);
    }

    [Fact]
    public async Task Bounded_DeliversEveryChunkInOrder_AndReleases()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        ScriptedPipeline pipeline = new(_ => new ScriptedStream(25 + 24 * 2, 25, 24));
        IWorldSession session = service.OpenChunked(pipeline, Plan(73));
        session.SendAction("w-10");
        List<int> indices = [];
        await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
            indices.Add(frame.Index);

        Assert.Equal(Enumerable.Range(0, 73), indices);
        Assert.Equal(["w-10"], pipeline.Stream!.Actions);
        Assert.Equal(3, pipeline.Stream.ChunksGenerated);
        Assert.True(pipeline.Stream.Disposed);
        Assert.Equal(pipeline.Stream.GenerateThread, pipeline.Stream.DisposeThread);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (VideoFrame _ in session.StreamAsync(CancellationToken.None))
            {
            }
        });
    }

    [Fact]
    public async Task Unbounded_TakesLiveActions_StopsOnDispose_AndWaitsForTheConsumer()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        ScriptedPipeline pipeline = new(_ => new ScriptedStream(null, 25, 24));
        IWorldSession session = service.OpenChunked(pipeline, Plan(0, unbounded: true));
        int frames = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
            {
                frames++;
                if (frame.Index == 30)
                    session.SendAction("d");
                if (frame.Index == 100)
                    session.Dispose();
            }
        });

        Assert.Equal(101, frames);
        Assert.Contains("d", pipeline.Stream!.Actions);
        // The action sent while chunk 1 was being read is seen by a chunk generated after it, not by one already generated.
        int firstSeen = pipeline.Stream.ActionsSeenAtChunk.FindIndex(count => count > 0);
        Assert.InRange(firstSeen, 2, 3);
        Assert.True(pipeline.Stream.Disposed);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
        // Frame 100 lies in chunk 4, and the producer waits for room before each chunk: at most chunk 5 sits buffered.
        Assert.InRange(pipeline.Stream.ChunksGenerated, 5, 6);
    }

    [Fact]
    public async Task Failure_WhileOpening_ReachesTheConsumer_AndReleases()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        ScriptedPipeline pipeline = new(_ => throw new InvalidDataException("bad checkpoint"));
        IWorldSession session = service.OpenChunked(pipeline, Plan(49));
        await Assert.ThrowsAsync<InvalidDataException>(async () =>
        {
            await foreach (VideoFrame _ in session.StreamAsync(CancellationToken.None))
            {
            }
        });

        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
    }

    [Fact]
    public async Task Failure_InALaterChunk_ReachesTheConsumer_AfterTheEarlierFrames()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        ScriptedPipeline pipeline = new(_ => new ScriptedStream(73, 25, 24) { FailAtChunk = 1 });
        IWorldSession session = service.OpenChunked(pipeline, Plan(73));
        int frames = 0;
        await Assert.ThrowsAsync<InvalidDataException>(async () =>
        {
            await foreach (VideoFrame _ in session.StreamAsync(CancellationToken.None))
                frames++;
        });

        Assert.Equal(25, frames);
        Assert.True(pipeline.Stream!.Disposed);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
    }

    [Fact]
    public async Task Cancel_WhileAChunkIsInFlight_EndsAsCancelled()
    {
        using InferenceEngine engine = new("cpu");
        WorldService service = new(engine);
        using SemaphoreSlim gate = new(0);
        ScriptedPipeline pipeline = new(_ => new ScriptedStream(49, 25, 24) { Gate = gate });
        using CancellationTokenSource stop = new();
        IWorldSession session = service.OpenChunked(pipeline, Plan(49));
        Task run = Task.Run(async () =>
        {
            await foreach (VideoFrame _ in session.StreamAsync(stop.Token))
            {
            }
        });
        Assert.True(await pipeline.Entered.WaitAsync(Patience), "the first chunk never started");
        stop.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run.WaitAsync(Patience));
        Assert.True(pipeline.Stream!.Disposed);
        Assert.Equal(0, pipeline.Stream.ChunksGenerated);
        Assert.Equal(0, service.RunLedger.Outstanding);
        Assert.False(service.RunInFlight);
    }

    private static WorldPlan Plan(int frames, bool unbounded = false) => WorldPlanResolver.Resolve(
        WorldTestFixtures.Spec(null, "fixture-batch"),
        new WorldRequest { InitImage = WorldTestFixtures.Image(2, 2), Frames = Math.Max(frames, 2), Seed = 1 },
        "cpu") with { Streaming = true, Unbounded = unbounded, Frames = unbounded ? 0 : frames };

    /// <summary>A stream that makes blank 2x2 chunks: a first chunk of <c>firstChunk</c> frames, then <c>chunk</c> frames each,
    /// until <c>totalFrames</c> (or forever when null). It records actions and the threads that generate and dispose it.</summary>
    private sealed class ScriptedStream(int? totalFrames, int firstChunk, int chunk) : IChunkedWorldStream
    {
        public List<string> Actions { get; } = [];
        public List<int> ActionsSeenAtChunk { get; } = [];
        public bool Disposed { get; private set; }
        public int DisposeThread { get; private set; }
        public int GenerateThread { get; private set; }
        public SemaphoreSlim? Gate { get; init; }
        public SemaphoreSlim? Entered { get; set; }
        public int? FailAtChunk { get; init; }
        public int FrameWidth => 2;
        public int FrameHeight => 2;
        public int ChunksGenerated { get; private set; }
        public int FramesGenerated { get; private set; }
        public int NextChunkFrames => Completed ? 0 : ChunksGenerated == 0 ? firstChunk : chunk;
        public bool Completed => totalFrames is int total && FramesGenerated >= total;

        public void SendAction(string action)
        {
            lock (Actions)
                Actions.Add(action);
        }

        public IReadOnlyList<Video.VideoFrame> GenerateChunk(CancellationToken cancel, Action<GenerationProgress>? progress = null)
        {
            GenerateThread = Environment.CurrentManagedThreadId;
            Entered?.Release();
            Gate?.Wait(cancel);
            cancel.ThrowIfCancellationRequested();
            if (FailAtChunk == ChunksGenerated)
                throw new InvalidDataException($"chunk {ChunksGenerated} failed");
            lock (Actions)
                ActionsSeenAtChunk.Add(Actions.Count);
            int count = NextChunkFrames;
            Video.VideoFrame[] frames = new Video.VideoFrame[count];
            for (int i = 0; i < count; i++)
                frames[i] = new Video.VideoFrame(FramesGenerated + i, 2, 2, new byte[12]);
            FramesGenerated += count;
            ChunksGenerated++;
            return frames;
        }

        public void Dispose()
        {
            Disposed = true;
            DisposeThread = Environment.CurrentManagedThreadId;
        }
    }

    /// <summary>Opens the scripted stream <c>open</c> makes and replays the request's opening actions into it.</summary>
    private sealed class ScriptedPipeline(Func<ChunkedWorldRequest, ScriptedStream> open) : IChunkedWorldPipeline
    {
        public ScriptedStream? Stream { get; private set; }
        public SemaphoreSlim Entered { get; } = new(0);

        public IChunkedWorldStream OpenStream(ChunkedWorldRequest request, CancellationToken cancel)
        {
            Stream = open(request);
            Stream.Entered = Entered;
            foreach (string action in request.Actions)
                Stream.SendAction(action);
            return Stream;
        }
    }
}
