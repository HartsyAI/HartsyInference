using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// CPU-only lifecycle tests for Anima's backend-keyed RoPE table cache. The proxy records the exact
/// host tensors handed to a backend and can inject failures without constructing a device backend.
/// </summary>
public sealed class AnimaRopeGpuTableCacheTests
{
    private const int Frames = 1;
    private const int GridHeight = 3;
    private const int GridWidth = 5;
    private const int HeadDim = 12;

    [Fact]
    public void Cache_GeometryReplacement_EvictsOldPairBeforePublishingNewPair()
    {
        AnimaRope rope = CreateRope();
        IBackend backend = RecordingWeightBackend.Create(out RecordingWeightBackend recording);

        try
        {
            (Tensor Cos, Tensor Sin) first =
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);
            (Tensor Cos, Tensor Sin) replacement = rope.GetOrCreateTables(
                backend, tFrames: 2, hPatched: 2, wPatched: 4);

            Assert.Equal(2, recording.PreloadCallCount);
            Assert.Equal(1, recording.FreeCallCount);
            AssertBatch(recording.PreloadBatch(0), first);
            AssertBatch(recording.FreeBatch(0), first);
            AssertBatch(recording.PreloadBatch(1), replacement);
            Assert.NotSame(first.Cos, replacement.Cos);
            Assert.NotSame(first.Sin, replacement.Sin);
            Assert.False(recording.IsResident(first.Cos));
            Assert.False(recording.IsResident(first.Sin));
            Assert.True(recording.IsResident(replacement.Cos));
            Assert.True(recording.IsResident(replacement.Sin));
            Assert.Equal(new TensorShape(16, HeadDim), replacement.Cos.Shape);
            Assert.Equal(new TensorShape(16, HeadDim), replacement.Sin.Shape);
        }
        finally
        {
            rope.ReleaseGpuTables(backend);
            backend.Dispose();
        }
    }

    [Fact]
    public void Cache_TwoBackends_IsolatesIdentityReuseAndRelease()
    {
        AnimaRope rope = CreateRope();
        IBackend backendA = RecordingWeightBackend.Create(out RecordingWeightBackend recordingA);
        IBackend backendB = RecordingWeightBackend.Create(out RecordingWeightBackend recordingB);

        try
        {
            (Tensor Cos, Tensor Sin) firstA =
                rope.GetOrCreateTables(backendA, Frames, GridHeight, GridWidth);
            (Tensor Cos, Tensor Sin) firstB =
                rope.GetOrCreateTables(backendB, Frames, GridHeight, GridWidth);
            (Tensor Cos, Tensor Sin) reusedA =
                rope.GetOrCreateTables(backendA, Frames, GridHeight, GridWidth);
            (Tensor Cos, Tensor Sin) reusedB =
                rope.GetOrCreateTables(backendB, Frames, GridHeight, GridWidth);

            Assert.Same(firstA.Cos, reusedA.Cos);
            Assert.Same(firstA.Sin, reusedA.Sin);
            Assert.Same(firstB.Cos, reusedB.Cos);
            Assert.Same(firstB.Sin, reusedB.Sin);
            Assert.NotSame(firstA.Cos, firstB.Cos);
            Assert.NotSame(firstA.Sin, firstB.Sin);
            Assert.Equal(1, recordingA.PreloadCallCount);
            Assert.Equal(1, recordingB.PreloadCallCount);

            rope.ReleaseGpuTables(backendA);

            Assert.Equal(1, recordingA.FreeCallCount);
            Assert.Equal(0, recordingB.FreeCallCount);
            Assert.False(recordingA.IsResident(firstA.Cos));
            Assert.False(recordingA.IsResident(firstA.Sin));
            Assert.True(recordingB.IsResident(firstB.Cos));
            Assert.True(recordingB.IsResident(firstB.Sin));

            (Tensor Cos, Tensor Sin) stillReusedB =
                rope.GetOrCreateTables(backendB, Frames, GridHeight, GridWidth);
            (Tensor Cos, Tensor Sin) recreatedA =
                rope.GetOrCreateTables(backendA, Frames, GridHeight, GridWidth);

            Assert.Same(firstB.Cos, stillReusedB.Cos);
            Assert.Same(firstB.Sin, stillReusedB.Sin);
            Assert.NotSame(firstA.Cos, recreatedA.Cos);
            Assert.NotSame(firstA.Sin, recreatedA.Sin);
            Assert.Equal(2, recordingA.PreloadCallCount);
            Assert.Equal(1, recordingB.PreloadCallCount);
        }
        finally
        {
            rope.ReleaseGpuTables(backendA);
            rope.ReleaseGpuTables(backendB);
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    [Fact]
    public void Cache_PreloadFailure_RollsBackPartialResidencyWithoutPublication()
    {
        AnimaRope rope = CreateRope();
        IBackend backend = RecordingWeightBackend.Create(out RecordingWeightBackend recording);
        recording.FailNextPreloadAfterFirstTensor();

        try
        {
            InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth));
            Assert.Equal(RecordingWeightBackend.PreloadFailureMessage, error.Message);

            Assert.Equal(1, recording.PreloadCallCount);
            Assert.Equal(1, recording.FreeCallCount);
            Tensor[] failedBatch = recording.PreloadBatch(0);
            AssertBatch(recording.FreeBatch(0), (failedBatch[0], failedBatch[1]));
            Assert.Equal(0, recording.ResidentCount);

            (Tensor Cos, Tensor Sin) replacement =
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);

            Assert.Equal(2, recording.PreloadCallCount);
            Assert.NotSame(failedBatch[0], replacement.Cos);
            Assert.NotSame(failedBatch[1], replacement.Sin);
            AssertBatch(recording.PreloadBatch(1), replacement);
            Assert.True(recording.IsResident(replacement.Cos));
            Assert.True(recording.IsResident(replacement.Sin));
        }
        finally
        {
            rope.ReleaseGpuTables(backend);
            backend.Dispose();
        }
    }

    [Fact]
    public void Cache_FreeFailure_RetainsPublishedEntryForRetry()
    {
        AnimaRope rope = CreateRope();
        IBackend backend = RecordingWeightBackend.Create(out RecordingWeightBackend recording);

        try
        {
            (Tensor Cos, Tensor Sin) first =
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);
            recording.FailNextFreeBeforeMutation();

            InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
                rope.ReleaseGpuTables(backend));
            Assert.Equal(RecordingWeightBackend.FreeFailureMessage, error.Message);
            Assert.Equal(1, recording.FreeCallCount);
            Assert.True(recording.IsResident(first.Cos));
            Assert.True(recording.IsResident(first.Sin));

            (Tensor Cos, Tensor Sin) retained =
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);
            Assert.Same(first.Cos, retained.Cos);
            Assert.Same(first.Sin, retained.Sin);
            Assert.Equal(1, recording.PreloadCallCount);

            rope.ReleaseGpuTables(backend);
            Assert.Equal(2, recording.FreeCallCount);
            Assert.Equal(0, recording.ResidentCount);

            (Tensor Cos, Tensor Sin) recreated =
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);
            Assert.NotSame(first.Cos, recreated.Cos);
            Assert.NotSame(first.Sin, recreated.Sin);
            Assert.Equal(2, recording.PreloadCallCount);
        }
        finally
        {
            rope.ReleaseGpuTables(backend);
            backend.Dispose();
        }
    }

    [Fact]
    public async Task Cache_ConcurrentSameKeyCallers_PublishExactlyOnePair()
    {
        const int followerCount = 8;
        AnimaRope rope = CreateRope();
        IBackend backend = RecordingWeightBackend.Create(out RecordingWeightBackend recording);
        using ManualResetEventSlim preloadEntered = new(initialState: false);
        using ManualResetEventSlim continuePreload = new(initialState: false);
        using CountdownEvent followersStarted = new(followerCount);
        recording.GateNextPreload(preloadEntered, continuePreload);

        List<Task<(Tensor Cos, Tensor Sin)>> calls = [];
        try
        {
            calls.Add(StartConcurrentCall(() =>
                rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth)));

            bool preloadEnteredInTime = preloadEntered.Wait(TimeSpan.FromSeconds(10));
            Task<(Tensor Cos, Tensor Sin)>[] followers = preloadEnteredInTime
                ? Enumerable.Range(0, followerCount)
                    .Select(_ => StartConcurrentCall(() =>
                    {
                        followersStarted.Signal();
                        return rope.GetOrCreateTables(backend, Frames, GridHeight, GridWidth);
                    }))
                    .ToArray()
                : [];
            calls.AddRange(followers);

            bool followersStartedInTime = preloadEnteredInTime
                && followersStarted.Wait(TimeSpan.FromSeconds(10));
            continuePreload.Set();

            (Tensor Cos, Tensor Sin)[] results = await Task.WhenAll(calls);

            Assert.True(preloadEnteredInTime, "The first caller did not reach the gated preload.");
            Assert.True(followersStartedInTime, "Not all concurrent followers started before publication resumed.");
            Assert.Equal(followerCount + 1, results.Length);
            Assert.Equal(1, recording.PreloadCallCount);
            Assert.Equal(0, recording.FreeCallCount);
            Assert.Equal(2, recording.ResidentCount);
            foreach ((Tensor Cos, Tensor Sin) result in results)
            {
                Assert.Same(results[0].Cos, result.Cos);
                Assert.Same(results[0].Sin, result.Sin);
            }
            AssertBatch(recording.PreloadBatch(0), results[0]);
        }
        finally
        {
            continuePreload.Set();
            rope.ReleaseGpuTables(backend);
            backend.Dispose();
        }
    }

    private static AnimaRope CreateRope() =>
        new(HeadDim, theta: 10_000.0f, ropeScale: (2.0f, 1.25f, 0.75f));

    private static Task<(Tensor Cos, Tensor Sin)> StartConcurrentCall(
        Func<(Tensor Cos, Tensor Sin)> call) =>
        Task.Factory.StartNew(
            call,
            CancellationToken.None,
            TaskCreationOptions.LongRunning,
            TaskScheduler.Default);

    private static void AssertBatch(Tensor[] actual, (Tensor Cos, Tensor Sin) expected)
    {
        Assert.Equal(2, actual.Length);
        Assert.Same(expected.Cos, actual[0]);
        Assert.Same(expected.Sin, actual[1]);
    }

    /// <summary>A strict fake for the two backend lifecycle methods used by the RoPE cache.</summary>
    public class RecordingWeightBackend : DispatchProxy
    {
        internal const string PreloadFailureMessage = "Injected partial preload failure.";
        internal const string FreeFailureMessage = "Injected weight eviction failure.";

        private readonly object _sync = new();
        private readonly List<Tensor[]> _preloadBatches = [];
        private readonly List<Tensor[]> _freeBatches = [];
        private readonly HashSet<Tensor> _resident = new(ReferenceEqualityComparer.Instance);
        private bool _failNextPreload;
        private bool _failNextFree;
        private (ManualResetEventSlim Entered, ManualResetEventSlim Continue)? _nextPreloadGate;

        public int PreloadCallCount
        {
            get { lock (_sync) return _preloadBatches.Count; }
        }

        public int FreeCallCount
        {
            get { lock (_sync) return _freeBatches.Count; }
        }

        public int ResidentCount
        {
            get { lock (_sync) return _resident.Count; }
        }

        public static IBackend Create(out RecordingWeightBackend recording)
        {
            IBackend backend = DispatchProxy.Create<IBackend, RecordingWeightBackend>();
            recording = (RecordingWeightBackend)backend;
            return backend;
        }

        public Tensor[] PreloadBatch(int index)
        {
            lock (_sync)
                return (Tensor[])_preloadBatches[index].Clone();
        }

        public Tensor[] FreeBatch(int index)
        {
            lock (_sync)
                return (Tensor[])_freeBatches[index].Clone();
        }

        public bool IsResident(Tensor tensor)
        {
            lock (_sync)
                return _resident.Contains(tensor);
        }

        public void FailNextPreloadAfterFirstTensor()
        {
            lock (_sync)
                _failNextPreload = true;
        }

        public void FailNextFreeBeforeMutation()
        {
            lock (_sync)
                _failNextFree = true;
        }

        public void GateNextPreload(ManualResetEventSlim entered, ManualResetEventSlim @continue)
        {
            ArgumentNullException.ThrowIfNull(entered);
            ArgumentNullException.ThrowIfNull(@continue);
            lock (_sync)
                _nextPreloadGate = (entered, @continue);
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            args ??= [];

            if (targetMethod.Name == nameof(IBackend.PreloadWeights))
            {
                Preload(MaterializeWeights(args));
                return null;
            }
            if (targetMethod.Name == nameof(IBackend.FreeWeights))
            {
                Free(MaterializeWeights(args));
                return null;
            }
            if (targetMethod.Name == nameof(IDisposable.Dispose))
                return null;

            throw new NotSupportedException(
                $"RecordingWeightBackend does not implement {targetMethod.DeclaringType?.Name}.{targetMethod.Name}.");
        }

        private void Preload(Tensor[] batch)
        {
            bool fail;
            (ManualResetEventSlim Entered, ManualResetEventSlim Continue)? gate;
            lock (_sync)
            {
                _preloadBatches.Add(batch);
                fail = _failNextPreload;
                _failNextPreload = false;
                gate = _nextPreloadGate;
                _nextPreloadGate = null;
            }

            if (gate is { } activeGate)
            {
                activeGate.Entered.Set();
                if (!activeGate.Continue.Wait(TimeSpan.FromSeconds(30)))
                    throw new TimeoutException("Timed out waiting to release the fake preload gate.");
            }

            lock (_sync)
            {
                if (fail)
                {
                    if (batch.Length == 0)
                        throw new InvalidOperationException("Cannot inject a partial preload for an empty batch.");
                    _resident.Add(batch[0]);
                    throw new InvalidOperationException(PreloadFailureMessage);
                }

                foreach (Tensor tensor in batch)
                    _resident.Add(tensor);
            }
        }

        private void Free(Tensor[] batch)
        {
            lock (_sync)
            {
                _freeBatches.Add(batch);
                if (_failNextFree)
                {
                    _failNextFree = false;
                    throw new InvalidOperationException(FreeFailureMessage);
                }

                foreach (Tensor tensor in batch)
                    _resident.Remove(tensor);
            }
        }

        private static Tensor[] MaterializeWeights(object?[] args)
        {
            if (args.Length != 1 || args[0] is not IEnumerable<Tensor> weights)
                throw new InvalidOperationException("Expected one tensor sequence argument.");
            return weights.ToArray();
        }
    }
}
