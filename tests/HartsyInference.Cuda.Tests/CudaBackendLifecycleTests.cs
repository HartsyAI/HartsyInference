using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Adversarial lifecycle coverage for backend abandonment, same-device registration fallback, and
/// backend-owned persistent side caches. These are CUDA-serial because every test deliberately creates multiple
/// backends on device zero and inspects the process-wide transfer-state registry.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaBackendLifecycleTests
{
    private const int LifecycleActive = 0;
    private const int LifecycleCleaned = 2;
    private readonly ITestOutputHelper _output;

    public CudaBackendLifecycleTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void AbandonedSameDeviceBackend_ReaperReclaimsOnlyAbandonedState_AndFreshBackendWorks()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int dim = 64;
        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        long enqueuedBefore = CudaBackend.AbandonedCleanupEnqueuedCount;
        long completedBefore = CudaBackend.AbandonedCleanupCompletedCount;
        long failedBefore = CudaBackend.AbandonedCleanupFailedCount;

        using Tensor inputA = RandomF32(new TensorShape(2, dim), 4101);
        using Tensor weightA = RandomF32(new TensorShape(dim, dim), 4102);
        float expectedA = ExpectedFirst(inputA, weightA, dim);
        CudaBackend backendA = new(0, PtxDir());
        nint keyA = backendA.TransferState.Key;
        try
        {
            using (Tensor first = new(inputA.Shape, DType.F32))
                Assert.Equal(expectedA, RunLinear(backendA, inputA, weightA, first), 3);
            Assert.Equal(registryBaseline + 1, GpuTransferHelper.RegisteredStateCount);

            AbandonedProbe abandoned = CreateAbandonedBackendWithLiveCaches();
            GpuTransferHelper.State stateB = abandoned.State;
            Assert.NotEqual(keyA, stateB.Key);
            Assert.Equal(registryBaseline + 2, GpuTransferHelper.RegisteredStateCount);
            Assert.Contains(stateB.Key, GpuTransferHelper.RegisteredStateKeysForTests);
            Assert.Single(stateB.WeightCache);
            KeyValuePair<Tensor, (ulong gpuPtr, nuint bytes)> activationEntry = Assert.Single(stateB.ActivationCache);
            Assert.Equal((nuint)(4L << 20), activationEntry.Value.bytes);
            // CachedBytes deliberately measures permanent weight/cast residency; activation bytes are asserted
            // from their own cache entry above.
            Assert.True(stateB.CachedBytes >= 16L << 20,
                $"Expected at least 16 MiB of intentional B weight cache, got {stateB.CachedBytes >> 20} MiB.");

            // The only strong backend reference died in the NoInlining producer. Its finalizer performs no CUDA
            // work: it claims lifecycle ownership and queues the instance for the managed reaper thread.
            bool enqueued = false;
            for (int attempt = 0; attempt < 20 && !enqueued; attempt++)
            {
                ForceFullGc();
                enqueued = CudaBackend.AbandonedCleanupEnqueuedCount >= enqueuedBefore + 1;
                if (!enqueued) Thread.Sleep(10);
            }
            Assert.True(enqueued, "Abandoned backend was not finalized/enqueued after forced collections.");
            Assert.True(CudaBackend.WaitForAbandonedCleanup(completedBefore + 1, TimeSpan.FromSeconds(20)),
                "Timed out waiting for the abandoned-backend reaper.");
            Assert.True(SpinWait.SpinUntil(
                    () => stateB.Unregistered && !GpuTransferHelper.IsStateRegistered(stateB.Key),
                    TimeSpan.FromSeconds(20)),
                "A different queued cleanup advanced the global counter, but backend B never retired.");
            ForceFullGc(); // collect the reaper's completed work item after its worker drops the local reference

            Assert.True(stateB.Retiring);
            Assert.True(stateB.Unregistered);
            Assert.False(abandoned.Backend.TryGetTarget(out _),
                "Reaper completed but the abandoned backend object remained strongly rooted.");
            Assert.False(GpuTransferHelper.IsStateRegistered(stateB.Key));
            Assert.DoesNotContain(stateB.Key, GpuTransferHelper.RegisteredStateKeysForTests);
            Assert.Equal(registryBaseline + 1, GpuTransferHelper.RegisteredStateCount);
            Assert.Empty(stateB.WeightCache);
            Assert.Empty(stateB.ActivationCache);
            Assert.Empty(stateB.WeightCastCache);
            Assert.Empty(stateB.CachedPointers);
            Assert.Empty(stateB.PendingOrphans);
            Assert.Empty(stateB.PinnedActivations);
            Assert.Empty(stateB.SidecarCache);
            Assert.Equal(0, stateB.CachedBytes);
            Assert.Equal(0, stateB.StreamHandle);
            Assert.Null(stateB.Context);
            Assert.Null(stateB.StreamingCache);
            Assert.Equal(failedBefore, CudaBackend.AbandonedCleanupFailedCount);

            // A is on the same primary context. B's retirement must neither evict A's resident weight nor leave
            // the shared device/context unusable.
            Assert.Contains(weightA, backendA.TransferState.WeightCache.Keys);
            long hitsBefore = backendA.TransferState.Hits;
            using Tensor second = new(inputA.Shape, DType.F32);
            Assert.Equal(expectedA, RunLinear(backendA, inputA, weightA, second), 3);
            Assert.True(backendA.TransferState.Hits > hitsBefore);
            Assert.Contains(keyA, GpuTransferHelper.RegisteredStateKeysForTests);
            backendA.FreeWeights([weightA]);
        }
        finally
        {
            backendA.Dispose();
        }

        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);

        // A brand-new owner after explicit + abandoned teardown is the final context/stream-health smoke.
        using CudaBackend fresh = new(0, PtxDir());
        using Tensor freshOutput = new(inputA.Shape, DType.F32);
        Assert.Equal(expectedA, RunLinear(fresh, inputA, weightA, freshOutput), 3);
        fresh.FreeWeights([weightA]);
    }

    [Fact]
    public void ConcurrentDispose_ExecutesCleanupExactlyOnce_WithoutExceptions()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        CudaBackend backend = new(0, PtxDir());
        GpuTransferHelper.State state = backend.TransferState;
        using Tensor source = RandomF32(new TensorShape(1, 4096), 4201);
        using Tensor liveActivation = new(source.Shape, DType.F32);
        backend.Scale(liveActivation, source, 1.25f);
        backend.Sync();
        using Tensor lifecycleScaleWeight = new(new TensorShape(8, 8), DType.F8E4M3)
        {
            Fp8InputScaleFactor = 0.5f,
        };
        Assert.NotEqual(0UL, backend.EnsureFp8InputScaleForTest(lifecycleScaleWeight));
        (int Count, long Allocations, long Frees) scaleBeforeDispose = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(1, scaleBeforeDispose.Count);
        Assert.Equal(LifecycleActive, backend.LifecycleStateForTests);
        Assert.Equal(0, backend.CleanupExecutionCount);
        Assert.Contains(liveActivation, state.ActivationCache.Keys);

        const int callers = 16;
        using ManualResetEventSlim start = new(initialState: false);
        ConcurrentQueue<Exception> failures = new();
        Thread[] threads = new Thread[callers];
        for (int i = 0; i < callers; i++)
        {
            threads[i] = new Thread(() =>
            {
                try
                {
                    start.Wait();
                    backend.Dispose();
                }
                catch (Exception error)
                {
                    failures.Enqueue(error);
                }
            }) { IsBackground = true };
            threads[i].Start();
        }
        start.Set();
        foreach (Thread thread in threads)
            Assert.True(thread.Join(TimeSpan.FromSeconds(30)), "Concurrent Dispose caller did not finish.");

        Assert.Empty(failures);
        Assert.Equal(LifecycleCleaned, backend.LifecycleStateForTests);
        Assert.Equal(1, backend.CleanupExecutionCount);
        Assert.True(state.Unregistered);
        Assert.DoesNotContain(state.Key, GpuTransferHelper.RegisteredStateKeysForTests);
        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);
        (int Count, long Allocations, long Frees) scaleAfterDispose = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(0, scaleAfterDispose.Count);
        Assert.Equal(scaleBeforeDispose.Allocations, scaleAfterDispose.Allocations);
        Assert.Equal(scaleBeforeDispose.Frees + 1, scaleAfterDispose.Frees);

        // Later calls are idempotent and cannot execute native cleanup again.
        backend.Dispose();
        Assert.Equal(1, backend.CleanupExecutionCount);
    }

    [Fact]
    public void DisposeWithOpenStepGraphCapture_AbortsPurgesAndLeavesDeviceReusable()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        using Tensor source = FilledF32(new TensorShape(1, 1024), 1.5f);
        using Tensor warmupOutput = new(source.Shape, DType.F32);
        using Tensor capturedOutput = new(source.Shape, DType.F32);
        CudaBackend backend = new(0, PtxDir());
        GpuTransferHelper.State state = backend.TransferState;
        try
        {
            if (!backend.StepGraphSupported)
            {
                _output.WriteLine("SKIPPED: StepGraph unsupported");
                return;
            }

            // Establish a resident input before capture so the captured Scale is a pure kernel plus a fresh
            // graph-private output allocation. Dispose must abort the still-open capture and purge that allocation
            // before attempting a stream synchronization or ordinary cache free.
            backend.Scale(warmupOutput, source, 1f);
            backend.Sync();
            backend.StepGraphBegin();
            backend.Scale(capturedOutput, warmupOutput, 2f);
            Assert.True(state.TrackCaptureWindow);
            Assert.NotEmpty(state.CaptureAllocs);

            Exception? failure = Record.Exception(backend.Dispose);
            Assert.Null(failure);
            Assert.Equal(LifecycleCleaned, backend.LifecycleStateForTests);
            Assert.Equal(1, backend.CleanupExecutionCount);
            Assert.True(state.Unregistered);
            Assert.False(GpuTransferHelper.IsStateRegistered(state.Key));
            Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);
        }
        finally
        {
            backend.Dispose();
        }

        // An aborted capture must not poison the shared primary context for the next owner.
        using CudaBackend fresh = new(0, PtxDir());
        using Tensor freshOutput = new(source.Shape, DType.F32);
        fresh.Scale(freshOutput, source, 3f);
        Assert.Equal(4.5f, ((float*)freshOutput.DataPointer)[0]);
    }

    [Fact]
    public void ThreeSameDeviceRegistrations_RemovingNewestRepointsAmbientlessFallback()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        CudaBackend backendC = new(0, PtxDir());
        GpuTransferHelper.State stateA = backendA.TransferState;
        GpuTransferHelper.State stateB = backendB.TransferState;
        GpuTransferHelper.State stateC = backendC.TransferState;
        nint contextHandle = stateA.Context!.Handle;
        try
        {
            Assert.Equal(registryBaseline + 3, GpuTransferHelper.RegisteredStateCount);
            Assert.Equal(stateC.Key, GpuTransferHelper.ContextFallbackKey(contextHandle));
            Assert.Equal(stateC.Key, ResolveWithoutAmbient(stateA));

            backendC.Dispose();
            Assert.Equal(registryBaseline + 2, GpuTransferHelper.RegisteredStateCount);
            Assert.Equal(stateB.Key, GpuTransferHelper.ContextFallbackKey(contextHandle));
            Assert.Equal(stateB.Key, ResolveWithoutAmbient(stateA));

            backendB.Dispose();
            Assert.Equal(registryBaseline + 1, GpuTransferHelper.RegisteredStateCount);
            Assert.Equal(stateA.Key, GpuTransferHelper.ContextFallbackKey(contextHandle));
            Assert.Equal(stateA.Key, ResolveWithoutAmbient(stateA));
        }
        finally
        {
            backendC.Dispose();
            backendB.Dispose();
            backendA.Dispose();
        }
        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);
    }

    [Fact]
    public void RetiringAmbient_ThrowsInsteadOfFallingThroughToSameContextSibling()
    {
        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        CudaContext fakeContext = (CudaContext)RuntimeHelpers.GetUninitializedObject(typeof(CudaContext));
        GC.SuppressFinalize(fakeContext);
        FieldInfo contextField = typeof(CudaContext).GetField("_context", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("CudaContext native-handle field was not found.");
        contextField.SetValue(fakeContext, (nint)0x5A1E_0001);

        GpuTransferHelper.State retiring = GpuTransferHelper.Register(fakeContext, (nint)0x101, streamingCache: null);
        GpuTransferHelper.State sibling = GpuTransferHelper.Register(fakeContext, (nint)0x202, streamingCache: null);
        using ManualResetEventSlim ambientBound = new(initialState: false);
        using ManualResetEventSlim retirementPublished = new(initialState: false);
        Exception? resolutionFailure = null;
        Thread resolver = new(() =>
        {
            GpuTransferHelper.SetAmbient(retiring);
            ambientBound.Set();
            retirementPublished.Wait();
            try
            {
                _ = GpuTransferHelper.CurrentState;
            }
            catch (Exception error)
            {
                resolutionFailure = error;
            }
        }) { IsBackground = true };

        try
        {
            resolver.Start();
            Assert.True(ambientBound.Wait(TimeSpan.FromSeconds(5)), "Resolver thread did not bind its ambient state.");

            // Retirement happens on a different thread. The resolver therefore still has the now-stale ambient
            // in its TLS slot; silently falling through would select the one remaining same-context sibling.
            GpuTransferHelper.BeginRetire(retiring);
            Assert.True(retiring.Retiring);
            Assert.False(retiring.Unregistered);
            Assert.Equal(sibling.Key, ResolveFreshThreadWithoutAmbient());

            retirementPublished.Set();
            Assert.True(resolver.Join(TimeSpan.FromSeconds(5)), "Resolver thread did not finish.");
            ObjectDisposedException disposed = Assert.IsType<ObjectDisposedException>(resolutionFailure);
            Assert.Contains("retiring", disposed.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            retirementPublished.Set();
            if (resolver.IsAlive) resolver.Join(TimeSpan.FromSeconds(5));
            if (!retiring.Unregistered) GpuTransferHelper.CompleteRetire(retiring);
            if (!sibling.Retiring) GpuTransferHelper.BeginRetire(sibling);
            if (!sibling.Unregistered) GpuTransferHelper.CompleteRetire(sibling);
        }

        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);
    }

    [Fact]
    public void CudaKernels_LateConstructorFailure_RollsBackAllAdoptedModulesSynchronously()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaContext context = new(0);
        long liveBaseline = CudaModule.LiveHandleCountForTests;
        long enqueuedBefore = CudaBackend.AbandonedCleanupEnqueuedCount;
        long completedBefore = CudaBackend.AbandonedCleanupCompletedCount;
        long failedBefore = CudaBackend.AbandonedCleanupFailedCount;
        long liveImmediatelyBeforeFailure = -1;
        int injectedCount = 0;
        InvalidOperationException sentinel = new("late CudaKernels construction failure sentinel");
        Func<string, Exception?>? previousInjector = CudaKernels.ModuleLoadFailureForTests;
        try
        {
            CudaKernels.ModuleLoadFailureForTests = path =>
            {
                if (!string.Equals(Path.GetFileName(path), "mul_mat_vec_q5k_q8_1.ptx", StringComparison.Ordinal))
                    return null;
                Interlocked.Increment(ref injectedCount);
                liveImmediatelyBeforeFailure = CudaModule.LiveHandleCountForTests;
                return sentinel;
            };

            InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() =>
                _ = new CudaKernels(PtxDir()));
            Assert.Same(sentinel, thrown);
            Assert.Equal(1, injectedCount);
            Assert.True(liveImmediatelyBeforeFailure > liveBaseline,
                $"The injected failure was not late: baseline={liveBaseline}, before failure={liveImmediatelyBeforeFailure}.");

            // Constructor catch owns rollback. Counts must be back at baseline before a GC/finalizer/reaper wait.
            long liveImmediatelyAfterFailure = CudaModule.LiveHandleCountForTests;
            Assert.Equal(liveBaseline, liveImmediatelyAfterFailure);
            Assert.Equal(enqueuedBefore, CudaBackend.AbandonedCleanupEnqueuedCount);
            Assert.Equal(completedBefore, CudaBackend.AbandonedCleanupCompletedCount);
            Assert.Equal(failedBefore, CudaBackend.AbandonedCleanupFailedCount);
            _output.WriteLine(
                $"CudaKernels late rollback: live baseline={liveBaseline}, before failure={liveImmediatelyBeforeFailure}, "
                + $"immediate after={liveImmediatelyAfterFailure}.");
        }
        finally
        {
            CudaKernels.ModuleLoadFailureForTests = previousInjector;
        }

        Assert.Equal(liveBaseline, CudaModule.LiveHandleCountForTests);
    }

    [Fact]
    public void Fp8InputScaleCache_ConcurrentEnsureIsSingleFlight_AndFreeWeightsEvictsSelectively()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir());
        using Tensor weight = new(new TensorShape(16, 16), DType.F8E4M3)
        {
            Fp8InputScaleFactor = 0.125f,
        };
        (int Count, long Allocations, long Frees) before = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(0, before.Count);

        using Barrier start = new(3);
        ulong first = 0;
        ulong second = 0;
        Exception? firstFailure = null;
        Exception? secondFailure = null;
        Thread t1 = new(() => Ensure(ref first, ref firstFailure)) { IsBackground = true };
        Thread t2 = new(() => Ensure(ref second, ref secondFailure)) { IsBackground = true };
        t1.Start();
        t2.Start();
        start.SignalAndWait();
        Assert.True(t1.Join(TimeSpan.FromSeconds(20)));
        Assert.True(t2.Join(TimeSpan.FromSeconds(20)));
        Assert.Null(firstFailure);
        Assert.Null(secondFailure);
        Assert.NotEqual(0UL, first);
        Assert.Equal(first, second);

        (int Count, long Allocations, long Frees) populated = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(1, populated.Count);
        Assert.Equal(before.Allocations + 1, populated.Allocations);
        Assert.Equal(before.Frees, populated.Frees);

        weight.Fp8InputScaleFactor = 0.375f;
        ulong updatedPointer = backend.EnsureFp8InputScaleForTest(weight);
        float updatedScale = 0;
        CudaMemory.CopyDeviceToHost(&updatedScale, updatedPointer, sizeof(float));
        Assert.Equal(first, updatedPointer);
        Assert.Equal(0.375f, updatedScale);
        Assert.Equal(populated, backend.Fp8InputScaleDiagnostics);

        foreach (float invalid in new[] { 0f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            using Tensor invalidWeight = new(new TensorShape(1), DType.F8E4M3)
            {
                Fp8InputScaleFactor = invalid,
            };
            Assert.Equal(0UL, backend.EnsureFp8InputScaleForTest(invalidWeight));
        }
        Assert.Equal(populated, backend.Fp8InputScaleDiagnostics);

        using Tensor secondWeight = new(new TensorShape(16, 16), DType.F8E4M3)
        {
            Fp8InputScaleFactor = 0.25f,
        };
        ulong secondWeightPointer = backend.EnsureFp8InputScaleForTest(secondWeight);
        Assert.NotEqual(0UL, secondWeightPointer);
        (int Count, long Allocations, long Frees) twoWeights = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(2, twoWeights.Count);
        Assert.Equal(populated.Allocations + 1, twoWeights.Allocations);
        Assert.Equal(populated.Frees, twoWeights.Frees);

        backend.FreeWeights([weight]);
        (int Count, long Allocations, long Frees) selective = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(1, selective.Count);
        Assert.Equal(twoWeights.Allocations, selective.Allocations);
        Assert.Equal(twoWeights.Frees + 1, selective.Frees);
        Assert.Equal(secondWeightPointer, backend.EnsureFp8InputScaleForTest(secondWeight));
        Assert.Equal(selective, backend.Fp8InputScaleDiagnostics);

        backend.FreeWeights([secondWeight]);
        (int Count, long Allocations, long Frees) evicted = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(0, evicted.Count);
        Assert.Equal(twoWeights.Allocations, evicted.Allocations);
        Assert.Equal(twoWeights.Frees + 2, evicted.Frees);

        void Ensure(ref ulong result, ref Exception? failure)
        {
            try
            {
                start.SignalAndWait();
                result = backend.EnsureFp8InputScaleForTest(weight);
            }
            catch (Exception error)
            {
                failure = error;
            }
        }
    }

    [Fact]
    public void Fp8InputScaleCache_AllBulkEvictionBoundariesFreeExactlyOnce()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend backend = new(0, PtxDir()) { EnableModulateEmitFp8 = true };
        using Tensor consumerWeight = new(new TensorShape(8, 8), DType.F8E4M3)
        {
            Fp8InputScaleFactor = 0.25f,
        };
        (int Count, long Allocations, long Frees) baseline = backend.Fp8InputScaleDiagnostics;

        PopulateFp8ScaleThroughProducer(backend, consumerWeight);
        AssertFp8Diagnostics(backend, count: 1, baseline.Allocations + 1, baseline.Frees);
        backend.FreeWeights([consumerWeight]);
        AssertFp8Diagnostics(backend, count: 0, baseline.Allocations + 1, baseline.Frees + 1);

        PopulateFp8ScaleThroughProducer(backend, consumerWeight);
        AssertFp8Diagnostics(backend, count: 1, baseline.Allocations + 2, baseline.Frees + 1);
        backend.FreePreloadedWeights();
        AssertFp8Diagnostics(backend, count: 0, baseline.Allocations + 2, baseline.Frees + 2);

        PopulateFp8ScaleThroughProducer(backend, consumerWeight);
        AssertFp8Diagnostics(backend, count: 1, baseline.Allocations + 3, baseline.Frees + 2);
        backend.EvictGpuCache();
        AssertFp8Diagnostics(backend, count: 0, baseline.Allocations + 3, baseline.Frees + 3);

        PopulateFp8ScaleThroughProducer(backend, consumerWeight);
        AssertFp8Diagnostics(backend, count: 1, baseline.Allocations + 4, baseline.Frees + 3);
        backend.FreeAllDeviceMemory();
        AssertFp8Diagnostics(backend, count: 0, baseline.Allocations + 4, baseline.Frees + 4);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static AbandonedProbe CreateAbandonedBackendWithLiveCaches()
    {
        CudaBackend backend = new(0, PtxDir());

        // 16 MiB resident weight plus a 4 MiB device-only activation makes accidental non-reclamation visible in
        // both registry diagnostics and state byte accounting without putting pressure on modest GPUs.
        Tensor weight = new(new TensorShape(2048, 2048), DType.F32);
        _ = weight.DataPointer; // allocate deterministic zeroed host storage before the H2D preload
        backend.PreloadWeights([weight]);
        Tensor source = RandomF32(new TensorShape(1, 1024, 1024), 4301);
        Tensor activation = new(source.Shape, DType.F32);
        backend.Scale(activation, source, 1.5f);
        backend.Sync();

        return new AbandonedProbe(new WeakReference<CudaBackend>(backend), backend.TransferState);
        // Deliberately do not Dispose backend, weight, source, or activation. The state strongly owns the cached
        // weight/activation until the backend reaper clears it; source becomes ordinarily collectible.
    }

    private static void PopulateFp8ScaleThroughProducer(CudaBackend backend, Tensor consumerWeight)
    {
        const int rows = 2;
        const int dim = 8;
        using Tensor input = FilledF32(new TensorShape(rows, dim), 0.5f);
        using Tensor scaleTable = FilledF32(new TensorShape(1, dim), 0f);
        using Tensor rowIndex = RowIndices(rows, 0);
        using Tensor outputFp8 = new(input.Shape, DType.F8E4M3);

        backend.PreloadWeights([consumerWeight]);
        Assert.True(backend.TryAffineBroadcastRowIndexedToFp8(
            outputFp8, input, scaleTable, shiftTable: null, rowIndex, consumerWeight),
            "dit_fp8emit.ptx is required for the portable static-scale production trigger.");
        backend.Sync();
    }

    private static void AssertFp8Diagnostics(CudaBackend backend, int count, long allocations, long frees)
    {
        (int Count, long Allocations, long Frees) actual = backend.Fp8InputScaleDiagnostics;
        Assert.Equal(count, actual.Count);
        Assert.Equal(allocations, actual.Allocations);
        Assert.Equal(frees, actual.Frees);
    }

    private static nint ResolveWithoutAmbient(GpuTransferHelper.State contextOwner)
    {
        nint resolved = 0;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try
            {
                contextOwner.Context!.EnsureCurrent();
                resolved = GpuTransferHelper.CurrentState.Key;
            }
            catch (Exception error)
            {
                failure = error;
            }
        }) { IsBackground = true };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(20)));
        Assert.Null(failure);
        return resolved;
    }

    private static nint ResolveFreshThreadWithoutAmbient()
    {
        nint resolved = 0;
        Exception? failure = null;
        Thread thread = new(() =>
        {
            try { resolved = GpuTransferHelper.CurrentState.Key; }
            catch (Exception error) { failure = error; }
        }) { IsBackground = true };
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(5)));
        Assert.Null(failure);
        return resolved;
    }

    private static float RunLinear(CudaBackend backend, Tensor input, Tensor weight, Tensor output)
    {
        backend.HighPrecisionGemm = true;
        backend.PreloadWeights([weight]);
        backend.Linear(output, input, weight, bias: null);
        return ((float*)output.DataPointer)[0];
    }

    private static float ExpectedFirst(Tensor input, Tensor weight, int dim)
    {
        float* inputData = (float*)input.DataPointer;
        float* weightData = (float*)weight.DataPointer;
        double sum = 0;
        for (int k = 0; k < dim; k++) sum += inputData[k] * weightData[k];
        return (float)sum;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        Random random = new(seed);
        for (long i = 0; i < tensor.ElementCount; i++)
            data[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    private static Tensor FilledF32(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++) data[i] = value;
        return tensor;
    }

    private static Tensor RowIndices(int count, int value)
    {
        Tensor tensor = new(new TensorShape(count), DType.I32);
        int* data = (int*)tensor.DataPointer;
        for (int i = 0; i < count; i++) data[i] = value;
        return tensor;
    }

    private static void ForceFullGc()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
    }

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private readonly record struct AbandonedProbe(
        WeakReference<CudaBackend> Backend,
        GpuTransferHelper.State State);
}
