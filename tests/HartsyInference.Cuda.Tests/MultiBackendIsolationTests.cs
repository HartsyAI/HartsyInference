using System.Runtime.CompilerServices;
using System.Threading;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Guards the per-BACKEND transfer-state isolation: two <see cref="CudaBackend"/>s in one process — on
/// different GPUs or the SAME one — must not clobber each other's streams, caches, or activation callbacks.
/// History: with fully-static state, constructing a second backend retargeted everything at the new context
/// (cross-device frees → CUDA_ERROR_ILLEGAL_ADDRESS); with per-context keying, two same-device backends still
/// collapsed into one State (primary contexts are one-per-device). States are now keyed per backend and resolved
/// via the EnterOp thread ambient, so the SameDevice_* tests below are first-class contract, not a fallback.</summary>
[Collection("CudaSerial")]
public sealed unsafe class MultiBackendIsolationTests
{
    private readonly ITestOutputHelper _output;
    public MultiBackendIsolationTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>Runs a small Linear on the backend, preloading the weight (exercises the weight cache),
    /// and returns the first output element (forces the lazy D2H sync — exercises the activation callback).</summary>
    private static float RunLinear(CudaBackend backend, Tensor input, Tensor weight, Tensor output)
    {
        // This test guards state isolation, not GEMM numerics — pin full-F32 math so the 3-decimal assert
        // against the CPU reference can't sit on the default TF32 path's ~1e-3 error boundary (it did: seed 1/2
        // lands at -1.6545 CPU vs -1.6552 TF32 on Ada).
        backend.HighPrecisionGemm = true;
        backend.PreloadWeights(new[] { weight });
        backend.Linear(output, input, weight, bias: null);
        return ((float*)output.DataPointer)[0]; // lazy sync fires here, on this backend's stream/context
    }

    [Fact]
    public void TwoBackends_InterleavedOps_NoCrossContextCorruption()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int deviceCount = CudaContext.GetDeviceCount();
        // Prefer two physical GPUs; fall back to two backends on device 0 (shared primary context —
        // still exercises registration/unregistration, though not cross-device isolation).
        int secondOrdinal = deviceCount >= 2 ? 1 : 0;
        _output.WriteLine($"Devices: {deviceCount}; second backend on ordinal {secondOrdinal}.");

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);

        using Tensor inputA = RandomF32(ioShape, 1);
        using Tensor weightA = RandomF32(wShape, 2);
        using Tensor inputB = RandomF32(ioShape, 3);
        using Tensor weightB = RandomF32(wShape, 4);

        // CPU reference for both results.
        float ExpectedFirst(Tensor input, Tensor weight)
        {
            float* ip = (float*)input.DataPointer;
            float* wp = (float*)weight.DataPointer;
            double acc = 0;
            for (int k = 0; k < dim; k++) acc += ip[k] * wp[k]; // out[0,0] = input[0,:] · weight[0,:]
            return (float)acc;
        }
        float expectedA = ExpectedFirst(inputA, weightA);
        float expectedB = ExpectedFirst(inputB, weightB);

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(secondOrdinal, PtxDir());
        try
        {
            // Interleave ops A→B→A→B: with shared static state, B's construction/ops would have retargeted
            // A's stream + caches, and A's second op (or its lazy sync callback) would fault.
            using Tensor outA1 = new(ioShape, DType.F32);
            using Tensor outB1 = new(ioShape, DType.F32);
            using Tensor outA2 = new(ioShape, DType.F32);
            using Tensor outB2 = new(ioShape, DType.F32);

            float a1 = RunLinear(backendA, inputA, weightA, outA1);
            float b1 = RunLinear(backendB, inputB, weightB, outB1);
            float a2 = RunLinear(backendA, inputA, weightA, outA2);
            float b2 = RunLinear(backendB, inputB, weightB, outB2);

            Assert.Equal(expectedA, a1, 3);
            Assert.Equal(expectedB, b1, 3);
            Assert.Equal(a1, a2, 5);
            Assert.Equal(b1, b2, 5);

            backendA.FreeWeights(new[] { weightA });
            backendB.FreeWeights(new[] { weightB });
        }
        finally
        {
            // Dispose in construction order — the second dispose must not fault on the first's freed state.
            backendA.Dispose();
            backendB.Dispose();
        }

        // The process must remain healthy: a fresh backend on device 0 still works after both disposals.
        using CudaBackend backendC = new(0, PtxDir());
        using Tensor outC = new(ioShape, DType.F32);
        float c = RunLinear(backendC, inputA, weightA, outC);
        Assert.Equal(expectedA, c, 3);
        backendC.FreeWeights(new[] { weightA });
    }

    private static float ExpectedFirst(Tensor input, Tensor weight, int dim)
    {
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        double acc = 0;
        for (int k = 0; k < dim; k++) acc += ip[k] * wp[k]; // out[0,0] = input[0,:] · weight[0,:]
        return (float)acc;
    }

    /// <summary>Runs a Linear whose output is never read and never disposed, so its GPU cleanup callback stays
    /// planted and — once the tensor is GC-finalized — is queued into THIS backend's context bucket. The reference
    /// dies at method return (NoInlining keeps it from being kept alive by the caller frame).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void QueueAbandonedGpuTensor(CudaBackend backend, Tensor input, Tensor weight, TensorShape ioShape)
    {
        Tensor abandoned = new(ioShape, DType.F32);
        backend.Linear(abandoned, input, weight, bias: null);
        // Deliberately no read (no lazy D2H sync) and no Dispose — the dispose callback survives to the finalizer.
    }

    /// <summary>The concurrency guard the interleaved test can't provide: two backends on two DISTINCT GPUs generating
    /// on two threads AT THE SAME TIME, while each thread abandons GPU tensors and forces GC so finalizer-cleanup
    /// callbacks pile up. Before the queue was partitioned (first by context, now by per-backend StateKey), thread
    /// B's drain ran thread A's callbacks — mutating A's non-thread-safe GpuTransferHelper State while A's own
    /// thread mutated it → intermittent leak / throw / illegal-address. Requires 2 physical GPUs so the kernels
    /// genuinely execute concurrently; the same-device analogue (still serialized by the engine's DeviceGate until
    /// the concurrency milestone) is <see cref="SameDevice_TwoBackends_InterleavedOps_IndependentStreamsAndCaches"/>.</summary>
    [Fact]
    public void TwoBackends_ConcurrentThreads_NoFinalizerDrainRace()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        int deviceCount = CudaContext.GetDeviceCount();
        if (deviceCount < 2)
        {
            _output.WriteLine($"SKIPPED: needs 2 physical GPUs for cross-context concurrency (found {deviceCount}).");
            return;
        }

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor weightA = RandomF32(wShape, 11);
        using Tensor weightB = RandomF32(wShape, 22);
        using Tensor inputA = RandomF32(ioShape, 33);
        using Tensor inputB = RandomF32(ioShape, 44);
        float expectedA = ExpectedFirst(inputA, weightA, dim);
        float expectedB = ExpectedFirst(inputB, weightB, dim);

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(1, PtxDir());
        Exception? failure = null;
        try
        {
            backendA.PreloadWeights(new[] { weightA });
            backendB.PreloadWeights(new[] { weightB });
            using Barrier gate = new(2);
            const int iters = 200;

            void Worker(CudaBackend backend, Tensor input, Tensor weight, float expected, int gcMod)
            {
                try
                {
                    backend.HighPrecisionGemm = true; // isolation test, not a numerics test — keep off the TF32 tolerance boundary
                    gate.SignalAndWait();
                    for (int i = 0; i < iters; i++)
                    {
                        using Tensor output = new(ioShape, DType.F32);
                        backend.Linear(output, input, weight, bias: null);
                        float got = ((float*)output.DataPointer)[0]; // lazy sync fires on THIS backend's stream/context
                        Assert.Equal(expected, got, 3);
                        QueueAbandonedGpuTensor(backend, input, weight, ioShape);
                        if (i % gcMod == 0) { GC.Collect(); GC.WaitForPendingFinalizers(); }
                    }
                }
                catch (Exception ex) { Interlocked.CompareExchange(ref failure, ex, null); }
            }

            Thread tA = new(() => Worker(backendA, inputA, weightA, expectedA, 7)) { IsBackground = true };
            Thread tB = new(() => Worker(backendB, inputB, weightB, expectedB, 5)) { IsBackground = true };
            tA.Start();
            tB.Start();
            tA.Join();
            tB.Join();

            Assert.Null(failure);
            backendA.FreeWeights(new[] { weightA });
            backendB.FreeWeights(new[] { weightB });
        }
        finally
        {
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    /// <summary>Two backends on ONE device (shared primary context) must have fully independent transfer states:
    /// distinct StateKeys, per-backend weight caches, and correct interleaved results. This was the broken case the
    /// per-backend StateKey + ambient refactor exists for — context-handle keying collapsed both backends into one
    /// State with last-registered-wins stream bindings. Runs on any box with one CUDA GPU.</summary>
    [Fact]
    public void SameDevice_TwoBackends_InterleavedOps_IndependentStreamsAndCaches()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor inputA = RandomF32(ioShape, 101);
        using Tensor weightA = RandomF32(wShape, 102);
        using Tensor inputB = RandomF32(ioShape, 103);
        using Tensor weightB = RandomF32(wShape, 104);
        float expectedA = ExpectedFirst(inputA, weightA, dim);
        float expectedB = ExpectedFirst(inputB, weightB, dim);

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        try
        {
            Assert.NotSame(backendA.TransferState, backendB.TransferState);
            Assert.NotEqual(backendA.TransferState.Key, backendB.TransferState.Key);

            using Tensor outA1 = new(ioShape, DType.F32);
            using Tensor outB1 = new(ioShape, DType.F32);
            using Tensor outA2 = new(ioShape, DType.F32);
            using Tensor outB2 = new(ioShape, DType.F32);

            float a1 = RunLinear(backendA, inputA, weightA, outA1);
            float b1 = RunLinear(backendB, inputB, weightB, outB1);
            float a2 = RunLinear(backendA, inputA, weightA, outA2);
            float b2 = RunLinear(backendB, inputB, weightB, outB2);

            Assert.Equal(expectedA, a1, 3);
            Assert.Equal(expectedB, b1, 3);
            Assert.Equal(a1, a2, 5);
            Assert.Equal(b1, b2, 5);

            // Each backend caches ITS weight only — the sibling's cache is untouched.
            Assert.True(backendA.TransferState.WeightCache.ContainsKey(weightA));
            Assert.False(backendA.TransferState.WeightCache.ContainsKey(weightB));
            Assert.True(backendB.TransferState.WeightCache.ContainsKey(weightB));
            Assert.False(backendB.TransferState.WeightCache.ContainsKey(weightA));

            backendA.FreeWeights(new[] { weightA });
            backendB.FreeWeights(new[] { weightB });
        }
        finally
        {
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    /// <summary>Disposing one same-device backend must not free the sibling's live weights or unbind its stream.
    /// Before the per-backend split, B's Dispose ran EvictAll on the SHARED per-context State (wiping A's resident
    /// weights → dangling device pointers) and removed the compute-stream binding A still allocates through.</summary>
    [Fact]
    public void SameDevice_DisposeOne_SiblingWeightsSurvive()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor inputA = RandomF32(ioShape, 201);
        using Tensor weightA = RandomF32(wShape, 202);
        using Tensor inputB = RandomF32(ioShape, 203);
        using Tensor weightB = RandomF32(wShape, 204);
        float expectedA = ExpectedFirst(inputA, weightA, dim);

        CudaBackend backendA = new(0, PtxDir());
        try
        {
            using Tensor outA1 = new(ioShape, DType.F32);
            float a1 = RunLinear(backendA, inputA, weightA, outA1);
            Assert.Equal(expectedA, a1, 3);

            CudaBackend backendB = new(0, PtxDir());
            using (Tensor outB = new(ioShape, DType.F32))
            {
                RunLinear(backendB, inputB, weightB, outB);
            }
            backendB.Dispose();

            // A's weight is still resident (B's EvictAll must only hit B's state)...
            Assert.True(backendA.TransferState.WeightCache.ContainsKey(weightA));
            long hitsBefore = backendA.TransferState.Hits;

            // ...and A still computes correctly, via a cache HIT (not a silent re-upload after a wipe).
            using Tensor outA2 = new(ioShape, DType.F32);
            float a2 = RunLinear(backendA, inputA, weightA, outA2);
            Assert.Equal(expectedA, a2, 3);
            Assert.True(backendA.TransferState.Hits > hitsBefore,
                $"expected a weight-cache hit after sibling dispose (hits {hitsBefore} -> {backendA.TransferState.Hits})");

            backendA.FreeWeights(new[] { weightA });
        }
        finally
        {
            backendA.Dispose();
        }
    }

    /// <summary>Pin/unpin must bind the receiver's transfer state even when another backend was the last one used
    /// on the thread; otherwise sharded exception cleanup can leave the primary backend's carried latent pinned.</summary>
    [Fact]
    public void PinAndUnpin_RebindOwningBackendAfterAnotherBackendWasCurrent()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using Tensor host = RandomF32(new TensorShape(257), 291);
        using Tensor carried = new(new TensorShape(257), DType.F32);
        using Tensor other = new(new TensorShape(17), DType.F32);
        using CudaBackend backendA = new(0, PtxDir());
        using CudaBackend backendB = new(0, PtxDir());

        backendA.Scale(carried, host, 1f);
        backendA.PinActivation(carried);
        Assert.Contains(carried, backendA.TransferState.PinnedActivations);

        // Simulate a sharded forward/exception leaving backend B as the thread's ambient transfer state.
        backendB.Fill(other, 1f);
        backendA.UnpinActivation(carried);

        Assert.DoesNotContain(carried, backendA.TransferState.PinnedActivations);
        backendA.FreeActivations(trimPool: false);
        Assert.False(backendA.TransferState.ActivationCache.ContainsKey(carried));
    }

    /// <summary>One tensor carrying activation bindings from TWO backends must keep BOTH: with the old single-slot
    /// callbacks, backend B re-binding a tensor backend A had cached silently overwrote A's dispose hook, so A's
    /// device buffer (and cache entry) leaked forever. Multi-slot keyed bindings free both on Dispose.</summary>
    [Fact]
    public void SameDevice_ActivationReboundOnSecondBackend_BothBuffersFreed()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor inputA = RandomF32(ioShape, 301);
        using Tensor weightA = RandomF32(wShape, 302);
        using Tensor inputB = RandomF32(ioShape, 303);
        using Tensor weightB = RandomF32(wShape, 304);

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        try
        {
            backendA.HighPrecisionGemm = true;
            backendB.HighPrecisionGemm = true;
            Tensor shared = new(ioShape, DType.F32);

            backendA.Linear(shared, inputA, weightA, bias: null);
            backendA.Sync();
            Assert.True(backendA.TransferState.ActivationCache.ContainsKey(shared), "A should hold shared's activation");

            // B writes its own result into the SAME tensor object — the cross-backend in-place reuse that used to
            // orphan A's binding. (B reads inputB/weightB, allocates its own output buffer, re-binds shared.)
            backendB.Linear(shared, inputB, weightB, bias: null);
            backendB.Sync();
            Assert.True(backendB.TransferState.ActivationCache.ContainsKey(shared), "B should hold shared's activation");

            // Multi-slot: BOTH backends' dispose hooks must fire, emptying both caches — under the single-slot
            // scheme A's entry (and its device buffer) survived the Dispose as an unfreeable leak.
            shared.Dispose();
            Assert.False(backendA.TransferState.ActivationCache.ContainsKey(shared), "A leaked its activation entry after Dispose");
            Assert.False(backendB.TransferState.ActivationCache.ContainsKey(shared), "B leaked its activation entry after Dispose");

            backendA.FreeWeights(new[] { weightA });
            backendB.FreeWeights(new[] { weightB });
        }
        finally
        {
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    /// <summary>While backend A has a graph-capture arena active, backend B's allocations must NOT bump-allocate
    /// out of A's arena — the arena scalars live on each backend's own State now (they were single-valued on the
    /// shared per-context State, so B's allocations landed inside A's arena and died with it).</summary>
    [Fact]
    public void SameDevice_GraphArena_Isolation()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        try
        {
            GpuTransferHelper.SetAmbient(backendA.TransferState);
            ulong arenaBase = GpuTransferHelper.BeginGraphArena();
            if (arenaBase == 0)
            {
                _output.WriteLine("SKIPPED-SOFT: arena allocation failed (VRAM-tight) — nothing to isolate.");
                return;
            }
            try
            {
                ulong insideA = GpuTransferHelper.AllocateDevice(4096);
                Assert.True(GpuTransferHelper.IsArenaPtr(backendA.TransferState, insideA), "A's capture alloc should be arena-backed");

                GpuTransferHelper.SetAmbient(backendB.TransferState);
                ulong fromB = GpuTransferHelper.AllocateDevice(4096);
                Assert.False(GpuTransferHelper.IsArenaPtr(backendA.TransferState, fromB),
                    "backend B's allocation bump-allocated out of backend A's live capture arena");
                GpuTransferHelper.FreeDevice(fromB);
            }
            finally
            {
                GpuTransferHelper.SetAmbient(backendA.TransferState);
                GpuTransferHelper.EndGraphArena();
                GpuTransferHelper.FreeGraphArena(arenaBase);
            }
        }
        finally
        {
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    /// <summary>The step-graph capture-window diagnostic tracker (<c>TrackCaptureWindow</c>/<c>CaptureAllocs</c>)
    /// used to be process-wide statics on <see cref="CudaMemory"/>: backend B beginning its own capture cleared
    /// backend A's in-flight window, and any of A's allocations issued while B's window happened to be open got
    /// folded into B's leak-detection report. Now lives on <see cref="GpuTransferHelper.State"/> — per backend,
    /// including two backends sharing one GPU.</summary>
    [Fact]
    public void SameDevice_StepGraphCaptureWindow_IsolatedPerBackend()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        try
        {
            // A opens its capture window and allocates — recorded in A's tracker only.
            GpuTransferHelper.SetAmbient(backendA.TransferState);
            backendA.TransferState.TrackCaptureWindow = true;
            ulong aPtr = GpuTransferHelper.AllocateDevice(4096);
            Assert.Single(backendA.TransferState.CaptureAllocs);

            // B is NOT tracking; an ordinary allocation on B must not appear in A's window (the old bug: a
            // single process-wide flag meant ANY backend's alloc while capturing recorded into the wrong tracker).
            GpuTransferHelper.SetAmbient(backendB.TransferState);
            Assert.False(backendB.TransferState.TrackCaptureWindow);
            ulong bPtr = GpuTransferHelper.AllocateDevice(4096);
            Assert.Empty(backendB.TransferState.CaptureAllocs);
            Assert.Single(backendA.TransferState.CaptureAllocs, kv => kv.Key != bPtr);

            // B opening ITS OWN window must not clear or touch A's still-open one.
            backendB.TransferState.TrackCaptureWindow = true;
            Assert.Single(backendA.TransferState.CaptureAllocs);
            Assert.True(backendA.TransferState.TrackCaptureWindow, "B starting its own window must not close A's");

            GpuTransferHelper.SetAmbient(backendA.TransferState);
            GpuTransferHelper.FreeDevice(aPtr);
            Assert.Empty(backendA.TransferState.CaptureAllocs);
            Assert.Equal(1, backendA.TransferState.CaptureFreeCount);
            Assert.Equal(0, backendB.TransferState.CaptureFreeCount);

            GpuTransferHelper.SetAmbient(backendB.TransferState);
            GpuTransferHelper.FreeDevice(bPtr);
            backendA.TransferState.TrackCaptureWindow = false;
            backendB.TransferState.TrackCaptureWindow = false;
        }
        finally
        {
            backendA.Dispose();
            backendB.Dispose();
        }
    }

    /// <summary>Fidelity check on the real public API: two same-device backends each run a genuine
    /// StepGraphBegin/EndAndLaunch cycle with actual captured work (a Linear), interleaved, and each must produce
    /// correct output — the isolated tracker from the test above is what makes this safe, but this exercises the
    /// path an inference pipeline actually calls.</summary>
    [Fact]
    public void SameDevice_TwoBackends_InterleavedStepGraphCapture_ProducesCorrectResults()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor inputA = RandomF32(ioShape, 801);
        using Tensor weightA = RandomF32(wShape, 802);
        using Tensor inputB = RandomF32(ioShape, 803);
        using Tensor weightB = RandomF32(wShape, 804);
        float expectedA = ExpectedFirst(inputA, weightA, dim);
        float expectedB = ExpectedFirst(inputB, weightB, dim);

        CudaBackend backendA = new(0, PtxDir());
        CudaBackend backendB = new(0, PtxDir());
        try
        {
            if (!((IBackend)backendA).StepGraphSupported)
            {
                _output.WriteLine("SKIPPED: StepGraph unsupported");
                return;
            }
            backendA.HighPrecisionGemm = true;
            backendB.HighPrecisionGemm = true;
            backendA.StepGraphOwner = this;
            backendB.StepGraphOwner = this;

            using Tensor outA = new(ioShape, DType.F32);
            using Tensor outB = new(ioShape, DType.F32);

            backendA.PreloadWeights(new[] { weightA });
            backendA.StepGraphBegin();
            backendA.Linear(outA, inputA, weightA, bias: null);
            backendA.StepGraphEndAndLaunch();

            backendB.PreloadWeights(new[] { weightB });
            backendB.StepGraphBegin();
            backendB.Linear(outB, inputB, weightB, bias: null);
            backendB.StepGraphEndAndLaunch();

            backendA.Sync();
            backendB.Sync();
            Assert.Equal(expectedA, ((float*)outA.DataPointer)[0], 3);
            Assert.Equal(expectedB, ((float*)outB.DataPointer)[0], 3);

            backendA.FreeWeights(new[] { weightA });
            backendB.FreeWeights(new[] { weightB });
        }
        finally
        {
            backendA.StepGraphReset();
            backendB.StepGraphReset();
            if (ReferenceEquals(backendA.StepGraphOwner, this)) backendA.StepGraphOwner = null;
            if (ReferenceEquals(backendB.StepGraphOwner, this)) backendB.StepGraphOwner = null;
            backendA.Dispose();
            backendB.Dispose();
        }
    }
}
