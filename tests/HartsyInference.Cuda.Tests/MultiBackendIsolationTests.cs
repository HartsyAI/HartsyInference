using System.Runtime.CompilerServices;
using System.Threading;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Guards the per-context transfer-state isolation: two <see cref="CudaBackend"/>s in one process
/// (different GPUs when available, same GPU otherwise) must not clobber each other's streams, caches, or
/// activation callbacks. Before the per-context refactor, constructing the second backend silently retargeted
/// ALL of GpuTransferHelper's static state at the new context — interleaved ops then issued cross-device
/// stream/free calls → CUDA_ERROR_ILLEGAL_ADDRESS on both GPUs and a poisoned process.</summary>
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
    /// callbacks pile up. Before the queue was partitioned by context, thread B's EnsureCurrent drained thread A's
    /// callbacks — mutating A's non-thread-safe GpuTransferHelper State while A's own thread mutated it → intermittent
    /// leak / throw / illegal-address. Requires 2 physical GPUs: two backends on ONE device intentionally share a
    /// primary context (and thus per-context State), so concurrent use there is out of contract and would race by design.</summary>
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
}
