using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Production gates for the general-precision cuBLASLt plan cache. These tests deliberately assert
/// both native-plan diagnostics and numerical results: a cache hit is not useful if it reuses a stale dynamic
/// bias pointer, and a clean fallback counter is not useful if the GemmEx path was never actually run.</summary>
[Collection("CudaSerial")]
public sealed class LtGemmPlanCacheTests
{
    private const int M = 16;
    private const int N = 32;
    private const int K = 64;

    private readonly ITestOutputHelper _output;

    public LtGemmPlanCacheTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void HighPrecisionPolicy_IsForwardedExactlyToLtPlan()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir())
        {
            EnableEpilogueFusion = true,
            HighPrecisionGemm = true,
        };
        LtGemmExecutor lt = backend.LtGemm;
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        // 1.0003f is below one TF32 unit at 1.0. If the fused path silently reselects FAST_TF32,
        // 64 products lose roughly 0.038 instead of merely differing in the final few F32 ulps.
        using Tensor input = FilledF32(new TensorShape(M, K), 1.0003f);
        using Tensor weight = FilledF32(new TensorShape(N, K), 1.0003f);
        using Tensor bias = PatternF32(new TensorShape(N), i => ((i % 7) - 3) * 0.03125f);
        using Tensor result = new(new TensorShape(M, N), DType.F32);

        backend.PreloadWeights([input, weight, bias]);
        backend.Linear(result, input, weight, bias);
        backend.Sync();

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(CublasApi.CUBLAS_COMPUTE_32F, diagnostics.LastComputeType);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.Fallbacks);
        AssertLinearMatchesCpu(result, input, weight, bias, tolerance: 1e-3f);
    }

    [Fact]
    public void F32EnvironmentPolicies_ForwardFastTf32_NoTf32_AndFastF16Exactly()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using (CudaContext capabilityProbe = new(0))
        {
            if (capabilityProbe.ComputeCapabilityMajor < 8)
            {
                _output.WriteLine("SKIPPED: F32 tensor-core policies require SM 8.0+");
                return;
            }
        }

        string? previousNoTf32 = Environment.GetEnvironmentVariable("HARTSY_NO_TF32");
        string? previousFastF16 = Environment.GetEnvironmentVariable("HARTSY_GEMM_F16");
        string? previousHighPrecision = Environment.GetEnvironmentVariable("HARTSY_HIGH_PRECISION_GEMM");
        try
        {
            // Constructor-time policy must be isolated per backend. Explicitly clear the high-precision override
            // so an external test environment cannot collapse all three cases to plain COMPUTE_32F.
            Environment.SetEnvironmentVariable("HARTSY_HIGH_PRECISION_GEMM", null);

            int? defaultPolicy = RunF32PolicyCase(noTf32: false, fastF16: false);
            if (defaultPolicy is null)
            {
                _output.WriteLine("SKIPPED: cuBLASLt unavailable");
                return;
            }
            int? noTf32Policy = RunF32PolicyCase(noTf32: true, fastF16: false);
            int? fastF16Policy = RunF32PolicyCase(noTf32: false, fastF16: true);

            Assert.Equal(CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, defaultPolicy.Value);
            Assert.NotNull(noTf32Policy);
            Assert.NotNull(fastF16Policy);
            Assert.Equal(CublasApi.CUBLAS_COMPUTE_32F, noTf32Policy.Value);
            Assert.Equal(CublasApi.CUBLAS_COMPUTE_32F_FAST_16F, fastF16Policy.Value);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_NO_TF32", previousNoTf32);
            Environment.SetEnvironmentVariable("HARTSY_GEMM_F16", previousFastF16);
            Environment.SetEnvironmentVariable("HARTSY_HIGH_PRECISION_GEMM", previousHighPrecision);
        }
    }

    [Fact]
    public void CachedPlan_PatchesDifferentBiasPointer_WithoutRequery()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaContext context = new(0);
        using CudaStream stream = new(nonBlocking: false);
        using DeviceBuffers buffers = new();
        using LtGemmExecutor lt = new(context, stream.Handle);
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        ulong input = buffers.Upload(new float[M * K]);
        ulong zeroWeight = buffers.Upload(new float[N * K]);
        ulong resultA = buffers.AllocateF32(M * N);
        ulong resultB = buffers.AllocateF32(M * N);

        // Bias alignment is intentionally part of the cache key because cuBLASLt exposes no bias-alignment
        // preference. Two offsets with exactly the same 128-byte alignment prove pointer patching independently
        // of that key: distinct live addresses, one cached plan, one native heuristic query.
        ulong biasSlab = buffers.AllocateBytes(512);
        Assert.Equal(0UL, biasSlab & 255UL);
        ulong biasA = biasSlab + 128;
        ulong biasB = biasSlab + 384;
        float[] expectedA = Enumerable.Range(0, N).Select(i => (i + 1) * 0.03125f).ToArray();
        float[] expectedB = Enumerable.Range(0, N).Select(i => -(i + 3) * 0.015625f).ToArray();
        buffers.UploadTo(biasA, expectedA);
        buffers.UploadTo(biasB, expectedB);

        try
        {
            Assert.True(TryRunF32(lt, stream, zeroWeight, input, resultA, M, N, K, biasA));
            Assert.True(TryRunF32(lt, stream, zeroWeight, input, resultB, M, N, K, biasB));
            stream.Synchronize();
            AssertRowsEqualBias(buffers.Download(resultA, M * N), expectedA, M);
            AssertRowsEqualBias(buffers.Download(resultB, M * N), expectedB, M);
        }
        finally
        {
            stream.Synchronize();
        }

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.Count);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(1, diagnostics.Hits);
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.Fallbacks);
    }

    [Fact]
    public void HotShape_QueriesOnce_ThenOnlyHitsThePlanCache()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaContext context = new(0);
        using CudaStream stream = new(nonBlocking: false);
        using DeviceBuffers buffers = new();
        using LtGemmExecutor lt = new(context, stream.Handle);
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        ulong input = buffers.Upload(Enumerable.Repeat(0.25f, M * K).ToArray());
        ulong weight = buffers.Upload(Enumerable.Repeat(0.125f, N * K).ToArray());
        ulong bias = buffers.Upload(Enumerable.Repeat(0.5f, N).ToArray());
        ulong result = buffers.AllocateF32(M * N);

        const int iterations = 32;
        try
        {
            for (int i = 0; i < iterations; i++)
                Assert.True(TryRunF32(lt, stream, weight, input, result, M, N, K, bias));
            stream.Synchronize();

            float[] actual = buffers.Download(result, M * N);
            Assert.All(actual, value => Assert.Equal(2.5f, value));
        }
        finally
        {
            stream.Synchronize();
        }

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.Count);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(iterations - 1, diagnostics.Hits);
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.PlanDestroys);
        Assert.Equal(0, diagnostics.Fallbacks);
    }

    [Fact]
    public void CachedPlan_CapturesAndReplaysWithoutRequery_AndReadsLiveBiasData()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaContext context = new(0);
        using CudaStream stream = new(nonBlocking: false);
        using DeviceBuffers buffers = new();
        using LtGemmExecutor lt = new(context, stream.Handle);
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        ulong input = buffers.Upload(new float[M * K]);
        ulong weight = buffers.Upload(new float[N * K]);
        ulong bias = buffers.Upload(Enumerable.Repeat(0.25f, N).ToArray());
        ulong result = buffers.AllocateF32(M * N);

        try
        {
            // Descriptor creation and the heuristic are not capture work. Populate the exact plan eagerly so
            // capture contains only a cached cublasLtMatmul submission with stable device addresses.
            Assert.True(TryRunF32(lt, stream, weight, input, result, M, N, K, bias));
            stream.Synchronize();

            using CudaGraph graph = new(stream.Handle);
            graph.Capture(() =>
            {
                if (!TryRunF32(lt, stream, weight, input, result, M, N, K, bias))
                    throw new InvalidOperationException("A pre-cached cuBLASLt plan was rejected during graph capture.");
            });
            Assert.True(graph.IsReady);

            graph.Launch();
            stream.Synchronize();
            Assert.All(buffers.Download(result, M * N), value => Assert.Equal(0.25f, value));

            // Replay must dereference the stable bias pointer at launch time, not bake its capture-time contents.
            buffers.UploadTo(bias, Enumerable.Repeat(-0.75f, N).ToArray());
            graph.Launch();
            stream.Synchronize();
            Assert.All(buffers.Download(result, M * N), value => Assert.Equal(-0.75f, value));
        }
        finally
        {
            stream.Synchronize();
        }

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.Count);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(1, diagnostics.Hits); // capture records one managed submission; replay is graph-native
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.Fallbacks);
    }

    [Fact]
    public void ForcedNoAlgorithm_FallsBackToGemmEx_AndNegativeCachesTheContract()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir())
        {
            EnableEpilogueFusion = true,
            HighPrecisionGemm = true,
        };
        LtGemmExecutor lt = backend.LtGemm;
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }
        lt.ForceNoAlgorithmForTests = true;

        using Tensor input = PatternF32(new TensorShape(M, K), i => ((i % 13) - 6) * 0.0625f);
        using Tensor weight = PatternF32(new TensorShape(N, K), i => ((i % 9) - 4) * 0.03125f);
        using Tensor bias = PatternF32(new TensorShape(N), i => ((i % 5) - 2) * 0.125f);
        using Tensor resultA = new(new TensorShape(M, N), DType.F32);
        using Tensor resultB = new(new TensorShape(M, N), DType.F32);

        backend.PreloadWeights([input, weight, bias]);
        backend.Linear(resultA, input, weight, bias);
        backend.Linear(resultB, input, weight, bias);
        backend.Sync();

        AssertLinearMatchesCpu(resultA, input, weight, bias, tolerance: 2e-5f);
        AssertLinearMatchesCpu(resultB, input, weight, bias, tolerance: 2e-5f);

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.Count);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(1, diagnostics.Hits);
        Assert.Equal(1, diagnostics.NegativeHits);
        Assert.Equal(2, diagnostics.Fallbacks);
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(0, diagnostics.PlanCreates);
        Assert.Equal(CublasApi.CUBLAS_COMPUTE_32F, diagnostics.LastComputeType);
    }

    [Fact]
    public void CapacityTwo_UsesTrueLru_AndDisposeReleasesAllLivePlans()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaContext context = new(0);
        using CudaStream stream = new(nonBlocking: false);
        using DeviceBuffers buffers = new();
        long liveBaseline = LtGemmExecutor.LivePlanCountForTests;
        long leaseBaseline = LtGemmExecutor.LiveContextLeaseCountForTests;
        LtGemmExecutor lt = new(context, stream.Handle, planCacheCapacity: 2);

        bool allRunsSucceeded = true;
        LtGemmExecutor.CacheDiagnostics beforeDispose = default;
        LtGemmExecutor.CacheDiagnostics afterDispose = default;
        long liveWhileFull = liveBaseline;
        long leasesWhileFull = leaseBaseline;
        long leasesAfterDispose = leaseBaseline;
        try
        {
            if (!lt.IsSupported)
            {
                _output.WriteLine("SKIPPED: cuBLASLt unavailable");
                return;
            }

            const int maxM = 24;
            ulong input = buffers.Upload(Enumerable.Repeat(0.25f, maxM * K).ToArray());
            ulong weight = buffers.Upload(Enumerable.Repeat(0.125f, N * K).ToArray());
            ulong bias = buffers.Upload(Enumerable.Repeat(0.5f, N).ToArray());
            ulong result = buffers.AllocateF32(maxM * N);

            // A, B, A, C, B: the second A refreshes A, C evicts B, and the final B must miss and evict A.
            int[] mSequence = [8, 16, 8, 24, 16];
            foreach (int m in mSequence)
                allRunsSucceeded &= TryRunF32(lt, stream, weight, input, result, m, N, K, bias);
            stream.Synchronize();

            beforeDispose = lt.Diagnostics;
            liveWhileFull = LtGemmExecutor.LivePlanCountForTests;
            leasesWhileFull = LtGemmExecutor.LiveContextLeaseCountForTests;
            lt.Dispose();
            afterDispose = lt.Diagnostics;
            leasesAfterDispose = LtGemmExecutor.LiveContextLeaseCountForTests;
        }
        finally
        {
            stream.Synchronize();
            lt.Dispose();
        }

        Assert.True(allRunsSucceeded);
        Assert.Equal(2, beforeDispose.Count);
        Assert.Equal(1, beforeDispose.Hits);
        Assert.Equal(4, beforeDispose.Misses);
        Assert.Equal(4, beforeDispose.HeuristicQueries);
        Assert.Equal(4, beforeDispose.PlanCreates);
        Assert.Equal(2, beforeDispose.PlanDestroys);
        Assert.Equal(2, beforeDispose.Evictions);
        Assert.Equal(liveBaseline + 2, liveWhileFull);
        Assert.Equal(leaseBaseline + 1, leasesWhileFull);
        Assert.Equal(0, afterDispose.Count);
        Assert.Equal(4, afterDispose.PlanDestroys);
        Assert.Equal(leaseBaseline, leasesAfterDispose);
        Assert.Equal(liveBaseline, LtGemmExecutor.LivePlanCountForTests);
        Assert.Equal(leaseBaseline, LtGemmExecutor.LiveContextLeaseCountForTests);
    }

    [Fact]
    public void StandaloneExecutor_DisposesAfterBorrowedStreamAndContext_AcrossItsRetainedLease()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        long planBaseline = LtGemmExecutor.LivePlanCountForTests;
        long leaseBaseline = LtGemmExecutor.LiveContextLeaseCountForTests;
        CudaContext? context = null;
        CudaStream? stream = null;
        DeviceBuffers? buffers = null;
        LtGemmExecutor? lt = null;
        try
        {
            context = new CudaContext(0);
            stream = new CudaStream(nonBlocking: false);
            buffers = new DeviceBuffers();
            lt = new LtGemmExecutor(context, stream.Handle);
            Assert.Equal(leaseBaseline + 1, LtGemmExecutor.LiveContextLeaseCountForTests);
            if (!lt.IsSupported)
            {
                _output.WriteLine("SKIPPED: cuBLASLt unavailable");
                return;
            }

            ulong input = buffers.Upload(Enumerable.Repeat(0.25f, M * K).ToArray());
            ulong weight = buffers.Upload(Enumerable.Repeat(0.125f, N * K).ToArray());
            ulong bias = buffers.Upload(Enumerable.Repeat(0.5f, N).ToArray());
            ulong result = buffers.AllocateF32(M * N);
            Assert.True(TryRunF32(lt, stream, weight, input, result, M, N, K, bias));
            stream.Synchronize();
            Assert.Equal(planBaseline + 1, LtGemmExecutor.LivePlanCountForTests);

            // Device tensors are caller-owned, so release them while the borrowed owner is still intact. Then
            // deliberately invalidate both borrowed native handles before disposing Lt. Its independent primary-
            // context retain must rebind and context-sync without dereferencing the stale stream, then release
            // plans/workspace/handle.
            buffers.Dispose();
            stream.Dispose();
            context.Dispose();
            Assert.Equal(planBaseline + 1, LtGemmExecutor.LivePlanCountForTests);
            Assert.Equal(leaseBaseline + 1, LtGemmExecutor.LiveContextLeaseCountForTests);

            Exception? disposeFailure = Record.Exception(lt.Dispose);
            Assert.Null(disposeFailure);
            LtGemmExecutor.CacheDiagnostics afterDispose = lt.Diagnostics;
            Assert.Equal(0, afterDispose.Count);
            Assert.Equal(1, afterDispose.PlanDestroys);
            Assert.Equal(planBaseline, LtGemmExecutor.LivePlanCountForTests);
            Assert.Equal(leaseBaseline, LtGemmExecutor.LiveContextLeaseCountForTests);
        }
        finally
        {
            // This order is valid in both paths: standalone Lt teardown always drains its retained context and never
            // probes the lifetime of the borrowed stream. Every Dispose involved is idempotent.
            lt?.Dispose();
            buffers?.Dispose();
            stream?.Dispose();
            context?.Dispose();
        }

        Assert.Equal(planBaseline, LtGemmExecutor.LivePlanCountForTests);
        Assert.Equal(leaseBaseline, LtGemmExecutor.LiveContextLeaseCountForTests);
    }

    [Fact]
    public async Task ConcurrentSameStreamCalls_KeepPerCallBiasPointersIsolated()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaContext context = new(0);
        using CudaStream stream = new(nonBlocking: false);
        using DeviceBuffers buffers = new();
        using LtGemmExecutor lt = new(context, stream.Handle);
        if (!lt.IsSupported)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        ulong input = buffers.Upload(new float[M * K]);
        ulong weight = buffers.Upload(new float[N * K]);
        const int workers = 8;
        ulong[] biases = new ulong[workers];
        ulong[] results = new ulong[workers];
        ulong biasSlab = buffers.AllocateBytes((nuint)(workers * 256));
        Assert.Equal(0UL, biasSlab & 255UL);
        for (int worker = 0; worker < workers; worker++)
        {
            float expectedBias = (worker + 1) * 0.125f;
            // Every address is distinct but exactly 128-byte aligned, keeping all workers on one PlanKey while
            // they contend over the descriptor's dynamic bias-pointer attribute.
            biases[worker] = biasSlab + 128UL + (ulong)(worker * 256);
            buffers.UploadTo(biases[worker], Enumerable.Repeat(expectedBias, N).ToArray());
            results[worker] = buffers.AllocateF32(M * N);
        }

        using ManualResetEventSlim start = new(initialState: false);
        Task<bool>[] tasks = new Task<bool>[workers];
        try
        {
            for (int worker = 0; worker < workers; worker++)
            {
                int captured = worker;
                tasks[worker] = Task.Run(() =>
                {
                    context.EnsureCurrent();
                    start.Wait();
                    return TryRunF32(
                        lt, stream, weight, input, results[captured], M, N, K, biases[captured]);
                });
            }

            start.Set();
            bool[] submitted = await Task.WhenAll(tasks);
            Assert.All(submitted, Assert.True);
            context.EnsureCurrent();
            stream.Synchronize();

            for (int worker = 0; worker < workers; worker++)
            {
                float expectedBias = (worker + 1) * 0.125f;
                float[] actual = buffers.Download(results[worker], M * N);
                Assert.All(actual, value => Assert.Equal(expectedBias, value));
            }
        }
        finally
        {
            start.Set();
            await Task.WhenAll(tasks.Where(task => task is not null)!);
            context.EnsureCurrent();
            stream.Synchronize();
        }

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.Count);
        Assert.Equal(1, diagnostics.Misses);
        Assert.Equal(workers - 1, diagnostics.Hits);
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.Fallbacks);
    }

    [Fact]
    public void AbandonedBackend_WithPopulatedLtPlan_ReaperRestoresPlanAndRegistryBaselines()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        long livePlanBaseline = LtGemmExecutor.LivePlanCountForTests;
        long leaseBaseline = LtGemmExecutor.LiveContextLeaseCountForTests;
        int registryBaseline = GpuTransferHelper.RegisteredStateCount;
        long enqueuedBefore = CudaBackend.AbandonedCleanupEnqueuedCount;
        long completedBefore = CudaBackend.AbandonedCleanupCompletedCount;
        long failedBefore = CudaBackend.AbandonedCleanupFailedCount;

        AbandonedLtProbe? probe = CreateAbandonedBackendWithPopulatedLtPlan();
        if (probe is null)
        {
            _output.WriteLine("SKIPPED: cuBLASLt unavailable");
            return;
        }

        Assert.Equal(livePlanBaseline + 1, LtGemmExecutor.LivePlanCountForTests);
        Assert.Equal(leaseBaseline + 1, LtGemmExecutor.LiveContextLeaseCountForTests);
        Assert.Equal(registryBaseline + 1, GpuTransferHelper.RegisteredStateCount);
        Assert.True(GpuTransferHelper.IsStateRegistered(probe.State.Key));

        bool enqueued = false;
        for (int attempt = 0; attempt < 20 && !enqueued; attempt++)
        {
            ForceFullGc();
            enqueued = CudaBackend.AbandonedCleanupEnqueuedCount >= enqueuedBefore + 1;
            if (!enqueued) Thread.Sleep(10);
        }

        Assert.True(enqueued, "Abandoned Lt-owning backend was not finalized and queued.");
        Assert.True(
            CudaBackend.WaitForAbandonedCleanup(completedBefore + 1, TimeSpan.FromSeconds(20)),
            "Timed out waiting for abandoned Lt-owning backend cleanup.");
        Assert.True(
            SpinWait.SpinUntil(
                () => probe.State.Unregistered
                    && !GpuTransferHelper.IsStateRegistered(probe.State.Key)
                    && LtGemmExecutor.LivePlanCountForTests == livePlanBaseline
                    && LtGemmExecutor.LiveContextLeaseCountForTests == leaseBaseline,
                TimeSpan.FromSeconds(20)),
            "Reaper advanced globally without retiring this backend's Lt plan and transfer state.");
        ForceFullGc();

        Assert.False(probe.Backend.TryGetTarget(out _));
        Assert.True(probe.State.Retiring);
        Assert.True(probe.State.Unregistered);
        Assert.Equal(registryBaseline, GpuTransferHelper.RegisteredStateCount);
        Assert.Equal(livePlanBaseline, LtGemmExecutor.LivePlanCountForTests);
        Assert.Equal(leaseBaseline, LtGemmExecutor.LiveContextLeaseCountForTests);
        Assert.Equal(failedBefore, CudaBackend.AbandonedCleanupFailedCount);
    }

    private static bool TryRunF32(
        LtGemmExecutor lt,
        CudaStream stream,
        ulong weight,
        ulong input,
        ulong result,
        int m,
        int n,
        int k,
        ulong bias)
        => lt.TryRun(
            weight, input, result,
            m, n, k, 1.0f,
            CublasApi.CUDA_R_32F,
            CublasApi.CUDA_R_32F,
            CublasApi.CUBLAS_COMPUTE_32F,
            bias,
            CublasLtApi.CUBLASLT_EPILOGUE_BIAS,
            stream.Handle);

    private static int? RunF32PolicyCase(bool noTf32, bool fastF16)
    {
        Environment.SetEnvironmentVariable("HARTSY_NO_TF32", noTf32 ? "1" : null);
        Environment.SetEnvironmentVariable("HARTSY_GEMM_F16", fastF16 ? "1" : null);

        using CudaBackend backend = new(0, PtxDir())
        {
            EnableEpilogueFusion = true,
            HighPrecisionGemm = false,
        };
        LtGemmExecutor lt = backend.LtGemm;
        if (!lt.IsSupported) return null;

        using Tensor input = FilledF32(new TensorShape(M, K), 0.25f);
        using Tensor weight = FilledF32(new TensorShape(N, K), 0.125f);
        using Tensor bias = FilledF32(new TensorShape(N), 0.5f);
        using Tensor result = new(new TensorShape(M, N), DType.F32);
        backend.PreloadWeights([input, weight, bias]);
        backend.Linear(result, input, weight, bias);
        backend.Sync();

        LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
        Assert.Equal(1, diagnostics.HeuristicQueries);
        Assert.Equal(1, diagnostics.PlanCreates);
        Assert.Equal(0, diagnostics.Fallbacks);
        return diagnostics.LastComputeType;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static unsafe AbandonedLtProbe? CreateAbandonedBackendWithPopulatedLtPlan()
    {
        CudaBackend backend = new(0, PtxDir())
        {
            EnableEpilogueFusion = true,
            HighPrecisionGemm = true,
        };
        try
        {
            LtGemmExecutor lt = backend.LtGemm;
            if (!lt.IsSupported)
            {
                backend.Dispose();
                return null;
            }

            using Tensor input = FilledF32(new TensorShape(M, K), 0.25f);
            using Tensor weight = FilledF32(new TensorShape(N, K), 0.125f);
            using Tensor bias = FilledF32(new TensorShape(N), 0.5f);
            using Tensor result = new(new TensorShape(M, N), DType.F32);
            backend.PreloadWeights([input, weight, bias]);
            backend.Linear(result, input, weight, bias);
            backend.Sync();
            Assert.Equal(2.5f, ((float*)result.DataPointer)[0]);

            LtGemmExecutor.CacheDiagnostics diagnostics = lt.Diagnostics;
            Assert.Equal(1, diagnostics.Count);
            Assert.Equal(1, diagnostics.PlanCreates);
            return new(new WeakReference<CudaBackend>(backend), backend.TransferState);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    private static void ForceFullGc()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
    }

    private static unsafe Tensor PatternF32(TensorShape shape, Func<long, float> valueAt)
    {
        Tensor tensor = new(shape, DType.F32);
        float* values = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++) values[i] = valueAt(i);
        return tensor;
    }

    private static Tensor FilledF32(TensorShape shape, float value)
        => PatternF32(shape, _ => value);

    private static void AssertRowsEqualBias(float[] actual, float[] expected, int rows)
    {
        int columns = expected.Length;
        Assert.Equal(rows * columns, actual.Length);
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
                Assert.Equal(expected[column], actual[row * columns + column]);
        }
    }

    private static unsafe void AssertLinearMatchesCpu(
        Tensor result,
        Tensor input,
        Tensor weight,
        Tensor bias,
        float tolerance)
    {
        float* actual = (float*)result.DataPointer;
        float* x = (float*)input.DataPointer;
        float* w = (float*)weight.DataPointer;
        float* b = (float*)bias.DataPointer;
        int rows = (int)input.Shape[0];
        int inner = (int)input.Shape[1];
        int columns = (int)weight.Shape[0];
        float maxError = 0.0f;
        int maxIndex = 0;

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                double expected = b[column];
                for (int innerIndex = 0; innerIndex < inner; innerIndex++)
                    expected += (double)x[row * inner + innerIndex] * w[column * inner + innerIndex];
                int index = row * columns + column;
                float error = MathF.Abs(actual[index] - (float)expected);
                if (error > maxError)
                {
                    maxError = error;
                    maxIndex = index;
                }
            }
        }

        Assert.True(
            maxError <= tolerance,
            $"Linear result max error {maxError:E3} at {maxIndex} exceeds {tolerance:E3}.");
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(
                HartsyInference.Tests.Common.RepoRoot.Path,
                "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private sealed record AbandonedLtProbe(
        WeakReference<CudaBackend> Backend,
        GpuTransferHelper.State State);

    private sealed class DeviceBuffers : IDisposable
    {
        private readonly List<ulong> _owned = [];

        public unsafe ulong Upload(float[] values)
        {
            ulong pointer = AllocateF32(values.Length);
            fixed (float* source = values)
                CudaMemory.CopyHostToDevice(pointer, source, (nuint)(values.Length * sizeof(float)));
            return pointer;
        }

        public ulong AllocateF32(int count)
            => AllocateBytes((nuint)(count * sizeof(float)));

        public ulong AllocateBytes(nuint bytes)
        {
            ulong pointer = CudaMemory.AllocatePersistent(bytes);
            _owned.Add(pointer);
            return pointer;
        }

        public unsafe void UploadTo(ulong pointer, float[] values)
        {
            fixed (float* source = values)
                CudaMemory.CopyHostToDevice(pointer, source, (nuint)(values.Length * sizeof(float)));
        }

        public unsafe float[] Download(ulong pointer, int count)
        {
            float[] values = new float[count];
            fixed (float* destination = values)
                CudaMemory.CopyDeviceToHost(destination, pointer, (nuint)(count * sizeof(float)));
            return values;
        }

        public void Dispose()
        {
            List<Exception>? failures = null;
            for (int i = _owned.Count - 1; i >= 0; i--)
            {
                try { CudaMemory.Free(_owned[i]); }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
            _owned.Clear();
            if (failures is not null)
                throw new AggregateException("One or more Lt GEMM test buffers failed to release.", failures);
        }
    }
}
