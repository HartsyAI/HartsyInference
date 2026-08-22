using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>General-precision cuBLASLt GEMM with epilogue fusion (bias and/or activation folded into the matmul).</summary>
/// <remarks>
/// <para>Layout matches <see cref="CudaBackend.Linear"/>: row-major
/// <c>output[M, N] = input[M, K] * weight^T[N, K]</c>, dispatched as cuBLAS <c>OP_T</c> on the weight and
/// <c>OP_N</c> on the input.</para>
///
/// <para>Plans are cached per executor, and therefore per CUDA backend/context. A plan owns its operation and
/// matrix-layout descriptors plus the exact algorithm selected for one shape/type/precision/alignment contract.
/// Device tensor pointers are never owned by a plan; the bias pointer is patched immediately before submission.</para>
/// </remarks>
public sealed unsafe class LtGemmExecutor : IDisposable
{
    private const int DefaultPlanCacheCapacity = 256;
    private const int ScaleType = CublasApi.CUDA_R_32F;

    private readonly object _gate = new();
    private readonly int _planCacheCapacity;
    private readonly Dictionary<PlanKey, CacheEntry> _plans = new();
    private readonly LinkedList<PlanKey> _lru = new();
    private readonly CudaContext _ownerContext;
    private readonly int _deviceHandle;
    private readonly bool _backendAdopted;

    private nint _retainedContext;
    private bool _contextLeaseCounted;
    private nint _ltHandle;
    private ulong _workspace;
    private nuint _workspaceBytes;
    private nint _ownerStream;
    private bool _streamBound;
    private int _disposed;
    private bool _forceNoAlgorithmForTests;

    private long _cacheHits;
    private long _cacheMisses;
    private long _heuristicQueries;
    private long _negativeHits;
    private long _fallbacks;
    private long _planCreates;
    private long _planDestroys;
    private long _evictions;
    private int _lastComputeType;
    private static long _livePlanCountForTests;
    private static long _liveContextLeaseCountForTests;

    /// <summary>The complete identity of a reusable plan under this executor's fixed TN, non-batched, host-scalar contract. Device addresses are represented only by the alignment class visible to the heuristic; the actual bias address is patched per submission.</summary>
    internal readonly record struct PlanKey(
        int M, int N, int K,
        int AbType, int DType, int ComputeType, int ScaleType,
        int Epilogue, bool HasBias,
        uint AlignA, uint AlignB, uint AlignC, uint AlignD, ulong AlignBias,
        ulong WorkspaceLimitBytes);

    [StructLayout(LayoutKind.Sequential)]
    private struct MatmulAlgo
    {
        public fixed ulong Data[8];
    }

    // Native cublasLtMatmulHeuristicResult_t. CUDA 12/13 use a 64-byte algo, size_t workspace,
    // cublasStatus_t state, float wavesCount, and four reserved ints (96 bytes on the supported x64 targets).
    [StructLayout(LayoutKind.Sequential)]
    private struct HeuristicResult
    {
        public MatmulAlgo Algo;
        public nuint WorkspaceSize;
        public int State;
        public float WavesCount;
        public fixed int Reserved[4];
    }

    private sealed class Plan
    {
        public nint MatmulDesc;
        public nint LayoutA;
        public nint LayoutB;
        public nint LayoutD;
        public MatmulAlgo Algo;
        public nuint RequiredWorkspaceBytes;
        public bool Counted;
        public bool Released;
    }

    private sealed class CacheEntry
    {
        public CacheEntry(Plan? plan, LinkedListNode<PlanKey> node)
        {
            Plan = plan;
            Node = node;
        }

        // Null is a stable, exact-key "no supported algorithm" result.
        public Plan? Plan { get; }
        public LinkedListNode<PlanKey> Node { get; }
    }

    private enum BuildDisposition
    {
        Success,
        Unsupported,
        TransientFailure,
    }

    private readonly record struct BuildResult(BuildDisposition Disposition, Plan? Plan);

    internal readonly record struct CacheDiagnostics(
        int Count,
        long Hits,
        long Misses,
        long HeuristicQueries,
        long NegativeHits,
        long Fallbacks,
        long PlanCreates,
        long PlanDestroys,
        long Evictions,
        int LastComputeType);

    /// <summary>Whether a cuBLASLt handle is available. The workspace may be zero bytes after a recoverable allocation failure; in that case the heuristic can still select a zero-workspace plan.</summary>
    public bool IsSupported { get; }

    /// <summary>Creates an executor that binds to the first stream passed to <see cref="TryRun"/>. Subsequent calls must use that same stream because the executor owns one shared workspace. The calling thread must have an active registered CUDA backend so teardown can retain and rebind the exact owning context.</summary>
    public LtGemmExecutor()
        : this(ResolveAmbientOwnerContext(), ownerStream: 0, bindOwnerStream: false,
            DefaultPlanCacheCapacity, backendAdopted: false)
    {
    }

    /// <summary>Backend-owned constructor: binds the executor to its one compute stream from creation.</summary>
    internal LtGemmExecutor(
        CudaContext ownerContext, nint ownerStream, int planCacheCapacity = DefaultPlanCacheCapacity,
        bool backendAdopted = false)
        : this(ownerContext, ownerStream, bindOwnerStream: true, planCacheCapacity, backendAdopted)
    {
    }

    private LtGemmExecutor(
        CudaContext ownerContext, nint ownerStream, bool bindOwnerStream, int planCacheCapacity,
        bool backendAdopted)
    {
        ArgumentNullException.ThrowIfNull(ownerContext);
        if (planCacheCapacity < 0)
            throw new ArgumentOutOfRangeException(nameof(planCacheCapacity));

        _ownerContext = ownerContext;
        _deviceHandle = ownerContext.DeviceHandle;
        _backendAdopted = backendAdopted;
        _planCacheCapacity = planCacheCapacity;
        _ownerStream = ownerStream;
        _streamBound = bindOwnerStream;
        _ownerContext.EnsureCurrent();

        try
        {
            CudaDriverApi.cuDevicePrimaryCtxRetain(out _retainedContext, _deviceHandle).ThrowOnError();
            _contextLeaseCounted = true;
            Interlocked.Increment(ref _liveContextLeaseCountForTests);
            _ownerContext.EnsureRetainedCurrent(_retainedContext);

            int createStatus;
            try
            {
                createStatus = CublasLtApi.cublasLtCreate(out _ltHandle);
            }
            catch (DllNotFoundException)
            {
                _workspaceBytes = 0;
                IsSupported = false;
                return;
            }
            catch (EntryPointNotFoundException)
            {
                _workspaceBytes = 0;
                IsSupported = false;
                return;
            }

            if (createStatus != CublasLtApi.CUBLAS_STATUS_SUCCESS)
            {
                _workspaceBytes = 0;
                if (createStatus == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED || IsUnsupported(createStatus))
                {
                    // cuBLAS documents no returned handle on failure, but do not silently abandon one if a future
                    // implementation supplies it. A failed cleanup is a constructor failure, not "unsupported".
                    DestroyLtHandleForRollback();
                    IsSupported = false;
                    return;
                }
                createStatus.ThrowOnCublasError();
                throw new InvalidOperationException("Unreachable cuBLASLt create status classification.");
            }

            nuint requestedWorkspaceBytes = (nuint)CublasLtApi.DefaultWorkspaceBytes;
            try
            {
                _workspace = CudaMemory.AllocatePersistent(requestedWorkspaceBytes);
                _workspaceBytes = requestedWorkspaceBytes;
                IsSupported = true;
            }
            catch (OutOfVramException)
            {
                // Epilogue fusion is optional. Keep the valid handle and ask the heuristic for a zero-workspace
                // plan; if none exists, TryRun returns false and Linear uses GemmEx+BiasAdd. A best-effort
                // optimization must never turn 64 MiB of unavailable scratch into a request failure.
                _workspace = 0;
                _workspaceBytes = 0;
                IsSupported = true;
                try
                {
                    Logs.Warning("[Cuda] cuBLASLt workspace allocation failed; trying zero-workspace plans and falling back to GemmEx per shape.");
                }
                catch
                {
                    // Logging must not turn a successful zero-workspace degradation into a constructor failure.
                }
            }
        }
        catch (Exception primary)
        {
            List<Exception> cleanupFailures = [];
            RollbackConstruction(cleanupFailures);
            if (cleanupFailures.Count > 0)
                throw new AggregateException(
                    "cuBLASLt construction and synchronous native rollback both failed.",
                    [primary, .. cleanupFailures]);
            throw;
        }
    }

    private void DestroyLtHandleForRollback()
    {
        if (_ltHandle == 0) return;
        CublasLtApi.cublasLtDestroy(_ltHandle).ThrowOnCublasError();
        _ltHandle = 0;
    }

    private void RollbackConstruction(List<Exception> failures)
    {
        bool contextBound = _retainedContext == 0;
        if (_retainedContext != 0)
        {
            try
            {
                _ownerContext.EnsureRetainedCurrent(_retainedContext);
                contextBound = true;
            }
            catch (Exception error)
            {
                failures.Add(new InvalidOperationException(
                    "Failed to bind the retained CUDA context during cuBLASLt constructor rollback.", error));
            }
        }

        if (contextBound)
        {
            if (_workspace != 0)
            {
                try
                {
                    CudaMemory.Free(_workspace);
                    _workspace = 0;
                    _workspaceBytes = 0;
                }
                catch (Exception error)
                {
                    failures.Add(new InvalidOperationException(
                        "Failed to free the cuBLASLt workspace during constructor rollback.", error));
                }
            }

            if (_ltHandle != 0)
            {
                int failureCountBefore = failures.Count;
                nint handle = _ltHandle;
                TryNativeDestroy(
                    () => CublasLtApi.cublasLtDestroy(handle),
                    "cuBLASLt handle during constructor rollback", failures);
                if (failures.Count == failureCountBefore)
                    _ltHandle = 0;
            }
        }

        ReleaseContextLease(failures);
    }

    private void ReleaseContextLease(List<Exception> failures)
    {
        if (_retainedContext == 0) return;
        try
        {
            CudaDriverApi.cuDevicePrimaryCtxRelease(_deviceHandle).ThrowOnError();
            _retainedContext = 0;
            if (_contextLeaseCounted)
            {
                _contextLeaseCounted = false;
                Interlocked.Decrement(ref _liveContextLeaseCountForTests);
            }
        }
        catch (Exception error)
        {
            failures.Add(new InvalidOperationException("Failed to release the cuBLASLt primary-context lease.", error));
        }
    }

    /// <summary>Compatibility wrapper for direct callers. Backend dispatch should use <see cref="TryRun"/> and pass its resolved compute policy explicitly.</summary>
    public void Run(ulong weight, ulong input, ulong outPtr, int m, int n, int k, float alpha,
        int abType, int dType, ulong biasPtr, int epilogue, nint stream)
    {
        if (!TryRun(weight, input, outPtr, m, n, k, alpha, abType, dType,
                CublasApi.CUBLAS_COMPUTE_32F, biasPtr, epilogue, stream))
        {
            throw new InvalidOperationException(
                $"cuBLASLt found no supported algorithm for GEMM {m}x{n}x{k} " +
                $"(abType={abType}, dType={dType}, computeType={CublasApi.CUBLAS_COMPUTE_32F}, epilogue={epilogue}).");
        }
    }

    /// <summary>Attempts <c>output = alpha * input * weight^T + bias</c>. Returns <c>false</c> only when cuBLASLt or an algorithm for this exact contract is unavailable, allowing the caller to run GemmEx plus BiasAdd. Binding errors and execution/context failures still throw.</summary>
    public bool TryRun(ulong weight, ulong input, ulong outPtr, int m, int n, int k, float alpha,
        int abType, int dType, int computeType, ulong biasPtr, int epilogue, nint stream)
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            _ownerContext.EnsureRetainedCurrent(_retainedContext);
            Volatile.Write(ref _lastComputeType, computeType);
            if (!IsSupported)
            {
                _fallbacks++;
                return false;
            }

            ValidateArguments(weight, input, outPtr, m, n, k, biasPtr, epilogue);
            BindOrValidateStream(stream);

            bool hasBias = biasPtr != 0;
            PlanKey key = new(
                m, n, k,
                abType, dType, computeType, ScaleType,
                epilogue, hasBias,
                PointerAlignment(weight), PointerAlignment(input), PointerAlignment(outPtr),
                PointerAlignment(outPtr), FullPointerAlignment(biasPtr),
                (ulong)_workspaceBytes);

            Plan? plan;
            bool cached = false;
            bool newlyBuilt = false;
            if (_plans.TryGetValue(key, out CacheEntry? entry))
            {
                _cacheHits++;
                Touch(entry);
                if (entry.Plan is null)
                {
                    _negativeHits++;
                    _fallbacks++;
                    return false;
                }
                plan = entry.Plan;
                cached = true;
            }
            else
            {
                _cacheMisses++;
                BuildResult built = BuildPlan(key, biasPtr);
                if (built.Disposition == BuildDisposition.Unsupported)
                {
                    AddCacheEntry(key, plan: null);
                    _fallbacks++;
                    return false;
                }
                if (built.Disposition == BuildDisposition.TransientFailure)
                {
                    _fallbacks++;
                    return false;
                }

                plan = built.Plan!;
                newlyBuilt = true;
            }

            try
            {
                if (newlyBuilt && _planCacheCapacity > 0)
                    AddCacheEntry(key, plan, ref cached);

                if (hasBias)
                {
                    ulong biasArg = biasPtr;
                    int biasStatus = CublasLtApi.cublasLtMatmulDescSetAttribute(
                        plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                        &biasArg, (nuint)sizeof(ulong));
                    biasStatus.ThrowOnCublasError();
                }

                float a = alpha;
                float beta = 0.0f;
                MatmulAlgo algo = plan.Algo;
                int runStatus = CublasLtApi.cublasLtMatmul(
                    _ltHandle, plan.MatmulDesc,
                    &a,
                    weight, plan.LayoutA,
                    input, plan.LayoutB,
                    &beta,
                    outPtr, plan.LayoutD,
                    outPtr, plan.LayoutD,
                    (nint)(&algo), (nint)_workspace, _workspaceBytes, stream);

                if (runStatus == CublasLtApi.CUBLAS_STATUS_SUCCESS)
                    return true;

                if (IsUnsupported(runStatus) || runStatus == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED)
                {
                    if (cached)
                        RemoveCacheEntry(key);
                    // An execution-time rejection can be state-dependent (notably CUDA graph capture). Evict the
                    // rejected algorithm, but reserve negative entries for an exact heuristic no-algorithm result;
                    // otherwise one capture rejection would poison eager execution for the life of this backend.
                    _fallbacks++;
                    return false;
                }

                runStatus.ThrowOnCublasError();
                throw new InvalidOperationException("Unreachable cuBLASLt status classification.");
            }
            finally
            {
                if (!cached)
                    ReleasePlan(plan, throwOnError: true);
            }
        }
    }

    private BuildResult BuildPlan(PlanKey key, ulong biasPtr)
    {
        Plan plan = new();
        nint preference = 0;
        bool ownershipTransferred = false;
        try
        {
            int status = CublasLtApi.cublasLtMatmulDescCreate(
                out plan.MatmulDesc, key.ComputeType, key.ScaleType);
            if (status == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED)
                return new(BuildDisposition.TransientFailure, null);
            status.ThrowOnCublasError();

            int transA = CublasApi.CUBLAS_OP_T;
            int transB = CublasApi.CUBLAS_OP_N;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSA,
                &transA, sizeof(int)).ThrowOnCublasError();
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_TRANSB,
                &transB, sizeof(int)).ThrowOnCublasError();

            int epilogue = key.Epilogue;
            CublasLtApi.cublasLtMatmulDescSetAttribute(
                plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue, sizeof(int)).ThrowOnCublasError();
            if (key.HasBias)
            {
                ulong biasArg = biasPtr;
                int biasType = key.DType;
                CublasLtApi.cublasLtMatmulDescSetAttribute(
                    plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    &biasArg, (nuint)sizeof(ulong)).ThrowOnCublasError();
                CublasLtApi.cublasLtMatmulDescSetAttribute(
                    plan.MatmulDesc, CublasLtApi.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    &biasType, sizeof(int)).ThrowOnCublasError();
            }

            // weight [N,K] row-major -> A as KxN col-major (OP_T); input [M,K] row-major -> B as KxM;
            // output [M,N] row-major == D as NxM col-major.
            CublasLtApi.cublasLtMatrixLayoutCreate(
                out plan.LayoutA, key.AbType, (ulong)key.K, (ulong)key.N, key.K).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(
                out plan.LayoutB, key.AbType, (ulong)key.K, (ulong)key.M, key.K).ThrowOnCublasError();
            CublasLtApi.cublasLtMatrixLayoutCreate(
                out plan.LayoutD, key.DType, (ulong)key.N, (ulong)key.M, key.N).ThrowOnCublasError();

            CublasLtApi.cublasLtMatmulPreferenceCreate(out preference).ThrowOnCublasError();
            ulong workspaceLimit = key.WorkspaceLimitBytes;
            CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
                preference, CublasLtApi.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspaceLimit, (nuint)sizeof(ulong)).ThrowOnCublasError();
            SetAlignmentPreference(preference, CublasLtApi.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, key.AlignA);
            SetAlignmentPreference(preference, CublasLtApi.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, key.AlignB);
            SetAlignmentPreference(preference, CublasLtApi.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, key.AlignC);
            SetAlignmentPreference(preference, CublasLtApi.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, key.AlignD);

            HeuristicResult heuristic = default;
            _heuristicQueries++;
            int heuristicStatus;
            int returnedAlgoCount;
            if (_forceNoAlgorithmForTests)
            {
                // Exercise the real descriptor/preference construction and rollback path while deterministically
                // modeling the native heuristic's ordinary "no algorithm" result.
                heuristicStatus = CublasLtApi.CUBLAS_STATUS_SUCCESS;
                returnedAlgoCount = 0;
            }
            else
            {
                heuristicStatus = CublasLtApi.cublasLtMatmulAlgoGetHeuristic(
                    _ltHandle, plan.MatmulDesc,
                    plan.LayoutA, plan.LayoutB, plan.LayoutD, plan.LayoutD,
                    preference, 1, &heuristic, out returnedAlgoCount);
            }

            if (IsUnsupported(heuristicStatus))
                return new(BuildDisposition.Unsupported, null);
            if (heuristicStatus == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED)
                return new(BuildDisposition.TransientFailure, null);
            heuristicStatus.ThrowOnCublasError();

            if (returnedAlgoCount < 1)
                return new(BuildDisposition.Unsupported, null);
            if (IsUnsupported(heuristic.State))
                return new(BuildDisposition.Unsupported, null);
            if (heuristic.State == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED)
                return new(BuildDisposition.TransientFailure, null);
            heuristic.State.ThrowOnCublasError();

            if (heuristic.WorkspaceSize > _workspaceBytes)
            {
                throw new InvalidOperationException(
                    $"cuBLASLt heuristic requires {heuristic.WorkspaceSize} workspace bytes, above the " +
                    $"configured {_workspaceBytes} byte limit for {key.M}x{key.N}x{key.K}.");
            }

            plan.Algo = heuristic.Algo;
            plan.RequiredWorkspaceBytes = heuristic.WorkspaceSize;

            try
            {
                CublasLtApi.cublasLtMatmulPreferenceDestroy(preference).ThrowOnCublasError();
                preference = 0;
            }
            catch (Exception cleanup)
            {
                // Teardown failure is never an algorithm-capability result. Keep the native handle for the
                // rollback path to retry, and wrap the status so the Unsupported/ALLOC_FAILED filters below
                // cannot accidentally convert a failed destroy into an ordinary GemmEx fallback.
                throw new InvalidOperationException(
                    "Failed to destroy the cuBLASLt matmul preference after plan selection.", cleanup);
            }

            _planCreates++;
            Interlocked.Increment(ref _livePlanCountForTests);
            plan.Counted = true;
            ownershipTransferred = true;
            return new(BuildDisposition.Success, plan);
        }
        catch (CudaException error) when (error.ErrorCode == CublasLtApi.CUBLAS_STATUS_ALLOC_FAILED)
        {
            return new(BuildDisposition.TransientFailure, null);
        }
        catch (CudaException error) when (IsUnsupported(error.ErrorCode))
        {
            return new(BuildDisposition.Unsupported, null);
        }
        catch (Exception primary)
        {
            List<Exception> cleanupFailures = [];
            ReleaseBuildResources(plan, ref preference, cleanupFailures);
            if (cleanupFailures.Count > 0)
                throw new AggregateException("cuBLASLt plan construction and rollback both failed.", [primary, .. cleanupFailures]);
            throw;
        }
        finally
        {
            // Success transfers plan ownership to the caller. Unsupported/transient outcomes must still destroy
            // every descriptor allocated while probing; otherwise one unsupported dynamic shape leaks four native
            // objects on every invocation.
            if (!ownershipTransferred && (!plan.Released || preference != 0))
            {
                List<Exception> cleanupFailures = [];
                ReleaseBuildResources(plan, ref preference, cleanupFailures);
                if (cleanupFailures.Count > 0)
                    throw new AggregateException("cuBLASLt plan rollback failed.", cleanupFailures);
            }
        }
    }

    private void ReleaseBuildResources(Plan plan, ref nint preference, List<Exception> failures)
    {
        if (preference != 0)
        {
            nint ownedPreference = preference;
            int failureCountBefore = failures.Count;
            TryNativeDestroy(
                () => CublasLtApi.cublasLtMatmulPreferenceDestroy(ownedPreference),
                "cuBLASLt matmul preference", failures);
            if (failures.Count == failureCountBefore)
                preference = 0;
        }
        ReleasePlan(plan, throwOnError: false, failures);
    }

    private static void SetAlignmentPreference(nint preference, int attribute, uint alignment)
    {
        CublasLtApi.cublasLtMatmulPreferenceSetAttribute(
            preference, attribute, &alignment, (nuint)sizeof(uint)).ThrowOnCublasError();
    }

    private void AddCacheEntry(PlanKey key, Plan? plan)
    {
        bool ignoredPlanOwnership = false;
        AddCacheEntry(key, plan, ref ignoredPlanOwnership);
    }

    private void AddCacheEntry(PlanKey key, Plan? plan, ref bool cacheOwnsPlan)
    {
        if (_planCacheCapacity == 0)
            return;

        // Publish to both containers transactionally. Until both insertions succeed the caller still owns a
        // positive plan and its finally block must release it. Once cacheOwnsPlan flips true, even a subsequent
        // eviction-destroy failure cannot cause the caller to double-destroy the newly cached plan.
        LinkedListNode<PlanKey> node = new(key);
        _plans.Add(key, new CacheEntry(plan, node));
        try
        {
            _lru.AddFirst(node);
        }
        catch
        {
            _plans.Remove(key);
            throw;
        }
        if (plan is not null)
            cacheOwnsPlan = true;

        if (_plans.Count <= _planCacheCapacity)
            return;

        LinkedListNode<PlanKey> victimNode = _lru.Last!;
        PlanKey victimKey = victimNode.Value;
        CacheEntry victim = _plans[victimKey];
        _lru.Remove(victimNode);
        _plans.Remove(victimKey);
        _evictions++;
        if (victim.Plan is not null)
            ReleasePlan(victim.Plan, throwOnError: true);
    }

    private void RemoveCacheEntry(PlanKey key)
    {
        if (!_plans.Remove(key, out CacheEntry? entry))
            return;
        _lru.Remove(entry.Node);
        if (entry.Plan is not null)
            ReleasePlan(entry.Plan, throwOnError: true);
    }

    private void Touch(CacheEntry entry)
    {
        _lru.Remove(entry.Node);
        _lru.AddFirst(entry.Node);
    }

    private void BindOrValidateStream(nint stream)
    {
        if (!_streamBound)
        {
            _ownerStream = stream;
            _streamBound = true;
            return;
        }
        if (_ownerStream != stream)
        {
            throw new InvalidOperationException(
                $"LtGemmExecutor is stream-affine: owner=0x{_ownerStream:X}, requested=0x{stream:X}. " +
                "Use one executor and workspace per CUDA stream.");
        }
    }

    private static void ValidateArguments(
        ulong weight, ulong input, ulong outPtr, int m, int n, int k, ulong biasPtr, int epilogue)
    {
        if (weight == 0) throw new ArgumentOutOfRangeException(nameof(weight));
        if (input == 0) throw new ArgumentOutOfRangeException(nameof(input));
        if (outPtr == 0) throw new ArgumentOutOfRangeException(nameof(outPtr));
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (biasPtr == 0 && EpilogueRequiresBias(epilogue))
            throw new ArgumentException($"cuBLASLt epilogue {epilogue} requires a non-zero bias pointer.", nameof(biasPtr));
    }

    private static bool EpilogueRequiresBias(int epilogue)
        => epilogue == CublasLtApi.CUBLASLT_EPILOGUE_BIAS
            || epilogue == CublasLtApi.CUBLASLT_EPILOGUE_RELU_BIAS
            || epilogue == CublasLtApi.CUBLASLT_EPILOGUE_GELU_BIAS;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsUnsupported(int status)
        => status == CublasLtApi.CUBLAS_STATUS_NOT_SUPPORTED
            || status == CublasLtApi.CUBLAS_STATUS_ARCH_MISMATCH;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint PointerAlignment(ulong pointer)
    {
        if (pointer == 0) return 0;
        int power = Math.Min(BitOperations.TrailingZeroCount(pointer), 8);
        return 1u << power;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ulong FullPointerAlignment(ulong pointer)
    {
        if (pointer == 0) return 0;
        return 1UL << BitOperations.TrailingZeroCount(pointer);
    }

    private static CudaContext ResolveAmbientOwnerContext()
    {
        CudaContext? context = GpuTransferHelper.CurrentState.Context;
        return context ?? throw new InvalidOperationException(
            "LtGemmExecutor requires an owning CUDA context. Construct it through CudaBackend.LtGemm " +
            "while that backend is active, or use the context-aware internal constructor in CUDA tests.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
            throw new ObjectDisposedException(nameof(LtGemmExecutor));
    }

    internal CacheDiagnostics Diagnostics
    {
        get
        {
            lock (_gate)
            {
                return new(
                    _plans.Count,
                    _cacheHits,
                    _cacheMisses,
                    _heuristicQueries,
                    _negativeHits,
                    _fallbacks,
                    _planCreates,
                    _planDestroys,
                    _evictions,
                    Volatile.Read(ref _lastComputeType));
            }
        }
    }

    internal static long LivePlanCountForTests => Interlocked.Read(ref _livePlanCountForTests);
    internal static long LiveContextLeaseCountForTests => Interlocked.Read(ref _liveContextLeaseCountForTests);

    internal bool ForceNoAlgorithmForTests
    {
        get { lock (_gate) return _forceNoAlgorithmForTests; }
        set
        {
            lock (_gate)
            {
                ThrowIfDisposed();
                if (_forceNoAlgorithmForTests == value) return;
                _ownerContext.EnsureRetainedCurrent(_retainedContext);
                _forceNoAlgorithmForTests = value;
                ClearPlanCacheLocked(throwOnError: true);
            }
        }
    }

    internal void ClearPlanCacheForTests()
    {
        lock (_gate)
        {
            ThrowIfDisposed();
            _ownerContext.EnsureRetainedCurrent(_retainedContext);
            ClearPlanCacheLocked(throwOnError: true);
        }
    }

    private void SynchronizeOwnerStream()
    {
        if (!_streamBound) return;

        if (!_backendAdopted)
        {
            // A standalone executor only borrows the caller's stream. There is no status-safe way to probe whether
            // that raw CUstream was already destroyed: some drivers dereference the stale handle and segfault inside
            // cuStreamSynchronize instead of returning CUDA_ERROR_INVALID_HANDLE. Our independent primary-context
            // lease is authoritative, so drain the retained context directly without ever touching the borrowed
            // stream during teardown.
            CudaDriverApi.cuCtxSynchronize().ThrowOnError();
            return;
        }

        Exception streamFailure;
        try
        {
            int streamStatus = CudaDriverApi.cuStreamSynchronize(_ownerStream);
            if (streamStatus == 0) return;
            try
            {
                streamStatus.ThrowOnError();
                throw new InvalidOperationException("Unreachable CUDA stream synchronization status.");
            }
            catch (Exception error)
            {
                streamFailure = error;
            }
        }
        catch (Exception error)
        {
            streamFailure = error;
        }

        // Backend cleanup guarantees that its owned stream is still live here. If synchronization nevertheless
        // reports an error, the independently-retained context remains a safe drain fallback.
        try
        {
            CudaDriverApi.cuCtxSynchronize().ThrowOnError();
        }
        catch (Exception contextFailure)
        {
            throw new AggregateException(
                "Failed to drain both the cuBLASLt owner stream and its retained CUDA context.",
                streamFailure, contextFailure);
        }
    }

    private bool TrySynchronizeOwnerWorkForFinalizer()
    {
        if (!_streamBound) return true;
        // A finalizer can never prove the borrowed/owning stream's lifetime. Use only the retained context; passing
        // a stale CUstream to the driver can fault natively before a managed status fallback is possible.
        try { return CudaDriverApi.cuCtxSynchronize() == 0; }
        catch { return false; }
    }

    public void Dispose()
    {
        List<Exception> failures = [];
        lock (_gate)
        {
            // A previous checked cleanup may have failed while retaining this executor's independent context
            // lease. Permit Dispose to retry those remaining resources, but never accept new GEMMs after the
            // first cleanup claim.
            if (Volatile.Read(ref _disposed) != 0 && _retainedContext == 0) return;
            bool safeToReleaseNativeResources = true;
            try
            {
                _ownerContext.EnsureRetainedCurrent(_retainedContext);
                // The shared workspace can still be in use by already-enqueued matmuls after the host lock is
                // released. Backend-adopted instances drain their still-owned stream; standalone instances drain
                // the retained context without dereferencing a possibly-destroyed borrowed stream.
                SynchronizeOwnerStream();
            }
            catch (Exception error)
            {
                if (!_backendAdopted) throw;
                // Backend teardown is terminal and the child finalizer is intentionally suppressed. Do not issue
                // unsafe descriptor/workspace calls without a proven drain, but do record the failure and release
                // this executor's independent primary-context retain below so the backend can finish destroying
                // its own context instead of pinning it forever.
                failures.Add(new InvalidOperationException(
                    "Failed to bind or drain the backend-adopted cuBLASLt executor during terminal cleanup.", error));
                safeToReleaseNativeResources = false;
            }

            Volatile.Write(ref _disposed, 1);
            try
            {
                if (safeToReleaseNativeResources)
                {
                    ClearPlanCacheLocked(throwOnError: false, failures);

                    if (_workspace != 0)
                    {
                        try
                        {
                            CudaMemory.Free(_workspace);
                            _workspace = 0;
                            _workspaceBytes = 0;
                        }
                        catch (Exception error) { failures.Add(error); }
                    }

                    if (_ltHandle != 0)
                    {
                        int failureCountBefore = failures.Count;
                        nint handle = _ltHandle;
                        TryNativeDestroy(
                            () => CublasLtApi.cublasLtDestroy(handle),
                            "cuBLASLt handle", failures);
                        if (failures.Count == failureCountBefore)
                            _ltHandle = 0;
                    }
                }
            }
            catch (Exception error)
            {
                failures.Add(new InvalidOperationException(
                    "Unexpected failure during cuBLASLt terminal resource cleanup.", error));
            }
            finally
            {
                // Always attempt the independently-retained lease last. Backend-owned executors are detached
                // during teardown and cannot rely on a second managed Dispose call; retaining the primary context
                // after any earlier best-effort cleanup failure would pin it permanently.
                ReleaseContextLease(failures);
            }
        }

        if (failures.Count > 0)
            throw new AggregateException("One or more cuBLASLt executor resources failed to release.", failures);
        GC.SuppressFinalize(this);
    }

    private void ClearPlanCacheLocked(bool throwOnError, List<Exception>? failures = null)
    {
        List<Exception> localFailures = failures ?? [];
        CacheEntry[] entries = [.. _plans.Values];
        _plans.Clear();
        _lru.Clear();
        foreach (CacheEntry entry in entries)
        {
            if (entry.Plan is not null)
                ReleasePlan(entry.Plan, throwOnError: false, localFailures);
        }
        if (throwOnError && localFailures.Count > 0)
            throw new AggregateException("One or more cached cuBLASLt plans failed to release.", localFailures);
    }

    private void ReleasePlan(Plan plan, bool throwOnError, List<Exception>? failures = null)
    {
        if (plan.Released) return;
        plan.Released = true;

        List<Exception> localFailures = failures ?? [];
        int failureCountBefore = localFailures.Count;
        nint layoutA = plan.LayoutA;
        nint layoutB = plan.LayoutB;
        nint layoutD = plan.LayoutD;
        nint matmulDesc = plan.MatmulDesc;
        plan.LayoutA = 0;
        plan.LayoutB = 0;
        plan.LayoutD = 0;
        plan.MatmulDesc = 0;

        if (layoutA != 0)
            TryNativeDestroy(() => CublasLtApi.cublasLtMatrixLayoutDestroy(layoutA), "cuBLASLt A layout", localFailures);
        if (layoutB != 0)
            TryNativeDestroy(() => CublasLtApi.cublasLtMatrixLayoutDestroy(layoutB), "cuBLASLt B layout", localFailures);
        if (layoutD != 0)
            TryNativeDestroy(() => CublasLtApi.cublasLtMatrixLayoutDestroy(layoutD), "cuBLASLt D layout", localFailures);
        if (matmulDesc != 0)
            TryNativeDestroy(() => CublasLtApi.cublasLtMatmulDescDestroy(matmulDesc), "cuBLASLt matmul descriptor", localFailures);

        // Only fully selected plans increment these counters. Partial-build rollback still releases every handle,
        // but is not a live/cacheable plan for diagnostic purposes. A failed native destroy deliberately leaves the
        // live count elevated: zero would incorrectly claim that all native plan resources were released.
        if (plan.Counted && localFailures.Count == failureCountBefore)
        {
            plan.Counted = false;
            _planDestroys++;
            Interlocked.Decrement(ref _livePlanCountForTests);
        }

        if (throwOnError && localFailures.Count > 0)
            throw new AggregateException("A cuBLASLt plan failed to release cleanly.", localFailures);
    }

    private static void TryNativeDestroy(Func<int> destroy, string resource, List<Exception> failures)
    {
        try
        {
            int status = destroy();
            if (status != CublasLtApi.CUBLAS_STATUS_SUCCESS)
                status.ThrowOnCublasError();
        }
        catch (Exception error)
        {
            failures.Add(new InvalidOperationException($"Failed to destroy {resource}.", error));
        }
    }

    ~LtGemmExecutor()
    {
        try
        {
            lock (_gate)
            {
                if (_retainedContext == 0) return;
                Volatile.Write(ref _disposed, 1);
                List<Exception> ignoredFailures = [];
                try
                {
                    bool safeToReleaseNativeResources = false;
                    try
                    {
                        _ownerContext.EnsureRetainedCurrent(_retainedContext);
                        safeToReleaseNativeResources = TrySynchronizeOwnerWorkForFinalizer();
                    }
                    catch { }

                    // Never free a workspace that may still back queued work. If neither the exact stream nor the
                    // retained context can be drained, skip child-resource calls and release only this lease; the
                    // remaining primary-context owner (or its eventual teardown) retains native safety.
                    if (safeToReleaseNativeResources)
                    {
                        ClearPlanCacheLocked(throwOnError: false);
                        if (_workspace != 0)
                        {
                            try
                            {
                                CudaMemory.Free(_workspace);
                                _workspace = 0;
                                _workspaceBytes = 0;
                            }
                            catch { }
                        }
                        if (_ltHandle != 0)
                        {
                            try
                            {
                                if (CublasLtApi.cublasLtDestroy(_ltHandle) == CublasLtApi.CUBLAS_STATUS_SUCCESS)
                                    _ltHandle = 0;
                            }
                            catch { }
                        }
                    }
                }
                finally
                {
                    // Last operation by contract. Even if best-effort descriptor/workspace teardown reported a
                    // failure, releasing the final lease lets the driver reclaim the primary context when no other
                    // owner remains instead of permanently pinning the device context from the finalizer queue.
                    ReleaseContextLease(ignoredFailures);
                }
            }
        }
        catch
        {
            // Finalizers must never terminate the process. CudaBackend suppresses this finalizer once it adopts
            // the executor; its managed abandonment reaper performs the normal checked cleanup path.
        }
    }
}
