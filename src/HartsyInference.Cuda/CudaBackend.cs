using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

/// <summary>CUDA GPU backend implementing <see cref="IBackend"/>: cuBLAS GEMM for matmul, PTX kernels for element-wise/normalization ops.</summary>
/// <remarks>Uses activation caching to keep intermediate results on GPU between ops — lazy sync to CPU on DataPointer access.</remarks>
public sealed class CudaBackend : IBackend
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    /// <summary>Side stream for async weight uploads that overlap with compute on <see cref="_stream"/>.</summary>
    /// <remarks>Used by <see cref="_streamingCache"/>. Created non-blocking so it doesn't serialize with the
    /// compute stream — synchronization between the two is explicit via <c>cuEventRecord</c> /
    /// <c>cuStreamWaitEvent</c> inside the streaming cache.</remarks>
    private readonly CudaStream _uploadStream;
    private readonly CudaStreamingWeightCache _streamingCache;
    /// <summary>This backend's transfer state — its identity in <see cref="GpuTransferHelper"/>, bound to the calling thread by <see cref="EnterOp"/> at every op entry. Two same-device backends share a primary context but never a state.</summary>
    private readonly GpuTransferHelper.State _transferState;
    private readonly CudaKernels? _kernels;
    private nint _cublasHandle;
    private Fp8GemmExecutor? _fp8Executor;
    private LtGemmExecutor? _ltGemmExecutor;
    private TensorCoreGemm? _tensorCoreGemm;
    private readonly object _nativeExecutorLock = new();
    private readonly object _cudnnSdpaLock = new();
    private CudnnSdpa? _cudnnSdpa;
    private volatile bool _cudnnSdpaDead;   // set if cuDNN INIT throws once — never retry, fall back for the session

    /// <summary>Per-head-dim failure/backoff state — replaces a plain permanent-dead set.</summary>
    /// <remarks>A structural failure (e.g. D=256 on a build whose fused engine tops out at 128 — <see
    /// cref="CudnnStatusException.IsPermanent"/>) disables that dim forever, same as before; a transient failure
    /// (e.g. the host-RAM allocation error that motivated this — see the plan notes referenced from <see
    /// cref="TryCudnnSdpa"/>) gets bounded backoff instead, since the resource pressure that caused it is typically
    /// external to this process and does clear up. <see cref="ConcurrentDictionary{TKey,TValue}"/> (not a plain
    /// <c>HashSet</c>) because this backend is a process-wide DI singleton and nothing enforces
    /// <c>InferenceQueue.MaxConcurrency</c> stays at its default of 1 — a plain <c>HashSet</c> was never actually
    /// safe under a higher setting.</remarks>
    /// <param name="NextRetryAtTicks">Environment.TickCount64; 0 = eligible immediately.</param>
    private sealed record DimFailureState(int ConsecutiveFailures, long NextRetryAtTicks, bool Permanent);
    private readonly ConcurrentDictionary<long, DimFailureState> _cudnnSdpaDimState = new();

    /// <summary>Test-only fault hook for <see cref="TryCudnnSdpa"/>: a non-null return from this is thrown instead of the real call.</summary>
    /// <remarks>Exists so the classify/retry/backoff behavior can be tested deterministically — a real
    /// host-RAM-starvation failure can't be reproduced on demand in a fast unit test. Null in production.</remarks>
    internal Func<long, Exception?>? TestCudnnSdpaFaultInjector { get; set; }

    /// <summary>Per-dim diagnostic snapshot for <see cref="TryCudnnSdpa"/>'s classify/retry/backoff state.</summary>
    /// <remarks>Programmatically queryable observability surface (deliberately not wired into <c>/ready</c>: a
    /// degraded-but-correct fallback to the materialized attention path is not a "can't serve traffic" condition,
    /// matching this engine's existing health-check philosophy).</remarks>
    public IReadOnlyDictionary<long, (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt)> CudnnSdpaDimDiagnostics =>
        _cudnnSdpaDimState.ToDictionary(
            kv => kv.Key,
            kv => (kv.Value.ConsecutiveFailures, kv.Value.Permanent,
                   kv.Value.Permanent ? (DateTimeOffset?)null : DateTimeOffset.UtcNow.AddMilliseconds(kv.Value.NextRetryAtTicks - Environment.TickCount64)));

    /// <summary>True if head dim <paramref name="d"/> is currently eligible for a cuDNN SDPA attempt.</summary>
    /// <remarks>No failure on record (the overwhelmingly common case: a single dictionary miss, same O(1) cost as
    /// the plain <c>HashSet.Contains</c> this replaces), not permanently disabled, and any backoff window from a
    /// prior transient failure has elapsed.</remarks>
    private bool CudnnSdpaDimEligible(long d)
    {
        if (!_cudnnSdpaDimState.TryGetValue(d, out DimFailureState? s)) return true;
        if (s.Permanent) return false;
        return Environment.TickCount64 >= s.NextRetryAtTicks;
    }

    // Backoff schedule for a transient cuDNN SDPA failure: short at first (the common case — a brief host-RAM
    // blip clears within seconds), capped so a persistently-constrained box doesn't retry so often it's
    // effectively hammering itself, but never gives up permanently on a failure class that isn't structural.
    private static long CudnnSdpaBackoffMs(int consecutiveFailures)
    {
        const long baseMs = 2000, capMs = 5 * 60 * 1000;
        int shift = Math.Min(consecutiveFailures - 1, 8);
        return Math.Min(baseMs * (1L << shift), capMs);
    }

    private CudnnConv? _cudnnConv;
    private bool _cudnnConvDead;   // any cuDNN conv failure → session fallback to the im2col path

    private long _cudnnSdpaExecutionCount;
    private long _cudnnSdpaSessionGeneration;
    private long _cudnnSdpaDisposedSessionCount;

    /// <summary>True once the cuDNN fused-attention fast path has run at least once this session (confirms engagement vs fallback).</summary>
    public bool CudnnSdpaEngaged => CudnnSdpaExecutionCount > 0;

    /// <summary>Number of successfully enqueued cuDNN fused-attention executions this session.</summary>
    public long CudnnSdpaExecutionCount => Interlocked.Read(ref _cudnnSdpaExecutionCount);

    /// <summary>Test diagnostic incremented whenever a new cuDNN SDPA handle/plan cache is constructed.</summary>
    internal long CudnnSdpaSessionGeneration => Interlocked.Read(ref _cudnnSdpaSessionGeneration);

    /// <summary>Test diagnostic incremented only after a detached cuDNN SDPA session releases all owned resources.</summary>
    internal long CudnnSdpaDisposedSessionCount => Interlocked.Read(ref _cudnnSdpaDisposedSessionCount);

    /// <summary>True once the opt-in TF32 FlashAttention-v2 path has run at least once this session.</summary>
    internal bool FlashAttentionV2Engaged { get; private set; }

    private long _sageAttentionExecutionCount;

    /// <summary>True once the SageAttention INT8 fast path has run at least once this session.</summary>
    public bool SageAttentionEngaged => SageAttentionExecutionCount > 0;

    /// <summary>Number of successfully enqueued SageAttention INT8 executions this session.</summary>
    public long SageAttentionExecutionCount => Interlocked.Read(ref _sageAttentionExecutionCount);

    /// <summary>True once the cuDNN convolution fast path has run at least once — same diagnostic role as <see cref="CudnnSdpaEngaged"/>.</summary>
    public bool CudnnConvEngaged { get; private set; }
    private readonly string? _ptxDir;

    private const int LifecycleActive = 0;
    private const int LifecycleClaimed = 1;
    private const int LifecycleCleaned = 2;
    private int _lifecycleState;
    private int _cleanupExecutionCount;
    private int _constructorCompleted;
    private int _mempoolPolicyAcquired;
    private Exception? _cleanupFailure;
    private readonly ManualResetEventSlim _cleanupCompleted = new(initialState: false);

    private static long _abandonedCleanupEnqueuedCount;
    private static long _abandonedCleanupCompletedCount;
    private static long _abandonedCleanupFailedCount;
    private static readonly object _abandonedCleanupProgress = new();

    /// <summary>Dedicated managed worker for abandoned backends. Its type initializer is forced from the constructor, never from the finalizer; the finalizer itself performs only an atomic claim and a queue write.</summary>
    private static class BackendAbandonmentReaper
    {
        private static readonly BlockingCollection<CudaBackend> Queue = new();
        private static readonly Thread Worker = StartWorker();

        internal static void EnsureStarted() => GC.KeepAlive(Worker);

        internal static void Enqueue(CudaBackend backend)
        {
            Interlocked.Increment(ref _abandonedCleanupEnqueuedCount);
            Queue.Add(backend);
        }

        private static Thread StartWorker()
        {
            Thread worker = new(Run)
            {
                IsBackground = true,
                Name = "Hartsy CUDA abandonment reaper",
            };
            worker.Start();
            return worker;
        }

        private static void Run()
        {
            while (true)
            {
                CudaBackend? backend = Queue.Take();
                Process(backend);
                // Do not let the worker local root the most recently cleaned backend while it waits indefinitely
                // for its next item (observable as an otherwise permanent managed leak).
                backend = null;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void Process(CudaBackend backend)
        {
            try
            {
                Exception? failure = backend.RunClaimedCleanup(abandoned: true);
                if (failure is not null)
                {
                    Interlocked.Increment(ref _abandonedCleanupFailedCount);
                    SafeLogFailure("[Cuda] Abandoned backend cleanup completed with one or more teardown failures.", failure);
                }
            }
            catch (Exception failure)
            {
                // The instance cleanup contains every individual action, but the worker itself must also be
                // immortal if a future edit accidentally lets an exception escape that boundary.
                Interlocked.Increment(ref _abandonedCleanupFailedCount);
                SafeLogFailure("[Cuda] Abandoned backend reaper failed unexpectedly.", failure);
            }
            finally
            {
                Interlocked.Increment(ref _abandonedCleanupCompletedCount);
                lock (_abandonedCleanupProgress) Monitor.PulseAll(_abandonedCleanupProgress);
            }
        }

        private static void SafeLogFailure(string message, Exception failure)
        {
            try { HartsyInference.Core.Logging.Logs.Error(message, failure); }
            catch { /* A user-supplied logger must never kill the process-wide cleanup worker. */ }
        }
    }

    internal int LifecycleStateForTests => Volatile.Read(ref _lifecycleState);
    internal int CleanupExecutionCount => Volatile.Read(ref _cleanupExecutionCount);
    internal bool ConstructorCompletedForTests => Volatile.Read(ref _constructorCompleted) != 0;
    internal static long AbandonedCleanupEnqueuedCount => Interlocked.Read(ref _abandonedCleanupEnqueuedCount);
    internal static long AbandonedCleanupCompletedCount => Interlocked.Read(ref _abandonedCleanupCompletedCount);
    internal static long AbandonedCleanupFailedCount => Interlocked.Read(ref _abandonedCleanupFailedCount);

    internal static bool WaitForAbandonedCleanup(long completedTarget, TimeSpan timeout)
    {
        if (completedTarget < 0) throw new ArgumentOutOfRangeException(nameof(completedTarget));
        if (timeout < TimeSpan.Zero && timeout != Timeout.InfiniteTimeSpan) throw new ArgumentOutOfRangeException(nameof(timeout));
        long deadline = timeout == Timeout.InfiniteTimeSpan ? long.MaxValue
            : Environment.TickCount64 + Math.Max(0L, (long)Math.Ceiling(timeout.TotalMilliseconds));
        lock (_abandonedCleanupProgress)
        {
            while (Interlocked.Read(ref _abandonedCleanupCompletedCount) < completedTarget)
            {
                if (deadline == long.MaxValue)
                {
                    Monitor.Wait(_abandonedCleanupProgress);
                    continue;
                }
                long remaining = deadline - Environment.TickCount64;
                if (remaining <= 0) return false;
                Monitor.Wait(_abandonedCleanupProgress, (int)Math.Min(remaining, int.MaxValue));
            }
            return true;
        }
    }

    /// <summary>The device this backend targets.</summary>
    public DeviceKind Device { get; }

    /// <summary>Capabilities of this CUDA backend.</summary>
    public BackendCapabilities Capabilities { get; }

    /// <summary>The CUDA context used by this backend.</summary>
    public CudaContext Context => _context;

    /// <summary>The default compute stream.</summary>
    public CudaStream Stream => _stream;

    /// <summary>The loaded kernel table (null if PTX kernels unavailable); internal for tests and optional-kernel launch glue.</summary>
    internal CudaKernels? Kernels => _kernels;

    /// <summary>The upload stream for async weight transfers. For diagnostics/tests; production callers use <see cref="StreamingCache"/>.</summary>
    public CudaStream UploadStream => _uploadStream;

    /// <inheritdoc/>
    public IStreamingWeightCache? StreamingCache => _streamingCache;

    /// <summary>Opt-in: page-lock weight host sources before streaming uploads so async H2D overlaps compute. Defaults to <c>false</c>.</summary>
    /// <remarks>See <see cref="CudaStreamingWeightCache.PinUploadSource"/>. Beneficial for block-swap workloads
    /// that re-upload weights across steps.</remarks>
    public bool EnablePinnedWeightUploads
    {
        get => _streamingCache.PinUploadSource;
        set => _streamingCache.PinUploadSource = value;
    }

    /// <summary>The cuBLAS handle for GEMM operations.</summary>
    public nint CublasHandle => _cublasHandle;

    /// <summary>Opt-in flag for the native cuBLASLt FP8 GEMM path on Ada+ (SM 8.9+) GPUs. Defaults to <c>false</c>.</summary>
    /// <remarks>On Ampere and below the path is unsupported and the existing cast-to-F16 fallback is correct. Gated
    /// on this flag because it has not been end-to-end validated on Ada hardware in CI; flip on after benchmarking
    /// against the F16 fallback.</remarks>
    public bool EnableNativeFp8Gemm { get; set; }

    /// <summary>Use the division-free head-major rope kernel (<c>HARTSY_ROPE_V2=0</c> to fall back). Bit-identical.</summary>
    public bool EnableRopeHeadMajorV2 { get; set; }

    /// <summary>Quantize fp8 activations with the checkpoint's <c>.input_scale</c> instead of a per-call absmax (<c>HARTSY_FP8_STATIC_INPUT_SCALE=0</c> to force the dynamic path). Changes numerics — see the doc comment.</summary>
    public bool EnableStaticFp8InputScale { get; set; }

    /// <summary>Let a modulate producer write e4m3 for its consuming fp8 Linear (<c>HARTSY_MODULATE_EMIT_FP8=0</c> to disable).</summary>
    public bool EnableModulateEmitFp8 { get; set; }

    /// <summary>Device-resident copy of a weight's static activation scale, allocated once per weight.</summary>
    private readonly Dictionary<Tensor, (ulong Pointer, float Scale)> _fp8InputScaleDev = new();
    private readonly object _fp8InputScaleLock = new();
    private long _fp8InputScaleAllocationCount;
    private long _fp8InputScaleFreeCount;

    /// <summary>Returns the device pointer holding <paramref name="weight"/>'s static input scale, or 0 if it has none.</summary>
    private unsafe ulong EnsureFp8InputScaleDev(Tensor weight)
    {
        float scale = weight.Fp8InputScaleFactor;
        if (scale <= 0.0f || !float.IsFinite(scale)) return 0;
        lock (_fp8InputScaleLock)
        {
            if (_fp8InputScaleDev.TryGetValue(weight, out (ulong Pointer, float Scale) cached))
            {
                if (cached.Scale != scale)
                {
                    // Scale metadata is mutable. Drain prior readers, then update the existing scalar in place so
                    // mutation neither serves stale numerics nor churns persistent device allocations.
                    _stream.Synchronize();
                    CudaMemory.CopyHostToDevice(cached.Pointer, &scale, sizeof(float));
                    _fp8InputScaleDev[weight] = (cached.Pointer, scale);
                }
                return cached.Pointer;
            }
            // Persistent, not pool-allocated: it outlives the call and must survive stream-ordered frees. Serialize
            // first creation so concurrent Linears cannot leak duplicate four-byte allocations for one weight.
            ulong dev = CudaMemory.AllocatePersistent(sizeof(float));
            try
            {
                CudaMemory.CopyHostToDevice(dev, &scale, sizeof(float));
                _fp8InputScaleDev[weight] = (dev, scale);
                Interlocked.Increment(ref _fp8InputScaleAllocationCount);
                return dev;
            }
            catch
            {
                CudaMemory.Free(dev);
                throw;
            }
        }
    }

    private void FreeFp8InputScale(Tensor weight)
    {
        lock (_fp8InputScaleLock)
        {
            if (!_fp8InputScaleDev.TryGetValue(weight, out (ulong Pointer, float Scale) entry)) return;
            CudaMemory.Free(entry.Pointer);
            _fp8InputScaleDev.Remove(weight);
            Interlocked.Increment(ref _fp8InputScaleFreeCount);
        }
    }

    private void FreeAllFp8InputScales()
    {
        // These persistent pointers are read by kernels/GEMMs on the compute stream. Never release them merely
        // because a broader cache sweep was attempted; independently prove the consuming stream is drained here.
        if (_stream is not null) _stream.Synchronize();
        List<Exception>? failures = null;
        lock (_fp8InputScaleLock)
        {
            foreach ((Tensor weight, (ulong Pointer, float Scale) entry) in _fp8InputScaleDev.ToArray())
            {
                try
                {
                    CudaMemory.Free(entry.Pointer);
                    _fp8InputScaleDev.Remove(weight);
                    Interlocked.Increment(ref _fp8InputScaleFreeCount);
                }
                catch (Exception error)
                {
                    // Keep failed entries registered so a later explicit sweep can retry them.
                    (failures ??= []).Add(error);
                }
            }
        }
        if (failures is not null) throw new AggregateException("One or more FP8 input-scale buffers failed to release.", failures);
    }

    internal int Fp8InputScaleCacheCount
    {
        get { lock (_fp8InputScaleLock) return _fp8InputScaleDev.Count; }
    }

    internal (int Count, long Allocations, long Frees) Fp8InputScaleDiagnostics =>
        (Fp8InputScaleCacheCount, Interlocked.Read(ref _fp8InputScaleAllocationCount), Interlocked.Read(ref _fp8InputScaleFreeCount));

    internal ulong EnsureFp8InputScaleForTest(Tensor weight)
    {
        EnterOp();
        return EnsureFp8InputScaleDev(weight);
    }

    /// <summary>Low-VRAM lever: when <c>true</c> (default) the per-weight fp8/quant→F16 cast is cached resident, not recomputed each GEMM.</summary>
    /// <remarks>Cached is fastest but keeps BOTH the fp8 weight (~1 byte/param) and its F16 cast (~2 bytes/param) on
    /// the device, ≈3× the fp8 footprint. Set <c>false</c> to force the transient path (one weight cast at a time,
    /// freed per GEMM): ~3× less weight VRAM at the cost of re-casting each step. Needed to fit large fp8 DiTs
    /// (e.g. AuraFlow 6.8B) on a single 24 GB card.</remarks>
    public bool CacheWeightCasts { get; set; } = true;

    /// <summary>Lazily-initialized FP8 GEMM executor, exposed for diagnostic/benchmarking callers.</summary>
    /// <remarks>Production GEMM dispatch goes through <see cref="MatMul"/> / <see cref="Linear"/>.</remarks>
    public Fp8GemmExecutor Fp8Executor
    {
        get
        {
            EnsureActiveContextForLazyNativeResource();
            lock (_nativeExecutorLock)
            {
                EnsureActiveContextForLazyNativeResource();
                if (_fp8Executor is null)
                {
                    _fp8Executor = new Fp8GemmExecutor(_context.ComputeCapabilityMajor, _context.ComputeCapabilityMinor);
                    GC.SuppressFinalize(_fp8Executor);
                }
                return _fp8Executor;
            }
        }
    }

    /// <summary>Fuses a Linear bias add into the cuBLASLt GEMM epilogue. Enabled by default; set <c>HARTSY_EPILOGUE_FUSION=0</c> to disable it.</summary>
    /// <remarks>Works on every targeted SM, including the RTX 3060. A supported biased Linear runs as one
    /// <c>cublasLtMatmul</c>; unavailable libraries, unsupported shapes, and no-algorithm results fall back to
    /// <c>cublasGemmEx</c> plus the existing <c>BiasAdd</c> path with the same resolved precision policy.</remarks>
    public bool EnableEpilogueFusion { get; set; }

    /// <summary>int8-activation dp4a decode GEMV for Q4_K/Q6_K/Q8_0 weights (default ON, kill-switch <c>HARTSY_DP4A_ON=0</c>).</summary>
    /// <remarks>Lossy within the Q8_1 rounding bound (see Dp4aGemvGroundTruthTests); measured 2026-07-22:
    /// Llama-3.2-1B 159→195 tok/s, Qwen3-4B 71→90 tok/s (RTX 3060, graph-on).</remarks>
    public bool EnableDp4aGemv { get; set; }

    /// <summary>Opt-in W8A8 INT8 tensor-core (IMMA) GEMM path (<c>HARTSY_W8A8=1</c>, INFERENCE_ACCEL_GRIND §H5).</summary>
    /// <remarks>Large-M Linears with 16-bit-float weights run as per-channel-int8 weight (host-quantized once,
    /// cached) × per-row dynamic-int8 activation on <see cref="Int8Gemm"/>, dequantized by the w8a8.ptx epilogue.
    /// The Ampere lever — SM 8.6 has no fp8 MMA, and IMMA measured 3.2–3.7× over the F16 GEMM (chain 2.57× at
    /// relL2 5.5e-3, `W8A8ImmaGemmTests`, 3060). Lossy (int8 rounding); ships opt-in until per-model quality gates
    /// land.</remarks>
    public bool EnableW8A8 { get; set; }

    /// <summary>Lazily-initialized INT8 IMMA GEMM executor (see <see cref="EnableW8A8"/>).</summary>
    public Int8GemmExecutor Int8Gemm
    {
        get
        {
            EnsureActiveContextForLazyNativeResource();
            lock (_nativeExecutorLock)
            {
                EnsureActiveContextForLazyNativeResource();
                if (_int8Executor is null)
                {
                    _int8Executor = new Int8GemmExecutor();
                    GC.SuppressFinalize(_int8Executor);
                }
                return _int8Executor;
            }
        }
    }

    private Int8GemmExecutor? _int8Executor;

    // Per-weight W8A8 device cache: one persistent buffer holding [int8 N·K | pad to 256 | F32 wScale[N]].
    // Keyed by the weight Tensor object (same convention as the GpuTransferHelper weight cache); freed by
    // FreeW8A8Cache from FreeAllDeviceMemory / FreePreloadedWeights / Dispose.
    private readonly Dictionary<Core.Tensors.Tensor, ulong> _w8a8WeightCache = new();

    // SmoothQuant per-input-channel smoothing scale s[K] (W8A8_HANDOFF.md item 1, offline-gate-confirmed
    // 2026-07-24: relL2 drops ~40% on real Kandinsky5 layers at alpha~0.7-0.8). Two views of the same s,
    // set together by SetW8A8SmoothingScale: a device 1/s[K] buffer consumed every activation-quant call
    // (LaunchW8A8QuantRowwise's invScale arg), and a host s[K] copy folded into QuantizeWeightForW8A8's
    // per-output-channel weight quant (runs once, at weight-quant time, host-side). Calibration — deciding
    // WHAT s should be — is deliberately NOT here; this is storage + application only (advisor-directed:
    // don't build a production calibration API before SSIM proves the mechanism earns a permanent place).
    private readonly Dictionary<Core.Tensors.Tensor, ulong> _w8a8SmoothInvScaleDevice = new();
    private readonly Dictionary<Core.Tensors.Tensor, float[]> _w8a8SmoothScaleHost = new();

    // Per-weight F32 wScale[N] for a RESIDENT int8 weight (ComfyUI int8_tensorwise). Unlike the W8A8 cache above
    // there is no quantized-weight buffer to hold: the checkpoint's own int8 bytes are the weight, uploaded by the
    // ordinary GpuTransferHelper path. Only the scale needs a device home, and a per-tensor scale is expanded to
    // N entries here so the shared dequant epilogue never needs a broadcast variant. Freed by FreeW8A8Cache.
    private readonly Dictionary<Core.Tensors.Tensor, ulong> _int8RowScaleDevice = new();
    private readonly object _int8RowScaleLock = new();

    // Per-weight companions for a RESIDENT nvfp4 weight (ComfyUI `nvfp4`): the swizzled E4M3 block-scale bytes on the
    // device plus the two host scalars the dequant kernel folds in. One sixteenth of the weight's size, and unlike a
    // dtype cast it is part of the resident representation — keeping it is what makes the weight usable at all, so it
    // is not subject to the cast budget gate. Freed by FreeW8A8Cache.
    private readonly Dictionary<Core.Tensors.Tensor, Nvfp4WeightScales> _nvfp4ScaleDevice = new();
    private readonly object _nvfp4ScaleLock = new();

    /// <summary>Sets (or replaces) the SmoothQuant per-input-channel scale s[K] for <paramref name="weight"/>: X_hat = X/s, W_hat = W*s.</summary>
    /// <remarks>Product-preserving pre-quantization — migrates activation outlier difficulty into the weight (see
    /// src/HartsyInference.Cuda/Kernels/dequant/w8a8.cu's invScale param). Must be called BEFORE the weight's first W8A8 use to take
    /// effect on the initial quantization; calling it after evicts the already-cached quantized weight so the NEXT
    /// use re-quantizes smoothed (safe, just a re-pay of the one-time quant cost). <paramref name="s"/> length must
    /// equal the weight's K (in-dim).</remarks>
    public unsafe void SetW8A8SmoothingScale(Core.Tensors.Tensor weight, ReadOnlySpan<float> s)
    {
        EnterOp();
        int k = (int)weight.Shape[1];
        if (s.Length != k)
            throw new ArgumentException($"SmoothQuant scale length {s.Length} != weight K={k}.", nameof(s));
        float[] invScale = new float[k];
        for (int i = 0; i < k; i++)
        {
            if (!(s[i] > 0f) || !float.IsFinite(s[i]))
                throw new ArgumentOutOfRangeException(nameof(s), $"SmoothQuant scale s[{i}] must be positive and finite; got {s[i]}.");
            invScale[i] = 1f / s[i];
        }
        float[] hostScale = s.ToArray();
        _w8a8SmoothInvScaleDevice.EnsureCapacity(_w8a8SmoothInvScaleDevice.Count + 1);
        _w8a8SmoothScaleHost.EnsureCapacity(_w8a8SmoothScaleHost.Count + 1);
        ulong dev = GpuTransferHelper.AllocateDevice((nuint)(k * sizeof(float)));
        bool published = false;
        try
        {
            fixed (float* p = invScale)
                CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)p, (nuint)(k * sizeof(float)), _stream.Handle).ThrowOnError();
            _stream.Synchronize();

            // Invalidate the quantized weight while the old smoothing pair is still authoritative. If its
            // free fails, publication aborts and the cache remains numerically compatible with the old pair.
            if (_w8a8WeightCache.TryGetValue(weight, out ulong cachedQuant))
            {
                GpuTransferHelper.FreeDevice(cachedQuant);
                _w8a8WeightCache.Remove(weight);
            }

            if (_w8a8SmoothInvScaleDevice.TryGetValue(weight, out ulong old))
            {
                GpuTransferHelper.FreeDevice(old);
                _w8a8SmoothInvScaleDevice.Remove(weight);
                _w8a8SmoothScaleHost.Remove(weight);
            }
            _w8a8SmoothInvScaleDevice[weight] = dev;
            _w8a8SmoothScaleHost[weight] = hostScale;
            published = true;
        }
        catch (Exception primary) when (!published)
        {
            // An unexpected managed publication failure must not leave a dictionary pointing at the
            // unpublished device allocation that rollback is about to free.
            if (_w8a8SmoothInvScaleDevice.TryGetValue(weight, out ulong tracked) && tracked == dev)
                _w8a8SmoothInvScaleDevice.Remove(weight);
            if (_w8a8SmoothScaleHost.TryGetValue(weight, out float[]? trackedHost)
                && ReferenceEquals(trackedHost, hostScale))
                _w8a8SmoothScaleHost.Remove(weight);
            try { GpuTransferHelper.FreeDevice(dev); }
            catch (Exception cleanup)
            {
                throw new AggregateException("SmoothQuant scale publication and rollback both failed.", primary, cleanup);
            }
            throw;
        }
    }

    /// <summary>Frees every SmoothQuant device scale buffer (mirrors FreeW8A8Cache's scope/callers).</summary>
    private void FreeW8A8SmoothScaleCache(GpuTransferHelper.State? explicitState = null)
    {
        List<Exception>? failures = null;
        foreach ((Tensor weight, ulong ptr) in _w8a8SmoothInvScaleDevice.ToArray())
        {
            try
            {
                if (explicitState is null) GpuTransferHelper.FreeDevice(ptr);
                else GpuTransferHelper.FreeDevice(explicitState, ptr);
                _w8a8SmoothInvScaleDevice.Remove(weight);
                _w8a8SmoothScaleHost.Remove(weight);
            }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        if (_w8a8SmoothInvScaleDevice.Count == 0) _w8a8SmoothScaleHost.Clear();
        if (failures is not null) throw new AggregateException("One or more W8A8 SmoothQuant scales failed to release.", failures);
    }

    /// <summary>Rows a resident-int8 GEMM chunk covers, bounded by what is actually free on the device.</summary>
    /// <remarks><para>The int32 accumulator is 4 bytes per output element and the ConvRot scratch another
    /// <c>k · activation bytes</c> per row, so an unchunked video-length GEMM against a wide projection asks for
    /// gigabytes of transient device memory.</para>
    /// <para>A fixed budget is not enough: the whole point of a resident int8 DiT is that the weights fill the card,
    /// which leaves the transients competing with the activations for what little is left. A 256 MB fixed budget
    /// OOM'd MiniMax-H3's <c>mlp.fc1</c> (n=28672) with a 21 GB DiT resident on a 24 GB card. Taking an eighth of
    /// free VRAM keeps the chunk large where there is room and shrinks it rather than failing where there is not;
    /// <c>cuMemGetInfo</c> is a cheap driver query with no stream sync, which is why the H3 transformer already
    /// polls it per forward.</para></remarks>
    /// <summary>Output-column tile for the resident int8 GEMM, in units of N. Tiling over N (not over M, which the row chunk above already showed is monotonically worse — it shrinks the GEMM's m) keeps the int32 accumulator small enough to be consumed by the dequant epilogue while still in L2, instead of streamed to HBM and read straight back. 0 or >= n disables tiling. Override with HARTSY_INT8_N_CHUNK.</summary>
    private static int Int8ResidentColChunk(int n)
        => EngineKnobs.Int8NChunk.Value is int env ? (env <= 0 ? int.MaxValue : env) : DefaultInt8ColChunk;

    // OFF. MEASURED 2026-08-13 at LTX-2.5 768x512x97f: no tiling 1457.2 ms/step, 2048 -> 1474.0, 1024 -> 1596.3
    // — monotonically worse, the same shape of result as the row chunk's own L2 experiment. Making the int32
    // accumulator L2-resident does not pay for the extra launches and the smaller-n GEMM. The accumulator round
    // trip is only recoverable by an epilogue fused INTO the GEMM, which cuBLASLt refuses (see
    // Int8GemmEpilogueProbeTests) and which our own mma kernel is not yet fast enough to justify.
    private const int DefaultInt8ColChunk = int.MaxValue;

    /// <summary>HARTSY_INT8_ROW_BUDGET_MB — pins <see cref="Int8ResidentRowChunk"/>'s byte budget instead of deriving it from free VRAM. 0 keeps the derived behaviour.</summary>
    private static readonly long RowChunkBudgetOverrideBytes = EngineKnobs.Int8RowBudgetMb.Value << 20;

    private int Int8ResidentRowChunk(int m, int n, int k, int activationBytes)
    {
        // 256 MB is the measured optimum, not a guess: shrinking the chunk so the int32 IMMA accumulator becomes
        // L2-resident (64/32/16 MB) is monotonically SLOWER on LTX-2.5 at 768x512x97f — 1768/1854/1920/2070 ms per
        // step — because the extra chunks cost more in launches and smaller-m GEMM efficiency than the accumulator
        // round trip costs in HBM traffic. Do not re-chase L2 sizing here.
        const long CeilingBytes = 256L << 20;
        const long FloorBytes = 8L << 20;
        // Polled per call, and that is fine: cuMemGetInfo measures 5.2 µs (Int8ResidentHostCostTests), ~14 ms/step
        // across ~2,700 resident-int8 Linears — but caching it buys nothing, because this path is GPU-bound at
        // 99-100% SM and host queuing time never reaches the wall clock. See Int8GemmExecutor's remarks for the
        // interleaved 4-rep campaign that measured the same null on a 45k-call-per-step version of this idea.
        // HARTSY_INT8_ROW_BUDGET_MB pins the budget. Deriving it from free VRAM makes the chunk count — and with it
        // the launch count and the GEMM's M — a function of whatever else is transiently allocated, so any A/B that
        // changes device-memory pressure silently changes this too and stops being a controlled comparison.
        long budget = RowChunkBudgetOverrideBytes > 0 ? RowChunkBudgetOverrideBytes
            : Math.Clamp(CudaMemory.GetMemInfo().FreeBytes / 8, FloorBytes, CeilingBytes);
        long perRowBytes = (long)n * sizeof(int) + k + (long)k * activationBytes;
        return (int)Math.Min(m, Math.Max(1, budget / Math.Max(1, perRowBytes)));
    }

    /// <summary>Rounds a row count up to what cuBLASLt's int8 TN kernels want, matching comfy-kitchen's own padding.</summary>
    private static int PadInt8Rows(int rows) => (Math.Max(rows, 32) + 31) & ~31;

    /// <summary>Whether the IMMA chain can serve this resident int8 weight; false routes it through the dequant fallback.</summary>
    private bool CanRunResidentInt8(Tensor output, Tensor input, Tensor weight, QuantWeightInfo info,
        int weightRowOffset, int weightRowCount)
    {
        // A future format that also stores I8 with a per-row scale must not silently inherit this chain's
        // int8_tensorwise-specific dequant arithmetic.
        if (info.Format != "int8_tensorwise") return false;
        if (info.FullPrecisionMatMul || weightRowOffset != 0 || weightRowCount >= 0) return false;
        if (_kernels is null || !_kernels.HasW8A8Kernels || !Int8Gemm.IsSupported) return false;
        if (info.ConvRotGroupSize > 0 && !_kernels.HasConvRotKernels) return false;
        // Above ~16384 the rotation kernel's dynamic shared memory exceeds the 64 KB opt-out ceiling and the launch
        // fails opaquely; refuse well short of it so the layer falls back to the dequant path instead.
        if (info.ConvRotGroupSize > 4096) return false;
        if (weight.Shape.Rank != 2) return false;

        int n = (int)weight.Shape[0];
        int k = (int)weight.Shape[1];
        // K and N multiples of 4 are the cuBLASLt int8 TN lda/ldc requirement (see Int8GemmExecutor).
        if (k % 4 != 0 || n % 4 != 0) return false;
        if (info.ConvRotGroupSize > 0
            && (!Int8ConvRotCodec.IsValidGroupSize(info.ConvRotGroupSize) || k % info.ConvRotGroupSize != 0))
        {
            return false;
        }
        if (info.RowScale!.DType != DType.F32) return false;
        if (info.RowScale.ElementCount != n && info.RowScale.ElementCount != 1) return false;
        return (input.DType == DType.F16 || input.DType == DType.F32)
            && (output.DType == DType.F16 || output.DType == DType.F32);
    }

    /// <summary>Frees every resident-int8 device wScale buffer (mirrors FreeW8A8Cache's scope/callers).</summary>
    private void FreeInt8RowScaleCache(GpuTransferHelper.State? explicitState = null)
    {
        List<Exception>? failures = null;
        lock (_int8RowScaleLock)
        {
            foreach ((Tensor weight, ulong ptr) in _int8RowScaleDevice.ToArray())
            {
                try
                {
                    if (explicitState is null) GpuTransferHelper.FreeDevice(ptr);
                    else GpuTransferHelper.FreeDevice(explicitState, ptr);
                    _int8RowScaleDevice.Remove(weight);
                }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
        }
        if (failures is not null) throw new AggregateException("One or more resident-int8 weight scales failed to release.", failures);
    }

    /// <summary>Uploads (once) the F32 <c>wScale[n]</c> the dequant epilogue indexes per output column.</summary>
    /// <remarks>A checkpoint may ship the scale as <c>[N, 1]</c>, <c>[N]</c>, or a single per-tensor value; all three
    /// land here as a dense N-entry buffer. Persistent, not pool-allocated: it outlives the call and is read by the
    /// epilogue on the compute stream.</remarks>
    private unsafe ulong EnsureInt8RowScaleDev(Tensor weight, int n)
    {
        lock (_int8RowScaleLock)
        {
            if (_int8RowScaleDevice.TryGetValue(weight, out ulong cached)) return cached;

            Tensor rowScale = weight.QuantInfo!.RowScale!;
            float[] scales = new float[n];
            ReadOnlySpan<float> source = rowScale.AsReadOnlySpan<float>();
            if (source.Length == 1) scales.AsSpan().Fill(source[0]);
            else source[..n].CopyTo(scales);

            ulong dev = GpuTransferHelper.AllocateDevice((nuint)(n * sizeof(float)));
            try
            {
                fixed (float* p = scales)
                    CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)p, (nuint)(n * sizeof(float)), _stream.Handle).ThrowOnError();
                _stream.Synchronize();
                _int8RowScaleDevice[weight] = dev;
                return dev;
            }
            catch
            {
                GpuTransferHelper.FreeDevice(dev);
                throw;
            }
        }
    }

    /// <summary>Whether the transient-dequant path can serve this resident nvfp4 weight; false routes it through the host dequant fallback.</summary>
    /// <remarks>Deliberately does NOT consult <see cref="QuantWeightInfo.FullPrecisionMatMul"/>. That flag means "this
    /// layer must run a real GEMM rather than a quantized one", which is exactly what this path does — the weight is
    /// unpacked to F16/BF16 and handed to cuBLAS. Every nvfp4 layer in the Qwen3-VL AWQ encoder carries the flag, so
    /// honouring it the way the int8 IMMA gate does would disable the resident path wholesale.</remarks>
    private bool CanRunResidentNvfp4(Tensor weight, QuantWeightInfo info, int weightRowOffset, int weightRowCount)
    {
        if (info.Format != "nvfp4") return false;
        if (weightRowOffset != 0 || weightRowCount >= 0) return false;
        if (_kernels is null || !_kernels.HasNvfp4Kernels) return false;
        if (weight.Shape.Rank != 2) return false;

        Tensor blockScale = info.BlockScale!;
        if (blockScale.DType != DType.F8E4M3 || blockScale.Shape.Rank != 2) return false;
        if (info.GlobalScale!.DType != DType.F32 || info.GlobalScale.ElementCount != 1) return false;

        long n = weight.Shape[0];
        long k = weight.Shape[1];
        if (k % Nvfp4ResidentCodec.GroupSize != 0) return false;
        // Rows are padded up to 128 and block columns up to 4 by the blocked layout, so the stored scale tensor is
        // never smaller than the logical one; smaller means the companion does not belong to this weight.
        return blockScale.Shape[0] >= n && blockScale.Shape[1] >= k / Nvfp4ResidentCodec.GroupSize
            && blockScale.Shape[1] % 4 == 0;
    }

    /// <summary>Frees every resident-nvfp4 device block-scale buffer (mirrors FreeW8A8Cache's scope/callers).</summary>
    private void FreeNvfp4ScaleCache(GpuTransferHelper.State? explicitState = null)
    {
        List<Exception>? failures = null;
        lock (_nvfp4ScaleLock)
        {
            foreach ((Tensor weight, Nvfp4WeightScales scales) in _nvfp4ScaleDevice.ToArray())
            {
                try
                {
                    if (explicitState is null) GpuTransferHelper.FreeDevice(scales.BlockScaleDevice);
                    else GpuTransferHelper.FreeDevice(explicitState, scales.BlockScaleDevice);
                    _nvfp4ScaleDevice.Remove(weight);
                }
                catch (Exception error) { (failures ??= []).Add(error); }
            }
        }
        if (failures is not null) throw new AggregateException("One or more resident-nvfp4 weight scales failed to release.", failures);
    }

    /// <summary>Uploads (once) the swizzled E4M3 block scales the dequant kernel indexes, and reads the two host scalars.</summary>
    /// <remarks>Persistent, not pool-allocated: it outlives the call and is read on the compute stream every GEMM.
    /// The two scalars are captured here rather than at launch time because reading them means touching
    /// <c>DataPointer</c> on the host, which must happen before the call reaches the transfer caches.</remarks>
    private unsafe Nvfp4WeightScales EnsureNvfp4Scales(Tensor weight)
    {
        lock (_nvfp4ScaleLock)
        {
            if (_nvfp4ScaleDevice.TryGetValue(weight, out Nvfp4WeightScales cached)) return cached;

            Tensor blockScale = weight.QuantInfo!.BlockScale!;
            nuint byteSize = (nuint)blockScale.DType.ComputeByteCount(blockScale.ElementCount);
            ulong dev = GpuTransferHelper.AllocateDevice(byteSize);
            try
            {
                CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)blockScale.DataPointer, byteSize, _stream.Handle).ThrowOnError();
                _stream.Synchronize();
                Nvfp4WeightScales scales = new Nvfp4WeightScales(dev, blockScale.Fp8ScaleFactor,
                    ((float*)weight.QuantInfo.GlobalScale!.DataPointer)[0], (int)blockScale.Shape[1]);
                _nvfp4ScaleDevice[weight] = scales;
                return scales;
            }
            catch
            {
                GpuTransferHelper.FreeDevice(dev);
                throw;
            }
        }
    }

    /// <summary>Weight-side dtype materialization, substituting the nvfp4 block-scaled unpack for the plain cast.</summary>
    /// <remarks>Two allocator flavours exist at the call site — the cached cast owns its buffer through
    /// <see cref="GpuTransferHelper"/>, the transient one through <see cref="CudaMemory"/> — so this pair mirrors
    /// <see cref="CastOnGpu"/> and <see cref="CastIfNeeded"/> rather than replacing either.</remarks>
    private void MaterializeWeight(ulong destination, ulong source, Tensor weight, DType gemmDtype, in Nvfp4WeightScales nvfp4)
    {
        if (nvfp4.BlockScaleDevice == 0)
        {
            CastOnGpu(destination, source, weight.DType, gemmDtype, (int)weight.ElementCount);
            return;
        }
        _kernels!.LaunchNvfp4Dequant(destination, source, nvfp4.BlockScaleDevice,
            (int)weight.Shape[0], (int)(weight.Shape[1] / 2), nvfp4.PaddedCols,
            nvfp4.ScaleFactor, nvfp4.GlobalScale, _stream.Handle, outBf16: gemmDtype == DType.BF16);
    }

    /// <summary>Transient-buffer form of <see cref="MaterializeWeight"/>; <paramref name="castOut"/> is the caller's to free.</summary>
    private unsafe ulong MaterializeWeightIfNeeded(ulong source, Tensor weight, DType gemmDtype, out ulong castOut,
        in Nvfp4WeightScales nvfp4)
    {
        if (nvfp4.BlockScaleDevice == 0)
            return CastIfNeeded(source, weight.DType, gemmDtype, (int)weight.ElementCount, out castOut);
        castOut = CudaMemory.Allocate((nuint)(weight.ElementCount * gemmDtype.SizeInBytes));
        MaterializeWeight(castOut, source, weight, gemmDtype, nvfp4);
        return castOut;
    }

    /// <summary>Test-only hook: LinearImpl passes F32 snapshots of pre-quant input/weight of the first W8A8-eligible call, then clears.</summary>
    /// <remarks>Parallel to <c>CausalConv3d.DisableBatchedPath</c>. Lets an offline test capture a real Linear's
    /// real operands off a live forward pass (e.g. an activation-vs-weight quantization-error ablation) without
    /// adding a permanent capture path. Snapshots go through <see cref="SnapshotToF32ForTest"/> — a cache-hit peek +
    /// cache-aware free, NEVER <c>Tensor.DataPointer</c> — because a mid-forward DataPointer read on a
    /// device-cached tensor trips the lazy-sync consume and races the transfer caches (see the w8a8 eligibility
    /// comment above in LinearImpl). No effect on the hot path unless a test sets it. Args: input[M*K], M, K,
    /// weight[N*K], N, weight (the Tensor identity — safe to hold/pass around since only its float[] snapshot is
    /// read here, never its DataPointer; needed so a calibration harness can later correlate accumulated stats back
    /// to a specific weight via <see cref="SetW8A8SmoothingScale"/>).</remarks>
    public static Action<float[], int, int, float[], int, Tensor>? CaptureW8A8Operands;

    /// <summary>Test-only: D2H-snapshots a tensor to a host F32 array via a cache-hit peek, backing <see cref="CaptureW8A8Operands"/> only.</summary>
    /// <remarks>Uses <see cref="GpuTransferHelper.CopyToDevice"/> (non-destructive on a hit) + blocking
    /// <c>cuMemcpyDtoH</c> + cache-aware <see cref="GpuTransferHelper.FreeDevice"/> — never touches
    /// <c>Tensor.DataPointer</c>, so it can't trip the lazy-sync eviction race.</remarks>
    private unsafe float[] SnapshotToF32ForTest(Tensor t, long count)
    {
        ulong pDev = GpuTransferHelper.CopyToDevice(t);
        try
        {
            nuint byteSize = GpuTransferHelper.ByteSize(t);
            byte[] host = new byte[byteSize];
            fixed (byte* dst = host)
                CudaDriverApi.cuMemcpyDtoH((nint)dst, pDev, byteSize).ThrowOnError();
            float[] result = new float[count];
            DType dt = t.DType;
            fixed (byte* src = host)
                for (long i = 0; i < count; i++) result[i] = W8A8Read(src, dt, i);
            return result;
        }
        finally { GpuTransferHelper.FreeDevice(pDev); }
    }

    // Persistent, stream-serialized scratch buffers (dp4a activation quantization; two-stage argmax
    // partials). Reusing one resident buffer instead of per-call AllocateDevice/FreeDevice removes ~6
    // allocation/free NODES per Linear from every captured decode graph (~750 nodes on a 24-layer model)
    // and the matching pool churn from eager decode. Reuse is safe on the single compute stream: the next
    // op's writes are stream-ordered behind the previous op's reads. NEVER grown inside stream capture —
    // a capture-time cuMemAllocAsync becomes a GRAPH-OWNED allocation whose lifetime dies with the graph,
    // which must not back a pointer this class caches (callers fall back to transient per-call buffers in
    // that case; in practice the pre-capture warm-up forward pass sizes both buffers to their maximum
    // before any capture begins, so the fallback never fires on the graph path).
    private ulong _dp4aScratch;
    private nuint _dp4aScratchBytes;
    private ulong _argmaxScratch;
    private ulong _ssmDeltaScratch;
    private nuint _ssmDeltaScratchBytes;

    private bool StreamIsCapturing()
    {
        CudaDriverApi.cuStreamIsCapturing(_stream.Handle, out int status).ThrowOnError();
        return status != 0;   // CU_STREAM_CAPTURE_STATUS_NONE = 0
    }

    /// <summary>Returns the persistent dp4a scratch grown to at least <paramref name="bytes"/>, or 0 mid-capture (use transient buffers).</summary>
    private ulong EnsureDp4aScratch(nuint bytes)
    {
        if (bytes <= _dp4aScratchBytes) return _dp4aScratch;
        if (StreamIsCapturing()) return 0;
        if (_dp4aScratch != 0) GpuTransferHelper.FreeDevice(_dp4aScratch);
        _dp4aScratch = GpuTransferHelper.AllocateDevice(bytes);
        _dp4aScratchBytes = bytes;
        return _dp4aScratch;
    }

    /// <summary>Opt-in: mixed bf16/f32 GEMMs at F32 precision (cast bf16 weight UP, not truncate activation). Default <c>false</c>.</summary>
    /// <remarks>Default preserves the bf16 Tensor-Core fast path. Turn ON for precision-sensitive models whose
    /// bf16 weights stay resident to fit VRAM but whose activations must keep full F32 mantissa across many
    /// layers/steps (e.g. ACE-Step's 3.5B DiT on a 12 GB 3060).</remarks>
    public bool HighPrecisionGemm { get; set; }

    /// <summary>Opt-in: <see cref="Conv2D"/> pads the width axis of every convolution with wrapped edge pixels instead of zeros. Default <c>false</c>. Set for the duration of one generation only — this changes every conv's output on the hot path, so it must never leak into a request that didn't ask for it (Tier 3.6).</summary>
    public bool SeamlessTilingX { get; set; }

    /// <summary>Same as <see cref="SeamlessTilingX"/> for the height axis.</summary>
    public bool SeamlessTilingY { get; set; }

    /// <summary>Route the fp8 GEMM activation cast through F16 (10-bit) instead of BF16 (7-bit).</summary>
    /// <remarks>Safe for GELU-FFN models (Wan); needed for deep fp8 DiTs where BF16's coarser mantissa compounds a
    /// per-step bias into divergence.</remarks>
    public bool EnableFp8F16Gemm { get; set; }

    /// <summary>Compute the fp8 GEMM in F32 (max precision) — decisive test for F16/BF16 compute-error compounding.</summary>
    public bool EnableFp8F32Gemm { get; set; }

    /// <summary>Lazily-initialized general-precision cuBLASLt GEMM executor used by the epilogue-fusion path.</summary>
    public LtGemmExecutor LtGemm
    {
        get
        {
            EnsureActiveContextForLazyNativeResource();
            lock (_nativeExecutorLock)
            {
                EnsureActiveContextForLazyNativeResource();
                if (_ltGemmExecutor is null)
                {
                    _ltGemmExecutor = new LtGemmExecutor(_context, _stream.Handle, backendAdopted: true);
                    GC.SuppressFinalize(_ltGemmExecutor);
                }
                return _ltGemmExecutor;
            }
        }
    }

    /// <summary>Opt-in flag for the hand-written tensor-core HGEMM in the F16 Linear path. Defaults to <c>false</c>.</summary>
    /// <remarks>The kernel is validated against cuBLAS within an F16 tolerance (<c>TensorCoreGemmTests</c> asserts
    /// avg_err &lt; 0.05, NOT bit-exactness) and is the unoptimized
    /// one-warp-per-tile baseline, so it is opt-in pending a perf comparison against cuBLAS on the target GPU. Only
    /// dispatches when operands and output are F16 and dimensions are aligned (M%16, N%8, K%16 == 0); otherwise
    /// falls through to cuBLAS.</remarks>
    public bool EnableTensorCoreGemm { get; set; }

    /// <summary>Fused BF16/F16 decode GEMV for small-m (≤8) F32-activation matmuls; replaces cuBLAS GemmEx (slow at m=1). On by default.</summary>
    /// <remarks>Faster and at least as accurate as the cuBLAS BF16 path (activations stay F32). Set
    /// <c>HARTSY_BF16_GEMV=0</c> to fall back to cuBLAS for A/B.</remarks>
    public bool EnableBf16Gemv { get; set; } = EngineKnobs.Bf16Gemv.Value;

    /// <summary>Lazily-initialized tensor-core HGEMM launcher. Requires PTX directory and SM 8.0+.</summary>
    public TensorCoreGemm TensorCoreGemm
    {
        get
        {
            EnsureActiveContextForLazyNativeResource();
            lock (_nativeExecutorLock)
            {
                EnsureActiveContextForLazyNativeResource();
                if (_tensorCoreGemm is null)
                {
                    _tensorCoreGemm = new TensorCoreGemm(
                        _ptxDir ?? throw new InvalidOperationException("TensorCoreGemm requires a PTX directory; construct CudaBackend with ptxDir."),
                        _context.ComputeCapabilityMajor);
                    GC.SuppressFinalize(_tensorCoreGemm);
                }
                return _tensorCoreGemm;
            }
        }
    }

    private void EnsureActiveContextForLazyNativeResource()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _lifecycleState) != LifecycleActive, this);
        _context.EnsureCurrent();
    }

    /// <summary>Native-F16 SageAttention policy; enabled by default and disabled with <c>HARTSY_SAGE_ATTN=0</c>.</summary>
    private static bool UseSageAttn => EngineKnobs.SageAttn.Value;

    /// <summary>The explicit <c>=1</c> sense of the Sage switch, distinct from the default-ON <see cref="UseSageAttn"/>.</summary>
    private static bool SageExplicitlyEnabled => EngineKnobs.SageAttnExplicit.Value;

    /// <summary>Query-tiled LTX-2.5 na3d kernel; <c>HARTSY_LTX25_NA3D_TILED=0</c> falls back to the per-query one. Changes numerics: the tiled path is an online softmax over the tile's union window, not one dense pass.</summary>
    private static bool UseLtx25Na3dTiled => EngineKnobs.Ltx25Na3dTiled.Value;

    /// <summary>True only when the caller explicitly accepts Sage's F32-to-F16 V-storage narrowing.</summary>
    /// <remarks>The current Sage prologue materializes V as F16, so an F32 value outside the finite F16 range
    /// becomes infinity. Requiring a second, plainly unsafe opt-in quarantines that path until V is range-safe.</remarks>
    internal static bool SageF32ValueNarrowingEnabled =>
        SageExplicitlyEnabled && EngineKnobs.SageUnsafeF32VNarrow.Value;

    /// <summary>TF32 tensor-core math for F32-operand GEMMs on Ampere+ (SM ≥ 8.0) — PyTorch's default. Opt out: <c>HARTSY_NO_TF32=1</c>.</summary>
    /// <remarks>Plain-F32 GEMMs have no tensor-core path on consumer Ampere (a 3060 runs them at a fraction of
    /// tensor-core throughput), leaving F32 pipelines compute-bound far below the hardware. TF32 keeps F32 range
    /// with a 10-bit mantissa.</remarks>
    private readonly bool _allowTf32;

    /// <summary>HARTSY_GEMM_F16=1: F16-mantissa (COMPUTE_32F_FAST_16F) tensor-core math for F32 GEMMs — faster than TF32 on Ada.</summary>
    /// <remarks>F32 accumulate is kept. Opt-in per parity-checked model (Oasis interactive DiT).</remarks>
    private readonly bool _gemmFast16;

    /// <summary>HARTSY_SDPA_F16=1: force the F16 SDPA path on for ALL callers (not just allowF16 ones) — testing/override.</summary>
    private readonly bool _sdpaF16ForceOn;
    /// <summary>HARTSY_SDPA_NO_F16=1: global kill-switch for the F16 SDPA path even when a caller passes allowF16.</summary>
    private readonly bool _sdpaF16Disabled;
    /// <summary>Routes MHA (D∈{64,128,256}, no mask or broadcastable F32 mask) through cuDNN's fused attention, not materialized cuBLAS.</summary>
    /// <remarks>~34× on the Krea2 self-attention shape. Standard-profile default ON; HARTSY_SDPA_CUDNN=0 disables.
    /// Missing cuDNN or engine rejections fall back to the materialized paths automatically.</remarks>
    private readonly bool _sdpaCudnn;

    /// <summary>Routes F16/BF16 NCHW convolutions through cuDNN conv-forward engines instead of the im2col→cuBLAS GEMM path.</summary>
    /// <remarks>Standard-profile default ON; HARTSY_CONV_CUDNN=0 disables. Failures self-disable for the session and fall back to im2col.</remarks>
    private readonly bool _convCudnn;

    /// <summary>Routes the audio 1D convs and transposed convs (vocoders/codecs/VITS) through cuDNN (mapped to 2D, H=1).</summary>
    /// <remarks>Includes causal/asymmetric pads via the graph API's separate PRE/POST padding attributes. Standard-profile
    /// default ON; HARTSY_AUDIO_CONV_CUDNN=0 restores the direct kernels exactly. Failures self-disable for the session.</remarks>
    private readonly bool _audioConvCudnn;

    /// <summary>Compute type for a GEMM whose operands resolved to <paramref name="gemmType"/>.</summary>
    /// <remarks>FAST_TF32 for F32 operands when allowed (and <see cref="HighPrecisionGemm"/> — an explicit
    /// full-precision request — is off), otherwise plain 32F accumulate (mixed F16-in/F32-acc stays 32F).</remarks>
    private int Compute32F(int gemmType)
    {
        if (HighPrecisionGemm || gemmType != CublasApi.CUDA_R_32F) return CublasApi.CUBLAS_COMPUTE_32F;
        // HARTSY_GEMM_F16=1: F16-mantissa tensor-core matmul with F32 storage+accumulate — ~2× TF32 on Ada for
        // GEMM-heavy small-DiT loops (Oasis). Safer than F16 SDPA (which Oasis already tolerates at corr>0.9999)
        // since accumulation stays F32; opt-in per model that has been parity-checked.
        if (_gemmFast16) return CublasApi.CUBLAS_COMPUTE_32F_FAST_16F;
        return _allowTf32 ? CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32 : CublasApi.CUBLAS_COMPUTE_32F;
    }

    /// <summary>Creates a CUDA backend for the specified device ordinal. If ptxDir is provided, loads PTX kernels from that directory.</summary>
    public CudaBackend(int deviceOrdinal = 0, string? ptxDir = null)
    {
        // Force the managed worker to exist before this finalizable object acquires its first native resource.
        BackendAbandonmentReaper.EnsureStarted();
        _context = new CudaContext(deviceOrdinal);
        GC.SuppressFinalize(_context); // adopted: CudaBackend Dispose/reaper is now the sole finalization owner
        // Must use blocking stream (CU_STREAM_DEFAULT) because GpuTransferHelper uses synchronous
        // cuMemcpyHtoD/DtoH which operate on the NULL stream. A non-blocking stream does NOT
        // synchronize with the NULL stream, causing race conditions where kernels read incomplete
        // data from in-progress H2D transfers. Fix: switch to cuMemcpyHtoDAsync on this stream.
        _stream = new CudaStream(nonBlocking: false);
        GC.SuppressFinalize(_stream);
        // HARTSY_PROFILE_SYNC's per-op GPU-time attribution resolves ITS stream from the ambient backend State
        // (see NvtxRange.Dispose) — no registration needed here.
        // Upload stream is non-blocking so its in-flight work doesn't gate the compute
        // stream's NULL-stream "wait for everything" semantics — without that, prefetched
        // uploads would force compute to wait, defeating overlap. The streaming cache uses
        // explicit cuEventRecord/cuStreamWaitEvent for the parts that *do* need to sync.
        _uploadStream = new CudaStream(nonBlocking: true);
        GC.SuppressFinalize(_uploadStream);
        _streamingCache = new CudaStreamingWeightCache(_context, _stream.Handle, _uploadStream.Handle);
        _ptxDir = ptxDir;
        Device = DeviceKind.Cuda(deviceOrdinal);

        // Keep freed activation buffers warm in the stream-ordered pool (HARTSY_MEMPOOL_KEEP, default on): a 0
        // release threshold returns every freed activation to the driver and re-acquires it on the next alloc,
        // stalling the compute stream (Krea2 1024² measured ~13 s of pure alloc/free round-trips). The threshold is
        // DEVICE state, so it is owned by the refcounted DeviceMempoolPolicy — the first backend on the device
        // decides, and a later same-device backend's construction can no longer flip a live sibling's pool. The
        // OOM path (AllocateAsync → SyncStreamsAndReleasePool) and explicit cuMemPoolTrimTo still return memory on
        // demand either way.
        bool mempoolKeep = EngineKnobs.MempoolKeep.Value;
        DeviceMempoolPolicy.Acquire(_context.DeviceOrdinal, mempoolKeep);
        Volatile.Write(ref _mempoolPolicyAcquired, 1);
        // CUDA_VISIBLE_DEVICES defaults to fastest-first ordering, so an ordinal does not identify the card —
        // log the name or every perf/VRAM number gets attributed to the wrong GPU.
        HartsyInference.Core.Logging.Logs.Info($"[Cuda] device {deviceOrdinal}: {_context.DeviceName} "
            + $"(SM {_context.ComputeCapabilityMajor}.{_context.ComputeCapabilityMinor})");

        // Perf-flag wiring, two tiers (docs/PERFORMANCE.md). STANDARD PROFILE features default ON via
        // tri-state EnvSwitch (unset → documented default, "0"/"false" is the kill-switch) so every install
        // reproduces the published benchmark times with zero configuration. EXPERIMENTAL switches keep the
        // strict opt-in EnvFlag form ("1" only) for A/B benchmarking without recompiling.
        // cuBLASLt bias-epilogue GEMM: promoted to the standard profile 2026-07-09 — every biased Linear
        // otherwise pays a separate BiasAdd kernel + an output-sized HBM round-trip (~700/step on the SDXL
        // UNet, measured −0.16 s/gen). Falls back to GemmEx+BiasAdd when Lt is unavailable or shapes
        // don't qualify; HARTSY_EPILOGUE_FUSION=0 is the kill-switch.
        EnableEpilogueFusion = EngineKnobs.EpilogueFusion.Value;
        // dp4a int8-activation decode GEMV: promoted to the standard profile 2026-07-22 after the
        // full Q4_K/Q6_K/Q8_0 kernel set measured +13-27% end-to-end decode on both benchmark models
        // with ground-truth-bounded numerics (Dp4aGemvGroundTruthTests, LLM_DECODE_PERF_GRIND.md).
        EnableDp4aGemv = EngineKnobs.Dp4aOn.Value;
        EnableTensorCoreGemm = EngineKnobs.TensorcoreGemm.Value;
        // fp8 tensor-core GEMM (activation-quant e4m3) requires SM 8.9+ (Ada); older parts default to the
        // F16-cast path. Verified quality-clean fleet-wide in the standard Swarm config.
        bool fp8TensorCores = _context.ComputeCapabilityMajor > 8
            || (_context.ComputeCapabilityMajor == 8 && _context.ComputeCapabilityMinor >= 9);
        EnableNativeFp8Gemm = EngineKnobs.Fp8Native.Value ?? fp8TensorCores;
        EnableRopeHeadMajorV2 = EngineKnobs.RopeV2.Value;
        EnableStaticFp8InputScale = EngineKnobs.Fp8StaticInputScale.Value;
        EnableModulateEmitFp8 = EngineKnobs.ModulateEmitFp8.Value;
        EnableW8A8 = EngineKnobs.W8a8.Value;
        HighPrecisionGemm = EngineKnobs.HighPrecisionGemm.Value;
        EnableFp8F16Gemm = EngineKnobs.Fp8F16.Value;
        EnableFp8F32Gemm = EngineKnobs.Fp8F32.Value;
        _allowTf32 = _context.ComputeCapabilityMajor >= 8 && !EngineKnobs.NoTf32.Value;
        _gemmFast16 = _context.ComputeCapabilityMajor >= 8 && EngineKnobs.GemmF16.Value;
        // F16 SDPA is gated PER-CALL via the allowF16 arg (callers with bounded/RMS-normed scores like Wan pass true);
        // safe by default because unbounded-score archs (Z-Image fp8) don't pass it. Env: force-on all callers, or kill.
        _sdpaF16ForceOn = EngineKnobs.SdpaF16.Value;
        _sdpaF16Disabled = EngineKnobs.SdpaNoF16.Value;
        // cuDNN fused flash attention: default ON — resolution failures self-disable per session (and engine
        // rejections per head-dim), falling back to the materialized paths, so machines without cuDNN lose
        // speed, never correctness.
        // Probe cuDNN ONCE up front: locate/provision a CUDA-major-matched build, guard against a mismatch (which
        // would hang mid-inference), and log a single clear line so a "why is it slow?" report is one log check.
        // Every cuDNN fast path is AND-gated on availability, so a missing/mismatched cuDNN cleanly stays on the
        // im2col+cuBLAS / custom-flash fallbacks instead of throwing per-op or hanging.
        CudnnRuntime.LogStatus();
        _sdpaCudnn = CudnnRuntime.SupportsSdpa && EngineKnobs.SdpaCudnn.Value;
        // cuDNN convolution forward: default ON — replaces im2col→GEMM for F16/BF16 NCHW convs (the SDXL
        // UNet/VAE cost). Same self-disable-on-failure contract as the fused SDPA path.
        _convCudnn = CudnnRuntime.Available && EngineKnobs.ConvCudnn.Value;
        // Audio conv1d cuDNN (1D→2D, H=1, F32 output via TF32 tensor cores): default ON. The Oobleck/EnCodec
        // vocoder decoders are conv-bound — routing their forward convs through cuDNN is ~1.6-2× on ACE-Step
        // end-to-end (bigger on longer audio, where the VAE decode dominates) with corr 0.9999 vs the direct
        // kernel (same-seed A/B, verified coherent across ACE-Step/MusicGen/AudioGen). Same session-sticky
        // self-disable-on-failure contract as the 2D path, so shapes cuDNN can't serve fall back safely.
        _audioConvCudnn = CudnnRuntime.Available && EngineKnobs.AudioConvCudnn.Value;
        // Each result dir self-documents the config it ran under: log the resolved flag set once.
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] perf flags: SdpaCudnn={_sdpaCudnn} ConvCudnn={_convCudnn} NativeFp8Gemm={EnableNativeFp8Gemm} MempoolKeep={mempoolKeep} " +
            $"EpilogueFusion={EnableEpilogueFusion} Dp4aGemv={EnableDp4aGemv} TensorCoreGemm={EnableTensorCoreGemm} " +
            $"HighPrecisionGemm={HighPrecisionGemm} CacheWeightCasts={CacheWeightCasts} " +
            $"AutoPromoteWeights={GpuTransferHelper.AutoPromoteWeights} Tf32Gemm={_allowTf32}.");

        CublasApi.cublasCreate(out _cublasHandle).ThrowOnCublasError();
        CublasApi.cublasSetStream(_cublasHandle, _stream.Handle).ThrowOnCublasError();

        // Report which cuBLAS we actually loaded. Blackwell (SM 12.x) tensor-core GEMM needs CUDA 12.8+
        // cuBLAS (version >= 120800); an older system cuBLAS silently falls back to a ~6 TFLOPS generic
        // path — the cause of the Ideogram-4 ~50x slowdown vs ComfyUI (which bundles its own 12.8 cuBLAS).
        try
        {
            int cublasVer = -1;
            if (CublasApi.cublasGetVersion(_cublasHandle, out int v) == 0) cublasVer = v;
            string cublasPath = "(path unknown)";
            if (OperatingSystem.IsLinux() && File.Exists("/proc/self/maps"))
            {
                foreach (string line in File.ReadLines("/proc/self/maps"))
                {
                    if (line.IndexOf("libcublas", StringComparison.OrdinalIgnoreCase) < 0) continue;
                    int slash = line.IndexOf('/');
                    cublasPath = slash >= 0 ? line[slash..].Trim() : line.Trim();
                    break;
                }
            }
            HartsyInference.Core.Logging.Logs.Info($"[Cuda] cuBLAS version {cublasVer} (~{cublasVer / 10000}.{(cublasVer / 100) % 100}) loaded from {cublasPath}.");
            if (_context.ComputeCapabilityMajor >= 12 && cublasVer is >= 0 and < 120800)
            {
                HartsyInference.Core.Logging.Logs.Warning($"[Cuda] cuBLAS {cublasVer} predates Blackwell (SM {_context.ComputeCapabilityMajor}.x) support (need >= 120800 / CUDA 12.8). " +
                    "GEMMs will run a slow non-tensor-core fallback. Fix: point LD_LIBRARY_PATH at a CUDA 12.8+ libcublas " +
                    "(e.g. PyTorch's bundled nvidia/cublas/lib) or install the CUDA 12.8+ runtime.");
            }
        }
        catch (Exception ex)
        {
            HartsyInference.Core.Logging.Logs.Warning($"[Cuda] couldn't query cuBLAS version: {ex.Message}");
        }

        // Register this backend's transfer state (context + stream + streaming cache). The state IS this
        // backend's identity: every op entry binds it as the thread ambient (EnterOp), which is how a second
        // backend — on another GPU or the SAME one — gets its own caches/stream instead of silently retargeting
        // this one's (the multi-GPU CUDA 700 poison, and its same-device sequel). Transient allocations route
        // through the stream-ordered pool on the same compute stream, so they reuse the memory their
        // cuMemFreeAsync frees return instead of stranding it in the pool (the Ideogram-4 ~100s/step thrash).
        // Persistent weights/workspaces stay on the sync allocator.
        _transferState = GpuTransferHelper.Register(_context, _stream.Handle, _streamingCache);
        _streamingCache.BindState(_transferState);
        GpuTransferHelper.SetAmbient(_transferState);

        // Two-stage argmax partials (64 float + 64 int slots). Allocated EAGERLY: ArgMaxInto is captured
        // into decode graphs, and the pre-capture warm-up forward doesn't call it — a lazy allocation
        // would bake the slow single-block fallback into the first captured graph permanently.
        // (The ambient bind above is what routes this allocation onto OUR stream.)
        _argmaxScratch = GpuTransferHelper.AllocateDevice(512);

        if (ptxDir != null && Directory.Exists(ptxDir))
        {
            _kernels = new CudaKernels(ptxDir);
            GC.SuppressFinalize(_kernels);
        }

        Capabilities = new BackendCapabilities
        {
            Name = $"CUDA ({_context.DeviceName}, SM {_context.ComputeCapabilityMajor}.{_context.ComputeCapabilityMinor})",
            SupportsF32 = true,
            SupportsF16 = true,
            SupportsBF16 = _context.ComputeCapabilityMajor >= 8,
            SupportsQuantized = true,
            SupportsConv2D = true,
            BandsIm2Col = true,
            Im2ColWorkspaceCapBytes = Im2ColBandCapBytes,
            SupportsSdpa = true,
            SupportsFft = false,
            MaxRank = 6,
        };
        Volatile.Write(ref _constructorCompleted, 1);
    }

    /// <summary>This backend's transfer state, for the multi-backend isolation tests.</summary>
    internal GpuTransferHelper.State TransferState => _transferState;

    /// <summary>Op-entry guard: binds this backend as the calling thread's ambient transfer state, binds the CUDA context, and drains THIS backend's finalizer-queued GPU cleanups. Replaces the bare <c>_context.EnsureCurrent()</c> at every op entry — context identity alone cannot name the owning backend when two backends share a device's primary context, and the cleanup buckets are keyed per backend.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    /// <summary>Binds this backend as the calling thread's ambient op target. For collective transports (<see cref="NcclComm"/>), which stage uploads through the ambient-based transfer helper without going through a public tensor op.</summary>
    internal void BindAmbient() => EnterOp();

    private void EnterOp()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _lifecycleState) != LifecycleActive, this);
        GpuTransferHelper.SetAmbient(_transferState);
        _context.EnsureCurrent();
        Tensor.DrainPendingFinalizerGpuCleanup(_transferState.Key);
        // The previous op's finally blocks have run by now, so anything a CacheActivation rebind displaced and
        // nobody claimed is provably ownerless. Reclaiming it here is what makes the non-in-place case safe.
        GpuTransferHelper.SweepOrphans();
    }

    /// <summary>Largest im2col workspace Conv2D will allocate; larger convs run as output-row bands (bit-identical GEMMs, per-band offset).</summary>
    /// <remarks>Bounds the 512-ch 3×3 @1024² VAE conv (9.2 GB naive) so full-res decode fits next to resident model
    /// weights — 1 GB verified live for the Flux.2 VAE beside Ideogram 4's 18.6 GB resident DiTs (2 GB still OOM'd
    /// there); band size costs no GEMM efficiency (m stays ≥ tens of thousands of rows). Override via
    /// HARTSY_IM2COL_BAND_MB (also lets tests force banding on small shapes).</remarks>
    private static readonly long Im2ColBandCapBytes = EngineKnobs.Im2colBandMb.Value << 20;

    #region Linear Algebra

    /// <summary>Matrix multiply via cuBLAS GemmEx: output = a @ b. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MatMul");
        EnterOp();
        EnsureKernels();

        int m = (int)a.Shape[0];
        int k = (int)a.Shape[1];
        int n = (int)b.Shape[1];

        ulong pA = 0, pB = 0, pC = 0, pBCast = 0, pACast = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            // fp8_scaled operands carry a per-tensor scalar (real = fp8_byte * Fp8ScaleFactor); the CastIfNeeded
            // dequant below is scale-blind, so fold the factor(s) into cuBLAS alpha exactly like LinearImpl does.
            // Both default to 1.0, so plain checkpoints are unchanged.
            float alpha = a.Fp8ScaleFactor * b.Fp8ScaleFactor;
            float beta = 0.0f;

            // Joint dtype resolution — see ResolveGemmDtype(a, b) docs. Fp8 forces F16.
            DType gemmDtype = ResolveGemmDtype(a.DType, b.DType);
            ulong aPtr = CastIfNeeded(pA, a.DType, gemmDtype, (int)a.ElementCount, out pACast);
            ulong bPtr = CastIfNeeded(pB, b.DType, gemmDtype, (int)b.ElementCount, out pBCast);

            int gemmType = CublasDataType(gemmDtype);
            int cType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            CublasApi.cublasGemmEx(_cublasHandle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N, n, m, k, &alpha, bPtr, gemmType, n,
                aPtr, gemmType, k,
                &beta,
                pC, cType, n,
                Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            GpuTransferHelper.CacheActivation(output, pC, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (pACast != 0) CudaMemory.FreeAsync(pACast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pC);
        }
    }

    /// <summary>Byte offset of the F32 wScale[n] section inside a W8A8 combined weight buffer (int8 n·k | pad-to-256 | wScale[n]).</summary>
    private static long W8A8ScaleOffset(int n, int k) => ((long)n * k + 255) & ~255L;

    /// <summary>Host-quantizes a 16-bit/F32 weight [n,k] to per-output-channel int8, uploading one combined buffer (int8 | pad | wScale).</summary>
    /// <remarks>The weight's <see cref="Tensor.Fp8ScaleFactor"/> (the alpha carrier — branch damp on 16-bit
    /// weights) is folded into wScale so the dequant epilogue needs no extra factor. Runs once per weight (cached).</remarks>
    private unsafe ulong QuantizeWeightForW8A8(Tensor weight, int n, int k)
    {
        long scaleOff = W8A8ScaleOffset(n, k);
        nuint totalBytes = (nuint)(scaleOff + (long)n * sizeof(float));
        byte[] host = new byte[totalBytes];
        float alpha = weight.Fp8ScaleFactor;
        void* src = (void*)weight.DataPointer;
        DType dt = weight.DType;
        // SmoothQuant W_hat = W*s (per input channel k) — set via SetW8A8SmoothingScale, folded in here so
        // the int8 quant itself absorbs the migrated activation-outlier difficulty; null = no smoothing.
        float[]? smooth = _w8a8SmoothScaleHost.TryGetValue(weight, out float[]? sv) ? sv : null;
        fixed (byte* dst = host)
        {
            sbyte* q = (sbyte*)dst;
            float* scales = (float*)(dst + scaleOff);
            byte* dstCopy = dst; // avoid capturing the fixed pointer in the lambda closure directly
            Parallel.For(0, n, ni =>
            {
                float amax = 0f;
                for (int ki = 0; ki < k; ki++)
                {
                    float v = W8A8Read(src, dt, (long)ni * k + ki);
                    if (smooth is not null) v *= smooth[ki];
                    float a = MathF.Abs(v);
                    if (a > amax) amax = a;
                }
                float scale = amax > 0f ? amax / 127f : 1f;
                float inv = amax > 0f ? 127f / amax : 0f;
                ((float*)(dstCopy + scaleOff))[ni] = scale * alpha;
                sbyte* qr = (sbyte*)dstCopy + (long)ni * k;
                for (int ki = 0; ki < k; ki++)
                {
                    float v = W8A8Read(src, dt, (long)ni * k + ki);
                    if (smooth is not null) v *= smooth[ki];
                    int iv = (int)MathF.Round(v * inv);
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    qr[ki] = (sbyte)iv;
                }
            });
            // Stream-ordered pool allocation, NOT CudaMemory.AllocatePersistent: a synchronous cuMemAlloc can
            // reuse a VA whose deferred cuMemFreeAsync (from an earlier transient weight cast) hasn't executed
            // yet — that late free then silently destroys this buffer and the eventual cuMemFree double-free
            // fails INVALID_VALUE (bisected 2026-07-23, W8A8ReproTemp). Pool pointers ride the same
            // stream-ordered allocator as every other cache, so ordering is correct by construction.
            ulong dev = GpuTransferHelper.AllocateDevice(totalBytes);
            try
            {
                CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)dst, totalBytes, _stream.Handle).ThrowOnError();
                // Host buffer is stack-scoped (fixed byte[]) — drain the async copy before it goes out of scope.
                _stream.Synchronize();
                return dev;
            }
            catch (Exception primary)
            {
                try { GpuTransferHelper.FreeDevice(dev); }
                catch (Exception cleanup)
                {
                    throw new AggregateException("W8A8 weight upload and rollback both failed.", primary, cleanup);
                }
                throw;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float W8A8Read(void* src, DType dt, long i)
    {
        if (dt == DType.F32) return ((float*)src)[i];
        if (dt == DType.F16) return (float)((Half*)src)[i];
        // BF16: high 16 bits of an F32.
        uint bits = (uint)((ushort*)src)[i] << 16;
        return BitConverter.UInt32BitsToSingle(bits);
    }

    /// <summary>Frees every cached W8A8 int8 weight buffer (model switch / full eviction / dispose).</summary>
    private void FreeW8A8Cache(GpuTransferHelper.State? explicitState = null)
    {
        List<Exception>? failures = null;
        foreach ((Tensor weight, ulong ptr) in _w8a8WeightCache.ToArray())
        {
            try
            {
                if (explicitState is null) GpuTransferHelper.FreeDevice(ptr);
                else GpuTransferHelper.FreeDevice(explicitState, ptr);
                _w8a8WeightCache.Remove(weight);
            }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        try { FreeW8A8SmoothScaleCache(explicitState); }
        catch (Exception error) { (failures ??= []).Add(error); }
        try { FreeInt8RowScaleCache(explicitState); }
        catch (Exception error) { (failures ??= []).Add(error); }
        try { FreeNvfp4ScaleCache(explicitState); }
        catch (Exception error) { (failures ??= []).Add(error); }
        if (failures is not null) throw new AggregateException("One or more W8A8 cache buffers failed to release.", failures);
    }

    /// <summary>Kill switch for the fused GEMM+dequant mma kernel (<c>HARTSY_INT8_FUSED_MMA=0</c>). ON by default: −10.5 ms/step end-to-end (4 interleaved reps, all pairs same-sign, paired t = 4.15). It only got there once <see cref="UseFusedMmaGemm"/> was narrowed to the shapes it actually wins on — wired in less carefully it measured +38.7 ms/step, and +6.9 with only a row floor.</summary>
    internal static bool FusedMmaGemm = EngineKnobs.Int8FusedMma.Value;

    /// <summary>Rows below which the fused mma GEMM is not used. Its block tile is 128×256, so a few hundred rows is two M-blocks — a grid that covers a fraction of one wave across 128 SMs, where cuBLASLt's small-m heuristic wins outright. Every measured win is at m ≥ 1543 (ffn_up's smaller row chunk); everything below this floor — audio attention and FFN, the text-side k/v projections — was never measured and must not be assumed. Wiring the fused path in WITHOUT this floor cost +38.7 ms/step end-to-end while winning +5.2% on the three shapes the microbenchmark covered.</summary>
    private const int FusedMmaMinRows = 1024;

    /// <summary>Whether the fused mma GEMM beats the cuBLASLt-GEMM + separate-dequant pair for this shape.</summary>
    /// <remarks><para>Measured on a 4090 (<c>Int8MmaGemmTests</c>, min-of-batches, 3 runs each): attn_qkvo
    /// 4992×4096×4096 **+5.2%** and ffn_up 4992×16384×4096 **+2%**, but ffn_down 4992×4096×16384 **−26%**. The
    /// split is K against N: the fused kernel's win is the int32 round trip it deletes, which scales with the
    /// OUTPUT, while its mainloop still runs ~30% behind cuBLASLt's — so a deep-K shape spends all its time in
    /// the part we are worse at and has almost no accumulator traffic to save (ffn_down's dequant is 11% of the
    /// pair, attn's is 31%).</para>
    /// <para>The N bound is symmetric — roughly square, not just <c>k &lt;= 2n</c> — because a wide-N shape has
    /// proportionally more WEIGHT traffic, and this kernel is the bandwidth-hungry one. Measured against COLD L2
    /// (rotating weight buffers, see <c>ColdBuffers</c>), ffn_up 4992×16384×4096 is **−7.0%**, not the +2.2% the
    /// warm harness reported; only the square attn shape survives, at +6.1%.</para>
    /// <para>Every bound exists because a per-shape microbenchmark is NOT evidence about the workload: this gate
    /// must admit only the regime actually measured, under conditions that resemble a real step. Re-measure
    /// end-to-end, not per shape, before widening it — and on a different card before trusting it there.</para></remarks>
    /// <summary>Widens the N bound from <c>n &lt;= 2k</c> to <c>n &lt;= 4k</c> (<c>HARTSY_INT8_MMA_WIDE_GATE=1</c>), which admits ffn_up 4992×16384×4096 and nothing else at LTX-2.5's shapes; ffn_down stays excluded by the unchanged <c>k &lt;= 2n</c>. OFF by default and deliberately an env switch rather than an edit: ffn_up was −7.0% against cold L2 under the padded layout, so re-admitting it is a claim that the swizzle flipped that sign, and this file's rule is that such a claim is settled end-to-end, not per shape. An env arm is also the only way to A/B the gate without swapping the binary mid-campaign, which corrupts the whole run.</summary>
    private static readonly bool WideMmaGate = EngineKnobs.Int8MmaWideGate.Value;

    /// <summary>The f16-staged wide ConvRot+quant kernel, bit-identical to the rotate-then-quant pair it replaces. **OFF, and measured**: it cuts that pair's 7 bytes/element to 3, and at LTX-2.5's 1280x736x145f FFN-down (17480x16384, 96 calls/step) `Int8.Quant` still went 1557.7 -> 1607.8 ms over 3 steps, with the end-to-end A/B inside its own 19 ms spread. Staging a 16384-wide row costs 40 KB of shared, which is 2 blocks/SM against the split pair's simple high-occupancy streaming kernels — the traffic saving does not pay for the occupancy. Kept unit-pinned (<c>ConvRotFusedQuantTests</c>) as the record: recomputing the byte ratio is not evidence. <c>HARTSY_CONVROT_WIDE=1</c> re-enables.</summary>
    private static readonly bool UseWideConvRotQuant = EngineKnobs.ConvrotWide.Value;

    private bool UseFusedMmaGemm(bool outF16, int rows, int n, int k) =>
        FusedMmaGemm && outF16 && rows >= FusedMmaMinRows
        && k <= 2L * n && n <= (WideMmaGate ? 4L : 2L) * k && _kernels!.HasInt8MmaGemm(rows, n, k);

    /// <summary>One resident-int8 projection for <see cref="RunResidentInt8"/>: device pointers already resolved, so the helper never touches the transfer caches (see the prologue ordering note in <c>LinearImpl</c>).</summary>
    private readonly record struct Int8ResidentTarget(ulong Weight, ulong Output, ulong BiasF32, ulong RowScaleDev,
        int N, bool OutF16, bool FuseGelu);

    /// <summary>The resident int8 chain — ConvRot, per-row quant, IMMA GEMM, dequant epilogue — over one activation and one or more weights. Every target shares the quantized activation, which is the point: the rotate+quant pass is ~40% of the LTX-2.5 step's quant traffic precisely because an attention re-derived it per projection. Bit-identical to running the targets one at a time (same pointer, same k, same kernel).</summary>
    /// <remarks>Chunked over rows because the int32 accumulator is 4 bytes per output element: at video token counts
    /// a single unchunked m·n·4 buffer runs to gigabytes (H3's mlp.fc1 is n=28672). Each chunk is padded up to the
    /// 32-row granularity cuBLASLt's int8 TN kernels want, exactly as comfy-kitchen's own _int8_matmul_accumulate
    /// does — 31 wasted rows of int8 compute beats materializing the weight.</remarks>
    /// <summary>An LTX-2 per-head output gate to apply as the activation is quantized, instead of as its own pass. <c>Logits</c> = 0 means no gate.</summary>
    private readonly record struct Int8PreGate(ulong Logits, int Heads, int HeadDim);

    private unsafe void RunResidentInt8(ulong pInput, DType inputDType, int m, int k, int group,
        ReadOnlySpan<Int8ResidentTarget> targets, Int8PreGate preGate = default)
    {
        bool srcF16 = inputDType == DType.F16;
        int inputElementBytes = inputDType.SizeInBytes;
        int nMax = 0;
        foreach (Int8ResidentTarget t in targets) nMax = Math.Max(nMax, t.N);
        int colChunk = Int8ResidentColChunk(nMax);
        // Sized by the WIDEST target so one accumulator serves the group; a narrower target just leaves it partly unused.
        int rowChunk = Int8ResidentRowChunk(m, Math.Min(nMax, colChunk), k, group > 0 ? inputElementBytes : 0);
        int paddedChunk = PadInt8Rows(rowChunk);
        int accCols = Math.Min(nMax, colChunk);

        // The rotated full-width intermediate only exists for the split rotate-then-quant pair; every fused
        // variant reads the activation once. Skipping the allocation matters beyond the pool: `rowChunk` above
        // is sized off free VRAM, so a 134 MB transient nobody reads shrinks the next Linear's chunk.
        bool fusedQuant = group > 0 && (preGate.Logits != 0 || _kernels!.HasFusedConvRotQuant(k, group)
            || (UseWideConvRotQuant && _kernels.HasWideConvRotQuant(k, group, srcF16)));

        ulong pRot = 0, pAct8 = 0, pRowScale = 0, pOut32 = 0;
        try
        {
            if (group > 0 && !fusedQuant)
                pRot = GpuTransferHelper.AllocateDevice((nuint)((long)rowChunk * k * inputElementBytes));
            pAct8 = GpuTransferHelper.AllocateDevice((nuint)((long)paddedChunk * k));
            pRowScale = GpuTransferHelper.AllocateDevice((nuint)((long)paddedChunk * sizeof(float)));
            pOut32 = GpuTransferHelper.AllocateDevice((nuint)((long)paddedChunk * accCols * sizeof(int)));
            // The pad rows are deliberately left unwritten rather than zeroed: every byte is a valid int8,
            // k·127² stays an order of magnitude inside int32 for any shape here (2.48e8 at k=15360 against
            // int32's 2.15e9), and the epilogue below is launched with `rows`, so those accumulators are
            // never read. A memset would cost a full-stream sync (cuMemsetD8 runs on the legacy null
            // stream) on every single Linear.

            for (int firstRow = 0; firstRow < m; firstRow += rowChunk)
            {
                int rows = Math.Min(rowChunk, m - firstRow);
                ulong inputChunk = pInput + (ulong)((long)firstRow * k * inputElementBytes);
                ulong quantSource = inputChunk;
                // Sub-scopes: "Linear" is FOUR kernels (rotate, quant, GEMM, dequant) and the whole-chain
                // label cannot say which one costs. HARTSY_PROFILE_FINE only — thousands of pushes per step.
                using (NvtxRange.PushFine("Int8.Quant"))
                if (preGate.Logits != 0)
                {
                    // The caller's per-head gate folds into this load, so its own full read+write pass over the
                    // activation disappears. Eligibility was settled before the chunk loop (CanPreGate).
                    _kernels!.LaunchConvRotQuantRowwiseGated(pAct8, pRowScale, inputChunk, rows, k, group,
                        _stream.Handle, preGate.Logits + (ulong)((long)firstRow * preGate.Heads * sizeof(ushort)),
                        preGate.Heads, preGate.HeadDim);
                }
                else if (group > 0 && _kernels!.HasFusedConvRotQuant(k, group))
                {
                    // Rotation + per-row quant in one pass: the split pair wrote a full-width rotated
                    // activation to HBM only for the quantizer to read it straight back.
                    _kernels.LaunchConvRotQuantRowwise(pAct8, pRowScale, inputChunk, rows, k, group, _stream.Handle, srcF16);
                }
                else if (group > 0 && UseWideConvRotQuant && _kernels!.HasWideConvRotQuant(k, group, srcF16))
                {
                    // Same win one row width up: LTX-2.5's FFN-down activation is 16384 wide, too wide for the
                    // f32-staged kernel above, and was paying 7 bytes/element instead of 3.
                    _kernels.LaunchConvRotQuantRowwiseWide(pAct8, pRowScale, inputChunk, rows, k, group, _stream.Handle);
                }
                else
                {
                    if (group > 0)
                    {
                        _kernels!.LaunchConvRotRotate(pRot, inputChunk, (long)rows * k, group, _stream.Handle, srcF16);
                        quantSource = pRot;
                    }
                    _kernels!.LaunchW8A8QuantRowwise(pAct8, pRowScale, quantSource, rows, k, _stream.Handle, srcF16);
                }
                int padded = PadInt8Rows(rows);
                foreach (Int8ResidentTarget t in targets)
                {
                    int n = t.N;
                    int outElementBytes = t.OutF16 ? 2 : 4;
                    ulong outputChunk = t.Output + (ulong)((long)firstRow * n * outElementBytes);
                    uint actMode = t.FuseGelu ? 1u : 0u;
                    if (colChunk >= n)
                    {
                        // One kernel for GEMM + dequant, so the [rows, n] int32 accumulator never reaches HBM.
                        // Only where it actually WINS — see UseFusedMmaGemm; elsewhere the cuBLASLt pair is faster.
                        // It takes `rows`, not `padded`: M is predicated inside, so the pad rows cost nothing.
                        if (UseFusedMmaGemm(t.OutF16, rows, n, k))
                        {
                            using (NvtxRange.PushFine("Int8.GemmDequant"))
                                _kernels!.LaunchInt8MmaGemmDequant(outputChunk, pAct8, t.Weight, pRowScale,
                                    t.RowScaleDev, t.BiasF32, rows, n, k, actMode, _stream.Handle);
                            continue;
                        }
                        using (NvtxRange.PushFine("Int8.Gemm"))
                            Int8Gemm.Run(t.Weight, pAct8, pOut32, padded, n, k, _stream.Handle);
                        using (NvtxRange.PushFine("Int8.Dequant"))
                            _kernels!.LaunchW8A8DequantBias(outputChunk, pOut32, pRowScale, t.RowScaleDev, t.BiasF32,
                                rows, n, _stream.Handle, outF16: t.OutF16, actMode: actMode);
                        continue;
                    }
                    // Tile over N so the int32 accumulator is a small slice consumed by the dequant while
                    // it is still in L2, instead of a full [rows, n] buffer streamed out to HBM and read
                    // straight back. The weight is [N, K] row-major, so a column slice of the output is a
                    // contiguous ROW range of the weight — just a pointer offset, no repack.
                    for (int firstCol = 0; firstCol < n; firstCol += colChunk)
                    {
                        int cols = Math.Min(colChunk, n - firstCol);
                        Int8Gemm.Run(t.Weight + (ulong)((long)firstCol * k), pAct8, pOut32, padded, cols, k, _stream.Handle);
                        _kernels!.LaunchW8A8DequantBiasStrided(
                            outputChunk + (ulong)((long)firstCol * outElementBytes),
                            pOut32, pRowScale,
                            t.RowScaleDev + (ulong)((long)firstCol * sizeof(float)),
                            t.BiasF32 == 0 ? 0 : t.BiasF32 + (ulong)((long)firstCol * sizeof(float)),
                            rows, cols, n, _stream.Handle,
                            outF16: t.OutF16, actMode: actMode);
                    }
                }
            }
        }
        finally
        {
            if (pRot != 0) GpuTransferHelper.FreeDevice(pRot);
            if (pAct8 != 0) GpuTransferHelper.FreeDevice(pAct8);
            if (pRowScale != 0) GpuTransferHelper.FreeDevice(pRowScale);
            if (pOut32 != 0) GpuTransferHelper.FreeDevice(pOut32);
        }
    }

    /// <summary>Linear layer via cuBLAS GemmEx with transpose: output = input × weight^T + bias.</summary>
    /// <remarks>Supports mixed F32/F16/F8 dtypes. For a quantized weight the dequantized F16 cast is cached per
    /// preloaded weight (fast, but the cast occupies F16-sized VRAM).</remarks>
    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
        => LinearImpl(output, input, weight, bias, cacheWeightCast: true);

    /// <summary>Kill switch for folding LTX-2's per-head gate into the activation quantization (<c>HARTSY_LTX2_GATEFUSE=0</c>).</summary>
    internal static bool FuseHeadGateIntoQuant = EngineKnobs.Ltx2Gatefuse.Value;

    /// <summary><c>Linear(gate(input))</c> where <c>gate</c> is LTX-2's per-head output scaling — folded into the activation's rotate+quant pass when the resident int8 chain can serve it, so the gate costs no traffic of its own. Falls back to the explicit gate-then-Linear pair, which mutates <paramref name="input"/> in place exactly as the caller's own sequence did.</summary>
    /// <remarks>The gate is a full read AND write of a <c>[seq, heads·headDim]</c> tensor — 81.8 MB per call at
    /// LTX-2.5's video shape — whose only consumer is this projection, which then reads the same tensor again to
    /// quantize it. Bit-identical either way: the fused kernel reproduces the separate pass's f16 store.</remarks>
    public unsafe void LinearHeadGated(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        Tensor gateLogits, int heads, int headDim)
    {
        bool fused = FuseHeadGateIntoQuant && input.DType == DType.F16 && gateLogits.DType == DType.F16
            && weight.DType == DType.I8 && weight.QuantInfo is QuantWeightInfo qi && qi.ConvRotGroupSize > 0
            && CanRunResidentInt8(output, input, weight, qi, 0, -1) && _kernels is not null
            && _kernels.HasFusedGatedConvRotQuant((int)weight.Shape[1], qi.ConvRotGroupSize, srcF16: true, heads, headDim);
        if (!fused)
        {
            Ltx2HeadGate(input, gateLogits, (int)input.Shape[0], heads, headDim);
            Linear(output, input, weight, bias);
            return;
        }
        LinearImpl(output, input, weight, bias, cacheWeightCast: true, preGate: gateLogits, preGateHeads: heads,
            preGateHeadDim: headDim);
    }

    /// <summary>Fused quantized matmul (the low-VRAM path): same GEMM as <see cref="Linear"/> but the dequantized weight is never cached.</summary>
    /// <remarks>The quantized weight bytes stay resident (preloaded), so a Q4/Q5/Q6/Q8 model keeps its on-device
    /// footprint at the quant size instead of expanding to F16. Trades a per-call dequant (memory-bound, cheap) for
    /// the VRAM saving.</remarks>
    public void QuantizedMatMul(Tensor output, Tensor input, Tensor quantWeight, Tensor? bias)
        // cacheWeightCast was hard-false here ("low-VRAM = never keep dequantized copies"), which made EVERY
        // prefill re-dequantize the whole weight set (llama-1B TTFT 144 ms vs llama.cpp's 6 — the entire gap).
        // The budget gate inside LinearImpl already provides the actual low-VRAM protection: casts cache only
        // for preloaded weights AND only while free VRAM stays above a 2 GB floor (a 1.2 GB GLM head cast is
        // refused and stays transient). Master kill-switch: CacheWeightCasts=false.
        => LinearImpl(output, input, quantWeight, bias, cacheWeightCast: true);

    /// <summary>GEMM against a contiguous ROW RANGE of <paramref name="weight"/>, sharing the resident weight (and its cached dtype cast) rather than materializing a slice. Rows are contiguous in <c>[outDim, inDim]</c>, so this is a pointer offset — which is the whole point: a real sub-tensor would be a separate cache identity and would upload a second copy of an already-resident weight. Lets a fused projection be evaluated in parts (see <c>MiniMaxH3Transformer</c>'s chunked attention, which projects k+v and then q from one packed qkv weight).</summary>
    public void LinearWeightRows(Tensor output, Tensor input, Tensor weight, Tensor? bias, int weightRowOffset, int weightRowCount)
        => LinearImpl(output, input, weight, bias, cacheWeightCast: true, weightRowOffset, weightRowCount);

    /// <summary>Linear whose GELU is folded into the GEMM's dequant epilogue where the path allows it (the resident int8-ConvRot chain), else a plain Linear followed by <see cref="Gelu"/>. A DiT feed-forward's up-projection is always immediately GELU'd, and that intermediate is the widest tensor in the block — writing it once instead of writing, re-reading and re-writing it is worth a full pass over ~328 MB per call.</summary>
    public unsafe void LinearGelu(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        bool fused = weight.DType == DType.I8 && weight.QuantInfo is QuantWeightInfo qi
            && CanRunResidentInt8(output, input, weight, qi, 0, -1);
        LinearImpl(output, input, weight, bias, cacheWeightCast: true, fuseGelu: fused);
        if (!fused) Gelu(output, output);
    }

    /// <summary>Kill switch for grouped resident-int8 Linears (<c>HARTSY_GROUPED_LINEAR=0</c>). Also the seam the bit-identity test flips to run the grouped and per-op routes against one another on one backend.</summary>
    internal static bool GroupedLinear = EngineKnobs.GroupedLinear.Value;

    /// <summary>Projections sharing one input, sharing one activation rotate+quant pass. Ops the resident int8 chain cannot serve — or that disagree on k or on the ConvRot group, since those decide the quantized bytes — fall out to an ordinary <see cref="Linear"/> each, so a mixed group is served, not refused.</summary>
    public unsafe void LinearMulti(Tensor input, ReadOnlySpan<LinearOp> ops)
    {
        if (ops.Length == 0) return;
        EnterOp();
        EnsureKernels();

        int k = 0, group = 0, grouped = 0;
        Span<bool> eligible = ops.Length <= 16 ? stackalloc bool[ops.Length] : new bool[ops.Length];
        for (int i = 0; i < ops.Length; i++)
        {
            LinearOp op = ops[i];
            eligible[i] = GroupedLinear
                && op.Weight.DType == DType.I8 && op.Weight.QuantInfo is { RowScale: not null } qi
                && CanRunResidentInt8(op.Output, input, op.Weight, qi, 0, -1)
                && (k == 0 || ((int)op.Weight.Shape[1] == k && qi.ConvRotGroupSize == group));
            if (!eligible[i]) continue;
            k = (int)op.Weight.Shape[1];
            group = op.Weight.QuantInfo!.ConvRotGroupSize;
            grouped++;
        }
        // One eligible op is just a Linear; grouping it would only duplicate that method's prologue.
        if (grouped < 2)
        {
            foreach (LinearOp op in ops) Linear(op.Output, input, op.Weight, op.Bias);
            return;
        }

        int m = (int)(input.ElementCount / k);
        Int8ResidentTarget[] targets = new Int8ResidentTarget[grouped];
        ulong[] pOutputs = new ulong[grouped];
        nuint[] outputBytes = new nuint[grouped];
        bool[] cachedOutputs = new bool[grouped];
        // Every host companion read happens BEFORE the first CopyToDevice below: a mid-forward DataPointer read on a
        // device-cached tensor trips the lazy-sync consume and the outer finally would then double-free (bisected
        // 2026-07-23, W8A8ReproTemp). With a group that loop runs once, up front, for all of them.
        int slot = 0;
        for (int i = 0; i < ops.Length; i++)
        {
            if (!eligible[i]) continue;
            targets[slot++] = new(0, 0, 0, EnsureInt8RowScaleDev(ops[i].Weight, (int)ops[i].Weight.Shape[0]),
                (int)ops[i].Weight.Shape[0], ops[i].Output.DType == DType.F16, false);
        }

        ulong pInput = 0;
        ulong[] pWeights = new ulong[grouped];
        ulong[] pBiases = new ulong[grouped];
        ulong[] biasCasts = new ulong[grouped];
        // Scoped as "Linear" so a grouped step's profile stays comparable with an ungrouped one; the ineligible
        // stragglers below push their own, outside this scope, so nothing is counted twice.
        using (NvtxRange.Push(NvtxRange.ProfileShapes ? $"Linear grp m={m} g={grouped}" : "Linear"))
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            slot = 0;
            for (int i = 0; i < ops.Length; i++)
            {
                if (!eligible[i]) continue;
                LinearOp op = ops[i];
                outputBytes[slot] = GpuTransferHelper.ByteSize(op.Output);
                pOutputs[slot] = GpuTransferHelper.AllocateDevice(outputBytes[slot]);
                pWeights[slot] = GpuTransferHelper.CopyToDevice(op.Weight);
                ulong biasF32 = 0;
                if (op.Bias is not null)
                {
                    pBiases[slot] = GpuTransferHelper.CopyToDevice(op.Bias);
                    biasF32 = CastIfNeeded(pBiases[slot], op.Bias.DType, DType.F32,
                        (int)op.Bias.ElementCount, out biasCasts[slot]);
                }
                targets[slot] = targets[slot] with
                {
                    Weight = pWeights[slot],
                    Output = pOutputs[slot],
                    BiasF32 = biasF32,
                };
                slot++;
            }

            RunResidentInt8(pInput, input.DType, m, k, group, targets);

            // Publish only after every launch succeeds, and unpublish on a partial failure — same rule as Split.
            try
            {
                for (int t = 0; t < grouped; t++)
                {
                    GpuTransferHelper.CacheActivation(GetGroupedOutput(ops, eligible, t), pOutputs[t], outputBytes[t]);
                    cachedOutputs[t] = true;
                }
            }
            catch
            {
                for (int t = 0; t < grouped; t++)
                {
                    if (GpuTransferHelper.TryUncacheActivation(GetGroupedOutput(ops, eligible, t), pOutputs[t]))
                        cachedOutputs[t] = false;
                }
                throw;
            }
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            for (int t = 0; t < grouped; t++)
            {
                GpuTransferHelper.FreeDevice(pWeights[t]);
                GpuTransferHelper.FreeDevice(pBiases[t]);
                if (biasCasts[t] != 0) CudaMemory.FreeAsync(biasCasts[t], _stream.Handle);
                if (!cachedOutputs[t]) GpuTransferHelper.FreeDevice(pOutputs[t]);
            }
        }

        for (int i = 0; i < ops.Length; i++)
        {
            if (!eligible[i]) Linear(ops[i].Output, input, ops[i].Weight, ops[i].Bias);
        }
    }

    private static Tensor GetGroupedOutput(ReadOnlySpan<LinearOp> ops, ReadOnlySpan<bool> eligible, int slot)
    {
        for (int i = 0; i < ops.Length; i++)
        {
            if (eligible[i] && slot-- == 0) return ops[i].Output;
        }
        throw new ArgumentOutOfRangeException(nameof(slot));
    }

    private unsafe void LinearImpl(Tensor output, Tensor input, Tensor weight, Tensor? bias, bool cacheWeightCast,
        int weightRowOffset = 0, int weightRowCount = -1, bool fuseGelu = false,
        Tensor? preGate = null, int preGateHeads = 0, int preGateHeadDim = 0)
    {
        using NvtxRange _nvtx = NvtxRange.Push(NvtxRange.ProfileShapes
            ? $"Linear m={input.ElementCount / weight.Shape[weight.Shape.Rank - 1]}x{weight.Shape[0]}x{weight.Shape[weight.Shape.Rank - 1]}"
            : "Linear");
        EnterOp();
        EnsureKernels();

        // A resident int8 weight has no dtype cast the generic path below could take — CastOnGpu has no I8 source —
        // so a layer the IMMA chain cannot serve (an explicit full_precision_matrix_mult, a shape the cuBLASLt int8
        // TN config rejects, a row range) has to be un-rotated and dequantized here or it cannot run at all.
        if (weight.DType == DType.I8 && weight.QuantInfo is { RowScale: not null } int8Info
            && !CanRunResidentInt8(output, input, weight, int8Info, weightRowOffset, weightRowCount))
        {
            using Tensor dequantized = Int8ConvRotCodec.DequantToBf16(weight, int8Info.RowScale, int8Info.ConvRotGroupSize);
            try
            {
                LinearImpl(output, input, dequantized, bias, cacheWeightCast: false, weightRowOffset, weightRowCount);
            }
            finally
            {
                // Drops the device copy keyed by this tensor (draining the stream first) before the bytes go away.
                FreeWeights([dequantized]);
            }
            return;
        }

        // A resident nvfp4 weight is 4 bits per element with its scales in a separate swizzled tensor, so the generic
        // CastOnGpu path cannot touch it (that helper is pointer-level and never sees the companions). A layer the
        // dequant kernel cannot serve — no PTX, a row range, a companion that does not describe this weight — is
        // unpacked on the host here or it cannot run at all.
        if (weight.DType == DType.F4E2M1 && weight.QuantInfo is { BlockScale: not null, GlobalScale: not null } nvfp4Info
            && !CanRunResidentNvfp4(weight, nvfp4Info, weightRowOffset, weightRowCount))
        {
            using Tensor dequantized = Nvfp4ResidentCodec.DequantToBf16(weight, nvfp4Info.BlockScale, nvfp4Info.GlobalScale);
            try
            {
                LinearImpl(output, input, dequantized, bias, cacheWeightCast: false, weightRowOffset, weightRowCount);
            }
            finally
            {
                // Drops the device copy keyed by this tensor (draining the stream first) before the bytes go away.
                FreeWeights([dequantized]);
            }
            return;
        }

        // A row range addresses the weight by byte offset, which block-quantized layouts (super-block scales
        // interleaved with packed nibbles) cannot express, and it makes the W8A8 int8 cache — keyed on the WHOLE
        // weight — describe the wrong rows. Both are refused rather than silently mis-slicing; every fused-GEMV
        // branch below is m<=8 (LLM decode) and so is unreachable from the chunked-DiT callers that need this.
        bool rowRange = weightRowOffset != 0 || weightRowCount >= 0;
        if (rowRange)
        {
            if (weight.DType.IsQuantized)
                throw new NotSupportedException($"LinearWeightRows cannot row-slice block-quantized weights (got {weight.DType}).");
            if (bias is not null && bias.DType != output.DType)
                throw new NotSupportedException(
                    $"LinearWeightRows needs bias dtype to match output ({bias.DType} vs {output.DType}) — a cast would rebase the slice.");
            if (weightRowOffset < 0 || weightRowCount <= 0 || (long)weightRowOffset + weightRowCount > weight.Shape[0])
                throw new ArgumentOutOfRangeException(nameof(weightRowOffset),
                    $"LinearWeightRows range [{weightRowOffset}, {weightRowOffset + weightRowCount}) is outside weight rows [0, {weight.Shape[0]}).");
        }

        int n = rowRange ? weightRowCount : (int)weight.Shape[0]; // outDim (or the requested row count)
        int k = (int)weight.Shape[1]; // inDim
        int m = (int)(input.ElementCount / k); // batch*seqLen

        // W8A8 eligibility decided BEFORE any device work: the int8 weight cache replaces the F16 weight on
        // device entirely, so the F16 upload is skipped — and the cache-miss host quant must read
        // weight.DataPointer BEFORE this call touches the transfer caches (a mid-forward DataPointer read on a
        // device-cached tensor trips the lazy-sync consume and the outer finally would double-free pWeight —
        // bisected 2026-07-23, W8A8ReproTemp).
        bool w8a8 = !rowRange && EnableW8A8 && _kernels!.HasW8A8Kernels && Int8Gemm.IsSupported && m >= 32
            && weight.Shape.Rank == 2 && k % 4 == 0 && n % 4 == 0
            && (weight.DType == DType.F16 || weight.DType == DType.BF16 || weight.DType == DType.F32)
            && (input.DType == DType.F16 || input.DType == DType.F32)
            && (output.DType == DType.F16 || output.DType == DType.F32);
        if (w8a8 && CaptureW8A8Operands is { } capture)
        {
            CaptureW8A8Operands = null;
            capture(SnapshotToF32ForTest(input, (long)m * k), m, k, SnapshotToF32ForTest(weight, (long)n * k), n, weight);
        }
        if (w8a8 && !_w8a8WeightCache.ContainsKey(weight))
        {
            _w8a8WeightCache[weight] = QuantizeWeightForW8A8(weight, n, k);
        }

        // Resident int8: the checkpoint's own weight bytes ARE the operand, so unlike w8a8 there is nothing to
        // quantize here — only the eligibility answer, already settled by the fallback gate above. The scale upload
        // reads the companion on the HOST, so it happens before this call touches the transfer caches, for the same
        // reason the W8A8 host quant above does.
        bool int8Resident = weight.DType == DType.I8 && weight.QuantInfo is { RowScale: not null };
        ulong int8RowScaleDev = int8Resident ? EnsureInt8RowScaleDev(weight, n) : 0;

        // Same ordering rule for nvfp4: the block-scale upload and the two scalar reads are HOST reads of the
        // companions, so they happen before the transfer caches are touched. Eligibility was already settled by the
        // fallback gate above, so reaching here with an F4E2M1 weight means the kernel can serve it.
        Nvfp4WeightScales nvfp4Scales = weight.DType == DType.F4E2M1
            && weight.QuantInfo is { BlockScale: not null, GlobalScale: not null } ? EnsureNvfp4Scales(weight)
                : default;

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        ulong pInputFp8 = 0, pFp8Scratch = 0;
        bool cachedOutput = false;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            if (!w8a8)
            {
                pWeight = GpuTransferHelper.CopyToDevice(weight);
            }
            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            // Fused quantized GEMV — the LLM decode hot path. When m is small (single-token or small-batch
            // decode) and the weight is Q4_K with F32 activations, read the quantized bytes ONCE and dequant
            // inline with F32 accumulate. This replaces "dequant whole weight to F16 (a full extra HBM pass +
            // F16-sized intermediate) then cuBLAS GEMM at m=1", cutting weight traffic ~4× and skipping the
            // temp — the dominant decode cost. k is always a multiple of 256 for Q4_K. Larger m (prefill)
            // falls through to the cuBLAS GEMM, which is efficient at m≥ a few hundred.
            // !rowRange: every fused GEMV below hands the kernel the un-offset pWeight and returns, so a row range
            // would silently read from row 0. They are decode-shaped (m<=8) and no row-range caller is, so forcing
            // the general GEMM costs nothing rather than needing an offset threaded through each launcher.
            if (!rowRange && m <= 8 && input.DType == DType.F32 && output.DType == DType.F32)
            {
                // dp4a int8-activation paths (standard profile, kill-switch HARTSY_DP4A_ON=0): quantize the
                // activation to int8
                // (Q8_1, per-32-block scale + int-sum) once per call, then run the GEMV as int8×int8 dot
                // products via __dp4a (4 MACs/instruction) instead of per-element float dequant — the fused
                // GEMV kernels are compute/memory CO-limited (74.6% ALU on Q4_K, ncu 2026-07-22), so cutting
                // per-element ALU cost is the remaining lever. Lossy (int8 activation rounding) but bounded —
                // ground-truth gates in Dp4aGemvGroundTruthTests derive the tolerance from the Q8_1 rounding
                // error rather than guessing. Q8_0/Q6_K are symmetric quants, so their kernels consume only
                // xq/xd; Q4_K's min term additionally needs the per-block int-sum xs.
                bool dp4a = EnableDp4aGemv
                    && (((weight.DType == DType.Q4_K || weight.DType == DType.Q5_K || weight.DType == DType.Q6_K) && k % 256 == 0)
                    || ((weight.DType == DType.Q8_0 || weight.DType == DType.Q4_0 || weight.DType == DType.Q5_0) && k % 32 == 0));
                if (dp4a)
                {
                    // Quantize-at-producer: the input's Q8_1 sidecar (emitted by RmsNormEmitQ8/
                    // AddRmsNormEmitQ8/GluActivateEmitQ8 in the same launch as the F32 output — identical
                    // bytes to the quantize kernel below) lets this call skip its own quantize launch.
                    if (m == 1 && GpuTransferHelper.TryGetSidecar(input, k, out ulong scXq, out ulong scXd, out ulong scXs))
                    {
                        if (weight.DType == DType.Q4_K)
                            _kernels!.LaunchMulMatVecQ4KQ8_1(pOutput, scXq, scXd, scXs, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q5_K)
                            _kernels!.LaunchMulMatVecQ5KQ8_1(pOutput, scXq, scXd, scXs, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q6_K)
                            _kernels!.LaunchMulMatVecQ6KQ8_1(pOutput, scXq, scXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q4_0)
                            _kernels!.LaunchMulMatVecQ4_0Q8_1(pOutput, scXq, scXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q5_0)
                            _kernels!.LaunchMulMatVecQ5_0Q8_1(pOutput, scXq, scXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else
                            _kernels!.LaunchMulMatVecQ8_0Q8_1(pOutput, scXq, scXd, pWeight, pBias, n, k, m, _stream.Handle);
                        GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                        cachedOutput = true;
                        return;
                    }
                    int kblocks = m * (k / 32);
                    // One combined [xq | xd | xs] buffer (256-aligned sections) from the persistent scratch —
                    // zero allocation/free nodes per call. Transient per-call buffers only if the scratch would
                    // have to grow mid-capture (see EnsureDp4aScratch).
                    nuint xqBytes = (nuint)(((long)m * k + 255) & ~255L);
                    nuint xdBytes = (nuint)(((long)kblocks * sizeof(float) + 255) & ~255L);
                    ulong scratch = EnsureDp4aScratch(xqBytes + 2 * xdBytes);
                    bool transient = scratch == 0;
                    ulong pXq = transient ? GpuTransferHelper.AllocateDevice((nuint)((long)m * k)) : scratch;
                    ulong pXd = transient ? GpuTransferHelper.AllocateDevice((nuint)((long)kblocks * sizeof(float))) : scratch + xqBytes;
                    ulong pXs = transient ? GpuTransferHelper.AllocateDevice((nuint)((long)kblocks * sizeof(float))) : scratch + xqBytes + xdBytes;
                    try
                    {
                        _kernels!.LaunchQuantizeActivationQ8_1(pXq, pXd, pXs, pInput, m, k, _stream.Handle);
                        if (weight.DType == DType.Q4_K)
                            _kernels!.LaunchMulMatVecQ4KQ8_1(pOutput, pXq, pXd, pXs, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q5_K)
                            _kernels!.LaunchMulMatVecQ5KQ8_1(pOutput, pXq, pXd, pXs, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q6_K)
                            _kernels!.LaunchMulMatVecQ6KQ8_1(pOutput, pXq, pXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q4_0)
                            _kernels!.LaunchMulMatVecQ4_0Q8_1(pOutput, pXq, pXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else if (weight.DType == DType.Q5_0)
                            _kernels!.LaunchMulMatVecQ5_0Q8_1(pOutput, pXq, pXd, pWeight, pBias, n, k, m, _stream.Handle);
                        else
                            _kernels!.LaunchMulMatVecQ8_0Q8_1(pOutput, pXq, pXd, pWeight, pBias, n, k, m, _stream.Handle);
                    }
                    finally
                    {
                        if (transient)
                        {
                            GpuTransferHelper.FreeDevice(pXq);
                            GpuTransferHelper.FreeDevice(pXd);
                            GpuTransferHelper.FreeDevice(pXs);
                        }
                    }
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q4_K && k % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ4KF32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q6_K && k % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ6KF32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q8_0 && k % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ8_0F32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Q5_0: llama.cpp's fallback quant (in Q4_K_M-style mixed schemes) for any tensor whose k isn't
                // a multiple of 256 — very common on odd hidden sizes (e.g. qwen2.5-0.5b's 896). Without this
                // branch those tensors missed every fused GEMV path and fell back to the ~10-20× slower
                // dequant-to-F16-then-cuBLAS route (measured: qwen2.5-0.5b decode was ~2.7× slower than its own
                // Q8_0 quant of the identical model, because nearly every projection is k=896 → Q5_0).
                if (weight.DType == DType.Q5_0 && k % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ5_0F32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Q4_0: common legacy/baseline GGUF quant. Previously missed every fused GEMV path and fell
                // back to the ~10-20× slower dequant-to-F16-then-cuBLAS route.
                if (weight.DType == DType.Q4_0 && k % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ4_0F32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q5_K && k % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ5KF32(pOutput, pInput, pWeight, pBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Dense 16-bit-float weights (BF16/F16 checkpoints, e.g. Orpheus and most audio LMs). cuBLAS
                // GemmEx is inefficient at m=1; the fused GEMV reads each weight row once with an F32 accumulate
                // (activation stays F32 — at least as accurate as the cuBLAS BF16 cast). On by default; set
                // HARTSY_BF16_GEMV=0 to fall back to cuBLAS.
                if (EnableBf16Gemv && _kernels!.HasFloatGemv && (weight.DType == DType.BF16 || weight.DType == DType.F16))
                {
                    // The fused GEMV kernels take an F32 bias pointer. Checkpoints that keep their small
                    // linears in 16-bit float ship the bias in the SAME dtype (Krea2 fp8_scaled: BF16
                    // time-embed MLP + modulation projections) — passing it raw reinterprets two 16-bit
                    // halves as one float and the timestep conditioning explodes → black frames. Transient
                    // F32 cast; F32 biases (the common LLM case) pass straight through.
                    ulong gemvBias = bias is null ? 0
                        : CastIfNeeded(pBias, bias.DType, DType.F32, (int)bias.ElementCount, out pBiasCast);
                    if (weight.DType == DType.BF16)
                        _kernels!.LaunchMulMatVecBf16F32(pOutput, pInput, pWeight, gemvBias, n, k, m, _stream.Handle);
                    else
                        _kernels!.LaunchMulMatVecF16F32(pOutput, pInput, pWeight, gemvBias, n, k, m, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
            }

            // For ComfyUI fp8_scaled checkpoints, every FP8 weight has a per-tensor scalar scale.
            // We store it on the Tensor itself; folding it into cuBLAS' alpha applies the scaling
            // for free during the GEMM (no extra kernel launch). Default Fp8ScaleFactor is 1.0.
            float alpha = weight.Fp8ScaleFactor;
            float beta = 0.0f;

            // Native FP8 GEMM path (Ada/Hopper, opt-in). The fp8 weight is consumed directly by fp8 tensor cores
            // (no per-call fp8→F16 weight cast — the dominant fp8-fallback cost) and an F32 activation is
            // quantized transiently to e4m3 with a per-tensor dynamic scale (absmax → x·448/amax), consumed by
            // the GEMM as a device-side B_SCALE_POINTER so the whole chain stays async. Alignment: cuBLASLt fp8
            // needs 16-byte leading dims (lda/ldb = k fp8 bytes; ldc = n · output element bytes); non-conforming
            // shapes fall through to the cast-to-F16 path below.
            if (EnableNativeFp8Gemm && weight.DType.IsFp8 && Fp8Executor.IsSupported
                && (input.DType.IsFp8 || input.DType == DType.F32 || input.DType == DType.F16)
                && (output.DType == DType.F16 || output.DType == DType.F32) && k % 16 == 0
                && (n * output.DType.SizeInBytes) % 16 == 0)
            {
                ulong inputFp8Ptr;
                ulong inputScaleDev = 0;
                if (input.DType.IsFp8)
                {
                    inputFp8Ptr = pInput;
                    alpha *= input.Fp8ScaleFactor;   // pre-quantized caller input: fold its per-tensor scale too
                }
                else
                {
                    // F32 or F16 activation → per-tensor dynamic e4m3 quantization (absmax → x·448/amax), weights
                    // stay PACKED fp8. The F16 branch is what keeps the DiT F16-activation path on the native fp8
                    // GEMM — without it, F16 input falls through to the cuBLAS path, which (with CacheWeightCasts
                    // off, the fp8 VRAM recipe) re-casts the whole fp8 weight set to F16 every step (Axis-B).
                    int count = (int)input.ElementCount;
                    pInputFp8 = CudaMemory.Allocate((nuint)count);
                    // A checkpoint-supplied static activation scale replaces the absmax pass entirely: two of the
                    // three launches go away, and with them a full extra read of the activation AND a grid-wide
                    // reduction the quantize kernel had to wait on before it could start. Measured on MiniMax-H3 at
                    // ~188 ms/step of quantization tax across 200 Linears. Falls back per-weight, so a Linear whose
                    // file ships no `.input_scale` (H3's mlp.fc2) still quantizes dynamically.
                    ulong staticScaleDev = EnableStaticFp8InputScale ? EnsureFp8InputScaleDev(weight) : 0;
                    if (staticScaleDev != 0)
                    {
                        if (input.DType == DType.F16)
                            _kernels!.LaunchFp8QuantF16ToE4M3(pInputFp8, pInput, staticScaleDev, count, _stream.Handle);
                        else
                            _kernels!.LaunchFp8QuantF32ToE4M3(pInputFp8, pInput, staticScaleDev, count, _stream.Handle);
                        inputFp8Ptr = pInputFp8;
                        inputScaleDev = staticScaleDev;
                    }
                    else
                    {
                        // Scratch layout: [0] = dequant scale (amax/448), [1..] = per-block maxes.
                        pFp8Scratch = CudaMemory.Allocate((nuint)((CudaKernels.Fp8AbsMaxBlockCount(count) + 1) * sizeof(float)));
                        if (input.DType == DType.F16)
                        {
                            _kernels!.LaunchFp8AbsMaxScaleF16(pInput, pFp8Scratch + sizeof(float), pFp8Scratch, count, _stream.Handle);
                            _kernels!.LaunchFp8QuantF16ToE4M3(pInputFp8, pInput, pFp8Scratch, count, _stream.Handle);
                        }
                        else
                        {
                            _kernels!.LaunchFp8AbsMaxScale(pInput, pFp8Scratch + sizeof(float), pFp8Scratch, count, _stream.Handle);
                            _kernels!.LaunchFp8QuantF32ToE4M3(pInputFp8, pInput, pFp8Scratch, count, _stream.Handle);
                        }
                        inputFp8Ptr = pInputFp8;
                        inputScaleDev = pFp8Scratch;
                    }
                }

                // This path consumes the PACKED fp8 weight directly and returns before the shared cast-resolution
                // offset below, so a row range has to be applied here too — fp8 is 1 byte per element, and the
                // rows of a [outDim, inDim] weight are contiguous.
                ulong fp8WeightPtr = rowRange ? pWeight + (ulong)((long)weightRowOffset * k * weight.DType.SizeInBytes)
                    : pWeight;
                Fp8Executor.Run(weight: fp8WeightPtr, input: inputFp8Ptr, outPtr: pOutput, m: m, n: n, k: k,
                    weightScale: alpha, stream: _stream.Handle,
                    inputScaleDev: inputScaleDev, outF32: output.DType == DType.F32);

                if (bias is not null)
                {
                    int totalElementsFp8 = m * n;
                    ulong biasPtr = rowRange ? pBias + (ulong)((long)weightRowOffset * bias!.DType.SizeInBytes) : pBias;
                    if (output.DType != bias!.DType)
                    {
                        pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                        CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                        biasPtr = pBiasCast;
                    }
                    if (output.DType == DType.F32)
                        _kernels!.LaunchBiasAdd(pOutput, biasPtr, n, 1, totalElementsFp8, _stream.Handle);
                    else
                        _kernels!.LaunchBiasAddF16(pOutput, biasPtr, n, 1, totalElementsFp8, _stream.Handle);
                }
                GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                cachedOutput = true;
                return;
            }

            // W8A8 IMMA path (HARTSY_W8A8=1, INFERENCE_ACCEL_GRIND §H5): large-m Linear with a 16-bit-float
            // (or F32) weight → per-channel int8 weight (host-quantized ONCE, persistent-cached with its
            // F32 wScale[n]) × per-row dynamic-int8 activation on the INT8 tensor cores, then the w8a8.ptx
            // dequant+bias epilogue. The Ampere lever: SM 8.6 has no fp8 MMA; measured chain 2.57× over the
            // F16 GEMM at relL2 5.5e-3 (W8A8ImmaGemmTests, 3060). m≥32 keeps the decode GEMV paths above
            // untouched; k%4/n%4 are the cuBLASLt int8 TN lda/ldc requirements. The weight's Fp8ScaleFactor
            // (the branch-damp/alpha carrier on 16-bit weights) folds into the cached wScale at quant time.
            if (w8a8)
            {
                ulong w8Combined = _w8a8WeightCache[weight];
                ulong wScaleDev = w8Combined + (ulong)W8A8ScaleOffset(n, k);

                ulong pAct8 = 0, pRowScale = 0, pOut32 = 0;
                try
                {
                    pAct8 = GpuTransferHelper.AllocateDevice((nuint)((long)m * k));
                    pRowScale = GpuTransferHelper.AllocateDevice((nuint)((long)m * sizeof(float)));
                    pOut32 = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * sizeof(int)));

                    ulong invScale = _w8a8SmoothInvScaleDevice.TryGetValue(weight, out ulong isv) ? isv : 0;
                    _kernels!.LaunchW8A8QuantRowwise(pAct8, pRowScale, pInput, m, k, _stream.Handle,
                        srcF16: input.DType == DType.F16, invScale: invScale);
                    Int8Gemm.Run(w8Combined, pAct8, pOut32, m, n, k, _stream.Handle);
                    // Bias rides the dequant epilogue as F32 (transient cast for 16-bit checkpoint biases).
                    ulong biasF32 = 0;
                    if (bias is not null)
                        biasF32 = CastIfNeeded(pBias, bias.DType, DType.F32, (int)bias.ElementCount, out pBiasCast);
                    _kernels!.LaunchW8A8DequantBias(pOutput, pOut32, pRowScale, wScaleDev, biasF32,
                        m, n, _stream.Handle, outF16: output.DType == DType.F16);
                }
                finally
                {
                    if (pAct8 != 0) GpuTransferHelper.FreeDevice(pAct8);
                    if (pRowScale != 0) GpuTransferHelper.FreeDevice(pRowScale);
                    if (pOut32 != 0) GpuTransferHelper.FreeDevice(pOut32);
                }
                GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                cachedOutput = true;
                return;
            }

            // Resident int8 (ComfyUI `int8_tensorwise`, ± `convrot`) — how the official LTX 2.5 and MiniMax-H3
            // quantized releases are shipped. Same IMMA chain as the W8A8 branch above, with two differences: the
            // weight arrives ALREADY quantized (its per-output-row scale comes from the file rather than a host
            // quant pass), so a 22B DiT stays at 1 byte/param instead of expanding to BF16; and when the quantizer
            // rotated the weight (W @ Hᵀ) the activation owes an x @ H first, which is what makes the product come
            // back out as x·Wᵀ — H is its own inverse. See RunResidentInt8 for the chain itself.
            if (int8Resident)
            {
                Int8ResidentTarget target = new(pWeight, pOutput, bias is null ? 0
                        : CastIfNeeded(pBias, bias!.DType, DType.F32, (int)bias.ElementCount, out pBiasCast),
                    int8RowScaleDev, n, output.DType == DType.F16, fuseGelu);
                Int8PreGate gate = preGate is null ? default
                    : new(GpuTransferHelper.CopyToDevice(preGate), preGateHeads, preGateHeadDim);
                try
                {
                    RunResidentInt8(pInput, input.DType, m, k, weight.QuantInfo!.ConvRotGroupSize, [target], gate);
                }
                finally
                {
                    if (gate.Logits != 0) GpuTransferHelper.FreeDevice(gate.Logits);
                }
                GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                cachedOutput = true;
                return;
            }

            // A pre-quantized fp8 caller input (Modulate-emitted e4m3, stored value = real/scale) is dequantized
            // SCALE-BLIND by CastIfNeeded below — fold its per-tensor scale into alpha exactly like the native
            // fp8 branch above (and Conv2D) do. Without this, every GEMM consuming such an input is off by that
            // factor on any card that can't take the native path (SM < 8.9) — the root cause of the MiniMax-H3
            // cross-device sharded mosaic (3060 tail blocks; confirmed by experiment 2026-08-06: disabling the
            // fp8 emit raised sharded SSIM 0.17 → 0.98).
            if (input.DType.IsFp8)
            {
                alpha *= input.Fp8ScaleFactor;
            }

            // Joint resolution: when fp8 is in play we run the whole GEMM in F16, casting the
            // F32 activation down too. Old behaviour resolved per-operand and ended up at F32,
            // forcing an oversized weight cast (151 MB for proj_mlp) plus a 75 MB intermediate
            // inside CastOnGpu's F8→F32 path.
            DType gemmDtype = ResolveGemmDtype(input.DType, weight.DType);
            ulong inputPtr = CastIfNeeded(pInput, input.DType, gemmDtype, (int)input.ElementCount, out pInputCast);

            // Weight cast (fp8/quant → BF16/F16): identical every forward, so for a preloaded weight we
            // compute it once and cache it. Re-casting the whole 9.3B weight set per Linear per step was the
            // dominant cost on the fp8 path (GEMMs themselves are tensor-core). pWeightCast stays 0 when the
            // cached copy is used so the finally block won't free the persistent cast.
            // HighPrecisionGemm upcasts a bf16 weight to F32 for the GEMM. Caching that cast would keep a full
            // F32 copy of every weight resident (≈2× the bf16 footprint — 14 GB for a 3.5B DiT, OOM on a 12 GB
            // 3060). Force the transient path so only one weight is F32 at a time, freed each call.
            bool hipUpcast = HighPrecisionGemm && weight.DType == DType.BF16 && gemmDtype == DType.F32;
            ulong weightPtr;
            if (weight.DType == gemmDtype)
            {
                weightPtr = pWeight;
            }
            else if (cacheWeightCast && CacheWeightCasts && !hipUpcast && GpuTransferHelper.IsWeightCached(weight))
            {
                if (!GpuTransferHelper.TryGetWeightCast(weight, out weightPtr))
                {
                    nuint castBytes = (nuint)(weight.ElementCount * gemmDtype.SizeInBytes);
                    // Budget gate: an unbounded cast cache keeps an F16 copy of EVERY preloaded quant weight —
                    // for a 12B fp8 DiT that's ~24 GB on top of the 12 GB fp8 originals, guaranteed OOM mid-first-
                    // forward on a 24 GB card (this was the Flux-fp8 eager-preload OOM). If caching this cast would
                    // leave less headroom than activations/transients need, cast transiently instead: correct output,
                    // per-call dequant cost, bounded memory. freeBytes counts pool reservations as used, so the gate
                    // errs conservative under transient churn — the safe direction.
                    (long freeBytes, long totalBytes) = CudaMemory.GetMemInfo();
                    // Quantized (GGUF) weights get a STRICTER floor: their cast sets can dwarf free VRAM
                    // (a 4B model's BF16 casts are ~8 GB), and prefill still needs large transients (head
                    // cast, activations, cuBLAS workspace) AFTER the cache stops growing — a 2 GB floor
                    // OOMed Qwen3-4B. 4 GB caches the hottest few GB of casts and never starves transients.
                    long headroom = weight.DType.IsQuantized ? Math.Max(4L << 30, totalBytes / 3)
                        : Math.Max(2L << 30, totalBytes / 8);
                    if (freeBytes > 0 && freeBytes - (long)castBytes < headroom)
                    {
                        System.Threading.Interlocked.Increment(ref _castTransientGated);
                        weightPtr = MaterializeWeightIfNeeded(pWeight, weight, gemmDtype, out pWeightCast, nvfp4Scales);
                    }
                    else
                    {
                        System.Threading.Interlocked.Increment(ref _castCachedNew);
                        weightPtr = GpuTransferHelper.AllocateDevice(castBytes);
                        MaterializeWeight(weightPtr, pWeight, weight, gemmDtype, nvfp4Scales);
                        GpuTransferHelper.CacheWeightCast(weight, weightPtr, castBytes);
                    }
                }
            }
            else
            {
                if (weight.DType.IsQuantized || weight.DType == DType.F8E4M3 || weight.DType == DType.BF16)
                {
                    System.Threading.Interlocked.Increment(ref _castTransientUncachedPath);
                    if (System.Threading.Interlocked.Increment(ref _castWhyLogged) % 200 == 1)
                        HartsyInference.Core.Logging.Logs.Info(
                            $"[Cuda] cast-transient-why: cacheParam={cacheWeightCast} propCache={CacheWeightCasts} " +
                            $"hip={hipUpcast} weightCached={GpuTransferHelper.IsWeightCached(weight)} " +
                            $"dtype={weight.DType} gemmDtype={gemmDtype} M={m} N={n} K={k}");
                }
                weightPtr = MaterializeWeightIfNeeded(pWeight, weight, gemmDtype, out pWeightCast, nvfp4Scales);
            }

            // Every branch above leaves weightPtr addressing gemmDtype elements over the WHOLE weight — casts are
            // deliberately computed and cached full-size so sibling row ranges share one cast — so the range is a
            // single offset applied here, after cast resolution rather than inside each branch.
            if (rowRange)
            {
                weightPtr += (ulong)((long)weightRowOffset * k * gemmDtype.SizeInBytes);
            }

            int gemmType = CublasDataTypeForGemm(gemmDtype, input.DType, weight.DType, output.DType, m, n, k);
            int outputType = CublasDataTypeForGemm(output.DType, input.DType, weight.DType, output.DType, m, n, k);

            // Bias prep shared by the fused-epilogue path and the separate-add fallback:
            // both want one bias value per output channel n, in the output dtype.
            ulong biasDevicePtr = 0;
            if (bias is not null)
            {
                // Bias is one value per output channel, so it slices with the same row range (guarded above to a
                // dtype that needs no cast, since casting would rebase off the full-length bias).
                biasDevicePtr = rowRange ? pBias + (ulong)((long)weightRowOffset * bias.DType.SizeInBytes) : pBias;
                if (output.DType != bias!.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasDevicePtr = pBiasCast;
                }
            }

            // A 16-bit output tensor that isn't the operand dtype can't be written by the GEMM directly (see the
            // cuBLAS branch below for the full reasoning); it needs a gemmDtype temp plus a cast. That applies to
            // cuBLASLt too: F32 operands writing BF16/F16 through the fused-bias epilogue returned ~2.6e38.
            bool outNeedsCast = output.DType != gemmDtype && output.DType != DType.F32;

            // Tensor-core HGEMM path (opt-in, validation-pending): F16 operands+output,
            // aligned dims. Produces the GEMM only; bias is added by the block below.
            bool ltBiasFused = false;
            if (EnableTensorCoreGemm && gemmDtype == DType.F16 && output.DType == DType.F16
                && TensorCoreGemm.IsSupported && Cuda.TensorCoreGemm.IsAligned(m, n, k))
            {
                TensorCoreGemm.Run(a: inputPtr, b: weightPtr, c: pOutput, m: m, n: n, k: k, alpha: alpha, stream: _stream.Handle);
            }
            else
            {
                // Fused path: fold the bias into the cuBLASLt epilogue, saving a BiasAdd launch plus an
                // output-sized HBM round-trip. Compute32F is the single precision-policy source for BOTH the
                // fused and GemmEx paths: HighPrecisionGemm/HARTSY_NO_TF32/HARTSY_GEMM_F16 must not change
                // semantics merely because this Linear has a bias. A missing Lt algorithm is an ordinary
                // per-shape capability result, so TryRun returns false and the existing GemmEx+BiasAdd path runs.
                if (EnableEpilogueFusion && bias is not null && !outNeedsCast)
                {
                    LtGemmExecutor lt = LtGemm;
                    if (lt.IsSupported)
                    {
                        ltBiasFused = lt.TryRun(
                            weight: weightPtr, input: inputPtr, outPtr: pOutput,
                            m: m, n: n, k: k, alpha: alpha,
                            abType: gemmType, dType: outputType, computeType: Compute32F(gemmType),
                            biasPtr: biasDevicePtr, epilogue: CublasLtApi.CUBLASLT_EPILOGUE_BIAS,
                            stream: _stream.Handle);
                    }
                }

                if (!ltBiasFused)
                {
                    // cuBLAS col-major: C_cm = op(A) × op(B) where op(A)=weight^T [n,k], op(B)=input [k,m]
                    // Row-major interpretation: output[m,n] = input[m,k] × weight^T[k,n].
                    //
                    // cublasGemmEx supports Ctype ∈ {Atype, F32} only. When the output tensor is a 16-bit type
                    // that differs from the operand gemmDtype — e.g. BF16 operands (fp8/quant weight × F32
                    // activation → BF16, chosen so the F32→16-bit cast can't overflow F16's 65504 in a SwiGLU
                    // MLP) but an F16 output tensor — there is NO BF16→F16 kernel and cuBLAS returns
                    // CUBLAS_STATUS_NOT_SUPPORTED. Run the GEMM into a temp of gemmDtype, then cast to the real
                    // output. (F32 output is always compatible with 16-bit operands, so it skips the temp.)
                    ulong gemmOut = pOutput;
                    int gemmOutType = outputType;
                    ulong pGemmTemp = 0;
                    try
                    {
                        if (outNeedsCast)
                        {
                            pGemmTemp = GpuTransferHelper.AllocateDevice((nuint)((long)m * n * gemmDtype.SizeInBytes));
                            gemmOut = pGemmTemp;
                            gemmOutType = gemmType;
                        }
                        CublasApi.cublasGemmEx(
                            _cublasHandle,
                            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            weightPtr, gemmType, k,
                            inputPtr, gemmType, k,
                            &beta,
                            gemmOut, gemmOutType, n,
                            Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                        if (outNeedsCast)
                            CastOnGpu(pOutput, pGemmTemp, gemmDtype, output.DType, m * n);
                    }
                    finally
                    {
                        if (pGemmTemp != 0) GpuTransferHelper.FreeDevice(pGemmTemp);
                    }
                }
            }

            // Bias add for every GEMM path except the cuBLASLt epilogue, which already fused it.
            if (bias is not null && !ltBiasFused)
            {
                int totalElements = m * n;
                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(pOutput, biasDevicePtr, n, 1, totalElements, _stream.Handle);
                else if (output.DType == DType.BF16)
                    _kernels!.LaunchBiasAddBf16(pOutput, biasDevicePtr, n, 1, totalElements, _stream.Handle);
                else
                    _kernels!.LaunchBiasAdd(pOutput, biasDevicePtr, n, 1, totalElements, _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            if (pInputCast != 0) CudaMemory.FreeAsync(pInputCast, _stream.Handle);
            if (pWeightCast != 0) CudaMemory.FreeAsync(pWeightCast, _stream.Handle);
            if (pBiasCast != 0) CudaMemory.FreeAsync(pBiasCast, _stream.Handle);
            if (pInputFp8 != 0) CudaMemory.FreeAsync(pInputFp8, _stream.Handle);
            if (pFp8Scratch != 0) CudaMemory.FreeAsync(pFp8Scratch, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
        }
    }

    /// <summary>Batched matrix multiply via cuBLAS strided batched GEMM. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void BatchedMatMul(Tensor output, Tensor a, Tensor b)
    {
        using NvtxRange _nvtx = NvtxRange.Push("BatchedMatMul");
        EnterOp();
        EnsureKernels();

        long batchSize = a.Shape[0];
        int m = (int)a.Shape[1];
        int k = (int)a.Shape[2];

        bool bIs2D = b.Shape.Rank == 2;
        int n = bIs2D ? (int)b.Shape[1] : (int)b.Shape[2];

        long strideA = m * k;
        long strideB = bIs2D ? 0 : k * n;
        long strideC = m * n;

        ulong pA = 0, pB = 0, pC = 0, pACast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pC = GpuTransferHelper.AllocateDevice(outBytes);

            // fp8_scaled operands: fold per-tensor scale(s) into cuBLAS alpha (see MatMul above). Defaults 1.0.
            float alpha = a.Fp8ScaleFactor * b.Fp8ScaleFactor;
            float beta = 0.0f;

            // Joint dtype resolution — see ResolveGemmDtype(a, b) docs.
            DType gemmDtype = ResolveGemmDtype(a.DType, b.DType);
            ulong aPtr = CastIfNeeded(pA, a.DType, gemmDtype, (int)a.ElementCount, out pACast);
            ulong bPtr = CastIfNeeded(pB, b.DType, gemmDtype, (int)b.ElementCount, out pBCast);

            int gemmType = CublasDataType(gemmDtype);
            int cType = output.DType == DType.F16 ? CublasApi.CUDA_R_16F : CublasApi.CUDA_R_32F;

            CublasApi.cublasGemmStridedBatchedEx(
                _cublasHandle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                n, m, k,
                &alpha,
                bPtr, gemmType, n, strideB,
                aPtr, gemmType, k, strideA,
                &beta,
                pC, cType, n, strideC,
                (int)batchSize,
                Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

            GpuTransferHelper.CacheActivation(output, pC, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (pACast != 0) CudaMemory.FreeAsync(pACast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pC);
        }
    }

    #endregion

    #region Convolution

    /// <summary>2D convolution via im2col + cuBLAS SGEMM. Supports arbitrary stride, padding, and kernel sizes.</summary>
    public unsafe void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW)
    {
        // Seamless tiling (Tier 3.6): neither the cuDNN graph path nor the im2col kernels below take a padding
        // *mode* — padH/padW is always zero-fill. Rather than touch either path, pre-materialize a wrapped-edge
        // copy of the input at the request pad size and recurse with pad=0; the recursive call's own output-shape
        // arithmetic ((inH+2*padH-kH)/stride+1) already matches `output`'s shape since the wrap adds the same
        // 2*padH/2*padW the caller asked for. Runs on every conv while the flag is set (UNet AND VAE, since both
        // route through this one method) — a decode-only wrap would leave a seamless latent that decodes with a
        // seam.
        if ((SeamlessTilingX || SeamlessTilingY) && (padH > 0 || padW > 0))
        {
            using Tensor wrapped = WrapPadForSeamlessTiling(input, padH, padW, wrapH: SeamlessTilingY, wrapW: SeamlessTilingX);
            Conv2D(output, wrapped, weight, bias, strideH, strideW, 0, 0);
            return;
        }

        using NvtxRange _nvtx = NvtxRange.Push("Conv2D");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int inCh = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];

        int outCh = (int)weight.Shape[0];
        int kH = (int)weight.Shape[2];
        int kW = (int)weight.Shape[3];

        int outH = (inH + 2 * padH - kH) / strideH + 1;
        int outW = (inW + 2 * padW - kW) / strideW + 1;

        // cuDNN conv-forward fast path: F16/BF16 same-dtype conv with no fp8 alpha folding. Skips the
        // im2col materialization entirely (tensor-core implicit-GEMM/Winograd engines) — the im2col
        // matrix is a kH·kW× input-sized HBM write+read per conv, the dominant conv cost on the SDXL
        // UNet (~50 convs/step) and VAE. Anything else (F32, fp8/quant weights, scale factors) keeps
        // the im2col path; any cuDNN failure self-disables the route for the session.
        if (_convCudnn && !_cudnnConvDead && input.DType == weight.DType && output.DType == input.DType
            && (input.DType == DType.F16 || input.DType == DType.BF16)
            && input.Fp8ScaleFactor == 1.0f && weight.Fp8ScaleFactor == 1.0f
            && TryCudnnConv(output, input, weight, bias, batch, inCh, inH, inW, outCh, kH, kW, outH, outW, strideH, strideW, padH, padW))
        {
            return;
        }

        int colRows = inCh * kH * kW;
        int colCols = outH * outW;

        bool is1x1 = kH == 1 && kW == 1 && strideH == 1 && strideW == 1 && padH == 0 && padW == 0;
        // Joint dtype resolution — fp8 on either side forces a 16-bit GEMM. The im2col
        // buffer matches the GEMM dtype so element size has to be derived from gemmDtype,
        // not the original input dtype.
        DType gemmDtype = ResolveGemmDtype(input.DType, weight.DType);
        int elemSize = gemmDtype.SizeInBytes;
        int outElemSize = output.DType.SizeInBytes;

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, colBuf = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        bool cachedOutput = false;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);

            ulong inputPtr = CastIfNeeded(pInput, input.DType, gemmDtype, (int)input.ElementCount, out pInputCast);
            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            // Banded im2col: cap the col workspace so one huge conv (512-ch 3x3 at 1024² = 9.2 GB) never
            // needs a single allocation that can't fit next to resident model weights. Each band feeds the
            // same GEMM with the output pointer offset to the band's rows and ldc = full outH·outW, so the
            // result is bit-identical to the unbanded call. bandRows is sized to the cap; 0 bands = unbanded.
            long colBytesFull = (long)colRows * colCols * elemSize;
            int bandRows = 0;
            if (!is1x1 && colBytesFull > Im2ColBandCapBytes)
            {
                bandRows = Math.Max(1, (int)(Im2ColBandCapBytes / ((long)colRows * outW * elemSize)));
                colBuf = CudaMemory.Allocate((nuint)((long)colRows * bandRows * outW * elemSize));
            }
            else if (!is1x1)
            {
                colBuf = CudaMemory.Allocate((nuint)colBytesFull);
            }

            // fp8_scaled conv weights: fold the per-tensor scale into the GEMM alpha (see MatMul). Defaults 1.0.
            float alpha = input.Fp8ScaleFactor * weight.Fp8ScaleFactor;
            float beta = 0.0f;

            ulong weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);

            int gemmType = CublasDataType(gemmDtype);
            int gemmOutType = CublasDataType(output.DType);

            for (int b = 0; b < batch; b++)
            {
                int inputBatchOffset = b * inCh;
                ulong outBatchPtr = pOutput + (ulong)((long)b * outCh * outH * outW * outElemSize);

                if (bandRows > 0)
                {
                    for (int ohBase = 0; ohBase < outH; ohBase += bandRows)
                    {
                        int rowsThis = Math.Min(bandRows, outH - ohBase);
                        int bandCols = rowsThis * outW;
                        _kernels!.LaunchIm2ColBanded(gemmDtype, colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outW, ohBase, rowsThis, inputBatchOffset,
                            _stream.Handle);
                        // C for this band = the band's rows within every output-channel plane: pointer offset
                        // ohBase·outW, ldc = full colCols so channel strides match the unbanded layout.
                        ulong outBandPtr = outBatchPtr + (ulong)((long)ohBase * outW * outElemSize);
                        CublasApi.cublasGemmEx(
                            _cublasHandle,
                            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                            bandCols, outCh, colRows,
                            &alpha,
                            colBuf, gemmType, bandCols,
                            weightPtr, gemmType, colRows,
                            &beta,
                            outBandPtr, gemmOutType, colCols,
                            Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                    }
                    continue;
                }

                ulong colPtr;
                if (is1x1)
                {
                    colPtr = inputPtr + (ulong)((long)b * inCh * inH * inW * elemSize);
                }
                else
                {
                    if (gemmDtype == DType.F16)
                        _kernels!.LaunchIm2ColF16(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    else if (gemmDtype == DType.BF16)
                        _kernels!.LaunchIm2ColBf16(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    else
                        _kernels!.LaunchIm2Col(
                            colBuf, inputPtr,
                            inCh, inH, inW, kH, kW,
                            padH, padW, strideH, strideW,
                            outH, outW, inputBatchOffset,
                            _stream.Handle);
                    colPtr = colBuf;
                }

                CublasApi.cublasGemmEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    colCols, outCh, colRows,
                    &alpha,
                    colPtr, gemmType, colCols,
                    weightPtr, gemmType, colRows,
                    &beta,
                    outBatchPtr, gemmOutType, colCols,
                    Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            // Add bias (cast if dtype mismatch)
            if (bias is not null)
            {
                int totalElements = batch * outCh * outH * outW;
                ulong biasPtr = pBias;

                if (output.DType != bias!.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasPtr = pBiasCast;
                }

                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
                else if (output.DType == DType.BF16)
                    _kernels!.LaunchBiasAddBf16(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
                else
                    _kernels!.LaunchBiasAdd(
                        pOutput, biasPtr,
                        outCh, outH * outW, totalElements,
                        _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            if (pInputCast != 0) CudaMemory.FreeAsync(pInputCast, _stream.Handle);
            if (pWeightCast != 0) CudaMemory.FreeAsync(pWeightCast, _stream.Handle);
            if (pBiasCast != 0) CudaMemory.FreeAsync(pBiasCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
            if (colBuf != 0) CudaMemory.FreeAsync(colBuf, _stream.Handle);
        }
    }

    /// <summary>Builds a [B,C,inH+2*padH,inW+2*padW] host copy of <paramref name="input"/> whose border comes from the opposite edge of the same tensor (circular/wrap padding) instead of zeros — the source data for <see cref="SeamlessTilingX"/>/<see cref="SeamlessTilingY"/>. <paramref name="wrapH"/>/<paramref name="wrapW"/> select which axes actually wrap (SwarmUI core's "X-Only"/"Y-Only"/"true" modes); the other axis's border is left at its allocated zero (<see cref="Tensor"/>'s lazy host buffer is zeroed on first touch), i.e. an ordinary zero-pad — same as passing the request straight through unset. Assumes <c>padH &lt;= inH</c> and <c>padW &lt;= inW</c>, true for every real conv kernel pad (1-3px) against any image/latent dimension. dtype-generic (raw byte copies keyed off <see cref="DType.SizeInBytes"/>) since UNet convs run F16/BF16 while VAE convs stay F32.</summary>
    private static unsafe Tensor WrapPadForSeamlessTiling(Tensor input, int padH, int padW, bool wrapH, bool wrapW)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int outH = inH + 2 * padH;
        int outW = inW + 2 * padW;
        int elemSize = input.DType.SizeInBytes;

        Tensor padded = new Tensor(new TensorShape(batch, channels, outH, outW), input.DType);
        byte* src = (byte*)input.DataPointer;
        byte* dst = (byte*)padded.DataPointer; // zeroed on first touch above — the fallback for a non-wrapped axis's border.

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                long srcPlane = ((long)b * channels + c) * inH * inW;
                long dstPlane = ((long)b * channels + c) * outH * outW;
                for (int oh = 0; oh < outH; oh++)
                {
                    bool rowInBounds = oh >= padH && oh < padH + inH;
                    if (!rowInBounds && !wrapH)
                    {
                        continue; // top/bottom border stays zeroed — Y axis not tiling.
                    }
                    int ih = wrapH ? ((oh - padH) % inH + inH) % inH : oh - padH;
                    byte* srcRow = src + (srcPlane + (long)ih * inW) * elemSize;
                    byte* dstRow = dst + (dstPlane + (long)oh * outW) * elemSize;

                    // Center: straight copy of the source row.
                    Buffer.MemoryCopy(srcRow, dstRow + (long)padW * elemSize, (long)inW * elemSize, (long)inW * elemSize);
                    if (padW > 0 && wrapW)
                    {
                        // Left border wraps from the row's right edge; right border wraps from its left edge.
                        Buffer.MemoryCopy(srcRow + (long)(inW - padW) * elemSize, dstRow, (long)padW * elemSize, (long)padW * elemSize);
                        Buffer.MemoryCopy(srcRow, dstRow + (long)(padW + inW) * elemSize, (long)padW * elemSize, (long)padW * elemSize);
                    }
                    // padW > 0 && !wrapW: left/right border stays zeroed — X axis not tiling.
                }
            }
        }

        return padded;
    }

    /// <summary>Attempts the cuDNN conv-forward route for <see cref="Conv2D"/>.</summary>
    /// <remarks>Returns false (after disabling the route for the session) on any cuDNN failure so the caller falls
    /// through to the im2col path — a rejection costs one warning, never a session kill. Bias is added by the same
    /// per-channel kernel as the im2col path, so the two routes differ only by GEMM-class accumulation order.</remarks>
    private unsafe bool TryCudnnConv(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int inCh, int inH, int inW, int outCh, int kH, int kW, int outH, int outW,
        int strideH, int strideW, int padH, int padW)
    {
        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, pBiasCast = 0;
        bool cachedOutput = false;
        try
        {
            _cudnnConv ??= new CudnnConv(_stream.Handle);

            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            int dataType = input.DType == DType.F16 ? CudnnApi.CUDNN_DATA_HALF : CudnnApi.CUDNN_DATA_BFLOAT16;
            _cudnnConv.Execute(pInput, pWeight, pOutput,
                batch, inCh, inH, inW, outCh, kH, kW, outH, outW, strideH, strideW, padH, padW, padW, dataType);

            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
                ulong biasPtr = pBias;
                if (output.DType != bias.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasPtr = pBiasCast;
                }
                int totalElements = batch * outCh * outH * outW;
                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(pOutput, biasPtr, outCh, outH * outW, totalElements, _stream.Handle);
                else
                    _kernels!.LaunchBiasAddBf16(pOutput, biasPtr, outCh, outH * outW, totalElements, _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
            cachedOutput = true;
            if (!CudnnConvEngaged)
            {
                CudnnConvEngaged = true;
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN conv] convolution-forward engine engaged (cuDNN {CudnnApi.cudnnGetVersion()})");
            }
            return true;
        }
        catch (Exception ex)
        {
            _cudnnConvDead = true;
            HartsyInference.Core.Logging.Logs.Warning($"[cuDNN conv] disabled for the session (falling back to im2col): {ex.Message}");
            return false;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            GpuTransferHelper.FreeDevice(pWeight);
            GpuTransferHelper.FreeDevice(pBias);
            if (pBiasCast != 0) CudaMemory.FreeAsync(pBiasCast, _stream.Handle);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
        }
    }

    #endregion

    #region Normalization

    /// <summary>Casts weight/bias down from F32 to <paramref name="target"/> (F16/BF16) — common for norm params in low-precision models.</summary>
    /// <returns>Pointers to use (the originals when no cast was needed) plus any allocated cast buffers, which the caller must free.</returns>
    // Any affine dtype other than the kernel's must be converted, not just F32 — an F16-checkpoint affine
    // fed raw to the BF16 GroupNorm reads as garbage scale/shift (flat-gray output; caught by the SeedVR2
    // BF16-VAE bring-up on the numz fp16 checkpoint, 2026-08-01).
    private (ulong wPtr, ulong bPtr, ulong wCast, ulong bCast) CastAffineDownIfF32(
        ulong pW, DType wDType, long wCount, ulong pB, DType bDType, long bCount, DType target)
    {
        ulong wPtr = pW, bPtr = pB, wCast = 0, bCast = 0;
        if (wDType != target)
        {
            wCast = CudaMemory.Allocate((nuint)(wCount * target.SizeInBytes));
            CastOnGpu(wCast, pW, wDType, target, (int)wCount);
            wPtr = wCast;
        }
        if (bDType != target)
        {
            bCast = CudaMemory.Allocate((nuint)(bCount * target.SizeInBytes));
            CastOnGpu(bCast, pB, bDType, target, (int)bCount);
            bPtr = bCast;
        }
        return (wPtr, bPtr, wCast, bCast);
    }

    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GroupNorm");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int spatial = 1;
        for (int d = 2; d < input.Shape.Rank; d++)
        {
            spatial *= (int)input.Shape[d];
        }

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                (ulong wPtr, ulong bPtr, pWCast, pBCast) = CastAffineDownIfF32(
                    pW, weight.DType, weight.ElementCount, pB, bias.DType, bias.ElementCount, DType.F16);
                _kernels!.LaunchGroupNormF16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else if (input.DType == DType.BF16)
            {
                (ulong wPtr, ulong bPtr, pWCast, pBCast) = CastAffineDownIfF32(
                    pW, weight.DType, weight.ElementCount, pB, bias.DType, bias.ElementCount, DType.BF16);
                _kernels!.LaunchGroupNormBf16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else
            {
                // F32 input path: the kernel reads weight/bias AS F32, so a F16/BF16 affine (common for repacked
                // VAEs, e.g. HunyuanVideo) must be upcast to F32 first — otherwise its raw bytes are misread as F32
                // (~1e38 garbage → washed-out / near-zero output). Mirrors the F16/BF16 paths' F32→low-precision cast.
                ulong wPtr = pW, bPtr = pB;
                if (weight.DType != DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 4));
                    CastOnGpu(pWCast, pW, weight.DType, DType.F32, (int)weight.ElementCount);
                    wPtr = pWCast;
                }
                if (bias.DType != DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 4));
                    CastOnGpu(pBCast, pB, bias.DType, DType.F32, (int)bias.ElementCount);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNorm(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("LayerNorm");
        EnterOp();
        EnsureKernels();

        int normDim = (int)input.Shape[input.Shape.Rank - 1];
        int totalRows = (int)(input.ElementCount / normDim);

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                (ulong wPtr, ulong bPtr, pWCast, pBCast) = CastAffineDownIfF32(
                    pW, weight.DType, weight.ElementCount, pB, bias.DType, bias.ElementCount, DType.F16);
                _kernels!.LaunchLayerNormF16(
                    pOut, pIn, wPtr, bPtr,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }
            else if (input.DType == DType.BF16)
            {
                (ulong wPtr, ulong bPtr, pWCast, pBCast) = CastAffineDownIfF32(
                    pW, weight.DType, weight.ElementCount, pB, bias.DType, bias.ElementCount, DType.BF16);
                _kernels!.LaunchLayerNormBf16(
                    pOut, pIn, wPtr, bPtr,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }
            else
            {
                // F32-input path: the kernel reads weight/bias as F32, so cast F16/BF16 affine UP to F32 first.
                // (A model cast to F16 keeps F16 norm affine; without this the F32 kernel reinterprets the F16
                // bytes → garbage affine → near-zero output. Same dtype-mismatch class as GroupNormSilu.)
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F16 || weight.DType == DType.BF16)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 4));
                    if (weight.DType == DType.F16) _kernels!.LaunchCastF16ToF32(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    else _kernels!.LaunchCastBf16ToF32(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F16 || bias.DType == DType.BF16)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 4));
                    if (bias.DType == DType.F16) _kernels!.LaunchCastF16ToF32(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    else _kernels!.LaunchCastBf16ToF32(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchLayerNorm(
                    pOut, pIn, wPtr, bPtr,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    /// <summary>Diagnostic counters for the prefill weight-cast paths (HARTSY_CAST_STATS=1 logs on read via <see cref="DumpCastStats"/>).</summary>
    /// <remarks>transient-because-budget-gate, transient-because-uncached-path (QuantizedMatMul / non-preloaded), and newly-cached casts.</remarks>
    private static long _castTransientGated, _castTransientUncachedPath, _castCachedNew, _castWhyLogged;

    /// <summary>Logs and resets the cast-path counters (called opportunistically; cheap).</summary>
    public static void DumpCastStats(string tag)
    {
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] cast-stats {tag}: transient-gated={System.Threading.Interlocked.Exchange(ref _castTransientGated, 0)} " +
            $"transient-uncachedPath={System.Threading.Interlocked.Exchange(ref _castTransientUncachedPath, 0)} " +
            $"cached-new={System.Threading.Interlocked.Exchange(ref _castCachedNew, 0)}");
    }

    private static readonly bool _quantAtProducer = EngineKnobs.QuantAtProducer.Value;

    /// <summary>Allocates the Q8_1 sidecar (xq/xd/xs) for a K-wide single row, returns the three pointers.</summary>
    private static (ulong xq, ulong xd, ulong xs) AllocSidecar(int k)
    {
        ulong xq = GpuTransferHelper.AllocateDevice((nuint)k);
        ulong xd = GpuTransferHelper.AllocateDevice((nuint)(k / 32 * sizeof(float)));
        ulong xs = GpuTransferHelper.AllocateDevice((nuint)(k / 32 * sizeof(float)));
        return (xq, xd, xs);
    }

    public unsafe void RmsNormEmitQ8(Tensor output, Tensor input, Tensor weight, float eps)
    {
        int rank = input.Shape.Rank;
        long lastDim = input.Shape[rank - 1];
        long rows = input.ElementCount / lastDim;
        // Sidecar only pays for the M=1 decode row feeding a dp4a GEMV; anything else runs the plain op.
        if (!_quantAtProducer || !EnableDp4aGemv || rows != 1 || lastDim % 32 != 0
            || input.DType != DType.F32 || weight.DType != DType.F32 || output.DType != DType.F32)
        {
            RmsNorm(output, input, weight, eps);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("RmsNormQ8");
        EnterOp();
        EnsureKernels();
        int k = (int)lastDim;
        ulong pOut = 0, pIn = 0, pWeight = 0, xq = 0, xd = 0, xs = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            (xq, xd, xs) = AllocSidecar(k);
            _kernels!.LaunchRmsNormQ8(pOut, xq, xd, xs, pIn, pWeight, k, 1, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            GpuTransferHelper.RegisterSidecar(output, xq, xd, xs, k);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput)
            {
                GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pWeight);
        }
    }

    public unsafe void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RmsNorm");
        EnterOp();
        int rank = input.Shape.Rank;
        long lastDim = input.Shape[rank - 1];
        long outerSize = input.ElementCount / lastDim;

        // GPU path: F32 activation + F32 weight. Keeps data resident — one block per row,
        // shared-mem reduction. Also serves per-head QK-RMSNorm (rows = B*L*heads, dim = headDim).
        if (input.DType == DType.F32 && weight.DType == DType.F32)
        {
            EnsureKernels();
            ulong pOut = 0, pIn = 0, pWeight = 0;
            bool cachedOutput = false;
            try
            {
                pIn = GpuTransferHelper.CopyToDevice(input);
                pWeight = GpuTransferHelper.CopyToDevice(weight);
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                pOut = GpuTransferHelper.AllocateDevice(outBytes);
                _kernels!.LaunchRmsNorm(pOut, pIn, pWeight, (int)lastDim, (int)outerSize, eps, _stream.Handle);
                GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                cachedOutput = true;
            }
            finally
            {
                if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(pIn);
                GpuTransferHelper.FreeDevice(pWeight);
            }
            return;
        }

        // Native F16 path (F16 activation I/O, F32 weight — the DiT F16 activation recipe): read F16, accumulate
        // F32, write F16, weight kept F32. Halves the norm's HBM traffic vs the upcast fallback below (which reads
        // F16 → F32 scratch → kernel → F16, three passes). Same reduction geometry + F32 shared-mem as the F32 kernel.
        if (input.DType == DType.F16 && weight.DType == DType.F32 && output.DType == DType.F16)
        {
            EnsureKernels();
            ulong pOut = 0, pIn = 0, pWeight = 0;
            bool cachedOutput = false;
            try
            {
                pIn = GpuTransferHelper.CopyToDevice(input);
                pWeight = GpuTransferHelper.CopyToDevice(weight);
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                pOut = GpuTransferHelper.AllocateDevice(outBytes);
                _kernels!.LaunchRmsNormF16(pOut, pIn, pWeight, (int)lastDim, (int)outerSize, eps, _stream.Handle);
                GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                cachedOutput = true;
            }
            finally
            {
                if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(pIn);
                GpuTransferHelper.FreeDevice(pWeight);
            }
            return;
        }

        // Native BF16 path (BF16 activation I/O, F32 weight — the DiT BF16 residual-stream recipe). Same reason as
        // the F16 branch above: the fallback below moves ~20 bytes/element (BF16→F32 cast, F32 norm, F32→BF16 cast)
        // where this moves 4. RmsNorm runs ~4× per DiT block, so the fallback dominated the BF16 body's norm cost.
        if (input.DType == DType.BF16 && weight.DType == DType.F32 && output.DType == DType.BF16)
        {
            EnsureKernels();
            ulong pOut = 0, pIn = 0, pWeight = 0;
            bool cachedOutput = false;
            try
            {
                pIn = GpuTransferHelper.CopyToDevice(input);
                pWeight = GpuTransferHelper.CopyToDevice(weight);
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                pOut = GpuTransferHelper.AllocateDevice(outBytes);
                _kernels!.LaunchRmsNormBf16(pOut, pIn, pWeight, (int)lastDim, (int)outerSize, eps, _stream.Handle);
                GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                cachedOutput = true;
            }
            finally
            {
                if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(pIn);
                GpuTransferHelper.FreeDevice(pWeight);
            }
            return;
        }

        // GPU path for non-F32 input/weight: cast operands to F32 on-device, run the F32 RMSNorm kernel, cast
        // the result back to the output dtype. This replaces a CPU loop over millions of activation elements
        // per norm (with a blocking D2H sync) that was the dominant cost of the fp8/quant DiT path — dozens of
        // norms per block × tens of blocks × every forward. Same math as the F32 kernel, so numerically matches.
        EnsureKernels();
        {
            ulong pIn = 0, pWeight = 0, pInF32 = 0, pWeightF32 = 0, pOutF32 = 0, pOut = 0;
            bool cachedOutput = false;
            try
            {
                pIn = GpuTransferHelper.CopyToDevice(input);
                pWeight = GpuTransferHelper.CopyToDevice(weight);

                ulong inF32 = pIn;
                if (input.DType != DType.F32)
                {
                    pInF32 = CudaMemory.Allocate((nuint)(input.ElementCount * 4));
                    CastOnGpu(pInF32, pIn, input.DType, DType.F32, (int)input.ElementCount);
                    inF32 = pInF32;
                }
                ulong wF32 = pWeight;
                if (weight.DType != DType.F32)
                {
                    pWeightF32 = CudaMemory.Allocate((nuint)(weight.ElementCount * 4));
                    CastOnGpu(pWeightF32, pWeight, weight.DType, DType.F32, (int)weight.ElementCount);
                    wF32 = pWeightF32;
                }

                if (output.DType == DType.F32)
                {
                    nuint outBytes = GpuTransferHelper.ByteSize(output);
                    pOut = GpuTransferHelper.AllocateDevice(outBytes);
                    _kernels!.LaunchRmsNorm(pOut, inF32, wF32, (int)lastDim, (int)outerSize, eps, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                    cachedOutput = true;
                }
                else
                {
                    pOutF32 = CudaMemory.Allocate((nuint)(output.ElementCount * 4));
                    _kernels!.LaunchRmsNorm(pOutF32, inF32, wF32, (int)lastDim, (int)outerSize, eps, _stream.Handle);
                    nuint outBytes = GpuTransferHelper.ByteSize(output);
                    pOut = GpuTransferHelper.AllocateDevice(outBytes);
                    CastOnGpu(pOut, pOutF32, DType.F32, output.DType, (int)output.ElementCount);
                    GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                    cachedOutput = true;
                }
            }
            finally
            {
                if (pInF32 != 0) GpuTransferHelper.FreeDevice(pInF32);
                if (pWeightF32 != 0) GpuTransferHelper.FreeDevice(pWeightF32);
                if (pOutF32 != 0) GpuTransferHelper.FreeDevice(pOutF32);
                if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(pIn);
                GpuTransferHelper.FreeDevice(pWeight);
            }
        }
    }

    public void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AffineBroadcast");
        // F32 activation, or F16/BF16 activation with F32 scale/shift (the DiT 16-bit recipe: activation
        // halved, tiny per-channel params kept F32 for precision). scale/shift must stay F32 in every case.
        bool f16 = input.DType == DType.F16 && output.DType == DType.F16;
        bool bf16 = input.DType == DType.BF16 && output.DType == DType.BF16;
        if ((!f16 && !bf16 && (output.DType != DType.F32 || input.DType != DType.F32))
            || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException($"CUDA AffineBroadcastLastDim supports F32, or F16/BF16 activation with F32 scale/shift — got output={output.DType}, input={input.DType}, scale={scale.DType}, shift={shift?.DType.Name ?? "null"}.");
        EnterOp();
        EnsureKernels();
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)input.Shape[rank - 2] : 1;

        ulong pOut = 0, pIn = 0, pScale = 0, pShift = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pScale = GpuTransferHelper.CopyToDevice(scale);
            if (shift is not null) pShift = GpuTransferHelper.CopyToDevice(shift);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (f16)
                _kernels!.LaunchAffineBroadcastLastDimF16(pOut, pIn, pScale, pShift, seqLen, dim, input.ElementCount, _stream.Handle);
            else if (bf16)
                _kernels!.LaunchAffineBroadcastLastDimBf16(pOut, pIn, pScale, pShift, seqLen, dim, input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchAffineBroadcastLastDim(pOut, pIn, pScale, pShift, seqLen, dim, input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pScale);
            if (shift is not null) GpuTransferHelper.FreeDevice(pShift);
        }
    }

    /// <inheritdoc/>
    public unsafe bool TryAffineBroadcastRowIndexedToFp8(Tensor outputFp8, Tensor input, Tensor scaleTable,
        Tensor? shiftTable, Tensor rowIndex, Tensor consumerWeight)
    {
        if (!EnableModulateEmitFp8 || outputFp8.DType != DType.F8E4M3 || input.DType != DType.F32
            || scaleTable.DType != DType.F32 || rowIndex.DType != DType.I32
            || (shiftTable is not null && shiftTable.DType != DType.F32))
        {
            return false;
        }
        EnterOp();
        EnsureKernels();
        if (!_kernels!.HasAffineBroadcastRowIndexedToFp8) return false;
        ulong pInputScale = EnsureFp8InputScaleDev(consumerWeight);
        if (pInputScale == 0) return false;

        using NvtxRange _nvtx = NvtxRange.Push("AffineBroadcastRowIndexedToFp8");
        int rank = input.Shape.Rank;
        int dim = (int)input.Shape[rank - 1];
        ulong pOut = 0, pIn = 0, pScale = 0, pShift = 0, pIdx = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pScale = GpuTransferHelper.CopyToDevice(scaleTable);
            pShift = shiftTable is not null ? GpuTransferHelper.CopyToDevice(shiftTable) : 0;
            pIdx = GpuTransferHelper.CopyToDevice(rowIndex);
            nuint outBytes = GpuTransferHelper.ByteSize(outputFp8);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAffineBroadcastRowIndexedToFp8(pOut, pIn, pScale, pShift, pIdx, pInputScale,
                dim, input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(outputFp8, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pScale);
            if (shiftTable is not null) GpuTransferHelper.FreeDevice(pShift);
            GpuTransferHelper.FreeDevice(pIdx);
        }
        return true;
    }

    public void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GatedResidual");
        // F32, or F16/BF16 activations (residual/value/output) with an F32 gate — the DiT 16-bit recipe.
        bool f16 = residual.DType == DType.F16 && value.DType == DType.F16 && output.DType == DType.F16;
        bool bf16 = residual.DType == DType.BF16 && value.DType == DType.BF16 && output.DType == DType.BF16;
        if ((!f16 && !bf16 && (output.DType != DType.F32 || residual.DType != DType.F32 || value.DType != DType.F32)) || gate.DType != DType.F32)
            throw new NotSupportedException($"CUDA GatedResidualLastDim supports F32, or F16/BF16 activations with an F32 gate — got output={output.DType}, residual={residual.DType}, value={value.DType}, gate={gate.DType}.");
        EnterOp();
        EnsureKernels();
        int rank = value.Shape.Rank;
        int dim = (int)value.Shape[rank - 1];
        int seqLen = rank >= 2 ? (int)value.Shape[rank - 2] : 1;

        ulong pOut = 0, pRes = 0, pVal = 0, pGate = 0;
        bool cachedOutput = false;
        try
        {
            pRes = GpuTransferHelper.CopyToDevice(residual);
            pVal = GpuTransferHelper.CopyToDevice(value);
            pGate = GpuTransferHelper.CopyToDevice(gate);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (f16)
                _kernels!.LaunchGatedResidualLastDimF16(pOut, pRes, pVal, pGate, seqLen, dim, value.ElementCount, _stream.Handle);
            else if (bf16)
                _kernels!.LaunchGatedResidualLastDimBf16(pOut, pRes, pVal, pGate, seqLen, dim, value.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchGatedResidualLastDim(pOut, pRes, pVal, pGate, seqLen, dim, value.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pRes);
            GpuTransferHelper.FreeDevice(pVal);
            GpuTransferHelper.FreeDevice(pGate);
        }
    }

    public void AffineBroadcastRowIndexed(Tensor output, Tensor input, Tensor scaleTable, Tensor? shiftTable, Tensor rowIndex)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AffineBroadcastRowIndexed");
        // Activation may be F32/F16/BF16; the modulation table stays F32 (tiny, precision-sensitive) —
        // same recipe as the lastdim twins above.
        DType act = ValidateRowIndexedDtypes("AffineBroadcastRowIndexed", output, input, scaleTable, shiftTable, rowIndex);
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        if (rowIndex.ElementCount < total / dim)
            throw new ArgumentException($"AffineBroadcastRowIndexed rowIndex has {rowIndex.ElementCount} entries, need {total / dim}.", nameof(rowIndex));
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0, pScale = 0, pShift = 0, pIdx = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pScale = GpuTransferHelper.CopyToDevice(scaleTable);
            if (shiftTable is not null) pShift = GpuTransferHelper.CopyToDevice(shiftTable);
            pIdx = GpuTransferHelper.CopyToDevice(rowIndex);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (act == DType.F16)
                _kernels!.LaunchAffineBroadcastRowIndexedF16(pOut, pIn, pScale, pShift, pIdx, dim, total, _stream.Handle);
            else if (act == DType.BF16)
                _kernels!.LaunchAffineBroadcastRowIndexedBf16(pOut, pIn, pScale, pShift, pIdx, dim, total, _stream.Handle);
            else
                _kernels!.LaunchAffineBroadcastRowIndexed(pOut, pIn, pScale, pShift, pIdx, dim, total, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pScale);
            if (shiftTable is not null) GpuTransferHelper.FreeDevice(pShift);
            GpuTransferHelper.FreeDevice(pIdx);
        }
    }

    public void GatedResidualRowIndexed(Tensor output, Tensor residual, Tensor value, Tensor gateTable, Tensor rowIndex)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GatedResidualRowIndexed");
        DType act = ValidateRowIndexedDtypes("GatedResidualRowIndexed", output, value, gateTable, null, rowIndex);
        if (residual.DType != act)
            throw new NotSupportedException($"CUDA GatedResidualRowIndexed needs residual in the activation dtype {act}, got {residual.DType}.");
        int dim = (int)value.Shape[value.Shape.Rank - 1];
        long total = value.ElementCount;
        if (rowIndex.ElementCount < total / dim)
            throw new ArgumentException($"GatedResidualRowIndexed rowIndex has {rowIndex.ElementCount} entries, need {total / dim}.", nameof(rowIndex));
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pRes = 0, pVal = 0, pGate = 0, pIdx = 0;
        bool cachedOutput = false;
        try
        {
            pRes = GpuTransferHelper.CopyToDevice(residual);
            pVal = GpuTransferHelper.CopyToDevice(value);
            pGate = GpuTransferHelper.CopyToDevice(gateTable);
            pIdx = GpuTransferHelper.CopyToDevice(rowIndex);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (act == DType.F16)
                _kernels!.LaunchGatedResidualRowIndexedF16(pOut, pRes, pVal, pGate, pIdx, dim, total, _stream.Handle);
            else if (act == DType.BF16)
                _kernels!.LaunchGatedResidualRowIndexedBf16(pOut, pRes, pVal, pGate, pIdx, dim, total, _stream.Handle);
            else
                _kernels!.LaunchGatedResidualRowIndexed(pOut, pRes, pVal, pGate, pIdx, dim, total, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pRes);
            GpuTransferHelper.FreeDevice(pVal);
            GpuTransferHelper.FreeDevice(pGate);
            GpuTransferHelper.FreeDevice(pIdx);
        }
    }

    /// <summary>Returns the activation dtype shared by output/input, having checked the table is F32 and the index I32.</summary>
    private static DType ValidateRowIndexedDtypes(string op, Tensor output, Tensor input, Tensor table, Tensor? table2, Tensor rowIndex)
    {
        DType act = input.DType;
        if (act != DType.F32 && act != DType.F16 && act != DType.BF16)
            throw new NotSupportedException($"CUDA {op} supports F32/F16/BF16 activations, got {act}.");
        if (output.DType != act)
            throw new NotSupportedException($"CUDA {op} needs a {act} output to match the activation, got {output.DType}.");
        if (table.DType != DType.F32 || (table2 is not null && table2.DType != DType.F32))
            throw new NotSupportedException($"CUDA {op} keeps the modulation table F32, got {table.DType}/{table2?.DType.Name ?? "null"}.");
        if (rowIndex.DType != DType.I32)
            throw new NotSupportedException($"CUDA {op} requires I32 rowIndex, got {rowIndex.DType}.");
        return act;
    }

    public void ModulationSplit4(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("ModulationSplit4");
        if (proj.DType != DType.F32)
            throw new NotSupportedException("CUDA ModulationSplit4 supports F32 only.");
        EnterOp();
        EnsureKernels();
        int dim = (int)scaleMsa.Shape[scaleMsa.Shape.Rank - 1];
        int batch = (int)(scaleMsa.ElementCount / dim);

        ulong pProj = 0, pSMsa = 0, pGMsa = 0, pSMlp = 0, pGMlp = 0;
        bool cached = false;
        try
        {
            pProj = GpuTransferHelper.CopyToDevice(proj);
            nuint bytes = GpuTransferHelper.ByteSize(scaleMsa);
            pSMsa = GpuTransferHelper.AllocateDevice(bytes);
            pGMsa = GpuTransferHelper.AllocateDevice(bytes);
            pSMlp = GpuTransferHelper.AllocateDevice(bytes);
            pGMlp = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchModulation4(pSMsa, pGMsa, pSMlp, pGMlp, pProj, dim, batch, _stream.Handle);
            GpuTransferHelper.CacheActivation(scaleMsa, pSMsa, bytes);
            GpuTransferHelper.CacheActivation(gateMsa, pGMsa, bytes);
            GpuTransferHelper.CacheActivation(scaleMlp, pSMlp, bytes);
            GpuTransferHelper.CacheActivation(gateMlp, pGMlp, bytes);
            cached = true;
        }
        finally
        {
            if (!cached)
            {
                GpuTransferHelper.FreeDevice(pSMsa);
                GpuTransferHelper.FreeDevice(pGMsa);
                GpuTransferHelper.FreeDevice(pSMlp);
                GpuTransferHelper.FreeDevice(pGMlp);
            }
            GpuTransferHelper.FreeDevice(pProj);
        }
    }

    /// <inheritdoc />
    public void PatchifyTokens(Tensor output, Tensor input, int patch, bool innerChannelFastest)
    {
        using NvtxRange _nvtx = NvtxRange.Push("PatchifyTokens");
        PatchTokenGeometry geometry = PatchTokenContract.ValidatePatchify(output, input, patch);
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (input.DType == DType.F32)
                _kernels!.LaunchDitPatchify(
                    pOut, pIn, geometry.Batch, geometry.Channels, geometry.Height, geometry.Width,
                    patch, innerChannelFastest, _stream.Handle);
            else
                _kernels!.LaunchDitPatchifyU16(
                    pOut, pIn, geometry.Batch, geometry.Channels, geometry.Height, geometry.Width,
                    patch, innerChannelFastest, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked,
        int patch, bool innerChannelFastest)
    {
        using NvtxRange _nvtx = NvtxRange.Push("UnpatchifyTokens");
        PatchTokenGeometry geometry = PatchTokenContract.ValidateUnpatchify(
            output, tokens, channels, hPacked, wPacked, patch);
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(tokens);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (tokens.DType == DType.F32)
                _kernels!.LaunchDitUnpatchify(
                    pOut, pIn, geometry.Batch, channels, hPacked, wPacked, patch, innerChannelFastest, _stream.Handle);
            else
                _kernels!.LaunchDitUnpatchifyU16(
                    pOut, pIn, geometry.Batch, channels, hPacked, wPacked, patch, innerChannelFastest, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void MoeTopKGate(Tensor weights, Tensor logits, int topK)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MoeTopKGate");
        if (logits.DType != DType.F32 || weights.DType != DType.F32)
            throw new NotSupportedException("CUDA MoeTopKGate supports F32 only.");
        EnterOp();
        EnsureKernels();
        int numExperts = (int)logits.Shape[logits.Shape.Rank - 1];
        long tokens = logits.ElementCount / numExperts;
        if (numExperts > 16)
            throw new NotSupportedException($"CUDA MoeTopKGate supports up to 16 experts, got {numExperts}.");

        ulong pW = 0, pL = 0;
        bool cachedOutput = false;
        try
        {
            pL = GpuTransferHelper.CopyToDevice(logits);
            nuint outBytes = GpuTransferHelper.ByteSize(weights);
            pW = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchMoeTopkGate(pW, pL, numExperts, topK, tokens, _stream.Handle);
            GpuTransferHelper.CacheActivation(weights, pW, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pL);
        }
    }

    public void RowGatedAccumulate(Tensor inout, Tensor value, Tensor gate, int numExperts, int expertIndex)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RowGatedAccum");
        if (inout.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("CUDA RowGatedAccumulate supports F32 only.");
        EnterOp();
        EnsureKernels();
        int dim = (int)inout.Shape[inout.Shape.Rank - 1];

        ulong pOut = 0, pVal = 0, pGate = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(inout);
            pVal = GpuTransferHelper.CopyToDevice(value);
            pGate = GpuTransferHelper.CopyToDevice(gate);
            _kernels!.LaunchRowGatedAccum(pOut, pVal, pGate, numExperts, expertIndex, dim, inout.ElementCount, _stream.Handle);

            // In-place on inout: clear stale callbacks before re-caching (pitfall #17).
            inout._gpuSyncCallback = null;
            inout._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(inout, pOut, GpuTransferHelper.ByteSize(inout));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pVal);
            GpuTransferHelper.FreeDevice(pGate);
        }
    }

    public void CfgEulerStep(Tensor z, Tensor pos, Tensor neg, float guidance, float delta)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("CfgEulerStep");
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CUDA CfgEulerStep supports F32 only.");
        if (!z.Shape.Equals(pos.Shape) || !z.Shape.Equals(neg.Shape))
            throw new ArgumentException(
                $"CUDA CfgEulerStep requires identical shapes; got z={z.Shape}, pos={pos.Shape}, neg={neg.Shape}.");
        if (z.ElementCount <= 0 || z.ElementCount > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(z),
                $"CUDA CfgEulerStep element count must be in [1,{int.MaxValue}]; got {z.ElementCount}.");
        if (z.HasOverlappingHostStorageWithoutSync(pos) || z.HasOverlappingHostStorageWithoutSync(neg))
            throw new ArgumentException(
                "CUDA CfgEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        EnterOp();
        EnsureKernels();

        ulong pZ = 0, pPos = 0, pNeg = 0;
        bool cachedZ = false;
        try
        {
            pZ = GpuTransferHelper.CopyToDevice(z);
            pPos = GpuTransferHelper.CopyToDevice(pos);
            pNeg = GpuTransferHelper.CopyToDevice(neg);
            _kernels!.LaunchCfgEuler(pZ, pPos, pNeg, guidance, delta, (int)z.ElementCount, _stream.Handle);

            // In-place on z: clear stale callbacks before re-caching so the old sync callback
            // doesn't FreeAsync the buffer we're keeping resident (pitfall #17).
            z._gpuSyncCallback = null;
            z._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(z, pZ, GpuTransferHelper.ByteSize(z));
            cachedZ = true;
        }
        finally
        {
            if (!cachedZ) GpuTransferHelper.FreeDevice(pZ);
            GpuTransferHelper.FreeDevice(pPos);
            GpuTransferHelper.FreeDevice(pNeg);
        }
    }

    /// <inheritdoc />
    public void AffineMix(Tensor output, Tensor x, Tensor y, float xScale, float yScale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AffineMix");
        long count = MixContract.ValidateAffineMix(output, x, y, xScale, yScale);
        EnterOp();
        EnsureKernels();

        ulong pOutput = 0, pX = 0, pY = 0;
        bool cachedOutput = false;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pY = ReferenceEquals(x, y) ? pX : GpuTransferHelper.CopyToDevice(y);
            nuint outputBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outputBytes);
            _kernels!.LaunchAffineMix(pOutput, pX, pY, xScale, yScale, count, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOutput, outputBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
            if (pY != pX) GpuTransferHelper.FreeDevice(pY);
            GpuTransferHelper.FreeDevice(pX);
        }
    }

    /// <inheritdoc />
    public void MaskedAffineMixInPlace(
        Tensor target,
        Tensor source,
        Tensor? noise,
        Tensor mask,
        float sourceScale,
        float noiseScale,
        MaskBroadcastLayout layout)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MaskedAffineMixInPlace");
        MaskedMixGeometry geometry = MixContract.ValidateMaskedAffineMix(
            target, source, noise, mask, sourceScale, noiseScale, layout);
        EnterOp();
        EnsureKernels();

        ulong pTarget = 0, pSource = 0, pNoise = 0, pMask = 0;
        bool cachedTarget = false;
        try
        {
            pTarget = GpuTransferHelper.CopyToDevice(target);
            pSource = GpuTransferHelper.CopyToDevice(source);
            pNoise = noise is null ? 0
                : ReferenceEquals(noise, source) ? pSource : GpuTransferHelper.CopyToDevice(noise);
            pMask = GpuTransferHelper.CopyToDevice(mask);

            if (layout == MaskBroadcastLayout.DenseNchwBroadcast)
            {
                _kernels!.LaunchMaskedAffineMixDense(
                    pTarget, pSource, pNoise, pMask, sourceScale, noiseScale,
                    geometry.Batch, geometry.Channels, geometry.Spatial, _stream.Handle);
            }
            else
            {
                _kernels!.LaunchMaskedAffineMixPacked(
                    pTarget, pSource, pNoise, pMask, sourceScale, noiseScale,
                    geometry.Tokens, geometry.FeatureDimension, geometry.PatchArea,
                    layout == MaskBroadcastLayout.PackedChannelInner, _stream.Handle);
            }

            // The target's existing allocation remains authoritative. Remove its stale D2H/free callbacks before
            // replanting the activation binding so it stays resident for the next denoise step.
            target._gpuSyncCallback = null;
            target._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(target, pTarget, GpuTransferHelper.ByteSize(target));
            cachedTarget = true;
        }
        finally
        {
            if (!cachedTarget) GpuTransferHelper.FreeDevice(pTarget);
            GpuTransferHelper.FreeDevice(pMask);
            if (pNoise != pSource) GpuTransferHelper.FreeDevice(pNoise);
            GpuTransferHelper.FreeDevice(pSource);
        }
    }

    /// <inheritdoc />
    public void CfgRenormEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float renormMin)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CUDA CfgRenormEulerStep supports F32 only.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CUDA CfgRenormEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.ElementCount <= 0)
            throw new ArgumentException("CUDA CfgRenormEulerStep requires nonempty tensors.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CUDA CfgRenormEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(renormMin) || renormMin < 0f || renormMin > 1f)
            throw new ArgumentOutOfRangeException(nameof(renormMin), "Renorm minimum must be finite and in [0,1].");

        long count = z.ElementCount;
        if (delta == 0f) return;

        EnterOp();
        EnsureKernels();
        const int threads = 256;
        int partialCount = (int)Math.Min(256L, 1L + (count - 1L) / threads);
        nuint scratchBytes = checked((nuint)(2L * partialCount * sizeof(double) + sizeof(float)));

        ulong pZ = 0, pCond = 0, pUncond = 0, pScratch = 0;
        bool cachedZ = false;
        try
        {
            pZ = GpuTransferHelper.CopyToDevice(z);
            pCond = GpuTransferHelper.CopyToDevice(cond);
            pUncond = GpuTransferHelper.CopyToDevice(uncond);
            pScratch = GpuTransferHelper.AllocateDevice(scratchBytes);
            _kernels!.LaunchCfgRenormEuler(
                pZ, pCond, pUncond, pScratch, partialCount,
                guidance, delta, renormMin, count, _stream.Handle);

            // In-place on z: keep its device buffer authoritative and resident after the update.
            z._gpuSyncCallback = null;
            z._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(z, pZ, GpuTransferHelper.ByteSize(z));
            cachedZ = true;
        }
        finally
        {
            if (!cachedZ) GpuTransferHelper.FreeDevice(pZ);
            GpuTransferHelper.FreeDevice(pCond);
            GpuTransferHelper.FreeDevice(pUncond);
            GpuTransferHelper.FreeDevice(pScratch);
        }
    }

    /// <inheritdoc />
    public void CfgNormalizedEulerStep(
        Tensor z, Tensor cond, Tensor uncond, float guidance, float delta, float eps = 1e-12f)
    {
        if (z.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException("CUDA CfgNormalizedEulerStep supports F32 only.");
        if (!z.Shape.Equals(cond.Shape) || !z.Shape.Equals(uncond.Shape))
            throw new ArgumentException(
                $"CUDA CfgNormalizedEulerStep requires identical shapes; got z={z.Shape}, cond={cond.Shape}, uncond={uncond.Shape}.");
        if (z.Shape.Rank < 1 || z.ElementCount <= 0 || z.Shape[z.Shape.Rank - 1] <= 0)
            throw new ArgumentException(
                "CUDA CfgNormalizedEulerStep requires a nonempty tensor with a positive last dimension.", nameof(z));
        if (z.HasOverlappingHostStorageWithoutSync(cond) || z.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException(
                "CUDA CfgNormalizedEulerStep inputs may alias each other, but neither may alias the in-place z tensor.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be finite.");
        if (!float.IsFinite(delta))
            throw new ArgumentOutOfRangeException(nameof(delta), "Euler delta must be finite.");
        if (!float.IsFinite(eps) || eps < 0f)
            throw new ArgumentOutOfRangeException(nameof(eps), "Normalization epsilon must be finite and nonnegative.");
        if (delta == 0f) return;

        long lastDim = z.Shape[z.Shape.Rank - 1];
        long rows = z.ElementCount / lastDim;
        EnterOp();
        EnsureKernels();

        ulong pZ = 0, pCond = 0, pUncond = 0;
        bool cachedZ = false;
        try
        {
            pZ = GpuTransferHelper.CopyToDevice(z);
            pCond = GpuTransferHelper.CopyToDevice(cond);
            pUncond = GpuTransferHelper.CopyToDevice(uncond);
            _kernels!.LaunchCfgNormalizedEuler(
                pZ, pCond, pUncond, guidance, delta, eps, rows, lastDim, _stream.Handle);

            z._gpuSyncCallback = null;
            z._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(z, pZ, GpuTransferHelper.ByteSize(z));
            cachedZ = true;
        }
        finally
        {
            if (!cachedZ) GpuTransferHelper.FreeDevice(pZ);
            GpuTransferHelper.FreeDevice(pCond);
            GpuTransferHelper.FreeDevice(pUncond);
        }
    }

    // ── Step-graph capture (CUDA-graph replay of a fixed per-step op sequence; see IBackend docs) ─────────
    private CudaGraph? _stepGraph;
    private bool _stepGraphCapturing;

    public bool SupportsF16Activations => true;

    /// <summary>True once the optional stepcache.ptx module is compiled/shipped; the step-cache stays disabled on CUDA without it.</summary>
    /// <remarks>Built via src/HartsyInference.Cuda/Kernels/dit/build.sh.</remarks>
    public bool SupportsDeviceStepCacheGate => _kernels is not null && _kernels.HasStepCacheKernels;

    /// <summary>Marks a tensor's device activation as surviving <see cref="FreeActivations()"/> (see IBackend doc).</summary>
    public void PinActivation(Tensor tensor)
    {
        EnterOp();
        GpuTransferHelper.PinActivation(tensor);
    }

    /// <summary>Removes a <see cref="PinActivation"/> mark.</summary>
    public void UnpinActivation(Tensor tensor)
    {
        EnterOp();
        GpuTransferHelper.UnpinActivation(tensor);
    }

    /// <summary>Materializes a cached activation to host and frees its device copy (see IBackend doc).</summary>
    public void OffloadActivation(Tensor tensor)
    {
        EnterOp();
        // Same hazard as FreeActivations: a captured graph bakes this activation's device pointer and the reload
        // hands back a different one. Gated on there being something to free so a no-op offload can't kill a graph.
        if (GpuTransferHelper.HasCachedActivation(tensor)) StepGraphInvalidateForActivationFree();
        GpuTransferHelper.OffloadActivation(tensor);
    }

    /// <summary>Offloads cached activations largest-first until <paramref name="targetBytes"/> is freed (see IBackend doc).</summary>
    public long OffloadActivations(long targetBytes)
    {
        EnterOp();
        if (targetBytes <= 0 || GpuTransferHelper.CachedActivationCount == 0) return 0;
        StepGraphInvalidateForActivationFree();
        return GpuTransferHelper.OffloadActivations(targetBytes);
    }

    public bool FlashDecodeSupported => true;

    public bool StepGraphSupported => true;

    public bool StepGraphReady => _stepGraph?.IsReady == true && !_stepGraphCapturing;

    /// <summary>Owner token for the single step-graph slot (see IBackend.StepGraphOwner).</summary>
    public object? StepGraphOwner { get; set; }

    public void StepGraphBegin()
    {
        EnterOp();
        // A still-open capture must abort+purge BEFORE the tracker clear below wipes its alloc records.
        if (_stepGraphCapturing)
            StepGraphReset();
        if (_stepGraph is null)
        {
            _stepGraph = new CudaGraph(_stream.Handle, autoFreeAllocationsOnRelaunch: true);
            GC.SuppressFinalize(_stepGraph);
        }
        _stepGraph.Reset();
        lock (_transferState.CaptureAllocs)
        {
            _transferState.CaptureAllocs.Clear();
            _transferState.CaptureAllocBytes = 0;
            _transferState.CaptureFreeBytes = 0;
            _transferState.CaptureAllocCount = 0;
            _transferState.CaptureFreeCount = 0;
            _transferState.TrackCaptureWindow = true;
        }
        _stepGraph.BeginCapture();
        _stepGraphCapturing = true;
    }

    public void StepGraphEndAndLaunch()
    {
        if (!_stepGraphCapturing || _stepGraph is null)
            return;
        _stepGraphCapturing = false;
        _transferState.TrackCaptureWindow = false;
        long outstanding;
        int outstandingCount;
        lock (_transferState.CaptureAllocs)
        {
            outstanding = _transferState.CaptureAllocBytes - _transferState.CaptureFreeBytes;
            outstandingCount = _transferState.CaptureAllocs.Count;
        }
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] step-graph capture window: allocs {_transferState.CaptureAllocCount} ({_transferState.CaptureAllocBytes >> 20} MB), " +
            $"frees {_transferState.CaptureFreeCount} ({_transferState.CaptureFreeBytes >> 20} MB), " +
            $"OUTSTANDING {outstandingCount} allocs / {outstanding >> 20} MB");
        _stepGraph.EndCaptureAndInstantiate();
        _stepGraph.Launch();
    }

    public void StepGraphLaunch()
    {
        EnterOp();
        if (_stepGraph is null || !_stepGraph.IsReady)
            throw new InvalidOperationException("StepGraphLaunch called with no captured graph.");
        _stepGraph.Launch();
    }

    public void StepGraphReset()
    {
        if (_stepGraphCapturing)
        {
            _stepGraph?.AbortCapture();
            _stepGraphCapturing = false;
            _transferState.TrackCaptureWindow = false;
            // A capture that never reached StepGraphEndAndLaunch leaves every activation cached mid-window
            // pointing at a graph-private VA the driver just released along with the discarded (never
            // instantiated) graph — purge them before anything tries to free one for real (see
            // GpuTransferHelper.PurgeAbortedCaptureAllocs for the CUDA_ERROR_INVALID_VALUE this prevents).
            GpuTransferHelper.PurgeAbortedCaptureAllocs(_transferState, _stream.Handle);
        }
        bool hadCapturedGraph = _stepGraph?.IsReady == true;
        _stepGraph?.Reset();
        if (hadCapturedGraph)
        {
            // Best-effort graph-pool trim after destroying a captured graph. NOTE (measured): the bulk of a
            // destroyed step graph's memory (~4.5 GB for the Chroma CFG pair) is a DRIVER-side lazily-
            // reclaimable cache that neither this trim nor cuMemPoolTrimTo returns — cuMemGetInfo reports it
            // used until a SYNCHRONOUS cuMemAlloc forces the reclaim (see CudaMemory.AllocateAsync's
            // sync-probe retry, which is what actually protects the next model's load). Sync first:
            // destroy/trim under a still-executing final replay is undefined.
            _stream.Synchronize();
            CudaDriverApi.cuDeviceGraphMemTrim(_context.DeviceHandle).ThrowOnError();
        }
    }

    /// <summary>Device copy into <paramref name="dst"/>'s EXISTING buffer (address-preserving — the captured-graph boundary refresh).</summary>
    /// <remarks>First call materializes a device buffer for dst; subsequent calls reuse it. Host src is uploaded;
    /// device src is DtoD'd, both stream-ordered on the compute stream.</remarks>
    public unsafe void CopyInto(Tensor dst, Tensor src)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CopyInto");
        EnterOp();
        EnsureKernels();
        ulong pSrc = 0;
        nuint bytes = GpuTransferHelper.ByteSize(dst);
        // The copy overwrites dst's buffer in full, so a first call must ALLOCATE, not upload: CopyToDevice would
        // stage dst's (here uninitialized) host bytes over PCIe and the DtoD below would immediately discard them.
        // LTX-2.5's CFG-pair split hits this once per step at 143 MB.
        ulong pDst = GpuTransferHelper.TryGetCachedDevice(dst, out ulong cachedDst)
            ? cachedDst : GpuTransferHelper.AllocateDevice(bytes);
        try
        {
            pSrc = GpuTransferHelper.CopyToDevice(src);
            CudaDriverApi.cuMemcpyDtoDAsync(pDst, pSrc, bytes, _stream.Handle).ThrowOnError();
            // In-place re-assert (the CfgEulerStep pattern): dst keeps this buffer across the copy.
            dst._gpuSyncCallback = null;
            dst._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(dst, pDst, bytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pSrc);
        }
    }

    public void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("ApplyRope");
        // F32, or F16 q/k with F32 cos/sin (DiT F16 recipe — cos/sin are tiny and precision-sensitive).
        bool f16 = q.DType == DType.F16;
        if ((!f16 && q.DType != DType.F32) || k.DType != q.DType || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRope supports F32, or F16 q/k with F32 cos/sin.");
        EnterOp();
        EnsureKernels();
        int numHeads = (int)q.Shape[2];
        int headDim = (int)q.Shape[3];
        long totalVecs = q.ElementCount / headDim;

        ulong pQ = 0, pK = 0, pCos = 0, pSin = 0;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            if (f16)
            {
                _kernels!.LaunchRopeF16(pQ, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle);
                _kernels!.LaunchRopeF16(pK, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle);
            }
            else
            {
                _kernels!.LaunchRope(pQ, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle);
                _kernels!.LaunchRope(pK, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle);
            }

            // In-place on q and k: clear stale callbacks before re-caching (pitfall #17).
            q._gpuSyncCallback = null;
            q._gpuDisposeCallback = null;
            k._gpuSyncCallback = null;
            k._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(q, pQ, GpuTransferHelper.ByteSize(q));
            GpuTransferHelper.CacheActivation(k, pK, GpuTransferHelper.ByteSize(k));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>LTX-2.5 channels-last 3D pixel shuffle — see <see cref="IBackend.Ltx25PixelShuffle"/>. The output spans every chunk, so it is allocated once and written in place rather than recached per call.</summary>
    public void Ltx25PixelShuffle(Tensor output, Tensor projected, int start, int h, int w,
        int strideT, int strideH, int strideW, int outChannels, int dropped, int outH, int outW)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx25PixelShuffle");
        if (output.DType != DType.F32 || projected.DType != DType.F32)
            throw new NotSupportedException($"CUDA Ltx25PixelShuffle supports F32 only — got output={output.DType}, projected={projected.DType}.");
        EnsureKernels();
        if (_kernels is null || !_kernels.HasLtx25PixelShuffle)
        {
            IBackend.Ltx25PixelShuffleReference(output, projected, start, h, w,
                strideT, strideH, strideW, outChannels, dropped, outH, outW);
            return;
        }

        EnterOp();
        ulong pSrc = 0;
        try
        {
            pSrc = GpuTransferHelper.CopyToDevice(projected);
            // The destination persists across the caller's chunk loop — allocate on the first chunk without an
            // upload (every destination element is written exactly once across the whole loop) and reuse after.
            if (!GpuTransferHelper.IsActivationCached(output))
            {
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                GpuTransferHelper.CacheActivation(output, GpuTransferHelper.AllocateDevice(outBytes), outBytes);
            }
            ulong pDst = GpuTransferHelper.CopyToDevice(output);
            _kernels.LaunchLtx25PixelShuffle(pDst, pSrc, start, h, w, strideT, strideH, strideW,
                outChannels, dropped, outH, outW, projected.ElementCount, _stream.Handle);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pSrc);
        }
    }

    /// <summary>3D neighborhood attention for the LTX-2.5 diffusion video decoder. Prefers the query-tiled kernel, which stages a whole tile's shared window instead of re-reading one window per query; falls back to the per-query kernel on shapes it does not cover, and to the managed reference when <c>ltx25_na_decoder.ptx</c> has not been deployed or the per-query kernel's scores would not fit 48 KB of shared memory.</summary>
    public void Na3d(Tensor output, Tensor q, Tensor k, Tensor v, int kernelT, int kernelH, int kernelW, float scale)
    {
        if (output.DType != DType.F32 || q.DType != DType.F32 || k.DType != DType.F32 || v.DType != DType.F32)
            throw new NotSupportedException($"CUDA Na3d supports F32 only — got output={output.DType}, q={q.DType}, k={k.DType}, v={v.DType}.");
        if (q.Shape.Rank != 6)
            throw new ArgumentException($"Na3d expects [batch, T, H, W, heads, headDim]; got {q.Shape}.", nameof(q));
        if (!q.Shape.Equals(k.Shape) || !q.Shape.Equals(v.Shape) || !IBackend.Na3dOutputShapeMatches(output.Shape, q.Shape))
            throw new ArgumentException($"Na3d requires identical shapes; got q={q.Shape}, k={k.Shape}, v={v.Shape}, output={output.Shape}.");
        if (kernelT <= 0 || kernelH <= 0 || kernelW <= 0)
            throw new ArgumentException($"Na3d kernel must be positive; got ({kernelT}, {kernelH}, {kernelW}).");
        if (ReferenceEquals(output, q) || ReferenceEquals(output, k) || ReferenceEquals(output, v))
            throw new ArgumentException("Na3d output must not alias q, k or v.", nameof(output));

        int batch = (int)q.Shape[0], dimT = (int)q.Shape[1], dimH = (int)q.Shape[2], dimW = (int)q.Shape[3];
        int heads = (int)q.Shape[4], headDim = (int)q.Shape[5];
        // An axis shorter than its kernel collapses to the whole axis, matching the reference.
        int kt = Math.Min(kernelT, dimT), kh = Math.Min(kernelH, dimH), kw = Math.Min(kernelW, dimW);

        EnsureKernels();
        (int tileH, int tileW) = _kernels is not null && UseLtx25Na3dTiled ? _kernels.Ltx25Na3dTile(dimH, dimW, headDim)
            : (0, 0);
        if (_kernels is null || !_kernels.HasLtx25NaKernels
            || (tileH == 0 && CudaKernels.Ltx25Na3dSharedBytes(headDim, kt, kh, kw) > 48 * 1024))
        {
            IBackend.Na3dReference(output, q, k, v, kernelT, kernelH, kernelW, scale);
            return;
        }

        EnterOp();
        ulong pOut = 0, pQ = 0, pK = 0, pV = 0; bool cached = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pV = GpuTransferHelper.CopyToDevice(v);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (tileH > 0)
            {
                _kernels.LaunchLtx25Na3dTiled(pOut, pQ, pK, pV, batch, dimT, dimH, dimW, heads, headDim,
                    kt, kh, kw, scale, tileH, tileW, _stream.Handle);
            }
            else
            {
                _kernels.LaunchLtx25Na3d(pOut, pQ, pK, pV, batch, dimT, dimH, dimW, heads, headDim,
                    kt, kh, kw, scale, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV);
        }
    }

    /// <summary>LTX-2.5 NA-decoder 3-axis interleaved rope, in place on x[1,T,H,W,heads,headDim].</summary>
    public void Ltx25NaRope3d(Tensor x, Tensor cosT, Tensor sinT, Tensor cosH, Tensor sinH, Tensor cosW, Tensor sinW,
        int splitT, int splitH)
    {
        IBackend.Ltx25NaRope3dGeometry(x, splitT, splitH, out int dimT, out int dimH, out int dimW, out int heads,
            out int headDim, out int pairsT, out int pairsH, out int pairsW);

        EnsureKernels();
        if (_kernels is null || !_kernels.HasLtx25NaKernels)
        {
            IBackend.Ltx25NaRope3dReference(x, cosT, sinT, cosH, sinH, cosW, sinW, splitT, splitH);
            return;
        }

        EnterOp();
        ulong pX = 0, pCt = 0, pSt = 0, pCh = 0, pSh = 0, pCw = 0, pSw = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCt = GpuTransferHelper.CopyToDevice(cosT); pSt = GpuTransferHelper.CopyToDevice(sinT);
            pCh = GpuTransferHelper.CopyToDevice(cosH); pSh = GpuTransferHelper.CopyToDevice(sinH);
            pCw = GpuTransferHelper.CopyToDevice(cosW); pSw = GpuTransferHelper.CopyToDevice(sinW);
            _kernels.LaunchLtx25NaRope3d(pX, pCt, pSt, pCh, pSh, pCw, pSw, dimT, dimH, dimW, heads, headDim,
                pairsT, pairsH, pairsW, splitT, splitT + splitH, _stream.Handle);
            // In-place on x: clear stale callbacks before re-caching (pitfall #17).
            x._gpuSyncCallback = null; x._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCt); GpuTransferHelper.FreeDevice(pSt);
            GpuTransferHelper.FreeDevice(pCh); GpuTransferHelper.FreeDevice(pSh);
            GpuTransferHelper.FreeDevice(pCw); GpuTransferHelper.FreeDevice(pSw);
        }
    }

    /// <summary>Oasis head split: frame-major qkv[token,3·dim] → out[b,heads,seq,headDim] (device port of the host loop).</summary>
    public void OasisSplitHeads(Tensor output, Tensor qkv, int frames, int sp, int heads, int headDim, int part, bool temporal)
    {
        if (output.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("CUDA OasisSplitHeads supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(qkv);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchOasisSplitHeads(pOut, pIn, frames, sp, heads, headDim, part, temporal, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    /// <summary>Interleaved (2i,2i+1) partial rope on x[b,heads,seq,headDim] in place; cos/sin are [seq,rotDim].</summary>
    public void OasisRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int batch, int heads, int seq, int headDim, int rotDim)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA OasisRopeInterleaved supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchOasisRopeInterleaved(pX, pCos, pSin, batch, heads, seq, headDim, rotDim, _stream.Handle);
            x._gpuSyncCallback = null; x._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally { GpuTransferHelper.FreeDevice(pCos); GpuTransferHelper.FreeDevice(pSin); }
    }

    /// <summary>Oasis head merge: attn[b,heads,seq,headDim] → out[token,dim] (inverse of split).</summary>
    public void OasisMergeHeads(Tensor output, Tensor attn, int frames, int sp, int heads, int headDim, bool temporal)
    {
        if (output.DType != DType.F32 || attn.DType != DType.F32) throw new NotSupportedException("CUDA OasisMergeHeads supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(attn);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchOasisMergeHeads(pOut, pIn, frames, sp, heads, headDim, temporal, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    /// <summary>Fused Oasis adaLN: out = LayerNorm(x)·(1+scale)+shift, scale/shift sliced from mod per frame.</summary>
    public void OasisAdaLn(Tensor output, Tensor input, Tensor mod, int dim, int sp, int totalRows, int modStride, int shiftOff, int scaleOff, float eps)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32 || mod.DType != DType.F32)
            throw new NotSupportedException("CUDA OasisAdaLn supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0, pMod = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pMod = GpuTransferHelper.CopyToDevice(mod);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchOasisAdaLn(pOut, pIn, pMod, dim, sp, totalRows, modStride, shiftOff, scaleOff, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    /// <summary>Oasis per-frame unpatchify: proj[t·sp, c·p²] → out[t,c,H,W] ([py,px,ci] inner layout).</summary>
    public void OasisUnpatchify(Tensor output, Tensor proj, int frames, int channels, int gh, int gw, int patch)
    {
        if (output.DType != DType.F32 || proj.DType != DType.F32) throw new NotSupportedException("CUDA OasisUnpatchify supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(proj);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchOasisUnpatchify(pOut, pIn, frames, channels, gh, gw, patch, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    /// <summary>TripoSR triplane grid-sample (device); <paramref name="outputF"/> stays resident so the NeRF MLP hits the activation cache.</summary>
    /// <remarks><paramref name="planes"/> uploads once (weight-cache resident via PreloadWeights) — no host round-trip per point.</remarks>
    public unsafe void TriplaneGridSample(Tensor outputF, Tensor planes, Tensor? coords, long chunkStart,
        int count, int channels, int planeH, int planeW, float radius, int gridRes)
    {
        if (outputF.DType != DType.F32 || planes.DType != DType.F32)
            throw new NotSupportedException("CUDA TriplaneGridSample supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pCoords = 0; bool cached = false; bool coordsTransient = false;
        try
        {
            ulong pPlanes = GpuTransferHelper.CopyToDevice(planes);
            if (coords is not null) { pCoords = GpuTransferHelper.CopyToDevice(coords); coordsTransient = true; }
            nuint outBytes = GpuTransferHelper.ByteSize(outputF);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchTriplaneGridSample(pOut, pPlanes, pCoords, (ulong)chunkStart, count, channels,
                planeH, planeW, radius, gridRes, _stream.Handle);
            GpuTransferHelper.CacheActivation(outputF, pOut, outBytes); cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            if (coordsTransient) GpuTransferHelper.FreeDevice(pCoords);
        }
    }

    /// <summary>2D transposed convolution (device gather kernel). Weight <c>[Cin,Cout,kH,kW]</c>.</summary>
    /// <remarks>Overrides the CPU scatter-add default — the TripoSR/YOLO/Demucs upsample was running on the host.</remarks>
    public unsafe void ConvTranspose2d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("ConvTranspose2d");
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException("CUDA ConvTranspose2d supports F32 only.");
        int n = (int)input.Shape[0], cIn = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int cOut = (int)output.Shape[1], oH = (int)output.Shape[2], oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2], kW = (int)weight.Shape[3];
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0, pW = 0, pB = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchConvTranspose2d(pOut, pIn, pW, pB, n, cIn, cOut, iH, iW, oH, oW,
                kH, kW, strideH, strideW, padH, padW, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn); GpuTransferHelper.FreeDevice(pW); GpuTransferHelper.FreeDevice(pB);
        }
    }

    /// <summary>GEGLU with exact erf gate (device): output[rows,inner] = proj[:,:inner]·gelu_erf(proj[:,inner:]).</summary>
    public unsafe void GegluErf(Tensor output, Tensor proj, long rows, int inner)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("GegluErf");
        if (output.DType != DType.F32 || proj.DType != DType.F32)
            throw new NotSupportedException("CUDA GegluErf supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0; bool cached = false;
        try
        {
            ulong pProj = GpuTransferHelper.CopyToDevice(proj);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGegluErf(pOut, pProj, rows, inner, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); }
    }

    /// <summary>Exact (erf) GELU, elementwise on device — PyTorch's default <c>nn.GELU()</c> (DINOv2 MLPs).</summary>
    public unsafe void GeluErf(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("GeluErf");
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("CUDA GeluErf supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0; bool cached = false;
        try
        {
            ulong pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGeluErf(pOut, pIn, output.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); }
    }

    /// <summary>DIAMOND pixel quantize to 256 levels: out = floor((clamp(v,-1,1)+1)·127.5)/127.5 − 1 (device).</summary>
    public void PixelQuantize(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32) throw new NotSupportedException("CUDA PixelQuantize supports F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchPixelQuantize(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    /// <summary>Fused LTX-2 QK path — see <see cref="IBackend.Ltx2QkNormRopeHeadMajor"/>. Falls back to the three-op sequence when the row is too wide to stage in shared memory; every LTX-2 geometry shipped today (inner 4096 video / 2048 audio) fits comfortably.</summary>
    public unsafe void Ltx2QkNormRopeHeadMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2QkNormRopeHeadMajor");
        bool f16 = input.DType == DType.F16;
        if ((input.DType != DType.F32 && !f16) || output.DType != input.DType)
            throw new NotSupportedException(
                $"CUDA Ltx2QkNormRopeHeadMajor needs matching F32 or F16 input/output, got in={input.DType}, out={output.DType}.");
        bool ropeF16 = cos is not null && cos.DType == DType.F16;
        // Dynamic shared is (256 + heads*headDim) floats (256 = CudaKernels.BlockSize); past the 48 KB static limit the launch would fail,
        // so hand those shapes back to the composed path rather than silently mis-sizing.
        if ((256L + (long)heads * headDim) * sizeof(float) > 48 * 1024)
        {
            if (ropeF16)
                throw new NotSupportedException($"F16 RoPE tables need the fused kernel, but heads*headDim={heads * headDim} exceeds its shared-memory budget.");
            Tensor normed = new Tensor(input.Shape, input.DType);
            try
            {
                RmsNorm(normed, input, normWeight, eps);
                if (cos is not null && sin is not null) Ltx2SplitRope(normed, cos, sin, seq, heads, headDim);
                Permute0213(output, normed, seq, heads, headDim);
            }
            finally { normed.Dispose(); }
            return;
        }
        EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, pW = 0, pCos = 0, pSin = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(normWeight);
            if (cos is not null) pCos = GpuTransferHelper.CopyToDevice(cos);
            if (sin is not null) pSin = GpuTransferHelper.CopyToDevice(sin);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchLtx2QkNormRopeHeadMajor(pOut, pIn, pW, pCos, pSin, seq, heads, headDim, eps, _stream.Handle, f16, ropeF16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pCos != 0) GpuTransferHelper.FreeDevice(pCos);
            if (pSin != 0) GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>Fused LTX-2 QK path, token-major — see <see cref="IBackend.Ltx2QkNormRopeTokenMajor"/>. Same shared-memory ceiling and composed fallback as the head-major twin.</summary>
    public unsafe void Ltx2QkNormRopeTokenMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2QkNormRopeTokenMajor");
        bool f16 = input.DType == DType.F16;
        if ((input.DType != DType.F32 && !f16) || output.DType != input.DType)
            throw new NotSupportedException(
                $"CUDA Ltx2QkNormRopeTokenMajor needs matching F32 or F16 input/output, got in={input.DType}, out={output.DType}.");
        bool ropeF16 = cos is not null && cos.DType == DType.F16;
        if ((256L + (long)heads * headDim) * sizeof(float) > 48 * 1024)
        {
            if (ropeF16)
                throw new NotSupportedException($"F16 RoPE tables need the fused kernel, but heads*headDim={heads * headDim} exceeds its shared-memory budget.");
            RmsNorm(output, input, normWeight, eps);
            if (cos is not null && sin is not null) Ltx2SplitRope(output, cos, sin, seq, heads, headDim);
            return;
        }
        EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, pW = 0, pCos = 0, pSin = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(normWeight);
            if (cos is not null) pCos = GpuTransferHelper.CopyToDevice(cos);
            if (sin is not null) pSin = GpuTransferHelper.CopyToDevice(sin);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchLtx2QkNormRopeTokenMajor(pOut, pIn, pW, pCos, pSin, seq, heads, headDim, eps, _stream.Handle, f16, ropeF16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pCos != 0) GpuTransferHelper.FreeDevice(pCos);
            if (pSin != 0) GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>Fused LTX-2 norm+modulate — see <see cref="IBackend.Ltx2RmsModulate"/>. The RMS weight is a ones vector at every call site, so it folds away rather than being read per element.</summary>
    public unsafe void Ltx2RmsModulate(Tensor output, Tensor input, Tensor onesWeight, Tensor scale, Tensor? shift, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2RmsModulate");
        bool f16 = input.DType == DType.F16;
        if ((input.DType != DType.F32 && !f16) || output.DType != input.DType || scale.DType != DType.F32
            || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException(
                $"CUDA Ltx2RmsModulate needs matching F32/F16 in/out with F32 scale+shift, got in={input.DType}, out={output.DType}, scale={scale.DType}.");
        int dim = (int)scale.Shape[scale.Shape.Rank - 1];
        int rows = (int)(input.ElementCount / dim);
        EnterOp();
        EnsureKernels();
        ulong pOut = 0, pIn = 0, pScale = 0, pShift = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pScale = GpuTransferHelper.CopyToDevice(scale);
            if (shift is not null) pShift = GpuTransferHelper.CopyToDevice(shift);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchLtx2RmsModulate(pOut, pIn, pScale, pShift, dim, rows, eps, _stream.Handle, f16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pScale);
            if (pShift != 0) GpuTransferHelper.FreeDevice(pShift);
        }
    }

    /// <summary>Fused LTX-2 per-head output gate — see <see cref="IBackend.Ltx2HeadGate"/>.</summary>
    /// <remarks>In-place: x's device buffer is mutated and re-cached to the SAME pointer, the
    /// <see cref="Ltx2SplitRope"/> pattern. Allocating a fresh output here would leave the mutation in a buffer
    /// nothing reads.</remarks>
    public unsafe void Ltx2HeadGate(Tensor x, Tensor logits, int seq, int heads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2HeadGate");
        bool f16 = x.DType == DType.F16;
        if ((x.DType != DType.F32 && !f16) || logits.DType != x.DType)
            throw new NotSupportedException(
                $"CUDA Ltx2HeadGate needs matching F32 or F16 x/logits, got x={x.DType}, logits={logits.DType}.");
        EnterOp();
        EnsureKernels();
        ulong pX = 0, pLogits = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);          // in-place target (cached activation)
            pLogits = GpuTransferHelper.CopyToDevice(logits);
            _kernels!.LaunchLtx2HeadGate(pX, pLogits, seq, heads, headDim, _stream.Handle, f16);
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));   // in-place: re-assert x -> pX
        }
        finally { GpuTransferHelper.FreeDevice(pLogits); }
    }

    /// <summary>Wan-Video interleaved in-place RoPE (shared cos/sin) — keeps q/k resident so RmsNorm→RoPE→SDPA never leaves the device.</summary>
    /// <remarks>In-place: x's device buffer is rotated and re-cached to the same pointer (no reallocation, no host round-trip).</remarks>
    public unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2SplitRope");
        // F32, or F16 activation x with F32 cos/sin (DiT F16 recipe), same as WanRopeInterleaved below.
        bool f16 = x.DType == DType.F16;
        if ((x.DType != DType.F32 && !f16) || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException(
                $"CUDA Ltx2SplitRope supports F32, or F16 x with F32 cos/sin — got x={x.DType}, cos={cos.DType}, sin={sin.DType}.");
        EnterOp();
        EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);       // in-place target (cached activation)
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchLtx2SplitRope(pX, pCos, pSin, seqLen, numHeads, headDim, _stream.Handle, f16);
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));   // in-place: re-assert x → pX
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    public unsafe void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleaved");
        // F32, or F16 activation x with F32 cos/sin (DiT F16 recipe). cos/sin stay F32 in both cases.
        bool f16 = x.DType == DType.F16;
        if ((!f16 && x.DType != DType.F32) || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA WanRopeInterleaved supports F32, or F16 x with F32 cos/sin.");
        EnterOp();
        EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);       // x's device buffer (cached activation, in-place target)
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            if (f16)
                _kernels!.LaunchWanRopeInterleavedF16(pX, pCos, pSin, seqLen, heads, headDim, _stream.Handle);
            else
                _kernels!.LaunchWanRopeInterleaved(pX, pCos, pSin, seqLen, heads, headDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));   // in-place: re-assert x → pX
        }
        finally
        {
            // pX is now x's cached buffer (freed on x.Dispose) — do NOT free it here. Free the cos/sin inputs
            // (FreeDevice skips them if they are cached activations).
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>Per-head interleaved RoPE (Matrix-Game 3.0 sigma_theta): cos/sin are <c>[heads, S, headDim]</c>.</summary>
    /// <remarks>Ported off the host loop that dominated the MG3 backbone (~1.9 s of a 2.05 s forward → ~0.15 s).</remarks>
    public unsafe void WanRopeInterleavedPerHead(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleavedPerHead");
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA WanRopeInterleavedPerHead supports F32 x/cos/sin.");
        EnterOp();
        EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchWanRopeInterleavedPerHead(pX, pCos, pSin, seqLen, heads, headDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    // ── Matrix-Game 3.0 ActionModule temporal-batched rearranges (GPU port of the host pointer-loops) ──

    public void Mg3SplitQkvTemporal(Tensor outT, Tensor qkv, int tt, int sp, int heads, int headDim, int part, int stride)
    {
        if (outT.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("CUDA Mg3SplitQkvTemporal F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(qkv);
            nuint outBytes = GpuTransferHelper.ByteSize(outT);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchMg3SplitQkvTemporal(pIn, pOut, tt, sp, heads, headDim, part, stride, _stream.Handle);
            GpuTransferHelper.CacheActivation(outT, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    public void Mg3MergeTemporal(Tensor outT, Tensor attn, int tt, int sp, int heads, int headDim)
    {
        if (outT.DType != DType.F32 || attn.DType != DType.F32) throw new NotSupportedException("CUDA Mg3MergeTemporal F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(attn);
            nuint outBytes = GpuTransferHelper.ByteSize(outT);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchMg3MergeTemporal(pIn, pOut, tt, sp, heads, headDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(outT, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); }
    }

    public void Mg3RopeBatched(Tensor x, Tensor cos, Tensor sin, int sp, int heads, int tt, int headDim, int gh, int gw, bool broadcastSpatial)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA Mg3RopeBatched F32 only.");
        EnterOp(); EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchMg3RopeBatched(pX, pCos, pSin, sp, heads, tt, headDim, gh, gw, broadcastSpatial ? 1 : 0, _stream.Handle);
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally { GpuTransferHelper.FreeDevice(pCos); GpuTransferHelper.FreeDevice(pSin); }
    }

    public void Mg3MouseMlpConcat(Tensor outT, Tensor hidden, Tensor mouseWin, int tt, int sp, int imgDim, int winFloats)
    {
        if (outT.DType != DType.F32 || hidden.DType != DType.F32 || mouseWin.DType != DType.F32)
            throw new NotSupportedException("CUDA Mg3MouseMlpConcat F32 only.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pHidden = 0, pWin = 0; bool cached = false;
        try
        {
            pHidden = GpuTransferHelper.CopyToDevice(hidden);
            pWin = GpuTransferHelper.CopyToDevice(mouseWin);
            nuint outBytes = GpuTransferHelper.ByteSize(outT);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchMg3MouseMlpConcat(pHidden, pWin, pOut, tt, sp, imgDim, winFloats, _stream.Handle);
            GpuTransferHelper.CacheActivation(outT, pOut, outBytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pHidden); GpuTransferHelper.FreeDevice(pWin); }
    }

    public void Mg3KvExpand(Tensor kOut, Tensor vOut, Tensor kv, int sp, int heads, int tt, int headDim)
    {
        if (kOut.DType != DType.F32 || vOut.DType != DType.F32 || kv.DType != DType.F32)
            throw new NotSupportedException("CUDA Mg3KvExpand F32 only.");
        EnterOp(); EnsureKernels();
        ulong pKv = 0, pK = 0, pV = 0; bool cachedK = false, cachedV = false;
        try
        {
            pKv = GpuTransferHelper.CopyToDevice(kv);
            nuint bytes = GpuTransferHelper.ByteSize(kOut);
            pK = GpuTransferHelper.AllocateDevice(bytes);
            pV = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchMg3KvExpand(pKv, pK, pV, sp, heads, tt, headDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(kOut, pK, bytes); cachedK = true;
            GpuTransferHelper.CacheActivation(vOut, pV, bytes); cachedV = true;
        }
        finally
        {
            if (!cachedK) GpuTransferHelper.FreeDevice(pK);
            if (!cachedV) GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pKv);
        }
    }

    /// <summary>Resolves the wan_vae kernel dtype: F32 → false, BF16 → true (bias, when present, must be F32 — the BF16 kernels read it as float).</summary>
    private static bool RequireVaeFrameDtype(DType a, DType b, Tensor? bias = null)
    {
        if (a != b || (a != DType.F32 && a != DType.BF16))
            throw new NotSupportedException($"CUDA VAE frame ops support matching F32 or BF16 tensors, got {a}/{b}.");
        if (a == DType.BF16 && bias is not null && bias.DType != DType.F32)
            throw new NotSupportedException($"CUDA VAE frame ops require an F32 bias on the BF16 path, got {bias.DType}.");
        return a == DType.BF16;
    }

    /// <summary>GPU temporal frame extract: output[B,C,H,W] = src[B,C,Tsrc,H,W][:,:,ti,:,:]. Keeps CausalConv3d slicing on-device (no D2H).</summary>
    public unsafe void ExtractVaeFrame(Tensor output, Tensor src, int ti)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("ExtractVaeFrame");
        EnterOp();
        EnsureKernels();
        int b = (int)output.Shape[0], c = (int)output.Shape[1];
        int frameHW = (int)(output.ElementCount / ((long)b * c));
        int tsrc = (int)src.Shape[2];
        ulong pOut = 0, pSrc = 0;
        bool cachedOutput = false;
        try
        {
            pSrc = GpuTransferHelper.CopyToDevice(src);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeExtractFrame(pOut, pSrc, ti, b, c, tsrc, frameHW, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, src.DType));
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pSrc);
        }
    }

    /// <summary>GPU temporal frame write (in-place): output[B,C,Tout,H,W][:, :, to, :, :] = acc[B,C,H,W] + bias[c].</summary>
    /// <remarks>Writes one slot of <paramref name="output"/>'s device buffer; accumulates CausalConv3d frames without leaving the device.</remarks>
    public unsafe void WriteVaeFrame(Tensor output, Tensor acc, Tensor? bias, int to)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("WriteVaeFrame");
        EnterOp();
        EnsureKernels();
        int b = (int)acc.Shape[0], c = (int)acc.Shape[1];
        int frameHW = (int)(acc.ElementCount / ((long)b * c));
        int tout = (int)output.Shape[2];
        ulong pOut = 0, pAcc = 0, pBias = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);   // in-place target: output's device buffer
            pAcc = GpuTransferHelper.CopyToDevice(acc);
            if (bias is not null) pBias = GpuTransferHelper.CopyToDevice(bias);
            _kernels!.LaunchWanVaeWriteFrame(pOut, pAcc, pBias, to, b, c, tout, frameHW, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, acc.DType, bias));
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));   // in-place re-assert
        }
        finally
        {
            // pOut is output's cached buffer — do NOT free. Free acc/bias inputs (skipped if cached).
            GpuTransferHelper.FreeDevice(pAcc);
            if (pBias != 0) GpuTransferHelper.FreeDevice(pBias);
        }
    }

    /// <summary>GPU build of the frame-major padded input for batched CausalConv3d: kt Conv2D calls, not tout·kt tiny ones.</summary>
    public unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0, bool reflectSpatial = false)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("BuildPaddedFrames");
        EnterOp();
        EnsureKernels();
        int paddedT = (int)padded.Shape[0], cIn = (int)padded.Shape[1];
        int h = (int)input.Shape[3], w = (int)input.Shape[4];
        int Tin = (int)input.Shape[2];
        int cacheLen = cache is null ? 0 : (int)cache.Shape[2];
        ulong pOut = 0, pIn = 0, pCache = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            if (cache is not null) pCache = GpuTransferHelper.CopyToDevice(cache);
            nuint outBytes = GpuTransferHelper.ByteSize(padded);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeBuildPadded(pOut, pIn, pCache, paddedT, cIn, Tin, cacheLen, zeroPad, h, w,
                padH, padW, replicateFirst, _stream.Handle,
                bf16: RequireVaeFrameDtype(padded.DType, input.DType), reflectSpatial: reflectSpatial);
            GpuTransferHelper.CacheActivation(padded, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            if (pCache != 0) GpuTransferHelper.FreeDevice(pCache);
        }
    }

    /// <summary>GPU MAGViT channel→space shuffle (SeedVR2 upsampler) — replaces the host per-element loop that forced a multi-GB D2H+H2D round trip per upsampler.</summary>
    public unsafe void SeedVr2PixelShuffle(Tensor output, Tensor input, int spatialRatio, int temporalRatio)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("SeedVr2PixelShuffle");
        EnterOp();
        EnsureKernels();
        bool bf16 = RequireVaeFrameDtype(output.DType, input.DType);
        int cIn = (int)input.Shape[1], f = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int c = (int)output.Shape[1], fFinal = (int)output.Shape[2], hOut = (int)output.Shape[3], wOut = (int)output.Shape[4];
        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchSeedVr2PixelShuffle(pOut, pIn, c, fFinal, hOut, wOut, cIn, f, h, w,
                spatialRatio, temporalRatio, dropDup: temporalRatio > 1, _stream.Handle, bf16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU asymmetric (0,1,0,1) zero pad (SeedVR2 downsampler) — replaces the host copy loop.</summary>
    public unsafe void SeedVr2PadBottomRight(Tensor output, Tensor input)
    {
        EnterOp();
        EnsureKernels();
        bool bf16 = RequireVaeFrameDtype(output.DType, input.DType);
        int h = (int)input.Shape[3], w = (int)input.Shape[4];
        long planes = input.ElementCount / ((long)h * w);
        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchSeedVr2PadBottomRight(pOut, pIn, planes, h, w, _stream.Handle, bf16);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU fill of the conv output with per-channel bias (init for the temporal accumulate).</summary>
    public unsafe void FillBias(Tensor output, Tensor? bias)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("FillBias");
        EnterOp();
        EnsureKernels();
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        int HW = (int)(output.ElementCount / ((long)cOut * tout));
        ulong pOut = 0, pBias = 0;
        bool cachedOutput = false;
        try
        {
            if (bias is not null) pBias = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeFillBias(pOut, pBias, cOut, tout, HW, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, output.DType, bias));
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (pBias != 0) GpuTransferHelper.FreeDevice(pBias);
        }
    }

    /// <summary>GPU temporal gather-sum (in place): output += the dt-shifted frames of convDt.</summary>
    public unsafe void AccumulateTap(Tensor output, Tensor convDt, int dt, int strideT)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("AccumulateTap");
        EnterOp();
        EnsureKernels();
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        int HW = (int)(output.ElementCount / ((long)cOut * tout));
        ulong pOut = 0, pConv = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);   // in-place accumulate target
            pConv = GpuTransferHelper.CopyToDevice(convDt);
            _kernels!.LaunchWanVaeAccumulateTap(pOut, pConv, dt, strideT, cOut, tout, HW, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, convDt.DType));
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));   // in-place re-assert
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pConv);   // pOut is output's cached buffer — do not free
        }
    }

    /// <summary>GPU channel-wise RMS norm for the Wan2.2 VAE (one thread per <c>[B, spatial]</c> position reduces over C).</summary>
    public unsafe void WanRmsNormChannel(Tensor output, Tensor input, Tensor? gamma, float eps)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("WanRmsNormChannel");
        EnterOp();
        EnsureKernels();
        int b = (int)input.Shape[0], c = (int)input.Shape[1];
        long spatial = input.ElementCount / ((long)b * c);
        long numPos = (long)b * spatial;
        float sqrtC = MathF.Sqrt(c);
        ulong pOut = 0, pIn = 0, pGamma = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            if (gamma is not null) pGamma = GpuTransferHelper.CopyToDevice(gamma);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeRmsNormChannel(pOut, pIn, pGamma, c, spatial, eps, sqrtC, numPos, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, input.DType, gamma));
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            if (pGamma != 0) GpuTransferHelper.FreeDevice(pGamma);
        }
    }

    /// <summary>GPU image-output conversion (CHW F32[-1,1] → HWC u8): converts on-device so only the 3 MB image crosses PCIe (~140→30 ms).</summary>
    public unsafe void ChwF32ToHwcU8(Tensor output, Tensor input)
    {
        EnterOp();
        EnsureKernels();
        int height = (int)input.Shape[2], width = (int)input.Shape[3];
        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchChwToHwcU8(pOut, pIn, height, width, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU Wan2.2 VAE unpatchify (pixel-shuffle), one thread per output element.</summary>
    public unsafe void UnpatchifyVae(Tensor output, Tensor input, int patchSize)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("UnpatchifyVae");
        EnterOp();
        EnsureKernels();
        int b = (int)input.Shape[0], packedC = (int)input.Shape[1], t = (int)input.Shape[2], h = (int)input.Shape[3], w = (int)input.Shape[4];
        int p = patchSize, c = packedC / (p * p);
        long numOut = output.ElementCount;
        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeUnpatchify(pOut, pIn, b, c, t, h, w, p, numOut, _stream.Handle,
                bf16: RequireVaeFrameDtype(output.DType, input.DType));
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU Wan2.2 VAE attention qkv split (channel↔token transpose into three [bt,1,hw,c] tensors).</summary>
    public unsafe void SplitVaeQkv(Tensor q, Tensor k, Tensor v, Tensor qkv, int bt, int c, int hw)
    {
        EnterOp();
        EnsureKernels();
        long numEl = (long)bt * c * hw;
        ulong pQ = 0, pK = 0, pV = 0, pSrc = 0;
        bool cached = false;
        try
        {
            pSrc = GpuTransferHelper.CopyToDevice(qkv);
            nuint bytes = GpuTransferHelper.ByteSize(q);
            pQ = GpuTransferHelper.AllocateDevice(bytes);
            pK = GpuTransferHelper.AllocateDevice(bytes);
            pV = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchWanVaeSplitQkv(pQ, pK, pV, pSrc, bt, c, hw, numEl, _stream.Handle);
            GpuTransferHelper.CacheActivation(q, pQ, bytes);
            GpuTransferHelper.CacheActivation(k, pK, bytes);
            GpuTransferHelper.CacheActivation(v, pV, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) { GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV); }
            GpuTransferHelper.FreeDevice(pSrc);
        }
    }

    /// <summary>GPU Wan2.2 VAE attention output un-transpose ([bt,1,hw,c] → [bt,c,h,w]).</summary>
    public unsafe void VaeTokensToFrame(Tensor output, Tensor attn, int bt, int c, int hw)
    {
        EnterOp();
        EnsureKernels();
        long numEl = (long)bt * c * hw;
        ulong pOut = 0, pAttn = 0;
        bool cachedOutput = false;
        try
        {
            pAttn = GpuTransferHelper.CopyToDevice(attn);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchWanVaeTokensToFrame(pOut, pAttn, bt, c, hw, numEl, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pAttn);
        }
    }

    /// <summary>In-place rotary embedding on a GPU-resident tensor <c>[B, L, numHeads, headDim]</c>; cos/sin are <c>[B, L, headDim]</c>.</summary>
    /// <remarks>Used by grouped-query attention where Q and K differ in head count (the paired <see cref="ApplyRope"/> would mis-stride K).</remarks>
    public void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRopeSingle supports F32 only.");
        using NvtxRange _nvtx = NvtxRange.Push("ApplyRopeSingle");
        EnterOp();
        EnsureKernels();
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        long totalVecs = x.ElementCount / headDim;

        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchRope(pX, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle, rotaryDim);

            // In-place: clear stale callbacks before re-caching (pitfall #17).
            x._gpuSyncCallback = null;
            x._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>In-place rotary embedding on a head-major GPU-resident tensor <c>[B, heads, L, headDim]</c>; cos/sin stay <c>[B, L, headDim]</c>.</summary>
    /// <remarks>Same rotation as <see cref="ApplyRopeSingle"/>; only the token index differs (<c>vec % seq</c> instead
    /// of <c>vec / numHeads</c>), so q/k written head-major by <see cref="QkvSplitNormHeadMajor"/> can be roped without
    /// a round-trip through token-major layout.</remarks>
    public void ApplyRopeSingleHeadMajor(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRopeSingleHeadMajor supports F32 only.");
        if (x.Shape.Rank != 4)
            throw new HartsyInferenceException($"ApplyRopeSingleHeadMajor needs x shaped [B,heads,seq,headDim], got {x.Shape}.");
        using NvtxRange _nvtx = NvtxRange.Push("ApplyRopeHeadMajor");
        EnterOp();
        EnsureKernels();
        int heads = (int)x.Shape[1];
        int seq = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // A token-major x is the same element count with heads/seq swapped; only the cos/sin row count catches it.
        long tableElements = x.Shape[0] * seq * headDim;
        if (cos.ElementCount != tableElements || sin.ElementCount != tableElements)
            throw new HartsyInferenceException(
                $"ApplyRopeSingleHeadMajor expects cos/sin [B,seq,headDim] for x {x.Shape}; got cos={cos.Shape}, sin={sin.Shape}.");
        long totalVecs = x.ElementCount / headDim;

        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            int batch = (int)x.Shape[0];
            if (EnableRopeHeadMajorV2 && _kernels!.CanUseRopeHeadMajorV2(heads, headDim, seq, batch, rotaryDim))
            {
                _kernels!.LaunchRopeHeadMajorV2(pX, pCos, pSin, heads, headDim, seq, batch, _stream.Handle, rotaryDim);
            }
            else
            {
                _kernels!.LaunchRopeHeadMajor(pX, pCos, pSin, heads, headDim, seq, totalVecs, _stream.Handle, rotaryDim);
            }

            // In-place: clear stale callbacks before re-caching (pitfall #17).
            x._gpuSyncCallback = null;
            x._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    /// <summary>In-place interleaved (GPT-J) RoPE on a GPU-resident <c>[B,L,numHeads,headDim]</c> tensor (pairs rotated by frequency i).</summary>
    /// <remarks>cos/sin are <c>[B, L, headDim]</c>. Without this override the shared <see cref="IBackend"/> CPU
    /// fallback would drag q/k off the device every layer (Sesame CSM / HeartMuLa use interleaved RoPE) — the whole
    /// RmsNorm→RoPE→SDPA chain stays resident.</remarks>
    public unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleaved");
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRopeInterleaved supports F32 only.");
        EnterOp();
        EnsureKernels();
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        long totalVecs = x.ElementCount / headDim;

        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchRopeInterleaved(pX, pCos, pSin, numHeads, headDim, rotaryDim, totalVecs, _stream.Handle);

            // In-place: clear stale callbacks before re-caching (pitfall #17).
            x._gpuSyncCallback = null;
            x._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pCos);
            GpuTransferHelper.FreeDevice(pSin);
        }
    }

    public void QkvRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qkv,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvRopeScatter");
        if (devicePos == 0 || qkv.DType != DType.F32 || kCache.DType != DType.F32 || vCache.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA QkvRopeScatterDecodeStep requires a device position buffer and F32 qkv/caches; got devicePos={devicePos}, " +
                $"qkv={qkv.DType}, kCache={kCache.DType}, vCache={vCache.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int maxSeq = (int)kCache.Shape[2];
        ulong pQkv = 0, pCos = 0, pSin = 0, pK = 0, pV = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQkv = GpuTransferHelper.CopyToDevice(qkv);
            pCos = GpuTransferHelper.CopyToDevice(cosTable);
            pSin = GpuTransferHelper.CopyToDevice(sinTable);
            pK = GpuTransferHelper.CopyToDevice(kCache);
            pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            ulong kOff = pQkv + (ulong)((long)hq * headDim * sizeof(float));
            ulong vOff = kOff + (ulong)((long)hkv * headDim * sizeof(float));
            _kernels!.LaunchQkvRopeScatter(pQ, pK, pV, pQkv, kOff, vOff, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, maxSeq, devicePos, _stream.Handle);
            // The caches were written in place — keep them resident without a host sync (same contract as
            // KvCacheAppendDev).
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQkv);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void QkRopeScatterVDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor v,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkRopeScatterV");
        if (devicePos == 0 || qk.DType != DType.F32 || v.DType != DType.F32 || kCache.DType != DType.F32 || vCache.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA QkRopeScatterVDecodeStep requires a device position buffer and F32 qk/v/caches; got devicePos={devicePos}, " +
                $"qk={qk.DType}, v={v.DType}, kCache={kCache.DType}, vCache={vCache.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int maxSeq = (int)kCache.Shape[2];
        ulong pQk = 0, pVi = 0, pCos = 0, pSin = 0, pK = 0, pV = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQk = GpuTransferHelper.CopyToDevice(qk);
            pVi = GpuTransferHelper.CopyToDevice(v);
            pCos = GpuTransferHelper.CopyToDevice(cosTable);
            pSin = GpuTransferHelper.CopyToDevice(sinTable);
            pK = GpuTransferHelper.CopyToDevice(kCache);
            pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            ulong kOff = pQk + (ulong)((long)hq * headDim * sizeof(float));
            _kernels!.LaunchQkvRopeScatter(pQ, pK, pV, pQk, kOff, pVi, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, maxSeq, devicePos, _stream.Handle);
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQk);
            GpuTransferHelper.FreeDevice(pVi);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void RopeScatterKvDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor q, Tensor k, Tensor v,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeScatterKv");
        if (devicePos == 0 || q.DType != DType.F32 || kCache.DType != DType.F32 || vCache.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA RopeScatterKvDecodeStep requires a device position buffer and F32 q/caches; got devicePos={devicePos}, " +
                $"q={q.DType}, kCache={kCache.DType}, vCache={vCache.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int maxSeq = (int)kCache.Shape[2];
        ulong pQi = 0, pKi = 0, pVi = 0, pCos = 0, pSin = 0, pK = 0, pV = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQi = GpuTransferHelper.CopyToDevice(q);
            pKi = GpuTransferHelper.CopyToDevice(k);
            pVi = GpuTransferHelper.CopyToDevice(v);
            pCos = GpuTransferHelper.CopyToDevice(cosTable);
            pSin = GpuTransferHelper.CopyToDevice(sinTable);
            pK = GpuTransferHelper.CopyToDevice(kCache);
            pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            _kernels!.LaunchQkvRopeScatter(pQ, pK, pV, pQi, pKi, pVi, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, maxSeq, devicePos, _stream.Handle);
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQi);
            GpuTransferHelper.FreeDevice(pKi);
            GpuTransferHelper.FreeDevice(pVi);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void QkvNormRopeScatterDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qkv,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvNormRopeScatter");
        if (devicePos == 0 || qkv.DType != DType.F32 || kCache.DType != DType.F32 || vCache.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA QkvNormRopeScatterDecodeStep requires a device position buffer and F32 qkv/caches; got devicePos={devicePos}, " +
                $"qkv={qkv.DType}, kCache={kCache.DType}, vCache={vCache.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int maxSeq = (int)kCache.Shape[2];
        ulong pQkv = 0, pQw = 0, pKw = 0, pCos = 0, pSin = 0, pK = 0, pV = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQkv = GpuTransferHelper.CopyToDevice(qkv);
            pQw = GpuTransferHelper.CopyToDevice(qNorm);
            pKw = GpuTransferHelper.CopyToDevice(kNorm);
            pCos = GpuTransferHelper.CopyToDevice(cosTable);
            pSin = GpuTransferHelper.CopyToDevice(sinTable);
            pK = GpuTransferHelper.CopyToDevice(kCache);
            pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            ulong kOff = pQkv + (ulong)((long)hq * headDim * sizeof(float));
            ulong vOff = kOff + (ulong)((long)hkv * headDim * sizeof(float));
            _kernels!.LaunchQkNormRopeScatter(pQ, pK, pV, pQkv, kOff, vOff, pQw, pKw, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos, _stream.Handle);
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQkv);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void QkNormRopeScatterVDecodeStep(Tensor qOut, Tensor kCache, Tensor vCache, Tensor qk, Tensor v,
        Tensor qNorm, Tensor kNorm, float eps,
        Tensor cosTable, Tensor sinTable, int hq, int hkv, int headDim, int rotaryDim, bool interleaved, ulong devicePos)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkNormRopeScatterV");
        if (devicePos == 0 || qk.DType != DType.F32 || v.DType != DType.F32 || kCache.DType != DType.F32 || vCache.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA QkNormRopeScatterVDecodeStep requires a device position buffer and F32 qk/v/caches; got devicePos={devicePos}, " +
                $"qk={qk.DType}, v={v.DType}, kCache={kCache.DType}, vCache={vCache.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int maxSeq = (int)kCache.Shape[2];
        ulong pQk = 0, pVi = 0, pQw = 0, pKw = 0, pCos = 0, pSin = 0, pK = 0, pV = 0, pQ = 0;
        bool cachedOutput = false;
        try
        {
            pQk = GpuTransferHelper.CopyToDevice(qk);
            pVi = GpuTransferHelper.CopyToDevice(v);
            pQw = GpuTransferHelper.CopyToDevice(qNorm);
            pKw = GpuTransferHelper.CopyToDevice(kNorm);
            pCos = GpuTransferHelper.CopyToDevice(cosTable);
            pSin = GpuTransferHelper.CopyToDevice(sinTable);
            pK = GpuTransferHelper.CopyToDevice(kCache);
            pV = GpuTransferHelper.CopyToDevice(vCache);
            nuint qBytes = GpuTransferHelper.ByteSize(qOut);
            pQ = GpuTransferHelper.AllocateDevice(qBytes);
            ulong kOff = pQk + (ulong)((long)hq * headDim * sizeof(float));
            _kernels!.LaunchQkNormRopeScatter(pQ, pK, pV, pQk, kOff, pVi, pQw, pKw, pCos, pSin,
                hq, hkv, headDim, rotaryDim, interleaved, eps, maxSeq, devicePos, _stream.Handle);
            kCache._gpuSyncCallback = null;
            kCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(kCache, pK, GpuTransferHelper.ByteSize(kCache));
            vCache._gpuSyncCallback = null;
            vCache._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(vCache, pV, GpuTransferHelper.ByteSize(vCache));
            GpuTransferHelper.CacheActivation(qOut, pQ, qBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQk);
            GpuTransferHelper.FreeDevice(pVi);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pQ);
        }
    }

    public void PleGatherDecodeStep(Tensor output, Tensor quantTable, float scale, ulong deviceTokenId)
    {
        using NvtxRange _nvtx = NvtxRange.Push("PleGather");
        if (quantTable.DType != DType.Q5_K || output.DType != DType.F32 || deviceTokenId == 0)
            throw new NotSupportedException("PleGatherDecodeStep requires a Q5_K table, F32 output, and a device token id.");
        EnterOp();
        EnsureKernels();
        int width = (int)output.ElementCount;
        ulong pTable = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pTable = GpuTransferHelper.CopyToDevice(quantTable);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchPleGatherQ5K(pOut, pTable, width, scale, deviceTokenId, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pTable);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    public void SsmConvSplitStep(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SsmConvSplit");
        EnterOp();
        EnsureKernels();
        int convDim = (int)qkvMixed.ElementCount;
        int keyDim = kHeads * skDim, valueDim = vHeads * svDim;
        ulong pQ = 0, pK = 0, pV = 0, pM = 0, pH = 0, pW = 0;
        bool cached = false;
        try
        {
            pM = GpuTransferHelper.CopyToDevice(qkvMixed);
            pH = GpuTransferHelper.CopyToDevice(history);
            pW = GpuTransferHelper.CopyToDevice(convW);
            pQ = GpuTransferHelper.AllocateDevice((nuint)(keyDim * sizeof(float)));
            pK = GpuTransferHelper.AllocateDevice((nuint)(keyDim * sizeof(float)));
            pV = GpuTransferHelper.AllocateDevice((nuint)(valueDim * sizeof(float)));
            _kernels!.LaunchSsmConvSplitStep(pQ, pK, pV, pM, pH, pW, convDim, keyDim, valueDim, kernel, _stream.Handle);
            _kernels!.LaunchSsmL2NormHeads(pQ, kHeads, skDim, eps, qScale, _stream.Handle);
            _kernels!.LaunchSsmL2NormHeads(pK, kHeads, skDim, eps, 1.0f, _stream.Handle);
            GpuTransferHelper.CacheActivation(q, pQ, (nuint)(keyDim * sizeof(float)));
            GpuTransferHelper.CacheActivation(k, pK, (nuint)(keyDim * sizeof(float)));
            GpuTransferHelper.CacheActivation(v, pV, (nuint)(valueDim * sizeof(float)));
            // history was mutated in place — keep it resident (KV-cache pattern, but retain the sync
            // callback so a host prefill can pull it back down).
            GpuTransferHelper.CacheActivation(history, pH, GpuTransferHelper.ByteSize(history));
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pM);
            GpuTransferHelper.FreeDevice(pW);
            if (!cached) { GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV); }
        }
    }

    public void SsmDeltaStep(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z,
        Tensor alphaRaw, Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW,
        int hv, int sv, int sk, int repeat, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SsmDeltaStep");
        EnterOp();
        EnsureKernels();
        ulong pO = 0, pSt = 0, pQ = 0, pK = 0, pV = 0, pZ = 0, pAl = 0, pBe = 0, pDt = 0, pSa = 0, pNw = 0;
        bool cached = false;
        try
        {
            pSt = GpuTransferHelper.CopyToDevice(state);
            pQ = GpuTransferHelper.CopyToDevice(q);
            pK = GpuTransferHelper.CopyToDevice(k);
            pV = GpuTransferHelper.CopyToDevice(v);
            pZ = GpuTransferHelper.CopyToDevice(z);
            pAl = GpuTransferHelper.CopyToDevice(alphaRaw);
            pBe = GpuTransferHelper.CopyToDevice(betaRaw);
            pDt = GpuTransferHelper.CopyToDevice(dtBias);
            pSa = GpuTransferHelper.CopyToDevice(ssmA);
            pNw = GpuTransferHelper.CopyToDevice(normW);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pO = GpuTransferHelper.AllocateDevice(outBytes);
            // Persistent scratch for the row-parallel delta kernel's pre-norm readout. Sized on
            // first use (the pre-capture warmup step), NEVER grown during capture — a graph would
            // bake a dead pointer. hv*sv is model-constant, so a capture-time miss means the
            // warmup was skipped; the launcher then falls back to the legacy single kernel.
            nuint scratchBytes = (nuint)((hv * sv + 2 * hv) * sizeof(float));   // readout + per-head gate scalars
            if (_ssmDeltaScratchBytes < scratchBytes && !StreamIsCapturing())
            {
                if (_ssmDeltaScratch != 0) GpuTransferHelper.FreeDevice(_ssmDeltaScratch);
                _ssmDeltaScratch = GpuTransferHelper.AllocateDevice(scratchBytes);
                _ssmDeltaScratchBytes = scratchBytes;
            }
            ulong oScratch = _ssmDeltaScratchBytes >= scratchBytes ? _ssmDeltaScratch : 0;
            _kernels!.LaunchSsmDeltaStep(pO, pSt, pQ, pK, pV, pZ, pAl, pBe, pDt, pSa, pNw,
                hv, sv, sk, repeat, eps, _stream.Handle, oScratch);
            GpuTransferHelper.CacheActivation(output, pO, outBytes);
            GpuTransferHelper.CacheActivation(state, pSt, GpuTransferHelper.ByteSize(state));   // mutated in place, keep resident
            cached = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ); GpuTransferHelper.FreeDevice(pK); GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pZ); GpuTransferHelper.FreeDevice(pAl); GpuTransferHelper.FreeDevice(pBe);
            GpuTransferHelper.FreeDevice(pDt); GpuTransferHelper.FreeDevice(pSa); GpuTransferHelper.FreeDevice(pNw);
            if (!cached) GpuTransferHelper.FreeDevice(pO);
        }
    }

    public void AddRmsNorm(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AddRmsNorm");
        // Unfused composition for the dtypes the fused kernel lacks; an ((IBackend)this) hop would re-enter here.
        if (residOut.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32 || weight.DType != DType.F32)
        {
            Add(residOut, a, b);
            RmsNorm(normOut, residOut, weight, eps);
            return;
        }
        EnterOp();
        EnsureKernels();
        int normDim = (int)weight.ElementCount;
        int rows = (int)(a.ElementCount / normDim);
        ulong pA = 0, pB = 0, pW = 0, pResid = 0, pNorm = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pW = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(residOut);
            pResid = GpuTransferHelper.AllocateDevice(outBytes);
            pNorm = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAddRmsNorm(pResid, pNorm, pA, pB, pW, normDim, rows, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(residOut, pResid, outBytes);
            GpuTransferHelper.CacheActivation(normOut, pNorm, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (!cachedOutput) { GpuTransferHelper.FreeDevice(pResid); GpuTransferHelper.FreeDevice(pNorm); }
        }
    }

    public void AddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
    {
        int normDim = (int)weight.ElementCount;
        long rows = a.ElementCount / normDim;
        if (!_quantAtProducer || !EnableDp4aGemv || rows != 1 || normDim % 32 != 0
            || residOut.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32 || weight.DType != DType.F32)
        {
            AddRmsNorm(residOut, normOut, a, b, weight, eps);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("AddRmsNormQ8");
        EnterOp();
        EnsureKernels();
        ulong pA = 0, pB = 0, pW = 0, pResid = 0, pNorm = 0, xq = 0, xd = 0, xs = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pW = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(residOut);
            pResid = GpuTransferHelper.AllocateDevice(outBytes);
            pNorm = GpuTransferHelper.AllocateDevice(outBytes);
            (xq, xd, xs) = AllocSidecar(normDim);
            _kernels!.LaunchAddRmsNormQ8(pResid, pNorm, xq, xd, xs, pA, pB, pW, normDim, 1, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(residOut, pResid, outBytes);
            GpuTransferHelper.CacheActivation(normOut, pNorm, outBytes);
            GpuTransferHelper.RegisterSidecar(normOut, xq, xd, xs, normDim);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            if (!cachedOutput)
            {
                GpuTransferHelper.FreeDevice(pResid);
                GpuTransferHelper.FreeDevice(pNorm);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
        }
    }

    public void NormAddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor w1, Tensor w2, float eps)
    {
        int normDim = (int)w2.ElementCount;
        long rows = a.ElementCount / normDim;
        if (residOut.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32
            || w1.DType != DType.F32 || w2.DType != DType.F32 || (int)w1.ElementCount != normDim)
        {
            // Unfused composition for the dtypes the fused kernel lacks; an ((IBackend)this) hop would re-enter here.
            using Tensor n1 = new Tensor(b.Shape, b.DType);
            RmsNorm(n1, b, w1, eps);
            AddRmsNormEmitQ8(residOut, normOut, a, n1, w2, eps);
            return;
        }
        bool sidecar = _quantAtProducer && EnableDp4aGemv && rows == 1 && normDim % 32 == 0;
        using NvtxRange _nvtx = NvtxRange.Push("NormAddRmsNorm");
        EnterOp();
        EnsureKernels();
        ulong pA = 0, pB = 0, pW1 = 0, pW2 = 0, pResid = 0, pNorm = 0, xq = 0, xd = 0, xs = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pW1 = GpuTransferHelper.CopyToDevice(w1);
            pW2 = GpuTransferHelper.CopyToDevice(w2);
            nuint outBytes = GpuTransferHelper.ByteSize(residOut);
            pResid = GpuTransferHelper.AllocateDevice(outBytes);
            pNorm = GpuTransferHelper.AllocateDevice(outBytes);
            if (sidecar) (xq, xd, xs) = AllocSidecar(normDim);
            _kernels!.LaunchNormAddRmsNormQ8(pResid, pNorm, xq, xd, xs, pA, pB, pW1, pW2, normDim, (int)rows, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(residOut, pResid, outBytes);
            GpuTransferHelper.CacheActivation(normOut, pNorm, outBytes);
            if (sidecar) GpuTransferHelper.RegisterSidecar(normOut, xq, xd, xs, normDim);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pW1);
            GpuTransferHelper.FreeDevice(pW2);
            if (!cachedOutput)
            {
                GpuTransferHelper.FreeDevice(pResid);
                GpuTransferHelper.FreeDevice(pNorm);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
        }
    }

    public void RmsNormAdd(Tensor output, Tensor a, Tensor b, Tensor weight, float eps)
    {
        int normDim = (int)weight.ElementCount;
        long rows = a.ElementCount / normDim;
        if (output.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32 || weight.DType != DType.F32)
        {
            // Unfused composition for the dtypes the fused kernel lacks; an ((IBackend)this) hop would re-enter here.
            using Tensor n = new Tensor(b.Shape, b.DType);
            RmsNorm(n, b, weight, eps);
            Add(output, a, n);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("RmsNormAdd");
        EnterOp();
        EnsureKernels();
        ulong pA = 0, pB = 0, pW = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pW = GpuTransferHelper.CopyToDevice(weight);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchRmsNormAdd(pOut, pA, pB, pW, normDim, (int)rows, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pA);
            GpuTransferHelper.FreeDevice(pB);
            GpuTransferHelper.FreeDevice(pW);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    public void GluActivateEmitQ8(Tensor output, Tensor gateUp, int ff, bool gelu)
    {
        long rows = gateUp.ElementCount / (2L * ff);
        if (!_quantAtProducer || !EnableDp4aGemv || rows != 1 || ff % 32 != 0
            || output.DType != DType.F32 || gateUp.DType != DType.F32)
        {
            GluActivate(output, gateUp, ff, gelu);
            return;
        }
        using NvtxRange _nvtx = NvtxRange.Push("GluActivateQ8");
        EnterOp();
        EnsureKernels();
        ulong pOut = 0, pGu = 0, xq = 0, xd = 0, xs = 0;
        bool cachedOutput = false;
        try
        {
            pGu = GpuTransferHelper.CopyToDevice(gateUp);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            (xq, xd, xs) = AllocSidecar(ff);
            _kernels!.LaunchGluActQ8(pOut, xq, xd, xs, pGu, 1, ff, gelu, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            GpuTransferHelper.RegisterSidecar(output, xq, xd, xs, ff);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pGu);
            if (!cachedOutput)
            {
                GpuTransferHelper.FreeDevice(pOut);
                GpuTransferHelper.FreeDevice(xq);
                GpuTransferHelper.FreeDevice(xd);
                GpuTransferHelper.FreeDevice(xs);
            }
        }
    }

    public void GluActivate(Tensor output, Tensor gateUp, int ff, bool gelu)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GluActivate");
        bool bf16 = output.DType == DType.BF16 && gateUp.DType == DType.BF16;
        bool f16 = output.DType == DType.F16 && gateUp.DType == DType.F16;
        if (!bf16 && !f16 && (output.DType != DType.F32 || gateUp.DType != DType.F32))
        {
            throw new NotSupportedException(
                $"CUDA GluActivate supports F32, F16 or BF16 (both sides same dtype); got output={output.DType}, gateUp={gateUp.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int rows = (int)(gateUp.ElementCount / (2L * ff));
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(gateUp);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (bf16)
                _kernels!.LaunchGluActBf16(pOut, pIn, rows, ff, gelu, _stream.Handle);
            else if (f16)
                _kernels!.LaunchGluActF16(pOut, pIn, rows, ff, gelu, _stream.Handle);
            else
                _kernels!.LaunchGluAct(pOut, pIn, rows, ff, gelu, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    public void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SliceLastDim");
        bool f16 = output.DType == DType.F16 && input.DType == DType.F16;
        if (!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            throw new NotSupportedException("CUDA SliceLastDim supports F32 or F16 (both sides same dtype).");
        EnterOp();
        EnsureKernels();
        int inDim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / inDim;
        int outDim = (int)(output.ElementCount / rows);

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (f16)
                _kernels!.LaunchSliceLastDimF16(pOut, pIn, outDim, inDim, offset, output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchSliceLastDim(pOut, pIn, outDim, inDim, offset, output.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void MaskRows(Tensor output, Tensor input, Tensor rowMask)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("MaskRows");
        if (output.DType != DType.F32 || input.DType != DType.F32 || rowMask.DType != DType.F32)
            throw new NotSupportedException("CUDA MaskRows supports F32 only.");
        EnterOp();
        EnsureKernels();
        int channels = (int)input.Shape[input.Shape.Rank - 1];

        ulong pOut = 0, pIn = 0, pMask = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pMask = GpuTransferHelper.CopyToDevice(rowMask);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchRowScale(pOut, pIn, pMask, channels, input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pMask);
        }
    }

    public void AddScalar(Tensor output, Tensor input, float scalar)
    {
        using NvtxRange _nvtx = NvtxRange.Push("AddScalar");
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("CUDA AddScalar supports F32 only.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAddScalar(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void LayerNormNoAffine(Tensor output, Tensor input, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("LayerNormNoAffine");
        // F32, or F16 activation I/O with F32 accumulation — the DiT F16 activation recipe.
        bool f16 = input.DType == DType.F16 && output.DType == DType.F16;
        if (!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            throw new NotSupportedException("CUDA LayerNormNoAffine supports F32 or F16 (matching input/output dtype).");
        EnterOp();
        EnsureKernels();
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / dim;

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (f16)
                _kernels!.LaunchLayerNormNoAffineF16(pOut, pIn, dim, (int)rows, eps, _stream.Handle);
            else
                _kernels!.LaunchLayerNormNoAffine(pOut, pIn, dim, (int)rows, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Fused adaLN modulation: out = (1+scale)·LayerNormNoAffine(in) + shift (F32 or F16 activation, F32 scale/shift).</summary>
    /// <remarks>Replaces LayerNormNoAffine + AddScalar(+1) + AffineBroadcast for the DiT NormModulate.</remarks>
    public void LayerNormModulate(Tensor output, Tensor input, Tensor scale, Tensor shift, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("LayerNormModulate");
        bool f16 = input.DType == DType.F16 && output.DType == DType.F16;
        if (!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            throw new NotSupportedException("CUDA LayerNormModulate supports F32 or F16 activations.");
        if (scale.DType != DType.F32 || shift.DType != DType.F32)
            throw new NotSupportedException("CUDA LayerNormModulate requires F32 scale/shift.");
        EnterOp();
        EnsureKernels();
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        int seqLen = input.Shape.Rank >= 2 ? (int)input.Shape[input.Shape.Rank - 2] : 1;
        long rows = input.ElementCount / dim;
        ulong pOut = 0, pIn = 0, pSc = 0, pSh = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pSc = GpuTransferHelper.CopyToDevice(scale);
            pSh = GpuTransferHelper.CopyToDevice(shift);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchLayerNormModulate(f16, pOut, pIn, pSc, pSh, dim, seqLen, (int)rows, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn); GpuTransferHelper.FreeDevice(pSc); GpuTransferHelper.FreeDevice(pSh);
        }
    }

    private static (int Tokens, int Heads, int HeadDim) ValidateQkvSplitNorm(
        Tensor q,
        Tensor k,
        Tensor v,
        Tensor qkv,
        Tensor qWeight,
        Tensor kWeight,
        float eps,
        bool headMajor)
    {
        string operation = headMajor ? "QkvSplitNormHeadMajor" : "QkvSplitNorm";
        if (qkv.DType != DType.F32 && qkv.DType != DType.F16)
            throw new NotSupportedException($"CUDA {operation} supports F32 or F16 activations, got {qkv.DType}.");
        if (q.DType != qkv.DType || k.DType != qkv.DType || v.DType != qkv.DType)
            throw new NotSupportedException(
                $"CUDA {operation} requires q/k/v to match qkv dtype {qkv.DType}, got q={q.DType}, k={k.DType}, v={v.DType}.");
        if (qWeight.DType != DType.F32 || kWeight.DType != DType.F32)
            throw new NotSupportedException(
                $"CUDA {operation} requires F32 norm weights, got qWeight={qWeight.DType}, kWeight={kWeight.DType}.");
        if (ReferenceEquals(q, k) || ReferenceEquals(q, v) || ReferenceEquals(k, v)
            || ReferenceEquals(q, qkv) || ReferenceEquals(k, qkv) || ReferenceEquals(v, qkv))
            throw new HartsyInferenceException($"CUDA {operation} requires distinct q, k, v, and qkv tensors.");
        if (qkv.Shape.Rank < 2)
            throw new HartsyInferenceException($"CUDA {operation} requires qkv rank >= 2 with fused QKV in the last dimension, got {qkv.Shape}.");
        if (qWeight.Shape.Rank < 1 || kWeight.Shape.Rank < 1)
            throw new HartsyInferenceException(
                $"CUDA {operation} requires non-empty Q/K norm weight shapes, got qWeight={qWeight.Shape}, kWeight={kWeight.Shape}.");
        if (!qWeight.Shape.Equals(kWeight.Shape))
            throw new HartsyInferenceException(
                $"CUDA {operation} requires identically shaped Q/K norm weights, got qWeight={qWeight.Shape}, kWeight={kWeight.Shape}.");
        if (!q.Shape.Equals(k.Shape) || !q.Shape.Equals(v.Shape))
            throw new HartsyInferenceException(
                $"CUDA {operation} requires identically shaped q/k/v outputs, got q={q.Shape}, k={k.Shape}, v={v.Shape}.");
        if (!float.IsFinite(eps) || eps <= 0f)
            throw new HartsyInferenceException($"CUDA {operation} requires a finite positive epsilon, got {eps}.");

        for (int i = 0; i < qWeight.Shape.Rank; i++)
        {
            if (qWeight.Shape[i] <= 0)
                throw new HartsyInferenceException($"CUDA {operation} requires positive norm-weight dimensions, got {qWeight.Shape}.");
        }
        long headDimLong = qWeight.Shape[qWeight.Shape.Rank - 1];
        if (headDimLong <= 0 || headDimLong > int.MaxValue || qWeight.ElementCount != headDimLong)
            throw new HartsyInferenceException(
                $"CUDA {operation} requires vector Q/K norm weights, got shape {qWeight.Shape} with {qWeight.ElementCount} elements.");

        long fusedWidth = qkv.Shape[qkv.Shape.Rank - 1];
        if (fusedWidth <= 0 || fusedWidth > int.MaxValue || fusedWidth % 3 != 0)
            throw new HartsyInferenceException(
                $"CUDA {operation} requires qkv last dimension to be a positive 3·W within Int32 range, got {fusedWidth}.");
        long width = fusedWidth / 3;
        if (width % headDimLong != 0)
            throw new HartsyInferenceException(
                $"CUDA {operation} requires W divisible by headDim, got W={width} and headDim={headDimLong}.");
        long headsLong = width / headDimLong;
        if (headsLong <= 0 || headsLong > int.MaxValue)
            throw new HartsyInferenceException($"CUDA {operation} computed an invalid head count {headsLong}.");

        long tokensLong = 1;
        for (int i = 0; i < qkv.Shape.Rank - 1; i++)
        {
            long dimension = qkv.Shape[i];
            if (dimension <= 0 || tokensLong > int.MaxValue / dimension)
                throw new HartsyInferenceException(
                    $"CUDA {operation} requires positive qkv dimensions with at most {int.MaxValue} tokens, got {qkv.Shape}.");
            tokensLong *= dimension;
        }
        if (qkv.ElementCount != tokensLong * fusedWidth)
            throw new HartsyInferenceException(
                $"CUDA {operation} detected qkv shape arithmetic overflow in {qkv.Shape}.");
        if (tokensLong * headsLong > int.MaxValue)
            throw new HartsyInferenceException(
                $"CUDA {operation} launch exceeds the supported grid: tokens={tokensLong}, heads={headsLong}.");
        long expectedOutputElements = tokensLong * width;
        long outputElements = 1;
        for (int i = 0; i < q.Shape.Rank; i++)
        {
            long dimension = q.Shape[i];
            if (dimension <= 0 || outputElements > expectedOutputElements / dimension)
                throw new HartsyInferenceException(
                    $"CUDA {operation} output shape must contain exactly {expectedOutputElements} positive-dimension elements, got {q.Shape}.");
            outputElements *= dimension;
        }
        if (outputElements != expectedOutputElements || q.ElementCount != expectedOutputElements)
            throw new HartsyInferenceException(
                $"CUDA {operation} output size mismatch: expected {expectedOutputElements} elements, got {q.ElementCount} in {q.Shape}.");

        if (headMajor)
        {
            if (q.Shape.Rank != 4 || q.Shape[0] > int.MaxValue || q.Shape[2] > int.MaxValue
                || q.Shape[1] != headsLong || q.Shape[3] != headDimLong || q.Shape[0] * q.Shape[2] != tokensLong)
                throw new HartsyInferenceException(
                    $"CUDA {operation} requires q/k/v [B,heads,seq,headDim] matching qkv; got {q.Shape}, " +
                    $"heads={headsLong}, headDim={headDimLong}, tokens={tokensLong}.");
        }
        else
        {
            bool flattenedHeads = q.Shape.Rank >= 1 && q.Shape[q.Shape.Rank - 1] == width;
            bool explicitHeads = q.Shape.Rank >= 2 && q.Shape[q.Shape.Rank - 2] == headsLong
                && q.Shape[q.Shape.Rank - 1] == headDimLong;
            if (!flattenedHeads && !explicitHeads)
                throw new HartsyInferenceException(
                    $"CUDA {operation} requires output ending in [W] or [heads,headDim], got {q.Shape} " +
                    $"for W={width}, heads={headsLong}, headDim={headDimLong}.");
        }

        return ((int)tokensLong, (int)headsLong, (int)headDimLong);
    }

    /// <summary>Fused QKV split + per-head QK-RMSNorm (F32 or F16 activation, F32 weights); writes q/k/v each [., W] laid [token, head, d].</summary>
    /// <remarks>Replaces SliceLastDim×3 + RmsNorm×2 per attention stream.</remarks>
    public void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvSplitNorm");
        (int tokens, int heads, int headDim) = ValidateQkvSplitNorm(q, k, v, qkv, qWeight, kWeight, eps, headMajor: false);
        bool f16 = qkv.DType == DType.F16;
        EnterOp();
        EnsureKernels();
        ulong pq = 0, pk = 0, pv = 0, pQkv = 0, pQw = 0, pKw = 0; bool cached = false;
        try
        {
            pQkv = GpuTransferHelper.CopyToDevice(qkv);
            pQw = GpuTransferHelper.CopyToDevice(qWeight);
            pKw = GpuTransferHelper.CopyToDevice(kWeight);
            nuint bytes = GpuTransferHelper.ByteSize(q);
            pq = GpuTransferHelper.AllocateDevice(bytes);
            pk = GpuTransferHelper.AllocateDevice(bytes);
            pv = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchQkvSplitNorm(f16, pq, pk, pv, pQkv, pQw, pKw, tokens, heads, headDim, eps, _stream.Handle);
            GpuTransferHelper.CacheActivation(q, pq, bytes);
            GpuTransferHelper.CacheActivation(k, pk, bytes);
            GpuTransferHelper.CacheActivation(v, pv, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) { GpuTransferHelper.FreeDevice(pq); GpuTransferHelper.FreeDevice(pk); GpuTransferHelper.FreeDevice(pv); }
            GpuTransferHelper.FreeDevice(pQkv); GpuTransferHelper.FreeDevice(pQw); GpuTransferHelper.FreeDevice(pKw);
        }
    }

    /// <summary>Head-major <see cref="QkvSplitNorm"/> (F32 or F16 activation, F32 weights): q/k/v come out [B, heads, seq, headDim].</summary>
    /// <remarks>Replaces SliceLastDim×3 + RmsNorm×2 + Permute0213×3 per attention stream — SDPA consumes the result directly.</remarks>
    public void QkvSplitNormHeadMajor(Tensor? q, Tensor? k, Tensor? v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvSplitNormHeadMajor");
        Tensor shapeRef = q ?? k ?? v
            ?? throw new HartsyInferenceException("QkvSplitNormHeadMajor needs at least one of q/k/v.");
        int tokens, heads, headDim, packStride, qSlot, kSlot, vSlot;
        if (q is not null && k is not null && v is not null
            && (int)qkv.Shape[qkv.Shape.Rank - 1] == 3 * (int)shapeRef.Shape[1] * (int)qWeight.Shape[qWeight.Shape.Rank - 1])
        {
            // Full fused call — unchanged, including its stricter validation, so it stays bit-exact.
            (tokens, heads, headDim) = ValidateQkvSplitNorm(q, k, v, qkv, qWeight, kWeight, eps, headMajor: true);
            packStride = 3; qSlot = 0; kSlot = 1; vSlot = 2;
        }
        else
        {
            // packStride comes from the SOURCE width, not from how many outputs were asked for: a full [q|k|v]
            // buffer can be read for only k and v, while a narrowed [k|v] or [q] carries only what it names.
            headDim = (int)qWeight.Shape[qWeight.Shape.Rank - 1];
            heads = (int)shapeRef.Shape[1];
            int w = heads * headDim;
            packStride = (int)qkv.Shape[qkv.Shape.Rank - 1] / w;
            if (packStride == 3)
            {
                qSlot = q is null ? -1 : 0; kSlot = k is null ? -1 : 1; vSlot = v is null ? -1 : 2;
            }
            else
            {
                int next = 0;
                qSlot = q is null ? -1 : next++; kSlot = k is null ? -1 : next++; vSlot = v is null ? -1 : next++;
                if (next != packStride)
                    throw new HartsyInferenceException(
                        $"QkvSplitNormHeadMajor: a {packStride}-wide packed source must carry exactly the requested "
                        + $"outputs, got q={q is not null} k={k is not null} v={v is not null}.");
            }
            tokens = (int)(qkv.ElementCount / ((long)packStride * w));
            if (shapeRef.Shape.Rank != 4 || shapeRef.Shape[3] != headDim || (long)shapeRef.Shape[0] * shapeRef.Shape[2] != tokens)
                throw new HartsyInferenceException(
                    $"QkvSplitNormHeadMajor layout mismatch: q {q?.Shape} k {k?.Shape} v {v?.Shape} vs qkv {qkv.Shape} "
                    + $"(packStride={packStride}, heads={heads}, headDim={headDim}, tokens={tokens}).");
        }
        bool f16 = qkv.DType == DType.F16;
        EnterOp();
        EnsureKernels();
        int seq = (int)shapeRef.Shape[2];
        ulong pq = 0, pk = 0, pv = 0, pQkv = 0, pQw = 0, pKw = 0; bool cached = false;
        try
        {
            pQkv = GpuTransferHelper.CopyToDevice(qkv);
            pQw = GpuTransferHelper.CopyToDevice(qWeight);
            pKw = GpuTransferHelper.CopyToDevice(kWeight);
            nuint bytes = GpuTransferHelper.ByteSize(shapeRef);
            if (q is not null) pq = GpuTransferHelper.AllocateDevice(bytes);
            if (k is not null) pk = GpuTransferHelper.AllocateDevice(bytes);
            if (v is not null) pv = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchQkvSplitNormHeadMajor(f16, pq, pk, pv, pQkv, pQw, pKw, tokens, heads, headDim, seq, eps,
                _stream.Handle, packStride, qSlot, kSlot, vSlot);
            if (q is not null) GpuTransferHelper.CacheActivation(q, pq, bytes);
            if (k is not null) GpuTransferHelper.CacheActivation(k, pk, bytes);
            if (v is not null) GpuTransferHelper.CacheActivation(v, pv, bytes);
            cached = true;
        }
        finally
        {
            if (!cached) { GpuTransferHelper.FreeDevice(pq); GpuTransferHelper.FreeDevice(pk); GpuTransferHelper.FreeDevice(pv); }
            GpuTransferHelper.FreeDevice(pQkv); GpuTransferHelper.FreeDevice(pQw); GpuTransferHelper.FreeDevice(pKw);
        }
    }

    /// <summary>FourierEmbedder over device coords [count,3] → dst [count, 3·(2·bands+1)] (F32).</summary>
    public void FourierEmbed(Tensor dst, Tensor coords, int count, int bands)
    {
        using NvtxRange _nvtx = NvtxRange.Push("FourierEmbed");
        if (dst.DType != DType.F32 || coords.DType != DType.F32)
            throw new NotSupportedException("CUDA FourierEmbed supports F32 only.");
        EnterOp();
        EnsureKernels();
        int dim = 3 * (2 * bands + 1);
        ulong pDst = 0, pCoords = 0; bool cached = false;
        try
        {
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            nuint outBytes = GpuTransferHelper.ByteSize(dst);
            pDst = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchFourierEmbed(pDst, pCoords, count, bands, dim, _stream.Handle);
            GpuTransferHelper.CacheActivation(dst, pDst, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pDst);
            GpuTransferHelper.FreeDevice(pCoords);
        }
    }

    /// <summary>Dense 3D convolution (gather form). Input/output [N,C,D,H,W], weight [Cout,Cin,kD,kH,kW] (F32).</summary>
    public unsafe void Conv3d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Conv3d");
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException("CUDA Conv3d supports F32 only.");
        if (input.Shape.Rank != 5 || output.Shape.Rank != 5 || weight.Shape.Rank != 5)
            throw new ArgumentException($"Conv3d requires 5D tensors; got input {input.Shape}, output {output.Shape}, weight {weight.Shape}.");
        EnterOp();
        EnsureKernels();
        int n = (int)input.Shape[0], cin = (int)input.Shape[1], iD = (int)input.Shape[2], iH = (int)input.Shape[3], iW = (int)input.Shape[4];
        int cout = (int)output.Shape[1], oD = (int)output.Shape[2], oH = (int)output.Shape[3], oW = (int)output.Shape[4];
        int kD = (int)weight.Shape[2], kH = (int)weight.Shape[3], kW = (int)weight.Shape[4];
        ulong pOut = 0, pIn = 0, pW = 0, pB = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchConv3d(pOut, pIn, pW, pB, n, cin, cout, iD, iH, iW, oD, oH, oW, kD, kH, kW,
                strideD, strideH, strideW, padD, padH, padW, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cached = true;
        }
        finally
        {
            if (!cached) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn); GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
        }
    }

    /// <summary>Scatters active-voxel feats onto a pre-zeroed device grid (in-place) — see IBackend.SparseScatterToGrid.</summary>
    public unsafe void SparseScatterToGrid(Tensor grid, Tensor feats, Tensor coords, int channels, int resolution)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SparseScatterToGrid");
        if (coords.DType != DType.I32) throw new NotSupportedException("SparseScatterToGrid requires I32 coords.");
        EnterOp(); EnsureKernels();
        int n = (int)(feats.ElementCount / channels);
        ulong pGrid = 0, pFeats = 0, pCoords = 0;
        try
        {
            pGrid = GpuTransferHelper.CopyToDevice(grid);
            pFeats = GpuTransferHelper.CopyToDevice(feats);
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            _kernels!.LaunchSparseGridScatterGather(true, pGrid, pFeats, pCoords, n, channels, resolution, _stream.Handle);
            grid._gpuSyncCallback = null; grid._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(grid, pGrid, GpuTransferHelper.ByteSize(grid));
        }
        finally { GpuTransferHelper.FreeDevice(pFeats); GpuTransferHelper.FreeDevice(pCoords); }
    }

    /// <summary>Gathers a device grid back to active-voxel feats — see IBackend.SparseGatherFromGrid.</summary>
    public unsafe void SparseGatherFromGrid(Tensor feats, Tensor grid, Tensor coords, int channels, int resolution)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SparseGatherFromGrid");
        if (coords.DType != DType.I32) throw new NotSupportedException("SparseGatherFromGrid requires I32 coords.");
        EnterOp(); EnsureKernels();
        int n = (int)(feats.ElementCount / channels);
        ulong pGrid = 0, pCoords = 0, pFeats = 0; bool cached = false;
        try
        {
            pGrid = GpuTransferHelper.CopyToDevice(grid);
            pCoords = GpuTransferHelper.CopyToDevice(coords);
            nuint bytes = GpuTransferHelper.ByteSize(feats);
            pFeats = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchSparseGridScatterGather(false, pGrid, pFeats, pCoords, n, channels, resolution, _stream.Handle);
            GpuTransferHelper.CacheActivation(feats, pFeats, bytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pFeats); GpuTransferHelper.FreeDevice(pCoords); }
    }

    /// <summary>Row gather: output[j] = input[indices[j]] — see IBackend.RowGather.</summary>
    public unsafe void RowGather(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        if (indices.DType != DType.I32) throw new NotSupportedException("RowGather requires I32 indices.");
        using NvtxRange _nvtx = NvtxRange.Push("RowGather");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0, pIdx = 0; bool cached = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input); pIdx = GpuTransferHelper.CopyToDevice(indices);
            nuint bytes = GpuTransferHelper.ByteSize(output); pOut = GpuTransferHelper.AllocateDevice(bytes);
            _kernels!.LaunchRowGatherScatter(true, pOut, pIn, pIdx, m, channels, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, bytes); cached = true;
        }
        finally { if (!cached) GpuTransferHelper.FreeDevice(pOut); GpuTransferHelper.FreeDevice(pIn); GpuTransferHelper.FreeDevice(pIdx); }
    }

    /// <summary>Row scatter-add (in-place accumulate): output[indices[j]] += input[j] — see IBackend.RowScatterAdd.</summary>
    public unsafe void RowScatterAdd(Tensor output, Tensor input, Tensor indices, int m, int channels)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("RowScatterAdd");
        if (indices.DType != DType.I32) throw new NotSupportedException("RowScatterAdd requires I32 indices.");
        EnterOp(); EnsureKernels();
        ulong pOut = 0, pIn = 0, pIdx = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output); pIn = GpuTransferHelper.CopyToDevice(input); pIdx = GpuTransferHelper.CopyToDevice(indices);
            _kernels!.LaunchRowGatherScatter(false, pOut, pIn, pIdx, m, channels, _stream.Handle);
            output._gpuSyncCallback = null; output._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));
        }
        finally { GpuTransferHelper.FreeDevice(pIn); GpuTransferHelper.FreeDevice(pIdx); }
    }

    public void IndexAddRows(Tensor h, Tensor table, Tensor indices)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("IndexAddRows");
        if (h.DType != DType.F32 || table.DType != DType.F32)
            throw new NotSupportedException("CUDA IndexAddRows supports F32 only.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("CUDA IndexAddRows requires I32 indices.");
        EnterOp();
        EnsureKernels();
        int dim = (int)h.Shape[h.Shape.Rank - 1];

        ulong pH = 0, pTable = 0, pIdx = 0;
        try
        {
            pH = GpuTransferHelper.CopyToDevice(h);
            pTable = GpuTransferHelper.CopyToDevice(table);
            pIdx = GpuTransferHelper.CopyToDevice(indices);
            _kernels!.LaunchIndexAdd(pH, pTable, pIdx, dim, h.ElementCount, _stream.Handle);

            // In-place on h: clear stale callbacks before re-caching (pitfall #17).
            h._gpuSyncCallback = null;
            h._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(h, pH, GpuTransferHelper.ByteSize(h));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pTable);
            GpuTransferHelper.FreeDevice(pIdx);
        }
    }

    public void ScatterRowsAfter(Tensor output, Tensor input, int headRows)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("CUDA ScatterRowsAfter supports F32 only.");
        EnterOp();
        EnsureKernels();
        int dim = (int)input.Shape[input.Shape.Rank - 1];

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchScatterRowsAfter(pOut, pIn, headRows, dim, output.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void ScatterSeqHeadMajor(Tensor output, Tensor input, int seqOffset)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("CUDA ScatterSeqHeadMajor supports F32 only.");
        EnterOp();
        int heads = (int)output.Shape[1], seq = (int)output.Shape[2], hd = (int)output.Shape[3];
        int c = (int)input.Shape[2];
        int elemSize = DType.F32.SizeInBytes;
        ulong pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            // The destination persists across calls — the FIRST chunk allocates it and every later one writes into
            // the same buffer, which is the whole point (a per-call allocate-and-recache would be a Concat again).
            if (!GpuTransferHelper.IsActivationCached(output))
            {
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                GpuTransferHelper.CacheActivation(output, GpuTransferHelper.AllocateDevice(outBytes), outBytes);
            }
            ulong pOut = GpuTransferHelper.CopyToDevice(output);
            // One stream-ordered DtoD per head: a head's chunk rows are contiguous, heads are not (dst stride is the
            // full seq, src stride the chunk) — the same per-slice shape Concat's dim>0 path issues, same launch count.
            nuint sliceBytes = (nuint)((long)c * hd * elemSize);
            for (int h = 0; h < heads; h++)
            {
                ulong dst = pOut + (ulong)((((long)h * seq + seqOffset) * hd) * elemSize);
                ulong src = pIn + (ulong)(((long)h * c * hd) * elemSize);
                CudaMemory.CopyDeviceToDeviceAsync(dst, src, sliceBytes, _stream.Handle);
            }
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("SliceRows");
        if (input.DType != output.DType || (output.DType != DType.F32 && output.DType != DType.F16 && output.DType != DType.BF16))
            throw new NotSupportedException("CUDA SliceRows supports F32, F16 or BF16 (matching input/output dtype).");
        EnterOp();
        EnsureKernels();
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long elemOffset = (long)rowOffset * dim;

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            // BF16 piggy-backs on the F16 kernel (pure 16-bit copy, see Permute0213).
            if (output.DType == DType.F16 || output.DType == DType.BF16)
                _kernels!.LaunchSliceRowsF16(pOut, pIn, elemOffset, output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchSliceRows(pOut, pIn, elemOffset, output.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Dtype-agnostic contiguous row-block slice — a raw byte-range D2D copy, so it needs no per-width kernel and works for any non-quantized dtype including fp8, unlike <see cref="SliceRows"/>'s F32/F16/BF16 guard. Carries <see cref="Tensor.Fp8ScaleFactor"/> onto the sliced chunk.</summary>
    public void SliceRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("SliceRowsGeneric");
        if (output.DType != input.DType)
            throw new ArgumentException($"CUDA SliceRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("CUDA SliceRowsGeneric does not support block-quantized dtypes.");
        EnterOp();
        int dim = (int)output.Shape[output.Shape.Rank - 1];
        long rowBytes = output.DType.ComputeByteCount(dim);
        long byteOffset = (long)rowOffset * rowBytes;

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            CudaMemory.CopyDeviceToDeviceAsync(pOut, pIn + (ulong)byteOffset, outBytes, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
        output.Fp8ScaleFactor = input.Fp8ScaleFactor;
    }

    public void ScatterRowsGeneric(Tensor output, Tensor input, int rowOffset)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("ScatterRowsGeneric");
        if (output.DType != input.DType)
            throw new ArgumentException($"CUDA ScatterRowsGeneric requires matching dtypes, got output {output.DType} vs input {input.DType}.");
        if (output.DType.IsQuantized)
            throw new NotSupportedException("CUDA ScatterRowsGeneric does not support block-quantized dtypes.");
        EnterOp();
        int dim = (int)input.Shape[input.Shape.Rank - 1];
        long byteOffset = (long)rowOffset * input.DType.ComputeByteCount(dim);
        nuint inBytes = GpuTransferHelper.ByteSize(input);
        ulong pIn = 0;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            // The destination persists across calls — the first chunk allocates it and every later one writes into
            // the same buffer, which is the whole point (a per-call allocate-and-recache would be a Concat again).
            if (!GpuTransferHelper.IsActivationCached(output))
            {
                nuint outBytes = GpuTransferHelper.ByteSize(output);
                GpuTransferHelper.CacheActivation(output, GpuTransferHelper.AllocateDevice(outBytes), outBytes);
            }
            ulong pOut = GpuTransferHelper.CopyToDevice(output);
            // Rows are contiguous, so unlike ScatterSeqHeadMajor's per-head loop this is one stream-ordered DtoD.
            CudaMemory.CopyDeviceToDeviceAsync(pOut + (ulong)byteOffset, pIn, inBytes, _stream.Handle);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps)
    {
        if (input.DType != DType.F32 || gamma.DType != DType.F32 || beta.DType != DType.F32)
            throw new NotSupportedException($"CUDA AdaInstanceNorm1d supports F32 only — got input {input.DType}, gamma {gamma.DType}, beta {beta.DType}.");
        EnterOp();
        EnsureKernels();

        // input: [B, C, T] channels-first; gamma/beta: [B, C] (per-batch) or [C] (broadcast).
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int t = (int)input.Shape[2];
        int totalRows = batch * channels;
        bool perBatch = gamma.Shape.Rank == 2;

        ulong pOut = 0, pIn = 0, pG = 0, pB = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pG = GpuTransferHelper.CopyToDevice(gamma);
            pB = GpuTransferHelper.CopyToDevice(beta);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchAudioAdaInstanceNorm1d(pOut, pIn, pG, pB, t, totalRows, channels, perBatch, eps, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pG);
            GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void LeakyRelu(Tensor output, Tensor input, float slope)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA LeakyRelu currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioLeakyRelu(pOut, pIn, slope, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Fused GroupNorm + SiLU via single PTX kernel. Eliminates intermediate allocation.</summary>
    public void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GroupNormSilu");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int spatial = 1;
        for (int d = 2; d < input.Shape.Rank; d++)
        {
            spatial *= (int)input.Shape[d];
        }

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, pWCast = 0, pBCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = GpuTransferHelper.CopyToDevice(bias);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
            {
                // Cast weight/bias to F16 if stored as F32 (common for norm params in FP16 models)
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToF16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormSiluF16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else if (input.DType == DType.BF16)
            {
                // BF16 path: chosen for SDXL VAE so resnet activations (which exceed
                // F16's 65504 range) stay finite. Weights/biases must match BF16 — cast
                // from F32 if needed. See PHASE_3_DEVIATIONS.md #36 for the F16-overflow
                // pattern; same family of bug, different op site.
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.F32)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.F32)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 2));
                    _kernels!.LaunchCastF32ToBf16(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormSiluBf16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else
            {
                // F32 activation path: the kernel reads weight/bias as F32. Norm params are often stored
                // in a narrower dtype (e.g. the Flux.2 VAE ships BF16 norm weights while the latent stays
                // F32) — cast them up to F32 first, else the kernel reinterprets BF16/F16 bytes as F32 and
                // produces a wrong affine (washed-out decode). Mirrors the F16/BF16 paths above.
                ulong wPtr = pW;
                ulong bPtr = pB;
                if (weight.DType == DType.BF16 || weight.DType == DType.F16)
                {
                    pWCast = CudaMemory.Allocate((nuint)(weight.ElementCount * 4));
                    if (weight.DType == DType.BF16)
                        _kernels!.LaunchCastBf16ToF32(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    else
                        _kernels!.LaunchCastF16ToF32(pWCast, pW, (int)weight.ElementCount, _stream.Handle);
                    wPtr = pWCast;
                }
                if (bias.DType == DType.BF16 || bias.DType == DType.F16)
                {
                    pBCast = CudaMemory.Allocate((nuint)(bias.ElementCount * 4));
                    if (bias.DType == DType.BF16)
                        _kernels!.LaunchCastBf16ToF32(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    else
                        _kernels!.LaunchCastF16ToF32(pBCast, pB, (int)bias.ElementCount, _stream.Handle);
                    bPtr = pBCast;
                }
                _kernels!.LaunchGroupNormSilu(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            GpuTransferHelper.FreeDevice(pB);
            if (pWCast != 0) CudaMemory.FreeAsync(pWCast, _stream.Handle);
            if (pBCast != 0) CudaMemory.FreeAsync(pBCast, _stream.Handle);
        }
    }

    /// <summary>GPU cast FP32 → FP16 via PTX kernel.</summary>
    public void CastToF16(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CastToF16");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchCastF32ToF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Dequantizes a GGUF weight (Q4_K/Q5_K/Q6_K/Q8_0/fp8) to F32 via <see cref="CastOnGpu"/>; test/tooling hook, not a hot path.</summary>
    public Tensor DequantizeToF32(Tensor quant)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("DequantizeToF32");
        EnterOp();
        EnsureKernels();
        int count = (int)quant.ElementCount;
        Tensor output = new Tensor(new TensorShape(count), DType.F32);
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(quant);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            CastOnGpu(pOut, pIn, quant.DType, DType.F32, count);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            return output;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU cast FP16 or BF16 → FP32 via PTX kernel.</summary>
    public void CastToF32(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CastToF32");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.BF16)
                _kernels!.LaunchCastBf16ToF32(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchCastF16ToF32(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GPU cast → BF16 via PTX kernel; routes non-F32 input through an F16→F32 cast first.</summary>
    public void CastToBf16(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("CastToBf16");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0, pIntermediate = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            ulong srcPtr = pIn;
            if (input.DType == DType.F16)
            {
                pIntermediate = CudaMemory.Allocate((nuint)(input.ElementCount * 4));
                _kernels!.LaunchCastF16ToF32(pIntermediate, pIn, (int)input.ElementCount, _stream.Handle);
                srcPtr = pIntermediate;
            }
            else if (input.DType != DType.F32)
            {
                throw new NotSupportedException($"CastToBf16: source dtype {input.DType} not supported.");
            }

            _kernels!.LaunchCastF32ToBf16(pOut, srcPtr, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            if (pIntermediate != 0) CudaMemory.FreeAsync(pIntermediate, _stream.Handle);
        }
    }

    #endregion

    #region Attention

    /// <summary>Scaled dot-product attention via cuBLAS batched GEMM: softmax(Q @ K^T * scale) @ V.</summary>
    public unsafe void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SDPA");
        ValidateScaledDotProductAttentionContract(output, query, key, value, mask, scale);
        EnterOp();
        EnsureKernels();

        long b = query.Shape[0];
        long h = query.Shape[1];
        long sq = query.Shape[2];
        long d = query.Shape[3];
        long skv = key.Shape[2];

        long totalHeads = b * h;

        // Memory-efficient dispatch: the GEMM path below materializes a [totalHeads, Sq, Skv] score matrix. For very
        // long sequences (Wan-Video full-res self-attention: 24 heads × ~14040² × 4B ≈ 19 GB) that OOMs alongside the
        // model weights. For the plain case — F32, no additive mask, full bidirectional MHA — avoid the full matrix:
        //   • default large-seq path = QUERY-TILED GEMM (SdpaTiledF32NoMask): tiles the query axis so only
        //     [totalHeads, Br, Skv] is materialized per tile, reusing the same TF32 tensor-core GEMMs + softmax kernel
        //     (numerically identical to the small-seq GEMM path, and ~10× faster than the online-softmax flash kernel
        //     which re-reads all K/V per query row).
        //   • HARTSY_SDPA_FORCE_FLASH=1 forces the online-softmax flash kernel (O(1) score memory; validation/fallback).
        // Gated on the score matrix eating most of free VRAM so small/medium attention keeps the plain GEMM path;
        // masked (Matrix-Game block-causal) callers always keep the plain GEMM path.
        // cuDNN fused flash-attention for NATIVE F16 Q/K/V/output (the DiT F16-activation path): zero casts —
        // the tensors are already in the engine's fp16 I/O dtype, so this is the pure fused execute (the F32 route
        // below pays 3+1 F32↔F16 cast kernels). The caller choosing F16 activations already asserts bounded scores
        // (RMS-normed Q/K), so no allowF16 gate. Same session kill-switch on any cuDNN failure.
        // An additive F32 [B,1,Sq,Skv]-broadcast mask (Chroma / padded-conditioning models) rides the fused
        // engine as a bias score-modifier; incompatible mask layouts fall through to the materialized path.
        // SageAttention F16-ingest (opt-in): native-F16 Q/K/V/out via the f16h prologues + f16io flash
        // kernel. Competes with the CAST-FREE cuDNN branch below, which Sage only beats at long seq —
        // gate high (HARTSY_SAGE_F16_MIN_SKV, default 8192) until the crossover is measured per-arch.
        if (SageF16Preferred(query, key, value, output, mask, sq, skv, d))
        {
            SageAttentionInt8(output, query, key, value, scale);
            return;
        }

        if (CudnnMaskCompatible(mask, b, sq, skv) && query.DType == DType.F16 && key.DType == DType.F16
            && value.DType == DType.F16 && output.DType == DType.F16
            && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(d) && CudnnSdpa.ShapeSupported(d)
            && TryCudnnSdpa(output, query, key, value, mask, scale))
        {
            return;
        }

        if (query.DType == DType.F32)
        {
            // ── HAZARD: the SageAttention branches below do NOT honor `allowF16`. ──────────────────────────
            // Sage quantizes Q/K to INT8 (smoothed, so their range is handled) but materializes V as an F16
            // transpose (LaunchSageVF16T). Any architecture whose |V| exceeds F16's 65504 therefore gets INF in
            // V, which softmax·V smears across every query row.
            //
            // This path now requires HARTSY_SAGE_UNSAFE_F32_V_NARROW=1 in addition to HARTSY_SAGE_ATTN=1.
            // allowF16 is not a V-range contract and therefore cannot make this narrowing safe.
            //
            // Diagnostic fingerprint, if a future model renders black/NaN: exactly ONE bad element per token in
            // the SDPA output (a single overflowing (head, dim) column of V, spread over all query rows by the
            // softmax). Check max|V| per block against 65504. Lens hit this at block 45 with max|V| growing
            // 18940 → 31281 → 71583 across forwards; its fix (LensTransformerBlock) scales V by 1/256 before
            // SDPA and back after — exact, since attention is linear in V, and power-of-two so exponent-only.
            //
            // MiniMax-H3 was measured against this and is CLEAR: peak max|V| = 1201 (1.83% of 65504, a 55x margin)
            // over a full 30-step generation, 1500 block-probes, zero non-finite (HARTSY_H3_VPROBE=1, 2026-08-08).
            // It grows with depth (81 at block 0 to ~1200 at block 48) but oscillates in a band across steps rather
            // than compounding like Lens did. Its documented ~2.7e6 residual never reaches V: norm1 precedes the
            // qkv projection, so V is a projection of a NORMALIZED tensor, not of the raw residual stream.
            // A model-agnostic fix belongs inside SageAttentionInt8, not here: a blanket V damp would push small
            // values toward F16 subnormals, so it needs its own range analysis.
            // ──────────────────────────────────────────────────────────────────────────────────────────────
            // SageAttention preference (opt-in, HARTSY_SAGE_ATTN=1): for no-mask F32 calls the INT8 flash
            // path beats the cuDNN-F16-cast branch below at large seq (110.6 vs 130.4 ms at 16384²/D=128,
            // 2026-07-22 BDN A/B) — and unlike it, keeps F32-fidelity accumulation. Gate on Skv ≥ 2048:
            // below that the quant prologue outweighs the win (small-seq shapes measured 0.93× vs cuDNN).
            if (SageF32ValueNarrowingEnabled && mask is null && (d == 64 || d == 128) && skv >= 2048)
            {
                EnsureKernels();
                if (_kernels!.HasSageAttentionKernels && (_kernels.HasSageV1 || sq % 32 == 0))
                {
                    SageAttentionInt8(output, query, key, value, scale);
                    return;
                }
            }

            // cuDNN fused flash-attention (HARTSY_SDPA_CUDNN): a single fused kernel — no materialized
            // [heads,Sq,Skv] score matrix — via cuDNN's runtime-compiled attention engine. ~34× over the
            // materialized cuBLAS path at Krea2 shape. MHA only, D∈{64,128}. Safe for RMS-normed-Q/K archs
            // (bounded scores) since we run fp16 I/O; callers gate the same way as the F16 path (allowF16).
            // Masked callers: the additive F32 mask is added to the fp32 scores INSIDE the engine (bias
            // score-modifier), so the mask never rounds through F16 — only Q/K/V do, same as unmasked.
            // Self-disables for the session if cuDNN init/exec ever throws, falling back to the paths below.
            if (CudnnMaskCompatible(mask, b, sq, skv)
                && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(d) && CudnnSdpa.ShapeSupported(d)
                && (allowF16 || _sdpaF16ForceOn) && !_sdpaF16Disabled
                && TryCudnnSdpa(output, query, key, value, mask, scale))
            {
                return;
            }
        }

        if (mask is null && query.DType == DType.F32)
        {

            // SageAttention-v1 INT8 flash attention (sage_attn_int8*.ptx): K-smoothed per-row INT8 QK^T on
            // the IMMA tensor cores, online softmax + PV in registers (mma.sync v1; wmma v0 fallback needs
            // Sq%32==0 — its WMMA Q-tile loads are unguarded). F32 V narrowing requires two explicit flags;
            // no-mask MHA, D∈{64,128}, Skv≥1024 (tiny-seq shapes keep the F32 fallback — prologue-bound).
            // Falls through when the PTX isn't built.
            if (SageF32ValueNarrowingEnabled && (d == 64 || d == 128) && skv >= 1024)
            {
                EnsureKernels();
                if (_kernels!.HasSageAttentionKernels && (_kernels.HasSageV1 || sq % 32 == 0))
                {
                    SageAttentionInt8(output, query, key, value, scale);
                    return;
                }
            }

            // Fused FlashAttention-2 (TF32 tensor cores, F32 accum, no materialized score matrix). Opt-in while
            // validating (HARTSY_SDPA_V2); MHA only (Hq==Hkv here — single B×H×S×D layout), D∈{64,128}.
            if (EngineKnobs.SdpaV2.Value
                && FlashAttentionV2ContractSatisfied(output, query, key, value, mask, scale, _allowTf32))
            {
                FlashAttentionV2(output, query, key, value, scale);
                return;
            }
            if (EngineKnobs.SdpaForceFlash.Value)
            {
                FlashAttention(output, query, key, value, (int)skv, kvGroup: 1, causal: false, qOffset: 0, scale);
                return;
            }
            ulong scoreBytesEst = (ulong)totalHeads * (ulong)sq * (ulong)skv * sizeof(float);
            (nuint freeBytes, _) = _context.GetMemoryInfo();
            if (EngineKnobs.SdpaForceTiled.Value || scoreBytesEst > (ulong)freeBytes / 2)
            {
                SdpaTiledF32(output, query, key, value, scale);
                return;
            }
        }

        // A key-only bias (one [Skv] row broadcast over every query) is compatible with the query-tiled path, so
        // a masked call whose score matrix would not fit is no longer forced into the full materialization. Without
        // this, a single transient cuDNN failure mid-generation demotes to a [heads,Sq,Skv] allocation that is far
        // larger than whatever the fused path could not get (Wan-Animate-2 at 480x800/61f: 5836 MB).
        if (query.DType == DType.F32 && MaskIsKeyOnly(mask, sq))
        {
            ulong keyBiasScoreBytes = (ulong)totalHeads * (ulong)sq * (ulong)skv * sizeof(float);
            (nuint keyBiasFree, _) = _context.GetMemoryInfo();
            if (EngineKnobs.SdpaForceTiled.Value || keyBiasScoreBytes > (ulong)keyBiasFree / 2)
            {
                SdpaTiledF32(output, query, key, value, scale, mask);
                return;
            }
        }

        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pMaskClamped = 0, pMaskCast = 0, pOut = 0, scoresBuf = 0;
        ulong pQCast = 0, pKCast = 0, pVCast = 0, pOutCast = 0, pMaskBroadcast = 0, pMaskOnes = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            long maskElemCount = 0;
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
                maskElemCount = mask.ElementCount;
                if (MaskIsKeyOnly(mask, sq))
                {
                    // This path indexes the mask per query row, so expand the stored [.., 1, Skv] into the
                    // [.., Sq, Skv] it stands for. Only small/medium shapes reach here — the large ones took the
                    // tiled branch above — so the expansion is bounded by the same budget as the score matrix.
                    long maskBlocks = maskElemCount / skv;
                    pMaskBroadcast = CudaMemory.Allocate((nuint)(maskBlocks * sq * skv * sizeof(float)));
                    pMaskOnes = CudaMemory.Allocate((nuint)(sq * sizeof(float)));
                    CudaMemory.Fill32(pMaskOnes, 0x3F80_0000u, (nuint)sq);   // 1.0f
                    for (long blk = 0; blk < maskBlocks; blk++)
                    {
                        AccumulateKeyBias(
                            pMaskBroadcast + (ulong)(blk * sq * skv * sizeof(float)),
                            pMask + (ulong)(blk * skv * sizeof(float)),
                            pMaskOnes, sq, skv, skv, beta: 0f);
                    }
                    maskElemCount = maskBlocks * sq * skv;
                }
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // BF16 path: SDPA's softmax kernel only has F16/F32 variants. For the
            // single VaeAttention call in the SDXL VAE, cast Q/K/V to F32 internally
            // and write the output back as BF16. The cost is one extra ~24 MB of temp
            // F32 (Q+K+V combined for VAE-typical 4096-token attention) — negligible
            // vs the precision cost of trying to squeeze SDXL VAE through F16. A
            // dedicated BF16 SDPA path can be a future optimization once it's hot.
            ulong opQ = pQ, opK = pK, opV = pV, opOut = pOut;
            DType opDtype = query.DType;
            if (query.DType == DType.BF16)
            {
                pQCast = CudaMemory.Allocate((nuint)(query.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pQCast, pQ, (int)query.ElementCount, _stream.Handle);
                pKCast = CudaMemory.Allocate((nuint)(key.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pKCast, pK, (int)key.ElementCount, _stream.Handle);
                pVCast = CudaMemory.Allocate((nuint)(value.ElementCount * 4));
                _kernels!.LaunchCastBf16ToF32(pVCast, pV, (int)value.ElementCount, _stream.Handle);
                pOutCast = CudaMemory.Allocate((nuint)(output.ElementCount * 4));
                opQ = pQCast;
                opK = pKCast;
                opV = pVCast;
                opOut = pOutCast;
                opDtype = DType.F32;
            }
            else if (query.DType == DType.F32 && mask is null && (allowF16 || _sdpaF16ForceOn) && !_sdpaF16Disabled)
            {
                // F16 speed path — enabled per-call via allowF16 (callers with bounded/normalized scores, e.g. Wan's
                // RMS-normed Q/K) or globally via HARTSY_SDPA_F16; disabled globally via HARTSY_SDPA_NO_F16. NOT safe
                // for unbounded-score archs (Z-Image fp8 → F16 overflow → black), which simply don't pass allowF16.
                // The non-tiled SDPA cost is dominated by the
                // [totalHeads, Sq, Skv] score matrix (Wan-1.3B self-attn: 12·4480²·4B ≈ 963 MB, written by QK then
                // re-read by softmax + AV — the profiled #1 GPU cost). Running in F16 halves that traffic and uses
                // F16 tensor cores. Scores are bounded (scale·RMS-normed Q·K), and softmax subtracts the row max, so
                // no F16 overflow. Output is cast back to F32 after AV.
                pQCast = CudaMemory.Allocate((nuint)(query.ElementCount * 2));
                _kernels!.LaunchCastF32ToF16(pQCast, pQ, (int)query.ElementCount, _stream.Handle);
                pKCast = CudaMemory.Allocate((nuint)(key.ElementCount * 2));
                _kernels!.LaunchCastF32ToF16(pKCast, pK, (int)key.ElementCount, _stream.Handle);
                pVCast = CudaMemory.Allocate((nuint)(value.ElementCount * 2));
                _kernels!.LaunchCastF32ToF16(pVCast, pV, (int)value.ElementCount, _stream.Handle);
                pOutCast = CudaMemory.Allocate((nuint)(output.ElementCount * 2));
                opQ = pQCast;
                opK = pKCast;
                opV = pVCast;
                opOut = pOutCast;
                opDtype = DType.F16;
            }

            bool isF16 = opDtype == DType.F16;
            int elemSize = opDtype.SizeInBytes;
            ulong opMask = pMaskBroadcast != 0 ? pMaskBroadcast : pMask;
            int maskElemSize = sizeof(float);
            if (mask is not null && isF16)
            {
                // Scores live in F16 on the native-F16/fallback path, while the public additive-mask contract is
                // F32. Convert once rather than reinterpreting packed float bytes as half values. Clamp before
                // conversion so common finite hard-mask sentinels such as -1e30 become -65504 rather than -Inf:
                // an entirely masked row then retains the repository's finite-sentinel/uniform-row convention
                // instead of evaluating -Inf - -Inf inside softmax and producing NaNs.
                pMaskClamped = CudaMemory.Allocate((nuint)(maskElemCount * sizeof(float)));
                _kernels!.LaunchClamp(
                    pMaskClamped, opMask, -65_504f, 65_504f, (int)maskElemCount, _stream.Handle);
                pMaskCast = CudaMemory.Allocate((nuint)(maskElemCount * sizeof(ushort)));
                _kernels!.LaunchCastF32ToF16(pMaskCast, pMaskClamped, (int)maskElemCount, _stream.Handle);
                opMask = pMaskCast;
                maskElemSize = sizeof(ushort);
            }

            nuint scoresBytes = (nuint)(totalHeads * sq * skv * elemSize);
            scoresBuf = CudaMemory.Allocate(scoresBytes);

            float alpha = scale;
            float beta = 0.0f;

            long strideQ = sq * d;
            long strideK = skv * d;
            long strideScores = sq * skv;

            // QK^T, all heads in one strided-batched launch (was totalHeads sequential cublasGemmEx calls —
            // at sq=1 decode shapes each call is a near-instant GEMV dressed as a GEMM, so per-call LAUNCH
            // overhead dominated: a step-decoding model issues this hundreds of times per step, thousands of
            // times per generation, and the accumulated launch overhead — not compute — was the wall-clock
            // cost. Same math as the per-head loop (each batch element is independent, offset by the stride),
            // just one driver call instead of totalHeads of them.
            if (isF16)
            {
                CublasApi.cublasGemmStridedBatchedEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    (int)skv, (int)sq, (int)d,
                    &alpha,
                    opK, CublasApi.CUDA_R_16F, (int)d, strideK,
                    opQ, CublasApi.CUDA_R_16F, (int)d, strideQ,
                    &beta,
                    scoresBuf, CublasApi.CUDA_R_16F, (int)skv, strideScores,
                    (int)totalHeads,
                    CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }
            else
            {
                // F32 QK^T via TF32 tensor cores (~8x over FP32 CUDA cores). TF32 keeps the F32
                // exponent range, so raw pre-softmax scores can't overflow the way F16 would.
                CublasApi.cublasGemmStridedBatchedEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                    (int)skv, (int)sq, (int)d,
                    &alpha,
                    opK, CublasApi.CUDA_R_32F, (int)d, strideK,
                    opQ, CublasApi.CUDA_R_32F, (int)d, strideQ,
                    &beta,
                    scoresBuf, CublasApi.CUDA_R_32F, (int)skv, strideScores,
                    (int)totalHeads,
                    CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            if (mask is not null)
            {
                long maskHeadStride = sq * skv;
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    long batchIndex = bh / h;
                    long headIndex = bh % h;
                    long maskBlock = mask.Shape.Rank switch
                    {
                        2 => 0,
                        3 => mask.Shape[0] == 1 ? 0 : headIndex,
                        4 => (mask.Shape[0] == 1 ? 0 : batchIndex) * mask.Shape[1]
                            + (mask.Shape[1] == 1 ? 0 : headIndex),
                        _ => throw new InvalidOperationException("Validated SDPA mask rank changed before dispatch."),
                    };
                    ulong scorePointer = scoresBuf + (ulong)(bh * strideScores * elemSize);
                    ulong maskPointer = opMask + (ulong)(maskBlock * maskHeadStride * maskElemSize);
                    if (isF16)
                        _kernels!.LaunchAddF16(scorePointer, scorePointer, maskPointer, (int)maskHeadStride, _stream.Handle);
                    else
                        _kernels!.LaunchAdd(scorePointer, scorePointer, maskPointer, (int)maskHeadStride, _stream.Handle);
                }
            }

            if (isF16)
                _kernels!.LaunchSoftmaxF16(scoresBuf, (int)skv, (int)(totalHeads * sq), _stream.Handle);
            else
                _kernels!.LaunchSoftmax(scoresBuf, (int)skv, (int)(totalHeads * sq), _stream.Handle);

            // attn_weights @ V, all heads in one strided-batched launch (see the QK^T comment above — same
            // per-call-launch-overhead problem, same fix).
            long strideV = skv * d;
            long strideOut = sq * d;
            float one = 1.0f;
            float zero = 0.0f;

            if (isF16)
            {
                CublasApi.cublasGemmStridedBatchedEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    (int)d, (int)sq, (int)skv,
                    &one,
                    opV, CublasApi.CUDA_R_16F, (int)d, strideV,
                    scoresBuf, CublasApi.CUDA_R_16F, (int)skv, strideScores,
                    &zero,
                    opOut, CublasApi.CUDA_R_16F, (int)d, strideOut,
                    (int)totalHeads,
                    CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }
            else
            {
                // attn_weights @ V via TF32 tensor cores (~8x over FP32 CUDA cores).
                CublasApi.cublasGemmStridedBatchedEx(
                    _cublasHandle,
                    CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                    (int)d, (int)sq, (int)skv,
                    &one,
                    opV, CublasApi.CUDA_R_32F, (int)d, strideV,
                    scoresBuf, CublasApi.CUDA_R_32F, (int)skv, strideScores,
                    &zero,
                    opOut, CublasApi.CUDA_R_32F, (int)d, strideOut,
                    (int)totalHeads,
                    CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            // If we did the BF16 internal-cast detour, the output is F32 in pOutCast — cast
            // it back to BF16 in pOut before caching.
            if (output.DType == DType.BF16 && pOutCast != 0)
            {
                _kernels!.LaunchCastF32ToBf16(pOut, pOutCast, (int)output.ElementCount, _stream.Handle);
            }
            else if (opDtype == DType.F16 && output.DType == DType.F32 && pOutCast != 0)
            {
                // F16 speed path produced an F16 result in pOutCast — cast it back to the F32 output tensor.
                _kernels!.LaunchCastF16ToF32(pOut, pOutCast, (int)output.ElementCount, _stream.Handle);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            GpuTransferHelper.FreeDevice(pMask);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (pMaskCast != 0) CudaMemory.FreeAsync(pMaskCast, _stream.Handle);
            if (pMaskClamped != 0) CudaMemory.FreeAsync(pMaskClamped, _stream.Handle);
            if (pMaskBroadcast != 0) CudaMemory.FreeAsync(pMaskBroadcast, _stream.Handle);
            if (pMaskOnes != 0) CudaMemory.FreeAsync(pMaskOnes, _stream.Handle);
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
            if (pQCast != 0) CudaMemory.FreeAsync(pQCast, _stream.Handle);
            if (pKCast != 0) CudaMemory.FreeAsync(pKCast, _stream.Handle);
            if (pVCast != 0) CudaMemory.FreeAsync(pVCast, _stream.Handle);
            if (pOutCast != 0) CudaMemory.FreeAsync(pOutCast, _stream.Handle);
        }
    }

    /// <summary>Always true: the cuDNN fused engine takes BSHD strides directly, and anything it declines is served by permuting into the head-major path, so the caller never has to carry a second layout.</summary>
    public bool SupportsTokenMajorAttention => true;

    /// <summary>Token-major SDPA — see <see cref="IBackend.ScaledDotProductAttentionTokenMajor"/>. cuDNN reads the <c>[S, heads*headDim]</c> buffers through BSHD strides at no throughput cost, which is what lets the LTX-2 caller drop the permute pair around attention. Anything cuDNN declines (wrong head dim, mask shape, a dead session) is permuted into the head-major dispatch so correctness never depends on the fast path.</summary>
    public unsafe void ScaledDotProductAttentionTokenMajor(Tensor output, Tensor query, Tensor key, Tensor value,
        Tensor? mask, int heads, int headDim, float scale, bool allowF16 = false)
    {
        using NvtxRange _nvtx = NvtxRange.Push(NvtxRange.ProfileShapes
            ? $"SDPA-TM {query.Shape[0]}x{key.Shape[0]}x{heads}x{headDim}" : "SDPA-TM");
        ValidateTokenMajorAttentionContract(output, query, key, value, mask, heads, headDim, scale);
        long sq = query.Shape[0], skv = key.Shape[0], d = headDim;
        // SageAttention has no token-major kernel, so this entry point reaches cuDNN and a long-sequence DiT
        // (LTX-2.5 at 17480 video tokens) runs fp16 flash while the INT8 path it qualifies for sits unused.
        // Declining cuDNN here routes the call through the permute pair below into the head-major dispatch, which
        // owns the Sage gate.
        //
        // OPT-IN, and here is the whole measurement. LTX-2.5 1280x736x145f on a 4090: the CLI harness (4 interleaved
        // reps, ~25 ms/step spread) puts it at a clean **-71 ms/step**, every pair the same sign. It still does not
        // ship on, for two reasons the CLI cannot see. (1) Through SwarmUI, N=3 warm per arm, it is unresolvable —
        // that harness drifts monotonically within a session and spans 8.4 s, so a 1.4 s effect is invisible.
        // (2) Sage's Q8/K8/scale/V-transpose workspaces raise peak VRAM by **494 MiB** (23539 -> 24033 of 24564),
        // and this geometry already runs 531 MiB from the ceiling — spending that headroom for 0.9% of wall clock
        // is the wrong trade for a user one setting away from an OOM. Quality is a third, smaller reason: SSIM
        // 0.9927 (min 0.9908) against the fp16 arm over 145 frames, audio relL2 0.18 at matched seed.
        if (!(SageTokenMajorDetour && SageF16Preferred(query, key, value, output, mask, sq, skv, d))
            && CudnnMaskCompatible(mask, 1, sq, skv)
            && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(d) && CudnnSdpa.ShapeSupported(d)
            && (query.DType == DType.F16 || (query.DType == DType.F32 && (allowF16 || _sdpaF16ForceOn) && !_sdpaF16Disabled)))
        {
            EnterOp();
            EnsureKernels();
            if (TryCudnnSdpa(output, query, key, value, mask, scale, CudnnSdpa.SdpaLayout.TokenMajor,
                    1, heads, sq, skv, d))
            {
                return;
            }
        }
        Tensor qMh = new Tensor(new TensorShape(1, heads, sq, headDim), query.DType);
        Tensor kMh = new Tensor(new TensorShape(1, heads, skv, headDim), key.DType);
        Tensor vMh = new Tensor(new TensorShape(1, heads, skv, headDim), value.DType);
        Tensor oMh = new Tensor(new TensorShape(1, heads, sq, headDim), output.DType);
        try
        {
            Permute0213(qMh, query, (int)sq, heads, headDim);
            Permute0213(kMh, key, (int)skv, heads, headDim);
            Permute0213(vMh, value, (int)skv, heads, headDim);
            ScaledDotProductAttention(oMh, qMh, kMh, vMh, mask, scale, allowF16);
            Permute0213(output, oMh, heads, (int)sq, headDim);
        }
        finally { qMh.Dispose(); kMh.Dispose(); vMh.Dispose(); oMh.Dispose(); }
    }

    /// <summary>Validates the rank-2 <c>[S, heads*headDim]</c> contract of the token-major SDPA entry point.</summary>
    internal static void ValidateTokenMajorAttentionContract(Tensor output, Tensor query, Tensor key, Tensor value,
        Tensor? mask, int heads, int headDim, float scale)
    {
        if (heads <= 0 || headDim <= 0)
            throw new ArgumentException($"Token-major SDPA needs positive heads/headDim; got heads={heads}, headDim={headDim}.");
        long inner = (long)heads * headDim;
        if (output.Shape.Rank != 2 || query.Shape.Rank != 2 || key.Shape.Rank != 2 || value.Shape.Rank != 2)
            throw new ArgumentException(
                $"Token-major SDPA requires rank-2 [S, heads*headDim] tensors; got output={output.Shape}, Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        if (query.Shape[1] != inner || key.Shape[1] != inner || value.Shape[1] != inner || output.Shape[1] != inner)
            throw new ArgumentException(
                $"Token-major SDPA rows must be heads*headDim={inner}; got Q={query.Shape}, K={key.Shape}, V={value.Shape}, output={output.Shape}.");
        if (output.Shape[0] != query.Shape[0])
            throw new ArgumentException($"Token-major SDPA output must have Q's row count; got output={output.Shape}, Q={query.Shape}.");
        if (key.Shape[0] != value.Shape[0])
            throw new ArgumentException($"Token-major SDPA K/V row counts must match; got K={key.Shape}, V={value.Shape}.");
        long sq = query.Shape[0], skv = key.Shape[0];
        if (sq <= 0 || skv <= 0)
            throw new ArgumentException($"Token-major SDPA dimensions must be positive; got Q={query.Shape}, K={key.Shape}.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "SDPA scale must be finite.");
        if (query.DType != DType.F32 && query.DType != DType.F16 && query.DType != DType.BF16)
            throw new NotSupportedException($"SDPA supports F32/F16/BF16 tensors; got {query.DType}.");
        if (output.DType != query.DType || key.DType != query.DType || value.DType != query.DType)
            throw new ArgumentException(
                $"SDPA requires matching output/Q/K/V dtypes; got output={output.DType}, Q={query.DType}, K={key.DType}, V={value.DType}.");
        if (sq * skv > int.MaxValue || query.ElementCount > int.MaxValue || key.ElementCount > int.MaxValue)
            throw new ArgumentException("SDPA tensors currently support at most Int32.MaxValue elements each.");
        if (mask is null) return;
        if (mask.DType != DType.F32)
            throw new NotSupportedException($"SDPA additive masks must be F32; got {mask.DType}.");
        bool validMask = mask.Shape.Rank switch
        {
            2 => mask.Shape[0] == sq && mask.Shape[1] == skv,
            3 => (mask.Shape[0] == 1 || mask.Shape[0] == heads) && mask.Shape[1] == sq && mask.Shape[2] == skv,
            4 => mask.Shape[0] == 1 && (mask.Shape[1] == 1 || mask.Shape[1] == heads)
                && mask.Shape[2] == sq && mask.Shape[3] == skv,
            _ => false,
        };
        if (!validMask)
            throw new ArgumentException(
                $"SDPA mask {mask.Shape} is not broadcastable to [1,{heads},{sq},{skv}].");
    }

    /// <summary>Validates the canonical rank-4 MHA contract shared by every SDPA dispatch branch.</summary>
    internal static void ValidateScaledDotProductAttentionContract(
        Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        if (output.Shape.Rank != 4 || query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape.Rank != 4)
            throw new ArgumentException(
                $"SDPA requires rank-4 [B,H,S,D] tensors; got output={output.Shape}, Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "SDPA scale must be finite.");
        if (query.DType != DType.F32 && query.DType != DType.F16 && query.DType != DType.BF16)
            throw new NotSupportedException($"SDPA supports F32/F16/BF16 tensors; got {query.DType}.");
        if (output.DType != query.DType || key.DType != query.DType || value.DType != query.DType)
            throw new ArgumentException(
                $"SDPA requires matching output/Q/K/V dtypes; got output={output.DType}, Q={query.DType}, K={key.DType}, V={value.DType}.");

        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
        if (b <= 0 || h <= 0 || sq <= 0 || skv <= 0 || d <= 0)
            throw new ArgumentException($"SDPA dimensions must be positive; got Q={query.Shape}, K={key.Shape}.");
        if (b > int.MaxValue || h > int.MaxValue || sq > int.MaxValue || skv > int.MaxValue || d > int.MaxValue)
            throw new ArgumentException("SDPA dimensions exceed the signed 32-bit CUDA/cuBLAS launch contract.");
        long totalHeads = b * h;
        if (totalHeads > int.MaxValue || totalHeads > int.MaxValue / sq)
            throw new ArgumentException("SDPA head/row products exceed the signed 32-bit CUDA/cuBLAS launch contract.");
        if (output.Shape != query.Shape)
            throw new ArgumentException($"SDPA output shape must equal Q; got output={output.Shape}, Q={query.Shape}.");
        if (key.Shape != value.Shape)
            throw new ArgumentException($"SDPA K/V shapes must match; got K={key.Shape}, V={value.Shape}.");
        if (key.Shape[0] != b || key.Shape[1] != h || key.Shape[3] != d)
            throw new ArgumentException($"SDPA K/V must match Q batch, heads, and head dimension; got Q={query.Shape}, K={key.Shape}.");

        if (output.ElementCount > int.MaxValue || query.ElementCount > int.MaxValue
            || key.ElementCount > int.MaxValue || value.ElementCount > int.MaxValue)
            throw new ArgumentException("SDPA tensors currently support at most Int32.MaxValue elements each.");

        if (mask is null) return;
        if (mask.DType != DType.F32)
            throw new NotSupportedException($"SDPA additive masks must be F32; got {mask.DType}.");
        if (mask.ElementCount > int.MaxValue || sq * skv > int.MaxValue)
            throw new ArgumentException("SDPA mask blocks currently support at most Int32.MaxValue elements.");

        bool validMask = mask.Shape.Rank switch
        {
            2 => (mask.Shape[0] == 1 || mask.Shape[0] == sq) && mask.Shape[1] == skv,
            3 => (mask.Shape[0] == 1 || mask.Shape[0] == h)
                && (mask.Shape[1] == 1 || mask.Shape[1] == sq) && mask.Shape[2] == skv,
            4 => (mask.Shape[0] == 1 || mask.Shape[0] == b) && (mask.Shape[1] == 1 || mask.Shape[1] == h)
                && (mask.Shape[2] == 1 || mask.Shape[2] == sq) && mask.Shape[3] == skv,
            _ => false,
        };
        if (!validMask)
            throw new ArgumentException(
                $"SDPA mask {mask.Shape} is not broadcastable to [{b},{h},{sq},{skv}]; "
                + "expected [1|Sq,Skv], [1|H,1|Sq,Skv], or [1|B,1|H,1|Sq,Skv].");
    }

    /// <summary>Query rows the additive mask actually stores. 1 means the mask depends only on the key and one row is broadcast over every query — the [Sq,Skv] duplicate of it is what makes a masked call expensive.</summary>
    internal static long MaskQueryRows(Tensor mask)
    {
        ArgumentNullException.ThrowIfNull(mask);
        return mask.Shape.Rank switch { 2 => mask.Shape[0], 3 => mask.Shape[1], _ => mask.Shape[2] };
    }

    /// <summary>True for a mask stored as one key row broadcast over more than one query row.</summary>
    internal static bool MaskIsKeyOnly(Tensor? mask, long sq)
        => mask is not null && mask.DType == DType.F32 && sq > 1 && MaskQueryRows(mask) == 1;

    /// <summary>Whether a mask can ride the cuDNN fused engine as an additive fp32 bias (no mask, or F32 broadcastable over heads).</summary>
    /// <remarks>Per-head ([B,H,Sq,Skv]) or non-F32 masks fall back to the materialized path.</remarks>
    private static bool CudnnMaskCompatible(Tensor? mask, long b, long sq, long skv)
    {
        if (mask is null) return true;
        if (mask.DType != DType.F32) return false;
        if (mask.Shape.Rank == 2)
            return (mask.Shape[0] == sq || mask.Shape[0] == 1) && mask.Shape[1] == skv;
        return mask.Shape.Rank == 4 && (mask.Shape[0] == 1 || mask.Shape[0] == b) && mask.Shape[1] == 1
            && (mask.Shape[2] == sq || mask.Shape[2] == 1) && mask.Shape[3] == skv;
    }

    /// <summary>cuDNN fused flash-attention fast path for the plain F32/no-mask/MHA case (F16 execute, F32 output).</summary>
    /// <remarks>Casts Q/K/V to F16, runs cuDNN's fused attention, casts the F16 result back to F32. Returns false (and
    /// permanently disables cuDNN for the session) on any failure so the caller falls through to the materialized/tiled
    /// paths. Q/out [B,H,Sq,D], K/V [B,H,Skv,D].</remarks>
    private unsafe bool TryCudnnSdpa(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
        => TryCudnnSdpa(output, query, key, value, mask, scale, CudnnSdpa.SdpaLayout.HeadMajor,
            query.Shape[0], query.Shape[1], query.Shape[2], key.Shape[2], query.Shape[3]);

    /// <summary>Layout-explicit form: the token-major caller's tensors are rank-2, so the dims cannot be read off the shapes.</summary>
    private unsafe bool TryCudnnSdpa(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale,
        CudnnSdpa.SdpaLayout layout, long b, long h, long sq, long skv, long d)
    {
        lock (_cudnnSdpaLock)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _lifecycleState) != LifecycleActive, this);

            // Callers perform this check before entering, but another request can publish a failure while this
            // request waits for the shared cuDNN handle. Recheck under the same lock that serializes attempts so
            // a queued request cannot bypass a newly established backoff window.
            if (_cudnnSdpaDead || !CudnnSdpaDimEligible(d))
                return false;

            return TryCudnnSdpaLocked(output, query, key, value, mask, scale, layout, b, h, sq, skv, d);
        }
    }

    private unsafe bool TryCudnnSdpaLocked(
        Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale,
        CudnnSdpa.SdpaLayout layout, long b, long h, long sq, long skv, long d)
    {
        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0, qF16 = 0, kF16 = 0, vF16 = 0, oF16 = 0;
        bool cachedOutput = false;
        try
        {
            if (_cudnnSdpa is null)
            {
                _cudnnSdpa = new CudnnSdpa(_stream.Handle);
                Interlocked.Increment(ref _cudnnSdpaSessionGeneration);
            }

            // Checked right after real (or already-cached) engine construction, so an injected failure is
            // classified the same way a real post-init failure (e.g. a BuildPlan/Execute failure) would be
            // — the catch below branches on whether _cudnnSdpa is null, i.e. whether construction itself
            // ever succeeded, and a fault injected before that point would be indistinguishable from a
            // genuine init failure (permanent, session-wide), defeating the point of testing per-dim
            // retry/backoff/classification.
            Exception? injected = TestCudnnSdpaFaultInjector?.Invoke(d);
            if (injected is not null) throw injected;

            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            long biasB = 1, biasSq = sq;
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
                // From the SHAPE, not the element count: a key-only [1,Skv] bias divides to 0 that way.
                biasSq = MaskQueryRows(mask);
                biasB = mask.Shape.Rank == 4 ? mask.Shape[0] : 1;
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (query.DType == DType.F16)
            {
                // Native F16 Q/K/V/output — the engine's fp16 I/O dtype already; execute directly, zero casts.
                _cudnnSdpa.Execute(pQ, pK, pV, pOut, b, h, sq, skv, d, scale, layout, pMask, biasB, biasSq);
            }
            else
            {
                qF16 = CudaMemory.Allocate((nuint)(query.ElementCount * 2));
                kF16 = CudaMemory.Allocate((nuint)(key.ElementCount * 2));
                vF16 = CudaMemory.Allocate((nuint)(value.ElementCount * 2));
                oF16 = CudaMemory.Allocate((nuint)(output.ElementCount * 2));
                _kernels!.LaunchCastF32ToF16(qF16, pQ, (int)query.ElementCount, _stream.Handle);
                _kernels!.LaunchCastF32ToF16(kF16, pK, (int)key.ElementCount, _stream.Handle);
                _kernels!.LaunchCastF32ToF16(vF16, pV, (int)value.ElementCount, _stream.Handle);

                _cudnnSdpa.Execute(qF16, kF16, vF16, oF16, b, h, sq, skv, d, scale, layout, pMask, biasB, biasSq);

                _kernels!.LaunchCastF16ToF32(pOut, oF16, (int)output.ElementCount, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            long executionCount = Interlocked.Increment(ref _cudnnSdpaExecutionCount);
            if (_cudnnSdpaDimState.TryRemove(d, out DimFailureState? recovered))
            {
                HartsyInference.Core.Logging.Logs.Info(
                    $"[cuDNN SDPA] D={d} recovered after {recovered.ConsecutiveFailures} prior failure(s) — fused path re-engaged.");
            }
            if (executionCount == 1)
            {
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN SDPA] fused flash-attention engaged (D={d}, cuDNN {CudnnApi.cudnnGetVersion()})");
            }
            return true;
        }
        catch (Exception ex)
        {
            // Init failure (lib missing) ⇒ session-dead — unconditional, this is never worth retrying (the
            // library either loads or it doesn't).
            if (_cudnnSdpa is null && ex is not OutOfVramException)
            {
                _cudnnSdpaDead = true;
                HartsyInference.Core.Logging.Logs.Warning($"[cuDNN SDPA] disabled for the session (init failed): {ex.Message}");
                return false;
            }
            // A failure AFTER a working init: classify by cuDNN's own status category
            // (CudnnStatusException.IsPermanent) when we can — structural failures (e.g. D=256 on a build
            // whose fused engine tops out at 128) are genuinely never going to succeed and stay permanently
            // disabled for this dim, same as before. Everything else, including the exact class of failure
            // that motivated this (a transient host-RAM allocation error), gets bounded backoff instead of
            // a permanent kill: the resource pressure that causes it is typically external to this process
            // (other processes on the box) and does clear up. An exception we can't positively classify
            // (not a CudnnStatusException) stays conservative and is treated as permanent, same as today.
            // A VRAM shortfall is the one failure class that is provably NOT structural: the fused path needs
            // LESS memory than the materialized fallback it demotes to, so killing it on an OOM guarantees the
            // next call asks for the [heads,Sq,Skv] score matrix and OOMs far harder. Always transient.
            bool permanent = ex switch
            {
                OutOfVramException => false,
                CudnnStatusException cse => cse.IsPermanent,
                _ => true,
            };
            if (permanent)
            {
                _cudnnSdpaDimState[d] = new DimFailureState(0, 0, Permanent: true);
                HartsyInference.Core.Logging.Logs.Warning(
                    $"[cuDNN SDPA] D={d} permanently disabled (structural failure) — this is the steady " +
                    $"state for the rest of the process: {ex.Message}");
            }
            else
            {
                int failures = _cudnnSdpaDimState.TryGetValue(d, out DimFailureState? previous)
                    ? previous.ConsecutiveFailures == int.MaxValue ? int.MaxValue : previous.ConsecutiveFailures + 1
                    : 1;
                long backoffMs = CudnnSdpaBackoffMs(failures);
                DimFailureState state = new(
                    failures,
                    Environment.TickCount64 + backoffMs,
                    Permanent: false);
                _cudnnSdpaDimState[d] = state;
                long? ramMb = HostMemoryInfo.AvailableBytes() is { } bytes ? bytes / (1024 * 1024) : null;
                HartsyInference.Core.Logging.Logs.Warning(
                    $"[cuDNN SDPA] D={d} transient failure #{state.ConsecutiveFailures} " +
                    $"(host RAM available={(ramMb is { } mb ? $"{mb}MB" : "unknown")}) — " +
                    $"retrying after {backoffMs / 1000.0:F1}s: {ex.Message}");
            }
            return false;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (pMask != 0) GpuTransferHelper.FreeDevice(pMask);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (qF16 != 0) CudaMemory.FreeAsync(qF16, _stream.Handle);
            if (kF16 != 0) CudaMemory.FreeAsync(kF16, _stream.Handle);
            if (vF16 != 0) CudaMemory.FreeAsync(vF16, _stream.Handle);
            if (oF16 != 0) CudaMemory.FreeAsync(oF16, _stream.Handle);
        }
    }

    /// <summary>Checks the complete safety contract for the experimental TF32 FlashAttention-v2 kernel.</summary>
    /// <remarks>The current kernel has no masked or GQA variant and its WMMA query load requires complete
    /// 32-row tiles. Rejecting a partial query tile prevents the final block from reading beyond Q.</remarks>
    internal static bool FlashAttentionV2ContractSatisfied(
        Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool tf32Available)
    {
        if (!tf32Available || mask is not null || !float.IsFinite(scale)) return false;
        if (output.Shape.Rank != 4 || query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape.Rank != 4)
            return false;
        if (output.DType != DType.F32 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32)
            return false;

        long b = query.Shape[0];
        long h = query.Shape[1];
        long sq = query.Shape[2];
        long d = query.Shape[3];
        long skv = key.Shape[2];
        if (b <= 0 || h <= 0 || sq <= 0 || skv <= 0 || d <= 0
            || !FlashAttentionV2GridDimensionsSupported(b, h) || sq > int.MaxValue
            || skv > int.MaxValue || d > int.MaxValue)
            return false;
        if ((d != 64 && d != 128) || sq % 32 != 0) return false;

        return output.Shape == query.Shape && key.Shape == value.Shape && key.Shape[0] == b && key.Shape[1] == h
            && key.Shape[3] == d;
    }

    /// <summary>Validates the CUDA grid Y/Z dimensions used directly for head and batch indices.</summary>
    internal static bool FlashAttentionV2GridDimensionsSupported(long batch, long heads)
        => batch is > 0 and <= 65_535 && heads is > 0 and <= 65_535;

    /// <summary>Fused FlashAttention-2 (TF32 tensor cores, F32 accumulate) — no-mask MHA, D∈{64,128}; never materializes the score matrix.</summary>
    private unsafe void FlashAttentionV2(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
        if (!FlashAttentionV2ContractSatisfied(output, query, key, value, null, scale, _allowTf32))
            throw new HartsyInferenceException("FlashAttention-v2 was invoked outside its F32 MHA, full-query-tile contract.");

        EnterOp();
        EnsureKernels();
        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchFlashAttentionV2Tf32(pOut, pQ, pK, pV, (int)b, (int)h, (int)sq, (int)skv, (int)d, scale, _stream.Handle);
            FlashAttentionV2Engaged = true;
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>HARTSY_LTX2_SAGE_TOKENMAJOR=1 — let a token-major SDPA call detour through the permute pair into the head-major SageAttention path. Off by default; the measurement is at the call site.</summary>
    private static readonly bool SageTokenMajorDetour = EngineKnobs.Ltx2SageTokenmajor.Value;

    /// <summary>Whether the native-F16 SageAttention ingest should take this call instead of cuDNN's fp16 flash. Shared by the head-major and token-major entry points: the token-major layout has no Sage kernel, so above the crossover it is worth permuting into head-major rather than keeping the layout.</summary>
    private bool SageF16Preferred(Tensor query, Tensor key, Tensor value, Tensor output, Tensor? mask,
        long sq, long skv, long d)
    {
        if (!UseSageAttn || mask is not null || (d != 64 && d != 128)) return false;
        if (query.DType != DType.F16 || key.DType != DType.F16
            || value.DType != DType.F16 || output.DType != DType.F16) return false;
        // BOTH sides must be long. The prologue (K quant, V transpose, K mean) is paid per KEY while the INT8 QK^T
        // saving is per query x key, so a short-query cross-attention over a long key stream is all prologue: at
        // LTX-2.5's 151-query / 17480-key audio->video attention Sage measured 2.53 ms/call against cuDNN's 0.51.
        if (skv < SageF16MinSkv() || sq < SageF16MinSkv()) return false;
        EnsureKernels();
        return _kernels!.HasSageAttentionKernels && _kernels.HasSageV1;
    }

    /// <summary>Min Skv (override: HARTSY_SAGE_F16_MIN_SKV) above which native-F16 SageAttention ingest is preferred over cuDNN.</summary>
    // Measured: 1.11x at 8192, 1.15x at 12288, parity at 4096 (3060).
    private static int SageF16MinSkv() => EngineKnobs.SageF16MinSkv.Value;

    /// <summary>SageAttention-v1 INT8 flash attention (src/HartsyInference.Cuda/Kernels/attention/sage_attn_int8.cu).</summary>
    /// <remarks>Four launches: K channel-mean → Q per-row INT8 quant (attn scale folded into the row scales) →
    /// (K−mean) per-row INT8 quant (the softmax-invariant smoothing that absorbs the DiT outlier channels) →
    /// fused INT8-QK^T flash loop with online softmax and TF32 PV. Workspace: Q8/K8 mirrors (¼ the F32 bytes),
    /// per-row scales, and the [B,H,D] mean — all transient device allocations, freed before return. Correctness
    /// oracle: SageAttentionReferenceTests (CPU int8 reference math) + SageAttnKernelTests (GPU vs CPU-SDPA parity).</remarks>
    private unsafe void SageAttentionInt8(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
        EnterOp();
        EnsureKernels();
        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
        bool f16 = query.DType == DType.F16;   // native-F16 contract: f16h prologues + f16io flash kernel
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        ulong pQ8 = 0, pQs = 0, pK8 = 0, pKs = 0, pKmean = 0, pVt16 = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            pKmean = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * d * sizeof(float)));
            pQ8 = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * sq * d));
            pQs = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * sq * sizeof(float)));
            pK8 = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * skv * d));
            pKs = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * skv * sizeof(float)));

            // Sub-ranges, not one "SDPA" label: four of these five launches are prologue, and the V transpose
            // alone moves ~14 GB/step on MiniMax-H3 — invisible while the whole call profiled as a single op.
            using (NvtxRange _rk = NvtxRange.PushFine("Sage.KMean"))
            {
                _kernels!.LaunchSageKMean(pKmean, pK, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            }
            // log2-domain softmax: fold log2(e) into the Q row scales so the flash kernels exponentiate via
            // native exp2 (one fewer multiply per score element; max/corr/l all commute with the constant).
            const float Log2E = 1.4426950408889634f;
            using (NvtxRange _rq = NvtxRange.PushFine("Sage.QuantQ"))
            {
                _kernels!.LaunchSageQuantQ(pQ8, pQs, pQ, (int)b, (int)h, (int)sq, (int)d, scale * Log2E, _stream.Handle, srcF16: f16);
            }
            using (NvtxRange _rkq = NvtxRange.PushFine("Sage.QuantK"))
            {
                _kernels!.LaunchSageQuantK(pK8, pKs, pK, pKmean, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            }
            if (_kernels!.UseSageV1)
            {
                using NvtxRange _rv = NvtxRange.PushFine("Sage.VF16T");
                // v1 prologue: one-shot [B,H,Skv,D]→[B,H,D,skvPad] F16 transpose (cp.async-able staging).
                long skvPad = (skv + 7L) & ~7L;
                if (skvPad >= 2048 && (skvPad & (skvPad - 1)) == 0) skvPad += 8;   // anti-aliasing pad (see sage_skv_pad)
                pVt16 = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * d * skvPad * 2));
                _kernels!.LaunchSageVF16T(pVt16, pV, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            }
            using (NvtxRange _ra = NvtxRange.PushFine("Sage.Flash"))
            {
                _kernels!.LaunchSageAttnInt8(pOut, pQ8, pQs, pK8, pKs, pV, pVt16, (int)b, (int)h, (int)sq, (int)skv, (int)d, _stream.Handle, f16Io: f16);
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            Interlocked.Increment(ref _sageAttentionExecutionCount);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pVt16);
            GpuTransferHelper.FreeDevice(pKmean);
            GpuTransferHelper.FreeDevice(pQ8);
            GpuTransferHelper.FreeDevice(pQs);
            GpuTransferHelper.FreeDevice(pK8);
            GpuTransferHelper.FreeDevice(pKs);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Query-tiled SDPA for plain F32/no-mask/MHA; materializes a <c>[totalHeads, Br, Skv]</c> tile at a time (Wan self-attn).</summary>
    /// <remarks>Never materializes the full <c>[totalHeads, Sq, Skv]</c> matrix (24×14040²×4B ≈ 19 GB). Each tile
    /// reuses the same TF32 tensor-core QK^T / softmax / scores·V ops as <see cref="ScaledDotProductAttention"/>, so
    /// results are numerically identical to the plain path. <c>Br</c> is sized to a quarter of free VRAM
    /// (override: <c>HARTSY_SDPA_TILE</c>).</remarks>
    private unsafe void SdpaTiledF32(Tensor output, Tensor query, Tensor key, Tensor value, float scale,
        Tensor? keyBias = null)
    {
        EnterOp();
        EnsureKernels();

        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
        long totalHeads = b * h;

        ulong pQ = 0, pK = 0, pV = 0, pOut = 0, scoresBuf = 0, pBias = 0, pOnes = 0;
        long biasBlocks = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            if (keyBias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(keyBias);
                // The mask may carry one row per batch/head ([H,1,Skv] or [B,H,1,Skv]), not just a single row
                // broadcast over everything — bh below indexes into it modulo this so each head/batch gets its
                // own row instead of every row silently reusing block 0's.
                biasBlocks = Math.Max(1, keyBias.ElementCount / skv);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // Query-tile height: fit [totalHeads, Br, Skv] into ~a quarter of free VRAM (leaves room for Q/K/V/out
            // and the model weights). Env override HARTSY_SDPA_TILE forces a fixed Br (benchmarking).
            (nuint freeBytes, _) = _context.GetMemoryInfo();
            long perRow = totalHeads * skv * sizeof(float);            // bytes for one query row across all heads
            long Br = (long)((ulong)freeBytes / 4) / Math.Max(1, perRow);
            if (Br < 1) Br = 1;
            if (Br > sq) Br = sq;
            if (EngineKnobs.SdpaTile.Value is int envBr && envBr > 0)
                Br = Math.Min(envBr, sq);

            scoresBuf = CudaMemory.Allocate((nuint)(totalHeads * Br * skv * sizeof(float)));
            if (pBias != 0)
            {
                pOnes = CudaMemory.Allocate((nuint)(Br * sizeof(float)));
                CudaMemory.Fill32(pOnes, 0x3F80_0000u, (nuint)Br);   // 1.0f
            }

            long strideQ = sq * d, strideK = skv * d, strideV = skv * d, strideOut = sq * d;
            float alpha = scale, beta = 0f, one = 1f, zero = 0f;

            for (long q0 = 0; q0 < sq; q0 += Br)
            {
                long curBr = Math.Min(Br, sq - q0);
                long tileStride = curBr * skv;   // per-head stride within the (packed) tile score buffer

                // QK^T per head: scores[curBr, Skv] = scale · Q[q0:q0+curBr] · Kᵀ  (TF32 tensor cores)
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    ulong qPtr = pQ + (ulong)((bh * strideQ + q0 * d) * sizeof(float));
                    ulong kPtr = pK + (ulong)(bh * strideK * sizeof(float));
                    ulong sPtr = scoresBuf + (ulong)(bh * tileStride * sizeof(float));
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        (int)skv, (int)curBr, (int)d,
                        &alpha,
                        kPtr, CublasApi.CUDA_R_32F, (int)d,
                        qPtr, CublasApi.CUDA_R_32F, (int)d,
                        &beta,
                        sPtr, CublasApi.CUDA_R_32F, (int)skv,
                        CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }

                // A key-only additive bias needs no [Sq,Skv] duplicate — one rank-1 accumulate covers a whole
                // head's curBr rows. Looped per head/batch (not one accumulate across totalHeads*curBr) so a
                // mask with more than one stored row applies the block that head/batch actually owns.
                if (pBias != 0)
                {
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        ulong sPtr = scoresBuf + (ulong)(bh * tileStride * sizeof(float));
                        ulong biasPtr = pBias + (ulong)((bh % biasBlocks) * skv * sizeof(float));
                        AccumulateKeyBias(sPtr, biasPtr, pOnes, curBr, skv, skv, beta: 1f);
                    }
                }

                // Row-softmax over Skv for the (totalHeads·curBr) packed rows.
                _kernels!.LaunchSoftmax(scoresBuf, (int)skv, (int)(totalHeads * curBr), _stream.Handle);

                // scores·V per head → out[q0:q0+curBr]  (TF32 tensor cores)
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    ulong sPtr = scoresBuf + (ulong)(bh * tileStride * sizeof(float));
                    ulong vPtr = pV + (ulong)(bh * strideV * sizeof(float));
                    ulong oPtr = pOut + (ulong)((bh * strideOut + q0 * d) * sizeof(float));
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        (int)d, (int)curBr, (int)skv,
                        &one,
                        vPtr, CublasApi.CUDA_R_32F, (int)d,
                        sPtr, CublasApi.CUDA_R_32F, (int)skv,
                        &zero,
                        oPtr, CublasApi.CUDA_R_32F, (int)d,
                        CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (pBias != 0) GpuTransferHelper.FreeDevice(pBias);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
            if (pOnes != 0) CudaMemory.FreeAsync(pOnes, _stream.Handle);
        }
    }

    /// <summary>Adds a key-only bias row to every row of a row-major <c>[rows, Skv]</c> F32 score block as a rank-1 GEMM (<c>ones ⊗ bias</c>) — no new kernel, and the bias stays stored once. <paramref name="beta"/> is 0 to materialize the broadcast and 1 to accumulate onto existing scores. Plain FP32 compute (k = 1, so it costs nothing) rather than TF32, so the bias value reaches the scores unrounded.</summary>
    private unsafe void AccumulateKeyBias(ulong dst, ulong bias, ulong ones, long rows, long skv, long ldc, float beta)
    {
        float alpha = 1f;
        CublasApi.cublasGemmEx(
            _cublasHandle,
            CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
            (int)skv, (int)rows, 1,
            &alpha,
            bias, CublasApi.CUDA_R_32F, (int)skv,
            ones, CublasApi.CUDA_R_32F, 1,
            &beta,
            dst, CublasApi.CUDA_R_32F, (int)ldc,
            CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
    }

    #endregion

    #region Activations

    private static int ValidateUnaryFloatOp(string operation, Tensor output, Tensor input)
    {
        if (!output.Shape.Equals(input.Shape))
            throw new HartsyInferenceException($"CUDA {operation} requires matching shapes; got output={output.Shape}, input={input.Shape}.");
        if (output.DType != input.DType)
            throw new NotSupportedException($"CUDA {operation} requires matching dtypes; got output={output.DType}, input={input.DType}.");
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException($"CUDA {operation} supports F32, F16, and BF16; got {input.DType}.");
        if (input.ElementCount <= 0 || input.ElementCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(input),
                $"CUDA {operation} element count must be in [1, {int.MaxValue}]; got {input.ElementCount}.");
        }

        return (int)input.ElementCount;
    }

    public void Gelu(Tensor output, Tensor input)
    {
        int count = ValidateUnaryFloatOp(nameof(Gelu), output, input);
        using NvtxRange _nvtx = NvtxRange.Push("Gelu");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F32)
                _kernels!.LaunchGelu(pOut, pIn, count, _stream.Handle);
            else if (input.DType == DType.F16)
                _kernels!.LaunchGeluF16(pOut, pIn, count, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchGeluBf16(pOut, pIn, count, _stream.Handle);
            else
                throw new NotSupportedException($"CUDA Gelu does not have a launcher for {input.DType}.");

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Sigmoid(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Sigmoid");
        if (input.DType != DType.F32 && !(input.DType == DType.F16 && output.DType == DType.F16))
            throw new NotSupportedException($"CUDA Sigmoid supports F32 or F16 — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            if (input.DType == DType.F16)
                _kernels!.LaunchDitSigmoidF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchAudioSigmoid(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Tanh(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Tanh");
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA Tanh currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchTanh(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Prelu(Tensor output, Tensor input, Tensor alpha)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA Prelu currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];
        int perCh = alpha.ElementCount > 1 ? 1 : 0;

        ulong pOut = 0, pIn = 0, pAlpha = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pAlpha = GpuTransferHelper.CopyToDevice(alpha);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioPrelu(pOut, pIn, pAlpha, batch, channels, timeDim, perCh, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pAlpha);
        }
    }

    public void RepeatTime(Tensor output, Tensor input, int numSamples)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA RepeatTime currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], inT = (int)input.Shape[2];

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioRepeatTime(pOut, pIn, batch, channels, inT, numSamples, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Elu(Tensor output, Tensor input, float alpha)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA Elu currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioElu(pOut, pIn, alpha, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA Snake currently supports F32 only — got {input.DType}.");
        EnterOp();
        EnsureKernels();

        // Snake operates on [B, C, T] with per-channel alpha (and optional per-channel beta).
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];

        ulong pOut = 0, pIn = 0, pAlpha = 0, pBeta = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pAlpha = GpuTransferHelper.CopyToDevice(alpha);
            pBeta = beta is null ? 0 : GpuTransferHelper.CopyToDevice(beta);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioSnake(pOut, pIn, pAlpha, pBeta, batch, channels, timeDim, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pAlpha);
            if (pBeta != 0) GpuTransferHelper.FreeDevice(pBeta);
        }
    }

    public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Conv1d");
        if (output.DType != DType.F32)
            throw new NotSupportedException($"CUDA Conv1d writes F32 output — got output {output.DType}.");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)output.Shape[1], tOut = (int)output.Shape[2], kernel = (int)weight.Shape[2];

        // cuDNN fast path: non-grouped 1D conv → 2D (H=1) so cuDNN's tensor-core (TF32 on Ampere)
        // implicit-GEMM/Winograd engines run it — faster than the direct kernel on the wide vocoder convs.
        // Asymmetric (causal) pads ride the graph API's separate PRE/POST padding attributes, so the
        // HeartCodec/EnCodec-style left-padded convs qualify too. Output stays F32. Falls back to the direct
        // kernel on grouped convs or any cuDNN failure (session-sticky).
        if (_audioConvCudnn && !_cudnnConvDead && groups == 1
            && TryCudnnConv1d(output, input, weight, bias, batch, cIn, cOut, tIn, tOut, kernel, stride, padLeft, padRight, dilation))
        {
            return;
        }

        // The conv1d_f32 kernel is F32; bf16/f16 inputs (e.g. ACE-Step's GLUMBConv on bf16 weights) are cast on-device.
        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, inCast = 0, wCast = 0, bCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            ulong inF32 = CastIfNeeded(pIn, input.DType, DType.F32, (int)input.ElementCount, out inCast);
            ulong wF32 = CastIfNeeded(pW, weight.DType, DType.F32, (int)weight.ElementCount, out wCast);
            ulong bF32 = bias is null ? 0 : CastIfNeeded(pB, bias.DType, DType.F32, (int)bias.ElementCount, out bCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchConv1d(pOut, inF32, wF32, bF32, batch, cIn, cOut, tIn, tOut, kernel,
                stride, padLeft, dilation, groups, bias is null ? 0 : 1, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
            if (wCast != 0) CudaMemory.FreeAsync(wCast, _stream.Handle);
            if (bCast != 0) CudaMemory.FreeAsync(bCast, _stream.Handle);
        }
    }

    /// <summary>cuDNN 1D convolution (mapped to 2D, H=1): F32 output via TF32 tensor cores; weights/inputs cast F32, bias added after.</summary>
    /// <remarks>Asymmetric (causal) pads pass straight through as PRE/POST paddings. Returns false (session-sticky) on any
    /// cuDNN failure so the caller falls back to the direct kernel.</remarks>
    private unsafe bool TryCudnnConv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int padLeft, int padRight, int dilation)
    {
        ulong pIn = 0, pW = 0, pB = 0, pOut = 0, inCast = 0, wCast = 0, bCast = 0;
        bool cachedOutput = false;
        try
        {
            _cudnnConv ??= new CudnnConv(_stream.Handle);
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            ulong inF32 = CastIfNeeded(pIn, input.DType, DType.F32, (int)input.ElementCount, out inCast);
            ulong wF32 = CastIfNeeded(pW, weight.DType, DType.F32, (int)weight.ElementCount, out wCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            // 1D → 2D: N,C,H=1,W=tIn ; filter K,C,R=1,S=kernel ; strideW=stride, padW=padLeft/padRight, dilationW=dilation.
            _cudnnConv.Execute(inF32, wF32, pOut,
                batch, cIn, 1, tIn, cOut, 1, kernel, 1, tOut, 1, stride, 0, padLeft, padRight, CudnnApi.CUDNN_DATA_FLOAT, 1, dilation);
            if (bias is not null)
            {
                pB = GpuTransferHelper.CopyToDevice(bias);
                ulong bF32 = CastIfNeeded(pB, bias.DType, DType.F32, (int)bias.ElementCount, out bCast);
                _kernels!.LaunchBiasAdd(pOut, bF32, cOut, tOut, batch * cOut * tOut, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (!CudnnConvEngaged)
            {
                CudnnConvEngaged = true;
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN conv] audio conv1d engine engaged (cuDNN {CudnnApi.cudnnGetVersion()})");
            }
            return true;
        }
        catch (Exception ex)
        {
            _cudnnConvDead = true;
            HartsyInference.Core.Logging.Logs.Warning(
                $"[cuDNN conv] audio conv1d disabled for the session (falling back to direct kernel) on shape "
                + $"[{batch},{cIn},{tIn}]⊛[{cOut},{cIn},{kernel}] stride={stride} pad=({padLeft},{padRight}) dil={dilation}: {ex.Message}");
            return false;
        }
        finally
        {
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
            if (wCast != 0) CudaMemory.FreeAsync(wCast, _stream.Handle);
            if (bCast != 0) CudaMemory.FreeAsync(bCast, _stream.Handle);
        }
    }

    public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padLeft, int padRight, int dilation, int groups)
    {
        if (output.DType != DType.F32)
            throw new NotSupportedException($"CUDA ConvTranspose1d writes F32 output — got output {output.DType}.");
        EnterOp();
        EnsureKernels();

        // ConvTranspose1d weight is [C_in, C_out/groups, K].
        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)output.Shape[1], tOut = (int)output.Shape[2], kernel = (int)weight.Shape[2];

        // cuDNN fast path: transposed conv = convolution-backward-data (1D → 2D, H=1). The [C_in, C_out, K]
        // weight is exactly cuDNN's dgrad filter layout [K=fwd-out, C=fwd-in, R=1, S=kernel], and the causal
        // right-crop (padRight = K − stride) maps onto the asymmetric PRE/POST paddings of the forward-conv
        // geometry. Same session-sticky fallback contract as the forward path.
        if (_audioConvCudnn && !_cudnnConvDead && groups == 1
            && TryCudnnConvTranspose1d(output, input, weight, bias, batch, cIn, cOut, tIn, tOut, kernel, stride, padLeft, padRight, dilation))
        {
            return;
        }

        // The conv_transpose1d_f32 kernel is F32; bf16/f16 inputs are cast on-device first.
        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, inCast = 0, wCast = 0, bCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            ulong inF32 = CastIfNeeded(pIn, input.DType, DType.F32, (int)input.ElementCount, out inCast);
            ulong wF32 = CastIfNeeded(pW, weight.DType, DType.F32, (int)weight.ElementCount, out wCast);
            ulong bF32 = bias is null ? 0 : CastIfNeeded(pB, bias.DType, DType.F32, (int)bias.ElementCount, out bCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchConvTranspose1d(pOut, inF32, wF32, bF32, batch, cIn, cOut, tIn, tOut, kernel,
                stride, padLeft, dilation, groups, bias is null ? 0 : 1, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
            if (wCast != 0) CudaMemory.FreeAsync(wCast, _stream.Handle);
            if (bCast != 0) CudaMemory.FreeAsync(bCast, _stream.Handle);
        }
    }

    /// <summary>cuDNN transposed 1D convolution via convolution-backward-data (mapped to 2D, H=1): F32 output via TF32 tensor cores; bias is added F32 after. Returns false on a geometry mismatch (without disabling the route) or on any cuDNN failure (session-sticky) so the caller falls back to the direct kernel.</summary>
    private unsafe bool TryCudnnConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int padLeft, int padRight, int dilation)
    {
        // dgrad's conv descriptor describes the forward geometry, which requires the exact length relation
        // tOut = (tIn−1)·stride + dilation·(kernel−1) + 1 − padLeft − padRight (no output_padding support).
        if (tOut != (tIn - 1) * stride + dilation * (kernel - 1) + 1 - padLeft - padRight) return false;
        ulong pIn = 0, pW = 0, pB = 0, pOut = 0, inCast = 0, wCast = 0, bCast = 0;
        bool cachedOutput = false;
        try
        {
            _cudnnConv ??= new CudnnConv(_stream.Handle);
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            ulong inF32 = CastIfNeeded(pIn, input.DType, DType.F32, (int)input.ElementCount, out inCast);
            ulong wF32 = CastIfNeeded(pW, weight.DType, DType.F32, (int)weight.ElementCount, out wCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            // DY = input [N, C_in, 1, tIn], W = [C_in, C_out, 1, K], DX = output [N, C_out, 1, tOut].
            _cudnnConv.ExecuteBackwardData(inF32, wF32, pOut,
                batch, cIn, cOut, 1, tIn, 1, kernel, 1, tOut, 1, stride, 0, padLeft, padRight, CudnnApi.CUDNN_DATA_FLOAT, 1, dilation);
            if (bias is not null)
            {
                pB = GpuTransferHelper.CopyToDevice(bias);
                ulong bF32 = CastIfNeeded(pB, bias.DType, DType.F32, (int)bias.ElementCount, out bCast);
                _kernels!.LaunchBiasAdd(pOut, bF32, cOut, tOut, batch * cOut * tOut, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (!CudnnConvEngaged)
            {
                CudnnConvEngaged = true;
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN conv] audio conv1d engine engaged (cuDNN {CudnnApi.cudnnGetVersion()})");
            }
            return true;
        }
        catch (Exception ex)
        {
            _cudnnConvDead = true;
            HartsyInference.Core.Logging.Logs.Warning(
                $"[cuDNN conv] audio conv_transpose1d disabled for the session (falling back to direct kernel) on shape "
                + $"[{batch},{cIn},{tIn}]⊛ᵀ[{cIn},{cOut},{kernel}] stride={stride} pad=({padLeft},{padRight}) dil={dilation}: {ex.Message}");
            return false;
        }
        finally
        {
            if (!cachedOutput && pOut != 0) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
            if (wCast != 0) CudaMemory.FreeAsync(wCast, _stream.Handle);
            if (bCast != 0) CudaMemory.FreeAsync(bCast, _stream.Handle);
        }
    }

    public unsafe void MaxPool2D(Tensor output, Tensor input, int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        if (output.DType != DType.F32 && output.DType != DType.F16)
            throw new NotSupportedException($"CUDA MaxPool2D supports F32/F16 output — got {output.DType}.");
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException($"CUDA MaxPool2D supports F32/F16/BF16 input — got {input.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        for (int dimension = 0; dimension < 4; dimension++)
        {
            if (input.Shape[dimension] <= 0 || output.Shape[dimension] <= 0)
                throw new ArgumentException($"MaxPool2D requires positive dimensions; got input {input.Shape} / output {output.Shape}.");
        }
        if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 || padH < 0 || padW < 0)
            throw new ArgumentException(
                $"MaxPool2D requires positive kernels/strides and non-negative padding; got kernel={kernelH}x{kernelW}, "
                + $"stride={strideH}x{strideW}, pad={padH}x{padW}.");
        if (input.Shape[0] != output.Shape[0] || input.Shape[1] != output.Shape[1])
            throw new ArgumentException($"MaxPool2D batch/channel mismatch: input {input.Shape}, output {output.Shape}.");

        long paddedH = checked(input.Shape[2] + 2L * padH);
        long paddedW = checked(input.Shape[3] + 2L * padW);
        if (paddedH < kernelH || paddedW < kernelW)
            throw new ArgumentException(
                $"MaxPool2D kernel exceeds padded input: input {input.Shape}, kernel={kernelH}x{kernelW}, pad={padH}x{padW}.");
        long expectedH = (paddedH - kernelH) / strideH + 1;
        long expectedW = (paddedW - kernelW) / strideW + 1;
        if (output.Shape[2] != expectedH || output.Shape[3] != expectedW)
            throw new ArgumentException(
                $"MaxPool2D output shape mismatch: expected [{input.Shape[0]}, {input.Shape[1]}, {expectedH}, {expectedW}], got {output.Shape}.");
        if (input.Shape[0] > int.MaxValue || input.Shape[1] > int.MaxValue || input.Shape[2] > int.MaxValue
            || input.Shape[3] > int.MaxValue || output.Shape[2] > int.MaxValue || output.Shape[3] > int.MaxValue)
            throw new ArgumentException("MaxPool2D dimensions must fit signed 32-bit launch arguments.");
        if (input.DType != output.DType && input.ElementCount > int.MaxValue)
            throw new ArgumentException("MaxPool2D dtype conversion currently supports at most Int32.MaxValue input elements.");
        EnterOp();
        EnsureKernels();

        int n = (int)input.Shape[0], c = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2], oW = (int)output.Shape[3];

        ulong pOut = 0, pIn = 0, inCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            ulong inTyped = input.DType == output.DType ? pIn
                : CastIfNeeded(pIn, input.DType, output.DType, (int)input.ElementCount, out inCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchMaxPool2D(output.DType, pOut, inTyped, n, c, iH, iW, oH, oW,
                kernelH, kernelW, strideH, strideW, padH, padW, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
        }
    }

    public unsafe void DeformableAttention(Tensor output, Tensor value, Tensor sampOff, Tensor attnLogits,
        Tensor refPoints, ReadOnlySpan<int> spatialShapes, ReadOnlySpan<int> levelStart,
        int heads, int levels, int points, int coords, int refQueryStride, int refLevelStride)
    {
        if (output.DType != DType.F32 || value.DType != DType.F32 || sampOff.DType != DType.F32
            || attnLogits.DType != DType.F32 || refPoints.DType != DType.F32)
            throw new NotSupportedException("CUDA DeformableAttention supports F32 only.");
        if (output.Shape.Rank != 3 || value.Shape.Rank != 3 || sampOff.Shape.Rank != 3 || attnLogits.Shape.Rank != 3)
            throw new ArgumentException(
                $"DeformableAttention expects output/value/offset/logit tensors with rank 3; got "
                + $"output={output.Shape}, value={value.Shape}, offsets={sampOff.Shape}, logits={attnLogits.Shape}.");
        if (output.Shape[0] != 1 || value.Shape[0] != 1 || sampOff.Shape[0] != 1 || attnLogits.Shape[0] != 1)
            throw new ArgumentException("CUDA DeformableAttention currently supports batch size 1 only.");
        if (heads <= 0 || levels <= 0 || points <= 0)
            throw new ArgumentException(
                $"DeformableAttention heads, levels, and points must be positive; got heads={heads}, levels={levels}, points={points}.");
        if (coords != 2 && coords != 4)
            throw new ArgumentException($"DeformableAttention coords must be 2 or 4, got {coords}.");
        if (refQueryStride < coords || refLevelStride < 0 || (refLevelStride > 0 && refLevelStride < coords))
            throw new ArgumentException(
                $"Invalid DeformableAttention reference strides: query={refQueryStride}, level={refLevelStride}, coords={coords}.");
        if (spatialShapes.Length != checked(levels * 2) || levelStart.Length != levels)
            throw new ArgumentException(
                $"DeformableAttention level metadata mismatch: levels={levels}, spatialShapes={spatialShapes.Length}, levelStart={levelStart.Length}.");

        long nqLong = output.Shape[1];
        long dLong = output.Shape[2];
        if (nqLong <= 0 || dLong <= 0 || nqLong > int.MaxValue || dLong > int.MaxValue)
            throw new ArgumentException($"DeformableAttention output dimensions must be in [1, Int32.MaxValue]; got {output.Shape}.");
        if (dLong % heads != 0)
            throw new ArgumentException($"DeformableAttention hidden size {dLong} must be divisible by heads {heads}.");
        if (value.Shape[2] != dLong)
            throw new ArgumentException($"DeformableAttention value width {value.Shape[2]} must match output width {dLong}.");

        long offsetWidth = checked((long)heads * levels * points * 2);
        long logitWidth = checked((long)heads * levels * points);
        if (sampOff.Shape[1] != nqLong || sampOff.Shape[2] != offsetWidth)
            throw new ArgumentException(
                $"DeformableAttention offsets must be [1, {nqLong}, {offsetWidth}], got {sampOff.Shape}.");
        if (attnLogits.Shape[1] != nqLong || attnLogits.Shape[2] != logitWidth)
            throw new ArgumentException(
                $"DeformableAttention logits must be [1, {nqLong}, {logitWidth}], got {attnLogits.Shape}.");

        long expectedStart = 0;
        for (int level = 0; level < levels; level++)
        {
            int levelH = spatialShapes[level * 2];
            int levelW = spatialShapes[level * 2 + 1];
            if (levelH <= 0 || levelW <= 0)
                throw new ArgumentException($"DeformableAttention level {level} has invalid shape {levelH}x{levelW}.");
            if (levelStart[level] != expectedStart)
                throw new ArgumentException(
                    $"DeformableAttention level {level} must start at {expectedStart}, got {levelStart[level]}.");
            expectedStart = checked(expectedStart + (long)levelH * levelW);
        }
        if (value.Shape[1] != expectedStart)
            throw new ArgumentException(
                $"DeformableAttention value sequence length {value.Shape[1]} does not match spatial total {expectedStart}.");

        long requiredRefs = checked((nqLong - 1) * refQueryStride + (long)(levels - 1) * refLevelStride + coords);
        if (refPoints.ElementCount < requiredRefs)
            throw new ArgumentException(
                $"DeformableAttention reference tensor has {refPoints.ElementCount} values; at least {requiredRefs} are required.");
        EnterOp();
        EnsureKernels();

        int nq = (int)output.Shape[1];
        int d = (int)output.Shape[output.Shape.Rank - 1];
        int hd = d / heads;
        nuint shBytes = (nuint)(spatialShapes.Length * sizeof(int));
        nuint lsBytes = (nuint)(levelStart.Length * sizeof(int));

        ulong pOut = 0, pVal = 0, pOff = 0, pAt = 0, pRef = 0, pShapes = 0, pLevelStart = 0;
        bool cachedOutput = false;
        try
        {
            pVal = GpuTransferHelper.CopyToDevice(value);
            pOff = GpuTransferHelper.CopyToDevice(sampOff);
            pAt = GpuTransferHelper.CopyToDevice(attnLogits);
            pRef = GpuTransferHelper.CopyToDevice(refPoints);

            pShapes = CudaMemory.AllocateAsync(shBytes, _stream.Handle);
            pLevelStart = CudaMemory.AllocateAsync(lsBytes, _stream.Handle);
            fixed (int* shp = spatialShapes)
                CudaMemory.CopyHostToDeviceAsync(pShapes, shp, shBytes, _stream.Handle);
            fixed (int* lsp = levelStart)
                CudaMemory.CopyHostToDeviceAsync(pLevelStart, lsp, lsBytes, _stream.Handle);

            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchMsdaForward(pOut, pVal, pOff, pAt, pRef, pShapes, pLevelStart,
                nq, heads, hd, levels, points, coords, refQueryStride, refLevelStride, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pVal);
            GpuTransferHelper.FreeDevice(pOff);
            GpuTransferHelper.FreeDevice(pAt);
            GpuTransferHelper.FreeDevice(pRef);
            if (pShapes != 0) CudaMemory.FreeAsync(pShapes, _stream.Handle);
            if (pLevelStart != 0) CudaMemory.FreeAsync(pLevelStart, _stream.Handle);
        }
    }

    public unsafe void Conv2dDepthwise(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Conv2dDepthwise");
        if (output.DType != DType.F32 && output.DType != DType.F16)
            throw new NotSupportedException($"CUDA Conv2dDepthwise supports F32/F16 output — got {output.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"Conv2dDepthwise requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        if (input.Shape[1] != weight.Shape[0] || output.Shape[1] != weight.Shape[0])
            throw new ArgumentException("Conv2dDepthwise requires input/output channel count to equal weight channel count.");
        EnterOp();
        EnsureKernels();

        int n = (int)input.Shape[0], c = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2], oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2], kW = (int)weight.Shape[3];

        ulong pOut = 0, pIn = 0, pW = 0, pB = 0, inCast = 0, wCast = 0, bCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pW = GpuTransferHelper.CopyToDevice(weight);
            pB = bias is null ? 0 : GpuTransferHelper.CopyToDevice(bias);
            ulong inTyped = CastIfNeeded(pIn, input.DType, output.DType, (int)input.ElementCount, out inCast);
            ulong wTyped = CastIfNeeded(pW, weight.DType, output.DType, (int)weight.ElementCount, out wCast);
            ulong bTyped = bias is null ? 0 : CastIfNeeded(pB, bias.DType, output.DType, (int)bias.ElementCount, out bCast);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            _kernels!.LaunchDepthwiseConv2D(output.DType, pOut, inTyped, wTyped, bTyped, bias is null ? 0 : 1,
                n, c, iH, iW, oH, oW, kH, kW, strideH, strideW, padH, padW, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pW);
            if (pB != 0) GpuTransferHelper.FreeDevice(pB);
            if (inCast != 0) CudaMemory.FreeAsync(inCast, _stream.Handle);
            if (wCast != 0) CudaMemory.FreeAsync(wCast, _stream.Handle);
            if (bCast != 0) CudaMemory.FreeAsync(bCast, _stream.Handle);
        }
    }

    public void Silu(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Silu");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchSiluF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchSiluBf16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchSilu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void Mish(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Mish");
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("CUDA Mish supports F32 only.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchAudioMish(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    #endregion

    #region Element-wise

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Add");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (a.DType == DType.F16)
                _kernels!.LaunchAddF16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else if (a.DType == DType.BF16)
                _kernels!.LaunchAddBf16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchAdd(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            // a and b can be the same tensor (e.g. an elementwise self-op) — CopyToDevice then returns the same
            // cached pointer for both, so freeing pB unconditionally double-frees it (CUDA_ERROR_INVALID_VALUE).
            if (pB != pA) GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Mul(Tensor output, Tensor a, Tensor b)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Mul");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pA = 0, pB = 0;
        bool cachedOutput = false;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (a.DType == DType.F16)
                _kernels!.LaunchMulF16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else if (a.DType == DType.BF16)
                _kernels!.LaunchMulBf16(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchMul(pOut, pA, pB, (int)a.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pA);
            // a and b can be the same tensor (e.g. Nemotron's relu(x)² does Mul(outp, outp, outp)) — CopyToDevice
            // then returns the same cached pointer for both, so freeing pB unconditionally double-frees it
            // (CUDA_ERROR_INVALID_VALUE).
            if (pB != pA) GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Scale(Tensor output, Tensor input, float scalar)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Scale");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchScaleF16(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchScaleBf16(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchScale(pOut, pIn, scalar, (int)input.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Device-side feature-cache gate metric Σ|a−b|/Σ|b| (stepcache.ptx); never pulls the operands to the host.</summary>
    /// <remarks>Cached activations stay cached. The 8-byte result readback is the only sync (once per gated forward).</remarks>
    public unsafe float RelativeL1Distance(Tensor a, Tensor b)
    {
        if (!a.Shape.Equals(b.Shape))
            throw new ArgumentException($"RelativeL1Distance shape mismatch: a={a.Shape}, b={b.Shape}.");
        if (a.DType != b.DType)
            throw new ArgumentException($"RelativeL1Distance dtype mismatch: a={a.DType}, b={b.DType}.");
        if (a.DType != DType.F32 && a.DType != DType.F16)
            throw new NotSupportedException($"RelativeL1Distance supports F32/F16; got {a.DType}.");
        EnterOp();
        EnsureKernels();
        if (!_kernels!.HasStepCacheKernels)
            throw new NotSupportedException(
                "stepcache.ptx not present — run src/HartsyInference.Cuda/Kernels/dit/build.sh and gate on SupportsDeviceStepCacheGate.");

        ulong pA = 0, pB = 0, pSums = 0;
        try
        {
            pA = GpuTransferHelper.CopyToDevice(a);
            pB = GpuTransferHelper.CopyToDevice(b);
            pSums = GpuTransferHelper.AllocateDevice(2 * sizeof(float));

            // Synchronous NULL-stream memset serializes correctly against the single blocking compute stream
            // (same ordering guarantee CudaMemory.cuMemsetD32 relies on).
            CudaDriverApi.cuMemsetD8(pSums, 0, 2 * sizeof(float)).ThrowOnError();
            _kernels!.LaunchStepCacheRelL1(pSums, pA, pB, a.ElementCount, a.DType == DType.F16, _stream.Handle);

            float* results = stackalloc float[2];
            CudaDriverApi.cuMemcpyDtoH((nint)results, pSums, 2 * sizeof(float)).ThrowOnError();
            if (!float.IsFinite(results[0]) || !float.IsFinite(results[1]))
                return float.NaN;
            return results[1] > 0f ? results[0] / results[1] : results[0] > 0f ? float.PositiveInfinity : 0f;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pSums);
            GpuTransferHelper.FreeDevice(pA);
            if (pB != pA) GpuTransferHelper.FreeDevice(pB);
        }
    }

    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Clamp");
        int count = ValidateUnaryFloatOp(nameof(Clamp), output, input);
        if (float.IsNaN(min) || float.IsNaN(max) || min > max)
            throw new ArgumentException($"CUDA Clamp requires ordered, non-NaN bounds; got min={min}, max={max}.");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F32)
                _kernels!.LaunchClamp(pOut, pIn, min, max, count, _stream.Handle);
            else if (input.DType == DType.F16)
                _kernels!.LaunchClampF16(pOut, pIn, min, max, count, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchClampBf16(pOut, pIn, min, max, count, _stream.Handle);
            else
                throw new NotSupportedException($"CUDA Clamp does not have a launcher for {input.DType}.");

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    #endregion

    #region FP8 Helpers

    /// <summary>Maps a HartsyInference dtype to its cuBLAS data-type constant for <c>cublasGemmEx</c> / <c>cublasGemmStridedBatchedEx</c>.</summary>
    /// <remarks>Handles F16, BF16, F32; throws otherwise (FP8 casts to F16/BF16 via <see cref="CastIfNeeded"/> before cuBLAS).</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CublasDataTypeForGemm(DType dtype, DType input, DType weight, DType output, int m, int n, int k)
    {
        if (dtype == DType.F32 || dtype == DType.F16 || dtype == DType.BF16) return CublasDataType(dtype);
        throw new NotSupportedException(
            $"cuBLAS GEMM does not support dtype {dtype} (input={input}, weight={weight}, output={output}, M={m}, N={n}, K={k}).");
    }

    private static int CublasDataType(DType dtype)
    {
        if (dtype == DType.F16) return CublasApi.CUDA_R_16F;
        if (dtype == DType.BF16) return CublasApi.CUDA_R_16BF;
        if (dtype == DType.F32) return CublasApi.CUDA_R_32F;
        throw new NotSupportedException($"cuBLAS GEMM does not support dtype {dtype}.");
    }

    /// <summary>Resolves GEMM compute dtype for two operands, preferring F16 over F32 when either is F16 (or fp8, which casts to F16).</summary>
    /// <remarks>cublasGemmEx COMPUTE_32F does not support F32×F32→F16, so when an F16 activation feeds an F16 output
    /// and the weight happens to be F32, the F32 weight must cast down to F16 — F16 also gets Tensor Core
    /// acceleration on Ampere+.</remarks>
    private DType ResolveGemmDtype(DType a, DType b)
    {
        // FP8 forces a 16-bit GEMM (Ampere has fast Tensor Cores in F16/BF16, not F32). Pick:
        //  - BF16 when the other operand is F32 — BF16 has F32's full dynamic range, so the
        //    F32→16-bit activation cast cannot produce ±Inf even when SwiGLU's `gated`
        //    intermediate momentarily exceeds 65504 in F32. F16 *would* overflow there
        //    (Z-Image L0 ffnOut had INF=9024 at step 1 of CUDA bring-up before this fix).
        //  - F16 otherwise — keeps the existing F16 fast path that Flux/SDXL FP8 paths rely on
        //    when their activations are already F16 (and therefore in-range).
        if (a.IsFp8 || b.IsFp8)
        {
            // F16 (10-bit mantissa) is more accurate than BF16 (7-bit) for the activation cast; BF16 is the default only
            // because SwiGLU MLPs can momentarily exceed F16's 65504. For GELU-FFN models (Wan) F16 is safe AND needed:
            // over a deep DiT (40 layers) + CFG, BF16's coarser mantissa lets a small per-step velocity bias compound
            // into a diverging trajectory. HARTSY_FP8_F16 opts the fp8 path into F16.
            if (EnableFp8F32Gemm) return DType.F32;
            if (EnableFp8F16Gemm) return DType.F16;
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        // GGUF quants always dequantize to F16 (or BF16 if the other operand is F32). The
        // dequant kernels emit F16 directly; routing through F32 would force an extra F16→F32
        // cast pass for no benefit. Same precedence rule as FP8 above.
        if (a.IsQuantized || b.IsQuantized)
        {
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        if (a == DType.F16 || b == DType.F16) return DType.F16;
        if (HighPrecisionGemm && (a == DType.BF16 || b == DType.BF16) && (a == DType.F32 || b == DType.F32))
            return DType.F32;
        if (a == DType.BF16 || b == DType.BF16) return DType.BF16;
        return a == DType.F32 || b == DType.F32 ? DType.F32 : a;
    }

    /// <summary>Ensures a GPU buffer of <paramref name="srcDtype"/> is available in <paramref name="dstDtype"/>, casting only if needed.</summary>
    /// <remarks>Returns the existing pointer if no cast is needed, or allocates + casts and writes the new dptr to
    /// <paramref name="castOut"/> (which the caller is responsible for freeing with <c>cuMemFreeAsync</c>). Hides
    /// the F8 special case so the four GEMM call sites all look the same.</remarks>
    private unsafe ulong CastIfNeeded(ulong srcPtr, DType srcDtype, DType dstDtype, int elementCount, out ulong castOut)
    {
        if (srcDtype == dstDtype)
        {
            castOut = 0;
            return srcPtr;
        }
        castOut = CudaMemory.Allocate((nuint)((long)elementCount * dstDtype.SizeInBytes));
        CastOnGpu(castOut, srcPtr, srcDtype, dstDtype, elementCount);
        return castOut;
    }

    /// <summary>Casts GPU data between dtypes via PTX kernels: F8↔F16, F16↔F32, and GGUF quantized → F16/F32 dequant (Q8_0/Q4_K/Q5_K/Q6_K).</summary>
    private void CastOnGpu(ulong output, ulong input, DType srcDtype, DType dstDtype, int count)
    {
        if (srcDtype == dstDtype) return;

        // ── GGUF dequant paths. F16 is the kernel's native output. F32 and BF16 stage through F16. ──
        if (srcDtype.IsQuantized && dstDtype == DType.F16)
        {
            LaunchGgufDequantToF16(output, input, srcDtype, count);
            return;
        }
        if (srcDtype.IsQuantized && dstDtype == DType.F32)
        {
            ulong tempF16 = CudaMemory.Allocate((nuint)((long)count * DType.F16.SizeInBytes));
            try
            {
                LaunchGgufDequantToF16(tempF16, input, srcDtype, count);
                _kernels!.LaunchCastF16ToF32(output, tempF16, count, _stream.Handle);
            }
            finally
            {
                CudaMemory.FreeAsync(tempF16, _stream.Handle);
            }
            return;
        }
        if (srcDtype.IsQuantized && dstDtype == DType.BF16)
        {
            // quant → F16 → F32 → BF16. F32 staging needed because BF16 conversion goes through F32 in our kernel set.
            ulong tempF16 = CudaMemory.Allocate((nuint)((long)count * DType.F16.SizeInBytes));
            ulong tempF32 = CudaMemory.Allocate((nuint)((long)count * DType.F32.SizeInBytes));
            try
            {
                LaunchGgufDequantToF16(tempF16, input, srcDtype, count);
                _kernels!.LaunchCastF16ToF32(tempF32, tempF16, count, _stream.Handle);
                _kernels!.LaunchCastF32ToBf16(output, tempF32, count, _stream.Handle);
            }
            finally
            {
                CudaMemory.FreeAsync(tempF16, _stream.Handle);
                CudaMemory.FreeAsync(tempF32, _stream.Handle);
            }
            return;
        }

        if (srcDtype.IsFp8 && dstDtype == DType.F16)
            _kernels!.LaunchCastF8E4M3ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F16 && dstDtype.IsFp8)
            _kernels!.LaunchCastF16ToF8E4M3(output, input, count, _stream.Handle);
        else if (srcDtype.IsFp8 && dstDtype == DType.F32)
        {
            // F8 → F16 → F32 (two-step via temp buffer)
            ulong temp = CudaMemory.Allocate((nuint)((long)count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastF8E4M3ToF16(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF32(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype.IsFp8 && dstDtype == DType.BF16)
        {
            // F8 → F32 → BF16 (the values FP8 represents are within F16 range, so we could go
            // F8→F16 first, but the F16→BF16 path also goes via F32; folding them avoids a
            // redundant intermediate). FP8 max ≈ 448, well within BF16's range.
            ulong temp = CudaMemory.Allocate((nuint)((long)count * DType.F32.SizeInBytes));
            // F8 → F32 (re-uses the two-step F8→F16→F32 ladder via recursion).
            CastOnGpu(temp, input, srcDtype, DType.F32, count);
            _kernels!.LaunchCastF32ToBf16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.BF16 && dstDtype.IsFp8)
        {
            // BF16 → F32 → F16 → F8. BF16 represents values up to 3.4e38; FP8's max is 448.
            // Going through F32 then F16 catches saturation at the F16 stage (which clips to ±Inf,
            // then the F16→F8 stage maps Inf to FP8's NaN encoding — so over-range values are
            // marked rather than wrapping silently).
            ulong temp32 = CudaMemory.Allocate((nuint)((long)count * DType.F32.SizeInBytes));
            ulong temp16 = CudaMemory.Allocate((nuint)((long)count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastBf16ToF32(temp32, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToF16(temp16, temp32, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF8E4M3(output, temp16, count, _stream.Handle);
            CudaMemory.FreeAsync(temp32, _stream.Handle);
            CudaMemory.FreeAsync(temp16, _stream.Handle);
        }
        else if (srcDtype == DType.F32 && dstDtype == DType.F16)
            _kernels!.LaunchCastF32ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F16 && dstDtype == DType.F32)
            _kernels!.LaunchCastF16ToF32(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F32 && dstDtype.IsFp8)
        {
            // F32 → F16 → F8 (two-step via temp buffer)
            ulong temp = CudaMemory.Allocate((nuint)((long)count * DType.F16.SizeInBytes));
            _kernels!.LaunchCastF32ToF16(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF16ToF8E4M3(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.BF16 && dstDtype == DType.F32)
            _kernels!.LaunchCastBf16ToF32(output, input, count, _stream.Handle);
        else if (srcDtype == DType.F32 && dstDtype == DType.BF16)
            _kernels!.LaunchCastF32ToBf16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.BF16 && dstDtype == DType.F16)
        {
            // BF16 → F32 → F16 (lossy via temp F32 buffer)
            ulong temp = CudaMemory.Allocate((nuint)((long)count * DType.F32.SizeInBytes));
            _kernels!.LaunchCastBf16ToF32(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToF16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else if (srcDtype == DType.F16 && dstDtype == DType.BF16)
        {
            // F16 → F32 → BF16 (round-trip via F32)
            ulong temp = CudaMemory.Allocate((nuint)((long)count * DType.F32.SizeInBytes));
            _kernels!.LaunchCastF16ToF32(temp, input, count, _stream.Handle);
            _kernels!.LaunchCastF32ToBf16(output, temp, count, _stream.Handle);
            CudaMemory.FreeAsync(temp, _stream.Handle);
        }
        else
            throw new NotSupportedException($"GPU cast from {srcDtype} to {dstDtype} not supported.");
    }

    /// <summary>Dispatches the per-DType GGUF dequant kernel. Count must respect the source dtype's block size (32 for Q8_0, 256 for Q*_K).</summary>
    private void LaunchGgufDequantToF16(ulong output, ulong input, DType srcDtype, int count)
    {
        if (srcDtype == DType.Q8_0)
            _kernels!.LaunchDequantQ8_0ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q4_0)
            _kernels!.LaunchDequantQ4_0ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q5_0)
            _kernels!.LaunchDequantQ5_0ToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q4_K)
            _kernels!.LaunchDequantQ4_KToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q5_K)
            _kernels!.LaunchDequantQ5_KToF16(output, input, count, _stream.Handle);
        else if (srcDtype == DType.Q6_K)
            _kernels!.LaunchDequantQ6_KToF16(output, input, count, _stream.Handle);
        else
            throw new NotSupportedException($"GPU dequant for {srcDtype} not yet implemented. Supported: Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K. Use CPU dequant via GgufDequantizer for other GGUF types.");
    }

    /// <summary>Test hook for the native-fp8 activation quantization kernels.</summary>
    /// <remarks>Computes the per-tensor e4m3 dequant scale and quantized bytes.</remarks>
    /// <remarks>Scale (<c>amax/448</c>) goes into <paramref name="scaleOut"/> (1-element F32), quantized bytes into
    /// <paramref name="fp8Out"/> (same element count as <paramref name="input"/>, F8E4M3). Runs on any CUDA GPU —
    /// the kernels are plain compute (only the GEMM needs Ada) — so the Ampere CI box can validate them.</remarks>
    internal void Fp8QuantizeActivationForTest(Tensor fp8Out, Tensor scaleOut, Tensor input)
    {
        EnterOp();
        EnsureKernels();
        int count = (int)input.ElementCount;
        ulong pIn = 0, pOut = 0, pScale = 0, pScratch = 0;
        bool cachedOut = false, cachedScale = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pOut = GpuTransferHelper.AllocateDevice((nuint)count);
            pScale = GpuTransferHelper.AllocateDevice(sizeof(float));
            pScratch = CudaMemory.Allocate((nuint)(CudaKernels.Fp8AbsMaxBlockCount(count) * sizeof(float)));
            _kernels!.LaunchFp8AbsMaxScale(pIn, pScratch, pScale, count, _stream.Handle);
            _kernels!.LaunchFp8QuantF32ToE4M3(pOut, pIn, pScale, count, _stream.Handle);
            GpuTransferHelper.CacheActivation(fp8Out, pOut, (nuint)count);
            cachedOut = true;
            GpuTransferHelper.CacheActivation(scaleOut, pScale, sizeof(float));
            cachedScale = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (pScratch != 0) CudaMemory.FreeAsync(pScratch, _stream.Handle);
            if (!cachedOut) GpuTransferHelper.FreeDevice(pOut);
            if (!cachedScale) GpuTransferHelper.FreeDevice(pScale);
        }
    }

    /// <summary>Implements CastF8E4M3ToF16 using the PTX cast kernel on GPU.</summary>
    public void CastF8E4M3ToF16(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("CastF8E4M3ToF16");
        EnterOp();
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchCastF8E4M3ToF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Implements CastF16ToF8E4M3 using the PTX cast kernel on GPU.</summary>
    public void CastF16ToF8E4M3(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("CastF16ToF8E4M3");
        EnterOp();
        EnsureKernels();
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchCastF16ToF8E4M3(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    private void EnsureKernels()
    {
        if (_kernels == null)
            throw new InvalidOperationException("PTX kernels not loaded. Provide a ptxDir to the CudaBackend constructor.");
    }

    /// <summary>Synchronizes the default compute stream. Only needed at pipeline boundaries or before explicit D2H.</summary>
    /// <remarks>Per-op sync removed — CUDA guarantees sequential execution on a single blocking stream.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sync()
    {
        EnterOp();
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
        // HARTSY_PROFILE_EACH=1: dump the accumulated per-op profile at each Sync (end of a generation) — the Swarm
        // ShutdownServer path does not reliably dispose the backend, so this is the reliable per-gen dump hook.
        if (EngineKnobs.ProfileEach.Value)
            Profiling.NvtxRange.DumpProfile(EngineKnobs.ProfileOut.Value ?? "/tmp/hartsy_profile.txt");
    }

    #endregion

    #region Transpose / Permute

    /// <summary>Batched 2D transpose: [B, D1, D2] -> [B, D2, D1] via PTX kernel.</summary>
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Transpose2D");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // BF16 piggy-backs on the F16 kernel — both are pure 16-bit byte shuffles
            // (no math, no precision concern), so the same kernel produces correct output.
            if (input.DType == DType.F16 || input.DType == DType.BF16)
                _kernels!.LaunchTranspose2DF16(pOut, pIn, d1, d2, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchTranspose2D(pOut, pIn, d1, d2, (int)output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>4D permute(0,2,1,3): [B, S, H, D] -> [B, H, S, D] via PTX kernel.</summary>
    public void Permute0213(Tensor output, Tensor input, int s, int h, int d)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Permute0213");
        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // BF16 piggy-backs on the F16 kernel (pure 16-bit byte shuffle, see Transpose2D).
            if (input.DType == DType.F16 || input.DType == DType.BF16)
                _kernels!.LaunchPermute0213F16(pOut, pIn, s, h, d, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchPermute0213(pOut, pIn, s, h, d, (int)output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>GQA K/V head repeat: [B, Hkv, L, D] → [B, Hkv*groupSize, L, D]. GPU-resident device-to-device gather.</summary>
    public void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("RepeatKvHeads");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new HartsyInferenceException($"CUDA RepeatKvHeads requires rank-4 input/output; got input={input.Shape}, output={output.Shape}.");
        if (kvHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(kvHeads), kvHeads, "KV head count must be positive.");
        if (groupSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(groupSize), groupSize, "KV repeat group size must be positive.");
        if (input.DType != output.DType)
            throw new NotSupportedException($"CUDA RepeatKvHeads requires matching dtypes; got output={output.DType}, input={input.DType}.");
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException($"CUDA RepeatKvHeads supports F32, F16, and BF16; got {input.DType}.");
        if (input.Shape[0] <= 0 || input.Shape[1] <= 0 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new HartsyInferenceException($"CUDA RepeatKvHeads requires positive input dimensions; got {input.Shape}.");
        if (input.Shape[1] != kvHeads)
            throw new HartsyInferenceException($"CUDA RepeatKvHeads kvHeads={kvHeads} does not match input head dimension {input.Shape[1]}.");
        if (input.Shape[2] > int.MaxValue || input.Shape[3] > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(input),
                $"CUDA RepeatKvHeads sequence/head dimensions exceed the launcher limit: {input.Shape}.");
        }

        long expandedHeads = (long)kvHeads * groupSize;
        if (expandedHeads > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(groupSize), groupSize, "Expanded KV head count exceeds the kernel index range.");
        if (output.Shape[0] != input.Shape[0] || output.Shape[1] != expandedHeads ||
            output.Shape[2] != input.Shape[2] || output.Shape[3] != input.Shape[3])
        {
            throw new HartsyInferenceException(
                $"CUDA RepeatKvHeads expected output [{input.Shape[0]}, {expandedHeads}, {input.Shape[2]}, {input.Shape[3]}], got {output.Shape}.");
        }

        EnterOp();
        EnsureKernels();

        int seqLen = (int)input.Shape[2];
        int headDim = (int)input.Shape[3];

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F32)
                _kernels!.LaunchRepeatKv(pOut, pIn, kvHeads, groupSize, seqLen, headDim, output.ElementCount, _stream.Handle);
            else if (input.DType == DType.F16)
                _kernels!.LaunchRepeatKvF16(pOut, pIn, kvHeads, groupSize, seqLen, headDim, output.ElementCount, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchRepeatKvBf16(pOut, pIn, kvHeads, groupSize, seqLen, headDim, output.ElementCount, _stream.Handle);
            else
                throw new NotSupportedException($"CUDA RepeatKvHeads does not have a launcher for {input.DType}.");

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>FlashAttention (online-softmax, GQA-aware, no materialized score matrix).</summary>
    /// <remarks>Requires F32 and head dimension in [1, 1024]. The launcher pads small dimensions to a full warp
    /// so the reduction's full-mask shuffle remains valid. Unsupported inputs fall back to the CPU reference.</remarks>
    public unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        ValidateFlashAttentionContract(
            output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);
        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        bool kernelOk = d > 0 && d <= 1024;
        // F16-storage KV cache (halved VRAM): key/value are the FixedKvCache buffers, which under kvDtype=F16
        // are __half-typed; Q/out/sink/alibi are unaffected (still F32). Both the monolithic and the split-K
        // path have an F16 twin, so the dtype no longer decides which kernel runs — only sink/alibi do.
        bool f16Kv = key.DType == DType.F16 && value.DType == DType.F16;
        if (!kernelOk)
        {
            AttentionReference.FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);
            return;
        }

        EnterOp();
        EnsureKernels();
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0, pSink = 0, pAlibi = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            if (sink is not null) pSink = GpuTransferHelper.CopyToDevice(sink);
            if (alibiSlopes is not null) pAlibi = GpuTransferHelper.CopyToDevice(alibiSlopes);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // Flash-decoding (split-K) for the plain path: when the base grid (b·hq·tq) under-occupies the
            // GPU — the decode case, e.g. 32 blocks on 128 SMs — split the key axis across more blocks and
            // merge with a combine kernel. Numerically exact vs the monolithic kernel (same per-key scores,
            // online-softmax merge). Sink/ALiBi/soft-cap/sliding-window keep the proven monolithic path.
            int baseBlocks = b * hq * tq;
            // Soft-cap and sliding-window are handled inside the split kernel (per-logit transform / key-range
            // clamp — see flash_attn_f32_split.cu); only sink and ALiBi still require the monolithic kernel.
            // This matters enormously for low-head-count windowed models: gemma3-1b decodes with FOUR query
            // heads, so the monolithic path put 4 blocks on 28 SMs (measured 5.4× slower than llama.cpp e2e).
            bool splitEligible = pSink == 0 && pAlibi == 0;
            bool forceSplit = EngineKnobs.FlashSplitForce.Value;
            // Occupancy-limited = the LLM decode case (tq=1, few heads → e.g. 16 blocks on 28 SMs). Splitting the
            // key axis there fills the GPU and is a large decode win (attention was ~30% of decode; split-K ≈ +38%
            // end-to-end on Qwen3). The split kernel is numerically exact vs monolithic (online-softmax merge). The
            // old kvLen≥1024 floor never engaged for decode (kvLen<300); gate on occupancy instead, floor 128.
            bool occLimited = baseBlocks < 2 * _context.MultiprocessorCount;
            // Severely under-occupied launches (fewer blocks than SMs — gemma3-1b: 4, qwen2.5-0.5b: 14) are
            // worth splitting even at short kvLen; the 128 floor only applies to moderately-occupied shapes.
            int engageLen = baseBlocks <= _context.MultiprocessorCount ? 32 : 128;
            int splits = 1;
            if (!EngineKnobs.FlashSplitOff.Value
                && splitEligible && (forceSplit || occLimited) && kvLen >= (forceSplit ? 8 : engageLen))
            {
                // Target/minChunk tuned the same way as FlashAttentionDev's graph-decode split formula below
                // (measured A/B sweep on Qwen3-4B/RTX 3060, kvLen~193: tok/s rose monotonically from the old
                // target=2×SM through ~4× more splits before plateauing) — same kernel, same occupancy physics,
                // this is the eager (non-graph-decode) dispatch of the identical split/combine kernel pair.
                int target = forceSplit ? 4 * baseBlocks : 16 * _context.MultiprocessorCount;
                int g = (target + baseBlocks - 1) / baseBlocks;
                int minChunk = forceSplit ? 1 : (occLimited ? 16 : 256);
                int maxG = Math.Max(1, kvLen / minChunk);
                g = Math.Clamp(g, 1, Math.Min(32, maxG));
                if (g >= 2) splits = g;
            }

            if (splits >= 2)
            {
                int chunk = (kvLen + splits - 1) / splits;
                splits = (kvLen + chunk - 1) / chunk;   // exact # of non-empty chunks covering kvLen
                long n = baseBlocks;
                ulong pM = 0, pL = 0, pAcc = 0;
                try
                {
                    pM = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
                    pL = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
                    pAcc = GpuTransferHelper.AllocateDevice((nuint)(n * splits * d * sizeof(float)));
                    _kernels!.LaunchFlashAttentionSplit(pM, pL, pAcc, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen,
                        kvGroup <= 0 ? 1 : kvGroup, causal, qOffset, scale, splits, chunk, _stream.Handle,
                        softcap: softcap, slidingWindow: slidingWindow, f16Kv: f16Kv);
                    _kernels!.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, tq, d, splits, _stream.Handle);
                }
                finally
                {
                    GpuTransferHelper.FreeDevice(pM); GpuTransferHelper.FreeDevice(pL); GpuTransferHelper.FreeDevice(pAcc);
                }
            }
            else if (f16Kv)
            {
                _kernels!.LaunchFlashAttentionF16Kv(pOut, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen, kvGroup <= 0 ? 1 : kvGroup, causal, qOffset, scale, softcap, pSink, slidingWindow, pAlibi, _stream.Handle);
            }
            else
            {
                _kernels!.LaunchFlashAttention(pOut, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen, kvGroup <= 0 ? 1 : kvGroup, causal, qOffset, scale, softcap, pSink, slidingWindow, pAlibi, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
            if (pSink != 0) GpuTransferHelper.FreeDevice(pSink);
            if (pAlibi != 0) GpuTransferHelper.FreeDevice(pAlibi);
        }
    }

    /// <summary>Allocates a fixed-capacity KV-cache buffer on device, registered resident, skipping the host-zeroed upload.</summary>
    /// <remarks>The tail beyond CurrentLength is never read (FlashAttention gets the exact kvLen), so leaving it
    /// uninitialized is correct. Idempotent: skip if already resident.</remarks>
    public unsafe void ResidentAllocateKv(Tensor buffer)
    {
        if (GpuTransferHelper.IsActivationCached(buffer)) return;
        EnterOp();
        nuint bytes = GpuTransferHelper.ByteSize(buffer);
        ulong dptr = GpuTransferHelper.AllocateDevice(bytes);
        // KvCacheAppend writes in place through this pointer; mark it cache-owned so no dispose/sync callback frees
        // it out from under the append (matches how KvCacheAppend re-caches the buffer below).
        buffer._gpuSyncCallback = null;
        buffer._gpuDisposeCallback = null;
        GpuTransferHelper.CacheActivation(buffer, dptr, bytes);
    }

    /// <summary>In-place KV-cache append (device-to-device strided write); no reallocation, buffer stays GPU-resident.</summary>
    public unsafe void KvCacheAppend(Tensor buffer, Tensor newKv, int offset)
    {
        // F16-storage KV cache (halved VRAM): the source stays F32 (straight out of the K/V projection) and
        // the destination buffer is F16 — a distinct dtype-converting kernel, not the plain-copy F32 path.
        bool f16Dest = buffer.DType == DType.F16 && newKv.DType == DType.F32;
        if (!f16Dest && (buffer.DType != DType.F32 || newKv.DType != DType.F32))
        {
            throw new NotSupportedException(
                $"CUDA KvCacheAppend supports F32→F32 or F32→F16 storage; got buffer={buffer.DType}, newKv={newKv.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int h = (int)buffer.Shape[1], maxSeq = (int)buffer.Shape[2], d = (int)buffer.Shape[3], tNew = (int)newKv.Shape[2];
        ulong pBuf = 0, pNew = 0;
        try
        {
            pBuf = GpuTransferHelper.CopyToDevice(buffer);
            pNew = GpuTransferHelper.CopyToDevice(newKv);
            if (f16Dest)
            {
                _kernels!.LaunchKvAppendF16(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle);
            }
            else
            {
                _kernels!.LaunchKvAppend(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle);
            }
            // buffer is updated in place; it is a cache-owned resident activation, so re-cache its pointer.
            buffer._gpuSyncCallback = null;
            buffer._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(buffer, pBuf, GpuTransferHelper.ByteSize(buffer));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pNew);
        }
    }

    /// <summary>Extracts a contiguous time-range from a KV-shaped tensor (device-to-device strided read).</summary>
    /// <remarks>Used by the paged KV cache to split multi-token appends across page boundaries and to gather
    /// a partially-filled page's occupied prefix.</remarks>
    public unsafe void SliceTimeRange(Tensor output, Tensor input, int start, int len)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA SliceTimeRange requires F32 output/input; got output={output.DType}, input={input.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int h = (int)input.Shape[1], tIn = (int)input.Shape[2], d = (int)input.Shape[3];
        ulong pIn = 0;
        nuint outBytes = GpuTransferHelper.ByteSize(output);
        ulong pOut = GpuTransferHelper.AllocateDevice(outBytes);
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            _kernels!.LaunchKvSliceTime(pOut, pIn, h, tIn, d, start, len, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Uploads a host span to a freshly allocated device buffer.</summary>
    private static unsafe ulong UploadArray<T>(ReadOnlySpan<T> values) where T : unmanaged
    {
        nuint bytes = (nuint)((long)values.Length * sizeof(T));
        ulong dptr = GpuTransferHelper.AllocateDevice(bytes);
        try
        {
            fixed (T* p = values) CudaDriverApi.cuMemcpyHtoD(dptr, (nint)p, bytes).ThrowOnError();
            return dptr;
        }
        catch
        {
            GpuTransferHelper.FreeDevice(dptr);
            throw;
        }
    }

    // ── Device-side decode position (autoregressive step-graph replay) ──────────────────────────────────────
    public bool GraphDecodeSupported => true;

    /// <summary>Persistent 2-int device buffer {kvLen, qOffset} the graph-decode kernels read each step.</summary>
    /// <remarks>Allocated once (outside capture) via the synchronous persistent allocator so it survives graph AUTO_FREE_ON_LAUNCH.</remarks>
    public ulong AllocDevicePos()
    {
        EnterOp();
        return CudaMemory.AllocatePersistent((nuint)(2 * sizeof(int)));
    }

    /// <summary>Refreshes the device position buffer for the next step.</summary>
    /// <remarks>Must be called OUTSIDE a capture region: the captured graph reads this buffer's address; only its contents change.</remarks>
    public unsafe void WriteDevicePos(ulong handle, int kvLen, int qOffset)
    {
        if (handle == 0) return;
        EnterOp();
        int* v = stackalloc int[2]; v[0] = kvLen; v[1] = qOffset;
        // HtoDAsync from pageable host memory stages synchronously (host buffer consumed before return), so the
        // stackalloc is safe; the device write is stream-ordered before the subsequent kernels/graph launch.
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)v, (nuint)(2 * sizeof(int)), _stream.Handle).ThrowOnError();
    }

    public void FreeDevicePos(ulong handle)
    {
        if (handle == 0) return;
        EnterOp();
        CudaMemory.Free(handle);
    }

    /// <summary>KV-cache append with the write slot read from <paramref name="devicePos"/> (graph-replayable).</summary>
    public unsafe void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos)
    {
        // F16 buffers scatter through lm_kv_append_f16, which takes the same device position. The source stays
        // F32 — the projection output is F32 and the kernel narrows on write, exactly as the eager path does.
        bool appendF16 = buffer.DType == DType.F16 && newKv.DType == DType.F32;
        if (devicePos == 0 || (!appendF16 && (buffer.DType != DType.F32 || newKv.DType != DType.F32)))
        {
            KvCacheAppend(buffer, newKv, offset);
            return;
        }
        EnterOp();
        EnsureKernels();
        // Flatten batch into heads: buffer [B,H,maxSeq,D] is contiguous == [B*H, maxSeq, D], so B*H heads appends
        // every batch element at the same slot (cond+uncond share the position). B=1 → unchanged for all callers.
        int h = (int)(buffer.Shape[0] * buffer.Shape[1]), maxSeq = (int)buffer.Shape[2], d = (int)buffer.Shape[3], tNew = (int)newKv.Shape[2];
        ulong pBuf = 0, pNew = 0;
        try
        {
            pBuf = GpuTransferHelper.CopyToDevice(buffer);
            pNew = GpuTransferHelper.CopyToDevice(newKv);
            if (appendF16)
            {
                _kernels!.LaunchKvAppendF16(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle, devicePos);
            }
            else
            {
                _kernels!.LaunchKvAppend(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle, devicePos);
            }
            buffer._gpuSyncCallback = null;
            buffer._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(buffer, pBuf, GpuTransferHelper.ByteSize(buffer));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pNew);
        }
    }

    /// <summary>Self-attention FlashAttention; kvLen/qOffset read from <paramref name="devicePos"/>, split-K decode, FIXED split count.</summary>
    /// <remarks>Chunk sized to the cache CAPACITY, not the current kvLen, so the grid is position-independent and
    /// graph-capturable while still filling the GPU at long context: splits whose chunk starts past the current
    /// kvLen early-exit in the kernel, so active parallelism grows with position.</remarks>
    public unsafe void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos,
        float softcap = 0f, int slidingWindow = 0)
    {
        // Graph decode can keep kvLen/qOffset exclusively on-device. Unsupported storage modes fall back to
        // eager attention, whose validator requires meaningful host positions; the device path validates the
        // same buffer/layout/GQA contract without pretending the zero host placeholders are current positions.
        bool devF16Kv = key.DType == DType.F16 && value.DType == DType.F16;
        if (devicePos == 0 || query.DType != DType.F32 || output.DType != DType.F32
            || (!devF16Kv && (key.DType != DType.F32 || value.DType != DType.F32)))
        {
            FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink: null, slidingWindow, alibiSlopes: null);
            return;
        }
        ValidateFlashAttentionContract(
            output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap,
            sink: null, slidingWindow, alibiSlopes: null, positionOnDevice: true);

        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        bool kernelOk = d > 0 && d <= 1024;
        // F16-storage KV is served here now that both device-position kernels have an F16 twin: the monolithic
        // lm_flash_attn_f16kv_f32 and the split lm_flash_attn_f16kv_f32_split. Q, out and the partials stay F32.
        // Without this the graph path had to force F32 storage, doubling the cache to ~576 KB/frame across the
        // guided pair and capping a 12 GB card near four minutes of song.
        if (!kernelOk)
        {
            throw new NotSupportedException("Device-position FlashAttention supports head dimensions up to 1024.");
        }
        EnterOp();
        EnsureKernels();

        // Fixed split count from CAPACITY (lk) so the grid never changes across steps. Target ~16× SM occupancy
        // (measured, not the original ~4×: an A/B sweep on Qwen3-4B/RTX 3060, kvLen capacity=193, showed tok/s
        // rising monotonically from splits=2 (61.7) through splits=12 (69.6, +12.9%) before plateauing at
        // splits=12-20 and slightly regressing beyond 24 — the fixed combine-kernel overhead per split eventually
        // outweighs the added parallelism. splits=12 sits in that plateau with margin either side).
        // chunk = ceil(capacity / splits) is constant, so early steps leave later splits empty (they early-exit).
        // Split-K is gated to b==1: the split/combine kernels have a latent batch>1 bug (they were only exercised by
        // the b=1 LLM decode), and for CFG-batched decode b=2 already yields 2·heads blocks — enough to fill the GPU
        // with the monolithic kernel — so batching keeps the (correct) monolithic path with no occupancy loss.
        int baseBlocks = b * hq * tq;
        int splits = 1;
        // Same engage rule as the eager dispatch: severely under-occupied launches (fewer blocks than SMs —
        // gemma3-1b decodes with 4 query heads, qwen2.5-0.5b with 14) are worth splitting even at short
        // capacity; the 128 floor only applies to moderately-occupied shapes.
        int devEngageLen = baseBlocks <= _context.MultiprocessorCount ? 32 : 128;
        if (b == 1 && lk >= devEngageLen)
        {
            int target = 16 * _context.MultiprocessorCount;
            int g = (target + baseBlocks - 1) / baseBlocks;
            int maxG = Math.Max(1, lk / 16);   // keep chunks ≥ 16 keys (was 64 — measured too conservative)
            splits = Math.Clamp(g, 1, Math.Min(32, maxG));
        }
        int chunk = (lk + splits - 1) / splits;         // fixed chunk over capacity
        splits = (lk + chunk - 1) / chunk;              // exact # of chunks covering capacity

        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            int grp = kvGroup <= 0 ? 1 : kvGroup;
            if (splits >= 2)
            {
                long n = baseBlocks;
                ulong pM = 0, pL = 0, pAcc = 0;
                try
                {
                    pM = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
                    pL = GpuTransferHelper.AllocateDevice((nuint)(n * splits * sizeof(float)));
                    pAcc = GpuTransferHelper.AllocateDevice((nuint)(n * splits * d * sizeof(float)));
                    _kernels!.LaunchFlashAttentionSplit(pM, pL, pAcc, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen,
                        grp, causal, qOffset, scale, splits, chunk, _stream.Handle, devicePos, softcap, slidingWindow,
                        f16Kv: devF16Kv);
                    _kernels!.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, tq, d, splits, _stream.Handle);
                }
                finally
                {
                    GpuTransferHelper.FreeDevice(pM); GpuTransferHelper.FreeDevice(pL); GpuTransferHelper.FreeDevice(pAcc);
                }
            }
            else
            {
                if (devF16Kv)
                {
                    _kernels!.LaunchFlashAttentionF16Kv(pOut, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen, grp,
                        causal, qOffset, scale, softcap, 0, slidingWindow, 0, _stream.Handle, devicePos);
                }
                else
                {
                    _kernels!.LaunchFlashAttention(pOut, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen, grp,
                        causal, qOffset, scale, softcap, 0, slidingWindow, 0, _stream.Handle, devicePos);
                }
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pQ);
            GpuTransferHelper.FreeDevice(pK);
            GpuTransferHelper.FreeDevice(pV);
        }
    }

    /// <summary>DtoH a resident device tensor WITHOUT freeing its buffer.</summary>
    /// <remarks>A normal DataPointer read syncs then frees, which would break a captured graph's fixed output-buffer address.</remarks>
    /// <remarks>Stream-syncs first so the pending launch has produced the data.</remarks>
    public unsafe void ReadResidentInto(Tensor src, float[] dst)
    {
        EnterOp();
        ulong p = GpuTransferHelper.CopyToDevice(src);   // resident activation → cached ptr (no re-upload, no free)
        _stream.Synchronize();
        fixed (float* d = dst)
            CudaDriverApi.cuMemcpyDtoH((nint)d, p, (nuint)((long)dst.Length * sizeof(float))).ThrowOnError();
    }

    // ── LLM decode-graph: RoPE / embed / argmax device state ────────────────────────────────────────────────
    // See IBackend's matching doc comments for the overall design. This backend implements all of it.

    public ulong AllocDeviceTokenId()
    {
        EnterOp();
        return CudaMemory.AllocatePersistent((nuint)sizeof(int));
    }

    public unsafe void WriteDeviceTokenId(ulong handle, int tokenId)
    {
        if (handle == 0) return;
        EnterOp();
        int v = tokenId;
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)(&v), (nuint)sizeof(int), _stream.Handle).ThrowOnError();
    }

    public void FreeDeviceTokenId(ulong handle)
    {
        if (handle == 0) return;
        EnterOp();
        CudaMemory.Free(handle);
    }

    /// <summary>D2H sync read of the 1-int device token-id buffer. Blocking (cuMemcpyDtoH), called once per decode step after LaunchGraph.</summary>
    /// <remarks>The sync waits on exactly the work that step's graph replay queued (not the whole stream's backlog).</remarks>
    public unsafe int ReadDeviceTokenId(ulong handle)
    {
        if (handle == 0) throw new NotSupportedException("ReadDeviceTokenId called with an unallocated buffer.");
        EnterOp();
        int v;
        CudaDriverApi.cuMemcpyDtoH((nint)(&v), handle, (nuint)sizeof(int)).ThrowOnError();
        return v;
    }

    /// <summary>Builds the RoPE table once, registered as a permanent resident tensor pair: stable across capture/replay, never evicted.</summary>
    /// <remarks>Same math as GenericTransformer.BuildRope, just for every position 0..maxPos-1 up front instead of one
    /// small per-step tensor.</remarks>
    public unsafe (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, RopeScaling scaling, bool splitHalfPartial = false)
    {
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int rotHalf = rdim / 2;
        int halfDim = headDim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, maxPos);
        Tensor cos = new(new TensorShape(maxPos, headDim), DType.F32);
        Tensor sin = new(new TensorShape(maxPos, headDim), DType.F32);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        // Partial rotary (rotaryDim < headDim) needs two incompatible layouts (see IBackend doc):
        // duplicate stride = rotaryDim/2 for the split-half kernels (they pair (i, i+rotaryDim/2) and
        // read raw index i across [0, rotaryDim)), headDim/2 for the interleaved kernels (one value per
        // pair from [0, headDim/2), identity in [rotaryDim/2, headDim/2) so un-rotated pairs pass
        // through — the GLM-4 fix caught by the 2026-07-22 popular-model benchmark sweep; entries left
        // as alloc garbage silently corrupt output). Full rotary: strides coincide, flag irrelevant.
        int dupStride = splitHalfPartial ? rotHalf : halfDim;
        for (int pos = 0; pos < maxPos; pos++)
        {
            long baseOff = (long)pos * headDim;
            for (int i = 0; i < headDim; i++) { pc[baseOff + i] = 1f; ps[baseOff + i] = 0f; }
            for (int i = 0; i < rotHalf; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float s = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + dupStride] = c;
                ps[baseOff + i] = s; ps[baseOff + i + dupStride] = s;
            }
        }
        PreloadWeights([cos, sin]);
        return (cos, sin);
    }

    public void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos)
    {
        if (devicePos == 0 || x.DType != DType.F32 || cosTable.DType != DType.F32 || sinTable.DType != DType.F32)
        {
            // No silent no-op fallback: skipping the rotation would produce plausible-looking wrong tokens.
            throw new NotSupportedException(
                $"CUDA RopeApplyDecodeStep requires a device position buffer and F32 operands; got devicePos={devicePos}, " +
                $"x={x.DType}, cos={cosTable.DType}, sin={sinTable.DType}.");
        }
        EnterOp();
        EnsureKernels();
        // Head count from the element count, not Shape[2]: at t=1 the [1,1,H,D] and [1,H,1,D] layouts are
        // byte-identical and graph decode passes the head-major form (permute-free path); this also covers
        // any batched [B,1,H,D] caller correctly (rotate all B·H heads, not just the first batch element).
        int headDim = (int)x.Shape[x.Shape.Rank - 1];
        int numHeads = (int)(x.ElementCount / headDim);
        ulong pX = GpuTransferHelper.CopyToDevice(x);
        ulong pCos = GpuTransferHelper.CopyToDevice(cosTable);
        ulong pSin = GpuTransferHelper.CopyToDevice(sinTable);
        if (interleaved)
            _kernels!.LaunchRopeDecodeInterleaved(pX, pCos, pSin, numHeads, headDim, devicePos, _stream.Handle);
        else
            _kernels!.LaunchRopeDecodeSplitHalf(pX, pCos, pSin, numHeads, headDim, rotaryDim, devicePos, _stream.Handle);
        x._gpuSyncCallback = null;
        x._gpuDisposeCallback = null;
        GpuTransferHelper.CacheActivation(x, pX, GpuTransferHelper.ByteSize(x));
    }

    public unsafe void EmbedGatherDecodeStep(Tensor output, Tensor embedTable, ulong tokenId)
    {
        if (tokenId == 0 || output.DType != DType.F32 || embedTable.DType != DType.F32)
        {
            throw new NotSupportedException("EmbedGatherDecodeStep requires an F32 embed table and a valid device token-id buffer.");
        }
        EnterOp();
        EnsureKernels();
        int hidden = (int)output.ElementCount;
        ulong pEmb = GpuTransferHelper.CopyToDevice(embedTable);
        nuint outBytes = GpuTransferHelper.ByteSize(output);
        ulong pOut = GpuTransferHelper.AllocateDevice(outBytes);
        _kernels!.LaunchEmbedGatherDecode(pOut, pEmb, tokenId, hidden, _stream.Handle);
        GpuTransferHelper.CacheActivation(output, pOut, outBytes);
    }

    public void ArgMaxInto(ulong outputTokenId, Tensor input)
    {
        if (outputTokenId == 0 || input.DType != DType.F32)
        {
            throw new NotSupportedException("ArgMaxInto requires F32 input and a valid device token-id buffer.");
        }
        EnterOp();
        EnsureKernels();
        int c = (int)input.Shape[input.Shape.Rank - 1];
        int rows = (int)(input.ElementCount / c);
        ulong pIn = GpuTransferHelper.CopyToDevice(input);
        _kernels!.LaunchArgMaxLastDim(outputTokenId, pIn, rows, c, _stream.Handle, _argmaxScratch);
    }

    // ── Graph-capture decode: repetition penalty ────────────────────────────────────────────────────────────

    public ulong AllocDeviceHistory(int capacity)
    {
        EnterOp();
        return CudaMemory.AllocatePersistent((nuint)(Math.Max(1, capacity) * sizeof(int)));
    }

    public void FreeDeviceHistory(ulong handle)
    {
        if (handle == 0) return;
        EnterOp();
        CudaMemory.Free(handle);
    }

    public ulong AllocDeviceCounter()
    {
        EnterOp();
        return CudaMemory.AllocatePersistent((nuint)sizeof(int));
    }

    public void FreeDeviceCounter(ulong handle)
    {
        if (handle == 0) return;
        EnterOp();
        CudaMemory.Free(handle);
    }

    public unsafe void WriteDeviceCounter(ulong handle, int value)
    {
        if (handle == 0) return;
        EnterOp();
        int v = value;
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)(&v), (nuint)sizeof(int), _stream.Handle).ThrowOnError();
    }

    public void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId)
    {
        if (history == 0 || historyCount == 0 || tokenId == 0) return;
        EnterOp();
        EnsureKernels();
        _kernels!.LaunchHistoryAppend(history, historyCount, tokenId, _stream.Handle);
    }

    public void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty)
    {
        if (history == 0 || historyCount == 0 || logits.DType != DType.F32)
        {
            throw new NotSupportedException("ApplyRepetitionPenaltyStep requires F32 logits and valid device history buffers.");
        }
        EnterOp();
        EnsureKernels();
        int vocabSize = (int)logits.Shape[logits.Shape.Rank - 1];
        ulong pLogits = GpuTransferHelper.CopyToDevice(logits);
        _kernels!.LaunchRepetitionPenalty(pLogits, history, historyCount, penalty, vocabSize, _stream.Handle);
        GpuTransferHelper.CacheActivation(logits, pLogits, GpuTransferHelper.ByteSize(logits));
    }

    /// <summary>Backend-agnostic graph capture (see IBackend); auto-free-on-relaunch since replays require repeated launches.</summary>
    /// <remarks>Validated on hardware (docs/Research/CUDA_GRAPH_FINDINGS.md): per-op stream-ordered activation
    /// allocations captured inside recordWork are freed before each relaunch and reuse the same virtual addresses,
    /// so pointers cached at capture time (e.g. the graph-decode session's cosTable/embedTable/devicePos) stay
    /// valid across replays.</remarks>
    public object? CaptureGraph(Action recordWork)
    {
        EnterOp();
        CudaGraph graph = new(_stream.Handle, autoFreeAllocationsOnRelaunch: true);
        // Route capture-time intermediate allocations through the persistent bump arena: without it,
        // every per-op AllocateDevice/free during capture becomes a memAlloc/memFree node that re-executes
        // on EVERY replay (measured on gemma3: 1672 of the graph's 2264 nodes — ~2-3 ms/token of pure
        // memory-node overhead on sub-2B models). Arena buffers persist for the backend's lifetime, so
        // captured pointers stay valid across replays; overflow falls back to pool nodes (correct, slower).
        if (EngineKnobs.GraphArena.Value)
        {
            graph.ArenaBase = GpuTransferHelper.BeginGraphArena();
            try { graph.Capture(recordWork); }
            finally { GpuTransferHelper.EndGraphArena(); }
        }
        else
        {
            graph.Capture(recordWork);
        }
        return graph;
    }

    public void LaunchGraph(object graphHandle)
    {
        EnterOp();
        ((CudaGraph)graphHandle).Launch();
    }

    public void DisposeGraph(object graphHandle)
    {
        CudaGraph graph = (CudaGraph)graphHandle;
        EnterOp();
        graph.Dispose();
        GpuTransferHelper.FreeGraphArena(graph.ArenaBase);
    }

    /// <summary>MoE row-gather: output[m] = input[rowIndices[m]] (collect an expert's routed tokens).</summary>
    public unsafe void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("GatherRows");
        if (ReferenceEquals(output, input))
            throw new ArgumentException("GatherRows does not support an in-place output.", nameof(output));
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA GatherRows requires F32 output/input; got output={output.DType}, input={input.DType}.");
        }
        if (input.Shape.Rank < 1)
            throw new ArgumentException("GatherRows input must have at least one dimension.", nameof(input));
        long width = input.Shape[input.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(input), $"GatherRows row width must be in [1,{int.MaxValue}]; got {width}.");
        long inputRows = input.ElementCount / width;
        long requiredOutputElements = checked((long)rowIndices.Length * width);
        if (output.ElementCount != requiredOutputElements)
            throw new ArgumentException(
                $"GatherRows output has {output.ElementCount} elements; {rowIndices.Length} rows of width {width} require {requiredOutputElements}.",
                nameof(output));
        for (int i = 0; i < rowIndices.Length; i++)
        {
            if (rowIndices[i] < 0 || (long)rowIndices[i] >= inputRows)
                throw new ArgumentOutOfRangeException(
                    nameof(rowIndices), rowIndices[i], $"Row index at position {i} is outside [0,{inputRows}).");
        }
        EnterOp();
        EnsureKernels();
        int k = (int)width;
        ulong total = (ulong)rowIndices.Length * (ulong)k;
        ulong pIn = 0, pIdx = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = UploadArray<int>(rowIndices);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchGatherRows(pOut, pIn, pIdx, k, total, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <inheritdoc />
    public bool TryGatherRowsResident(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        EnterOp();
        if (!GpuTransferHelper.IsWeightCached(input) && !GpuTransferHelper.IsActivationCached(input))
            return false;
        GatherRows(output, input, rowIndices);
        return true;
    }

    /// <summary>Per-row argmax over the last dim: indices[r] = argmax_c input[r,c]. On-device; only indices sync back, not full logit rows.</summary>
    public unsafe void ArgMaxLastDim(Tensor indices, Tensor input)
    {
        if (input.DType != DType.F32 || indices.DType != DType.I32)
        {
            throw new NotSupportedException(
                $"CUDA ArgMaxLastDim requires F32 input and I32 indices; got input={input.DType}, indices={indices.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int c = (int)input.Shape[input.Shape.Rank - 1];
        int rows = (int)(input.ElementCount / c);
        ulong pIn = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = (nuint)((long)rows * sizeof(int));
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchArgMaxLastDim(pOut, pIn, rows, c, _stream.Handle, _argmaxScratch);
            GpuTransferHelper.CacheActivation(indices, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Validates FlashAttention geometry before either the CUDA kernel or CPU reference can index it.</summary>
    internal static void ValidateFlashAttentionContract(
        Tensor output, Tensor query, Tensor key, Tensor value,
        int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap,
        Tensor? sink, int slidingWindow, Tensor? alibiSlopes, bool positionOnDevice = false)
    {
        if (output.Shape.Rank != 4 || query.Shape.Rank != 4 || key.Shape.Rank != 4 || value.Shape.Rank != 4)
            throw new ArgumentException(
                $"FlashAttention requires rank-4 [B,H,S,D] tensors; got output={output.Shape}, Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        if (output.DType != DType.F32 || query.DType != DType.F32)
            throw new NotSupportedException(
                $"FlashAttention requires F32 output/Q; got output={output.DType}, Q={query.DType}.");
        if (key.DType != value.DType || (key.DType != DType.F32 && key.DType != DType.F16))
            throw new NotSupportedException(
                $"FlashAttention K/V must both be F32 or both be F16; got K={key.DType}, V={value.DType}.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), "FlashAttention scale must be finite.");
        if (!float.IsFinite(softcap) || softcap < 0f)
            throw new ArgumentOutOfRangeException(nameof(softcap), "FlashAttention softcap must be finite and non-negative.");
        if (slidingWindow < 0)
            throw new ArgumentOutOfRangeException(nameof(slidingWindow), "FlashAttention sliding window must be non-negative.");

        long b = query.Shape[0], hq = query.Shape[1], tq = query.Shape[2], d = query.Shape[3];
        long hkv = key.Shape[1], lk = key.Shape[2];
        if (b <= 0 || hq <= 0 || tq <= 0 || d <= 0 || hkv <= 0 || lk <= 0)
            throw new ArgumentException($"FlashAttention dimensions must be positive; got Q={query.Shape}, K={key.Shape}.");
        if (b > int.MaxValue || hq > int.MaxValue || tq > int.MaxValue || d > int.MaxValue
            || hkv > int.MaxValue || lk > int.MaxValue)
            throw new ArgumentException("FlashAttention dimensions exceed the signed 32-bit launch contract.");
        long headRows = b * hq;
        if (headRows > int.MaxValue || headRows > int.MaxValue / tq)
            throw new ArgumentException("FlashAttention batch/head/query products exceed the signed 32-bit launch contract.");
        if (output.Shape != query.Shape)
            throw new ArgumentException($"FlashAttention output shape must equal Q; got output={output.Shape}, Q={query.Shape}.");
        if (key.Shape != value.Shape)
            throw new ArgumentException($"FlashAttention K/V shapes must match; got K={key.Shape}, V={value.Shape}.");
        if (key.Shape[0] != b || key.Shape[3] != d)
            throw new ArgumentException($"FlashAttention K/V must match Q batch and head dimension; got Q={query.Shape}, K={key.Shape}.");
        if (kvGroup <= 0 || (long)kvGroup * hkv != hq)
            throw new ArgumentException(
                $"FlashAttention requires Hq == Hkv * kvGroup with a positive group; got Hq={hq}, Hkv={hkv}, kvGroup={kvGroup}.");
        if (!positionOnDevice)
        {
            if (kvLen <= 0 || kvLen > lk)
                throw new ArgumentOutOfRangeException(nameof(kvLen), kvLen, $"FlashAttention kvLen must be in [1,{lk}].");
            if (qOffset < 0 || (causal && (long)qOffset + tq > kvLen))
                throw new ArgumentOutOfRangeException(
                    nameof(qOffset), qOffset, "FlashAttention causal query positions must lie inside the valid KV prefix.");
        }
        if (key.DType == DType.F16 && d > 1024)
            throw new NotSupportedException("F16-KV FlashAttention supports head dimensions up to 1024; no F16 CPU fallback exists.");

        if (sink is not null && (sink.DType != DType.F32 || sink.ElementCount != hq))
            throw new ArgumentException($"FlashAttention sink must be F32 with exactly Hq={hq} elements; got {sink.DType} {sink.Shape}.");
        if (alibiSlopes is not null && (alibiSlopes.DType != DType.F32 || alibiSlopes.ElementCount != hq))
            throw new ArgumentException(
                $"FlashAttention ALiBi slopes must be F32 with exactly Hq={hq} elements; got {alibiSlopes.DType} {alibiSlopes.Shape}.");
    }

    /// <summary>MoE weighted scatter-add (in place): output[rowIndices[m]] += scales[m]·input[m].</summary>
    /// <remarks>Output must be a resident, already-accumulating activation (pre-zeroed, then one call per expert).</remarks>
    public unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            throw new NotSupportedException(
                $"CUDA ScatterAddWeightedRows requires F32 output/input; got output={output.DType}, input={input.DType}.");
        }
        EnterOp();
        EnsureKernels();
        int k = (int)input.Shape[input.Shape.Rank - 1];
        ulong total = (ulong)rowIndices.Length * (ulong)k;
        ulong pIn = 0, pIdx = 0, pScale = 0, pOut = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);   // resident accumulator (cache hit)
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = UploadArray<int>(rowIndices);
            pScale = UploadArray<float>(scales);
            _kernels!.LaunchScatterAddWeightedRows(pOut, pIn, pIdx, pScale, k, total, _stream.Handle);
            output._gpuSyncCallback = null;
            output._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pIn);
            GpuTransferHelper.FreeDevice(pIdx);
            GpuTransferHelper.FreeDevice(pScale);
        }
    }

    /// <summary>GEGLU activation splitting each logical row as <c>[value | gate]</c> along its last dimension.</summary>
    public void GeGlu(Tensor output, Tensor input)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("GeGlu");
        if (input.Shape.Rank == 0 || output.Shape.Rank != input.Shape.Rank)
        {
            throw new HartsyInferenceException(
                $"CUDA GeGlu requires input/output with the same positive rank; got input={input.Shape}, output={output.Shape}.");
        }
        if (input.DType != output.DType)
            throw new NotSupportedException($"CUDA GeGlu requires matching dtypes; got output={output.DType}, input={input.DType}.");
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException($"CUDA GeGlu supports F32, F16, and BF16; got {input.DType}.");

        long inputLastDim = input.Shape[input.Shape.Rank - 1];
        if (inputLastDim <= 0 || (inputLastDim & 1) != 0)
            throw new HartsyInferenceException($"CUDA GeGlu requires an even, positive input last dimension; got {inputLastDim} in {input.Shape}.");
        long expectedInnerDim = inputLastDim / 2;
        for (int dim = 0; dim < input.Shape.Rank - 1; dim++)
        {
            if (output.Shape[dim] != input.Shape[dim])
                throw new HartsyInferenceException($"CUDA GeGlu output prefix must match input; got input={input.Shape}, output={output.Shape}.");
        }
        if (output.Shape[output.Shape.Rank - 1] != expectedInnerDim)
        {
            throw new HartsyInferenceException(
                $"CUDA GeGlu output last dimension must be {expectedInnerDim} for input {input.Shape}; got output={output.Shape}.");
        }
        if (expectedInnerDim > int.MaxValue || output.ElementCount <= 0 || output.ElementCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(output),
                $"CUDA GeGlu shape exceeds the 32-bit launcher range: input={input.Shape}, output={output.Shape}.");
        }

        EnterOp();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            int innerDim = (int)output.Shape[output.Shape.Rank - 1];
            if (input.DType == DType.F32)
                _kernels!.LaunchGeGlu(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);
            else if (input.DType == DType.F16)
                _kernels!.LaunchGeGluF16(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchGeGluBf16(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);
            else
                throw new NotSupportedException($"CUDA GeGlu does not have a launcher for {input.DType}.");

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>Broadcast add: hidden[b,c,s] += bias[b,c] in-place via PTX kernel.</summary>
    public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("BroadcastAdd");
        EnterOp();
        EnsureKernels();

        ulong pHidden = 0, pBias = 0;
        try
        {
            pHidden = GpuTransferHelper.CopyToDevice(hidden);
            pBias = GpuTransferHelper.CopyToDevice(bias);

            if (hidden.DType == DType.F16)
                _kernels!.LaunchBroadcastAddF16(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);
            else if (hidden.DType == DType.BF16)
                _kernels!.LaunchBroadcastAddBf16(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchBroadcastAdd(pHidden, pBias, channels, spatial, (int)hidden.ElementCount, _stream.Handle);

            // BroadcastAdd modifies hidden in-place. Clear old GPU callbacks before re-caching
            // to prevent CacheActivation's DataPointer access from firing the old sync callback
            // (which would FreeAsync the GPU pointer we're about to re-cache).
            hidden._gpuSyncCallback = null;
            hidden._gpuDisposeCallback = null;
            nuint hiddenBytes = GpuTransferHelper.ByteSize(hidden);
            GpuTransferHelper.CacheActivation(hidden, pHidden, hiddenBytes);
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pBias);
        }
    }

    #endregion

    #region Shape Operations

    /// <summary>Concatenates tensors along the specified dimension.</summary>
    public unsafe void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Concat");
        EnterOp();
        ulong[] gpuInputs = new ulong[inputs.Length];
        ulong pOut = 0;
        bool cachedOutput = false;
        try
        {
            for (int t = 0; t < inputs.Length; t++)
            {
                gpuInputs[t] = GpuTransferHelper.CopyToDevice(inputs[t]);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            int elemSize = output.DType.SizeInBytes;

            if (dim == 0)
            {
                ulong offset = 0;
                for (int t = 0; t < inputs.Length; t++)
                {
                    nuint byteSize = (nuint)(inputs[t].ElementCount * elemSize);
                    // Stream-ordered (async) DtoD: the sync cuMemcpyDtoD serialized the host against the null
                    // stream per slice AND invalidated CUDA-graph capture (the Krea2 step-graph 901).
                    CudaMemory.CopyDeviceToDeviceAsync(pOut + offset, gpuInputs[t], byteSize, _stream.Handle);
                    offset += (ulong)byteSize;
                }
            }
            else
            {
                long outerSize = 1;
                for (int d = 0; d < dim; d++)
                {
                    outerSize *= output.Shape[d];
                }

                long innerSize = 1;
                for (int d = dim + 1; d < output.Shape.Rank; d++)
                {
                    innerSize *= output.Shape[d];
                }

                // Fast path: a 2-input F32/F16 concat runs as ONE kernel (one thread per output element) instead of
                // `outer` × 2 async memcpys. The per-slice loop below was the dominant DiT cost — the Hunyuan3D
                // single-block cat(attn, mlp) (dim=last, outer=seqLen≈4442) issued ~8900 memcpys/concat → 8.4 ms/call
                // and ~280k graph nodes/forward. Covers every 2-input block concat; other cases keep the loop.
                bool f16 = output.DType == DType.F16;
                if (inputs.Length == 2 && (f16 || output.DType == DType.F32)
                    && inputs[0].DType == output.DType && inputs[1].DType == output.DType)
                {
                    EnsureKernels();
                    _kernels!.LaunchConcat2(f16, pOut, gpuInputs[0], gpuInputs[1],
                        (int)outerSize, (int)inputs[0].Shape[dim], (int)inputs[1].Shape[dim], (int)innerSize, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOut, outBytes);
                    cachedOutput = true;
                    return;
                }

                long outDimStride = output.Shape[dim] * innerSize;

                for (long outer = 0; outer < outerSize; outer++)
                {
                    long dimOffset = 0;
                    for (int t = 0; t < inputs.Length; t++)
                    {
                        long inputDimSize = inputs[t].Shape[dim];
                        long sliceSize = inputDimSize * innerSize;
                        nuint sliceBytes = (nuint)(sliceSize * elemSize);

                        long inDimStride = inputDimSize * innerSize;
                        ulong srcOffset = (ulong)((outer * inDimStride) * elemSize);
                        ulong dstOffset = (ulong)((outer * outDimStride + dimOffset) * elemSize);

                        CudaMemory.CopyDeviceToDeviceAsync(pOut + dstOffset, gpuInputs[t] + srcOffset, sliceBytes, _stream.Handle);
                        dimOffset += sliceSize;
                    }
                }
            }

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            for (int t = 0; t < gpuInputs.Length; t++)
            {
                GpuTransferHelper.FreeDevice(gpuInputs[t]);
            }
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>Splits <paramref name="input"/> into device-resident <paramref name="outputs"/> along <paramref name="dim"/>, preserving every F32/F16/BF16 payload bit.</summary>
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        SplitGeometry geometry = SplitContract.Validate(outputs, input, dim);
        using NvtxRange _nvtx = NvtxRange.Push("Split");
        EnterOp();

        // Large contiguous outer slices are fastest on the copy engine, but many tiny copies serialize launch
        // submission (last-dimension split can have millions of outer rows). Cap that path at 16 commands and
        // require at least 64 KiB per command; the general kernel then handles fragmented geometry in one launch
        // per output with coalesced reads/writes and a grid-stride loop.
        bool useDeviceCopies = dim == 0;
        if (!useDeviceCopies && geometry.Outer <= 16 / outputs.Length)
        {
            useDeviceCopies = true;
            for (int t = 0; t < outputs.Length; t++)
            {
                long sliceBytes = checked(outputs[t].Shape[dim] * geometry.Inner * geometry.ElementSize);
                if (sliceBytes < 64 * 1024)
                {
                    useDeviceCopies = false;
                    break;
                }
            }
        }
        if (!useDeviceCopies)
            EnsureKernels();

        ulong pInput = 0;
        ulong[] pOutputs = new ulong[outputs.Length];
        nuint[] outputBytes = new nuint[outputs.Length];
        bool[] cachedOutputs = new bool[outputs.Length];
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            for (int t = 0; t < outputs.Length; t++)
            {
                outputBytes[t] = checked((nuint)(outputs[t].ElementCount * geometry.ElementSize));
                pOutputs[t] = GpuTransferHelper.AllocateDevice(outputBytes[t]);
            }

            long splitOffset = 0;
            for (int t = 0; t < outputs.Length; t++)
            {
                long outputDimension = outputs[t].Shape[dim];
                if (useDeviceCopies)
                {
                    long inputStride = checked(geometry.InputDimension * geometry.Inner);
                    long outputStride = checked(outputDimension * geometry.Inner);
                    nuint sliceBytes = checked((nuint)(outputStride * geometry.ElementSize));
                    for (long outer = 0; outer < geometry.Outer; outer++)
                    {
                        ulong sourceOffsetBytes = checked((ulong)(
                            checked(outer * inputStride + checked(splitOffset * geometry.Inner))
                            * geometry.ElementSize));
                        ulong outputOffsetBytes = checked((ulong)(
                            checked(outer * outputStride) * geometry.ElementSize));
                        CudaMemory.CopyDeviceToDeviceAsync(
                            pOutputs[t] + outputOffsetBytes,
                            pInput + sourceOffsetBytes,
                            sliceBytes,
                            _stream.Handle);
                    }
                }
                else
                {
                    _kernels!.LaunchSplitSlice(
                        geometry.ElementSize == sizeof(ushort),
                        pOutputs[t],
                        pInput,
                        geometry.Outer,
                        geometry.InputDimension,
                        outputDimension,
                        geometry.Inner,
                        splitOffset,
                        _stream.Handle);
                }
                splitOffset = checked(splitOffset + outputDimension);
            }

            // Publish only after every copy/kernel launch succeeds. If a later binding fails, unpublish every
            // earlier result before propagating the exception so Split never exposes a partial new result set.
            try
            {
                for (int t = 0; t < outputs.Length; t++)
                {
                    GpuTransferHelper.CacheActivation(outputs[t], pOutputs[t], outputBytes[t]);
                    cachedOutputs[t] = true;
                }
            }
            catch
            {
                for (int t = 0; t < outputs.Length; t++)
                {
                    if (GpuTransferHelper.TryUncacheActivation(outputs[t], pOutputs[t]))
                        cachedOutputs[t] = false;
                }
                throw;
            }
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pInput);
            for (int t = 0; t < pOutputs.Length; t++)
            {
                if (!cachedOutputs[t])
                    GpuTransferHelper.FreeDevice(pOutputs[t]);
            }
        }
    }

    #endregion

    #region Sampling

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("UpsampleNearest2D");
        EnterOp();
        EnsureKernels();

        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        int outH = inH * scaleH;
        int outW = inW * scaleW;

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchUpsampleNearest2DF16(
                    pOut, pIn,
                    batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                    _stream.Handle);
            else if (input.DType == DType.BF16)
                _kernels!.LaunchUpsampleNearest2DBf16(
                    pOut, pIn,
                    batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                    _stream.Handle);
            else
                _kernels!.LaunchUpsampleNearest2D(
                    pOut, pIn,
                    batch, channels, inH, inW, outH, outW, scaleH, scaleW,
                    _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("UpsampleBilinear2D");
        throw new NotImplementedException("CUDA UpsampleBilinear2D not yet implemented");
    }

    #endregion

    #region Data Movement

    /// <summary>Copies tensor data between host and device or device to device.</summary>
    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        EnterOp();
        nuint byteSize = (nuint)(source.ElementCount * source.DType.SizeInBytes);

        bool srcGpu = source.Device.IsCuda;
        bool dstGpu = destination.Device.IsCuda;

        if (srcGpu && dstGpu)
        {
            CudaMemory.CopyDeviceToDevice(
                (ulong)(nint)destination.DataPointer,
                (ulong)(nint)source.DataPointer,
                byteSize);
        }
        else if (!srcGpu && dstGpu)
        {
            CudaMemory.CopyHostToDevice(
                (ulong)(nint)destination.DataPointer,
                source.DataPointer,
                byteSize);
        }
        else if (srcGpu && !dstGpu)
        {
            CudaMemory.CopyDeviceToHost(
                destination.DataPointer,
                (ulong)(nint)source.DataPointer,
                byteSize);
        }
        else
        {
            // Both CPU - direct memory copy
            Buffer.MemoryCopy(source.DataPointer, destination.DataPointer, (long)byteSize, (long)byteSize);
        }
    }

    /// <summary>This backend's cached device pointer for <paramref name="tensor"/> (weight or activation shadow), or 0 when none. Read-only peek for the peer-copy boundary; the owning backend must be quiescent (the per-device gate / stage sequencing guarantees the producing stage finished issuing work).</summary>
    internal ulong TryGetDevicePointer(Tensor tensor)
    {
        if (_transferState.WeightCache.TryGetValue(tensor, out ulong weightPtr))
        {
            return weightPtr;
        }
        return _transferState.ActivationCache.TryGetValue(tensor, out (ulong gpuPtr, nuint bytes) cached) ? cached.gpuPtr : 0;
    }

    /// <summary>Count of copies that took the direct peer path.</summary>
    private long _peerCopies;

    /// <inheritdoc/>
    public long GetPeerCopyCount() => Interlocked.Read(ref _peerCopies);

    /// <summary>Cross-backend boundary copy. When the source lives on another CUDA device and P2P is available, the device copy moves directly (event-ordered against the source's stream, no host round-trip and no eviction of the source backend's resident copy). Otherwise the source's device data is staged into the DESTINATION's host buffer — deliberately not the default interface path, which would fire the source tensor's demote hooks and evict it from the source backend.</summary>
    public unsafe void CopyFromPeer(Tensor destination, Tensor source, IBackend sourceBackend)
    {
        // Raw memcpy both ways below — a dtype mismatch would bit-reinterpret silently.
        if (destination.DType != source.DType)
        {
            throw new ArgumentException(
                $"CopyFromPeer dtype mismatch: source {source.DType} vs destination {destination.DType}.");
        }
        // ByteSize, not ElementCount*SizeInBytes: block-quantized dtypes have SizeInBytes==0.
        nuint byteSize = GpuTransferHelper.ByteSize(source);
        if (GpuTransferHelper.ByteSize(destination) != byteSize)
        {
            throw new ArgumentException(
                $"CopyFromPeer size mismatch: source {byteSize} bytes vs destination " +
                $"{GpuTransferHelper.ByteSize(destination)} bytes.");
        }
        if (sourceBackend is not CudaBackend srcCuda || ReferenceEquals(srcCuda, this))
        {
            _ = source.DataPointer;
            CopyTo(destination, source);
            return;
        }

        ulong srcPtr = srcCuda.TryGetDevicePointer(source);
        if (srcPtr == 0)
        {
            // No device shadow — the data already lives (only) in the source's host buffer.
            _ = source.DataPointer;
            CopyTo(destination, source);
            return;
        }

        if (CudaPeerAccess.TryEnable(_context, srcCuda._context))
        {
            // The ordering event must be CREATED and RECORDED under the SOURCE context (event/stream must share a
            // context); the cross-context part CUDA explicitly supports is waiting on it from OUR stream.
            srcCuda.EnterOp();
            CudaDriverApi.cuEventCreate(out nint evt, CudaDriverApi.CU_EVENT_DISABLE_TIMING).ThrowOnError();
            try
            {
                CudaDriverApi.cuEventRecord(evt, srcCuda._stream.Handle).ThrowOnError();
                EnterOp();
                ulong dstPtr = GpuTransferHelper.AllocateDevice(byteSize);
                try
                {
                    CudaDriverApi.cuStreamWaitEvent(_stream.Handle, evt, CudaDriverApi.CU_EVENT_WAIT_DEFAULT).ThrowOnError();
                    CudaDriverApi.cuMemcpyPeerAsync(dstPtr, _context.Handle, srcPtr, srcCuda._context.Handle, byteSize, _stream.Handle)
                        .ThrowOnError();
                }
                catch
                {
                    GpuTransferHelper.FreeDevice(dstPtr);
                    throw;
                }
                // Register as OUR activation shadow so downstream ops consume it device-resident and its lifecycle
                // (dispose/lazy-sync) is managed like any other op output.
                GpuTransferHelper.CacheActivation(destination, dstPtr, byteSize);
                Interlocked.Increment(ref _peerCopies);
            }
            finally
            {
                // Destroy under the event's own context; the driver defers teardown past the pending stream wait.
                srcCuda._context.EnsureCurrent();
                CudaDriverApi.cuEventDestroy(evt);
                _context.EnsureCurrent();
            }
            return;
        }

        // No P2P: drain the producing stream, then stage the source's DEVICE data into the DESTINATION's host
        // buffer. The source backend's resident copy is untouched (unlike source.DataPointer, which would demote
        // it), and the destination uploads lazily on its first use here.
        srcCuda.EnterOp();
        CudaDriverApi.cuStreamSynchronize(srcCuda._stream.Handle).ThrowOnError();
        CudaMemory.CopyDeviceToHost(destination.EnsureHostBuffer(), srcPtr, byteSize);
        EnterOp();
    }

    /// <summary>Fills a tensor with a constant float value. Works on CPU tensors directly.</summary>
    public unsafe void Fill(Tensor tensor, float value)
    {
        using NvtxRange _nvtxProf = NvtxRange.Push("Fill");
        // CPU-side fill — DataPointer access syncs the GPU copy out (if cached) and
        // disposes its dptr, so the next op will re-upload from the just-written CPU
        // buffer. Dtype-aware so VAE F16 codepaths can use this for shift/scale broadcasts.
        if (tensor.DType == DType.F16)
        {
            Half* ptr = (Half*)tensor.DataPointer;
            Half h = (Half)value;
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = h;
        }
        else if (tensor.DType == DType.BF16)
        {
            // BF16 = upper 16 bits of F32. Truncate via right-shift (RTNE not needed
            // for typical fill values; if `value` lands exactly between two BF16 grid
            // points the trunc bias is acceptable for init scalars).
            ushort* ptr = (ushort*)tensor.DataPointer;
            uint bits = *(uint*)&value;
            ushort bf = (ushort)(bits >> 16);
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = bf;
        }
        else if (tensor.DType == DType.F32)
        {
            float* ptr = (float*)tensor.DataPointer;
            for (long i = 0; i < tensor.ElementCount; i++) ptr[i] = value;
        }
        else
        {
            throw new NotSupportedException($"Fill not supported for dtype {tensor.DType}");
        }
    }

    #endregion

    #region Audio

    public void Fft(Tensor output, Tensor input)
    {
        throw new NotSupportedException("CUDA FFT not supported - use CPU backend for audio");
    }

    public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window)
    {
        throw new NotSupportedException("CUDA STFT not supported - use CPU backend for audio");
    }

    public void MelFilterbank(Tensor output, Tensor input, Tensor filters)
    {
        throw new NotSupportedException("CUDA MelFilterbank not supported - use CPU backend for audio");
    }

    #endregion

    #region GPU Cache Management

    /// <summary>Preloads weight tensors to GPU memory. Subsequent ops using these tensors skip H2D transfer.</summary>
    /// <remarks>Rolls the batch back on failure. A mid-loop OOM otherwise leaves the successfully-uploaded prefix
    /// registered against a model that will never finish constructing — its <see cref="Tensor"/> keys become
    /// unreachable, so nothing can ever <see cref="FreeWeights"/> them and the VRAM is held until the process
    /// exits, starving every other consumer of the card (including separate processes).</remarks>
    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        EnterOp();
        List<Tensor>? uploaded = null;
        try
        {
            foreach (Tensor weight in weights)
            {
                // Only weights this call actually uploaded are rollback candidates — one already resident from
                // an earlier phase (or from HARTSY_KEEP_MODELS) is not ours to free. PreloadWeight reports this
                // itself so ownership is decided by the same lookup that does the registration.
                if (GpuTransferHelper.PreloadWeight(weight))
                {
                    uploaded ??= new List<Tensor>();
                    uploaded.Add(weight);
                }
            }
        }
        catch (Exception ex)
        {
            HartsyInference.Core.Logging.Logs.Error(
                $"[Cuda] PreloadWeights failed after {uploaded?.Count ?? 0} weight(s) — rolling back this batch.", ex);
            try
            {
                if (uploaded is not null)
                {
                    GpuTransferHelper.FreeWeights(uploaded);
                }
                GpuTransferHelper.SyncStreamsAndReleasePool();
            }
            catch (Exception cleanupEx)
            {
                HartsyInference.Core.Logging.Logs.Error(
                    "[Cuda] PreloadWeights rollback failed — device memory may still be held.", cleanupEx);
            }
            throw;
        }
    }

    /// <summary>Frees specific weight tensors from GPU to reclaim VRAM (e.g., UNet weights before VAE decode).</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        EnterOp();
        List<Tensor> materialized = weights as List<Tensor> ?? [.. weights];
        GpuTransferHelper.FreeWeights(materialized);
        foreach (Tensor weight in materialized) FreeFp8InputScale(weight);
    }

    public void FreeActivations()
    {
        EnterOp();
        // A captured step graph bakes activation-pool device pointers (fixed latent / velocity buffers) —
        // freeing activations under it leaves the graph pointing at freed memory, and the next replay is a
        // context-poisoning CUDA 700. Reset the graph slot here so cross-generation graphs (Chroma) survive
        // ONLY as long as their buffers do; owners detect the external reset and re-warm.
        StepGraphInvalidateForActivationFree();
        GpuTransferHelper.FreeActivations();
    }

    public void FreeActivations(bool trimPool)
    {
        EnterOp();
        StepGraphInvalidateForActivationFree();
        GpuTransferHelper.FreeActivations(trimPool);
    }

    private void StepGraphInvalidateForActivationFree()
    {
        if (_stepGraph is not null)
        {
            StepGraphReset();
            StepGraphOwner = null;
        }
    }

    public void TrimMemoryPool()
    {
        EnterOp();
        GpuTransferHelper.TrimPool();
    }

    /// <inheritdoc/>
    public void ReleaseAttentionExecutionCache()
    {
        EnterOp();
        ReleaseAttentionExecutionCacheCore();
    }

    /// <summary>Discards cuDNN SDPA plans/workspaces without touching weights, activations, convolution plans, retry state, or cumulative engagement diagnostics. The lock covers both the stream drain and detach so a first-call plan build/enqueue cannot race phase-boundary teardown.</summary>
    private void ReleaseAttentionExecutionCacheCore()
    {
        lock (_cudnnSdpaLock)
        {
            // The interface promises a phase-boundary drain even when this backend used a custom attention path
            // and therefore never created a cuDNN session. Holding the same lock as TryCudnnSdpa keeps the drain
            // and possible session detach atomic with respect to cuDNN plan construction/enqueue.
            _stream.Synchronize();
            CudnnSdpa? session = _cudnnSdpa;
            if (session is null)
            {
                return;
            }
            // Detach first. If native teardown reports a failure, no later attention call can observe and execute
            // through the half-disposed session; it will construct a fresh one after this lock is released.
            _cudnnSdpa = null;
            session.Dispose();
            Interlocked.Increment(ref _cudnnSdpaDisposedSessionCount);
        }
    }

    public void FreeAllDeviceMemory()
    {
        EnterOp();
        long freeBefore = -1;
        long freeAfter = -1;
        Exception? firstError = null;
        Try(() => freeBefore = (long)_context.GetMemoryInfo().freeBytes);
        Try(StepGraphInvalidateForActivationFree);
        // Drop the cuDNN sessions too: their execution-plan + workspace caches held ~4.5 GB after a
        // Z-Image session (measured 2026-07-23 — enough to trip Ideogram's ≥20 GB guard after a model
        // switch). Both instances lazily recreate on next use; the only cost is one plan re-search.
        Try(ReleaseAttentionExecutionCacheCore);
        CudnnConv? conv = _cudnnConv;
        _cudnnConv = null;
        if (conv is not null)
            Try(conv.Dispose);
        // EvictAll clears weights + casts + activations (syncing the stream first); the trim then returns the
        // stream-ordered pool's reservations so cuMemGetInfo/persistent allocs see the memory as actually free.
        Try(() => FreeW8A8Cache());
        Try(GpuTransferHelper.EvictAll);
        Try(FreeAllFp8InputScales);
        Try(GpuTransferHelper.TrimPool);
        Try(() => freeAfter = (long)_context.GetMemoryInfo().freeBytes);
        if (freeBefore >= 0 && freeAfter >= 0)
        {
            HartsyInference.Core.Logging.Logs.Info(
                $"[Cuda] FreeAllDeviceMemory: free {freeBefore >> 20} MB → {freeAfter >> 20} MB");
        }
        if (firstError is not null)
            throw new InvalidOperationException(
                "One or more CUDA resources failed to release during the full device-memory sweep.", firstError);

        void Try(Action cleanup)
        {
            try { cleanup(); }
            catch (Exception error) { firstError ??= error; }
        }
    }

    public long FreeMemoryBytes()
    {
        EnterOp();
        return (long)_context.GetMemoryInfo().freeBytes;
    }

    /// <summary>Frees all preloaded weight memory from GPU and clears the cache.</summary>
    public void FreePreloadedWeights()
    {
        EnterOp();
        FreeAllWeightCachesCore();
    }

    /// <summary>Evicts all cached GPU weight buffers. Call between pipeline stages to free VRAM.</summary>
    public void EvictGpuCache()
    {
        EnterOp();
        FreeAllWeightCachesCore();
    }

    private void FreeAllWeightCachesCore()
    {
        List<Exception>? failures = null;
        void Attempt(Action cleanup)
        {
            try { cleanup(); }
            catch (Exception error) { (failures ??= []).Add(error); }
        }
        // W8A8 frees are stream ordered; the transfer-cache sweep drains that stream before static FP8 scale
        // pointers are synchronously released. Every independent owner is attempted even after an earlier error.
        Attempt(() => FreeW8A8Cache());
        Attempt(GpuTransferHelper.EvictAll);
        Attempt(FreeAllFp8InputScales);
        if (failures is not null) throw new AggregateException("One or more CUDA weight caches failed to release.", failures);
    }

    /// <summary>Returns GPU cache stats: (cachedBytes, hits, misses).</summary>
    public (long cachedBytes, long hits, long misses) GetGpuCacheStats()
    {
        EnterOp();
        return GpuTransferHelper.GetStats();
    }

    /// <summary>Number of lazy D2H syncs since <see cref="ResetD2hSyncCount"/> — each is a full GPU stall plus a copy.</summary>
    public long GetD2hSyncCount()
    {
        EnterOp();
        return GpuTransferHelper.GetSyncCount();
    }

    /// <summary>Resets the device-to-host sync counter (call before a region you want to measure for residency).</summary>
    public void ResetD2hSyncCount()
    {
        EnterOp();
        GpuTransferHelper.ResetSyncCount();
    }

    /// <summary>Foundation check for graph-based decode: capture a Scale kernel, replay, change input, replay again.</summary>
    /// <remarks>A working capture/replay returns (input0·3, input1·3) = (6, 15) — proving replay reads live buffer content
    /// (stable pointers) and the async-pool memory model is capture-compatible.</remarks>
    public unsafe (float first, float second) GraphSmokeTest()
    {
        EnterOp();
        EnsureKernels();
        const int n = 256;
        ulong dIn = CudaMemory.AllocatePersistent((nuint)(n * sizeof(float)));
        ulong dOut = CudaMemory.AllocatePersistent((nuint)(n * sizeof(float)));
        try
        {
            CudaMemory.Fill32(dIn, BitConverter.SingleToUInt32Bits(2.0f), n);
            _stream.Synchronize();
            using CudaGraph graph = new(_stream.Handle);
            graph.Capture(() => _kernels!.LaunchScale(dOut, dIn, 3.0f, n, _stream.Handle));
            graph.Launch();
            _stream.Synchronize();
            float first;
            CudaMemory.CopyDeviceToHost(&first, dOut, sizeof(float));

            CudaMemory.Fill32(dIn, BitConverter.SingleToUInt32Bits(5.0f), n);
            _stream.Synchronize();
            graph.Launch();
            _stream.Synchronize();
            float second;
            CudaMemory.CopyDeviceToHost(&second, dOut, sizeof(float));
            return (first, second);
        }
        finally
        {
            CudaMemory.Free(dIn);
            CudaMemory.Free(dOut);
        }
    }

    /// <summary>Device memory (free, total) in bytes via cuMemGetInfo.</summary>
    public (long FreeBytes, long TotalBytes) GetVramInfo()
    {
        EnterOp();
        return CudaMemory.GetMemInfo();
    }

    /// <inheritdoc/>
    public void ResetOpProfile() => NvtxRange.ResetProfile();

    /// <inheritdoc/>
    public void DumpOpProfile(string label)
    {
        string basePath = EngineKnobs.ProfileOut.Value ?? "/tmp/hartsy_profile.txt";
        NvtxRange.DumpProfile($"{basePath}.{label}");
    }

    #endregion

    #region Disposal

    /// <summary>Releases every resource owned by this backend exactly once.</summary>
    /// <remarks>Callers must quiesce inference work (the server's DeviceGate/request drain) before disposal. Concurrent
    /// Dispose callers are supported; Dispose racing an in-flight tensor operation is intentionally not.</remarks>
    public void Dispose()
    {
        Exception? failure = null;
        int observed = Interlocked.CompareExchange(ref _lifecycleState, LifecycleClaimed, LifecycleActive);
        if (observed == LifecycleActive)
        {
            failure = RunClaimedCleanup(abandoned: false);
        }
        else if (observed == LifecycleClaimed)
        {
            // Concurrent Dispose callers observe deterministic completion without racing a partially torn-down
            // native object graph. The winner alone executes cleanup.
            _cleanupCompleted.Wait();
            failure = Volatile.Read(ref _cleanupFailure);
        }
        GC.SuppressFinalize(this);
        if (failure is not null) throw failure;
    }

    private Exception? RunClaimedCleanup(bool abandoned)
    {
        Interlocked.Increment(ref _cleanupExecutionCount);
        List<Exception>? failures = null;
        GpuTransferHelper.State? state = _transferState;

        void Attempt(string resource, Action cleanup)
        {
            try { cleanup(); }
            catch (Exception error)
            {
                (failures ??= []).Add(new InvalidOperationException($"CUDA cleanup failed for {resource}.", error));
            }
        }

        try
        {
            // This is deliberately the first cleanup transition. The State remains strongly available through the
            // local explicit handle, but Resolve/_sole/_byContext/StatesOnDevice can no longer select it while native
            // streams, caches, and handles are being dismantled. It is inside the terminal try/finally so even an
            // unexpected managed retirement failure cannot strand LifecycleClaimed or deadlock another Dispose.
            if (state is not null) Attempt("transfer-state route retirement", () => GpuTransferHelper.BeginRetire(state));

            if (!abandoned && NvtxRange.ProfileEnabled)
                Attempt("NVTX profile dump", () => NvtxRange.DumpProfile(
                    EngineKnobs.ProfileOut.Value ?? "/tmp/hartsy_profile.txt"));

            if (_context is not null) Attempt("context binding", _context.EnsureCurrent);
            CudaGraph? graph = _stepGraph;
            _stepGraph = null;
            bool graphCapturing = _stepGraphCapturing;
            _stepGraphCapturing = false;
            if (graph is not null && graphCapturing) Attempt("open graph capture abort", graph.AbortCapture);
            if (state is not null && graphCapturing)
            {
                state.TrackCaptureWindow = false;
                nint captureStream = state.StreamHandle;
                if (captureStream != 0)
                    Attempt("aborted graph-private cache purge", () => GpuTransferHelper.PurgeAbortedCaptureAllocs(state, captureStream));
            }
            if (graph is not null) Attempt("step graph", graph.Dispose);

            // Synchronizing an actively capturing stream is illegal, so capture is aborted/purged above first.
            if (_stream is not null) Attempt("compute-stream drain", _stream.Synchronize);
            if (_uploadStream is not null) Attempt("upload-stream drain", _uploadStream.Synchronize);

            ulong dp4aScratch = _dp4aScratch;
            _dp4aScratch = 0;
            _dp4aScratchBytes = 0;
            ulong argmaxScratch = _argmaxScratch;
            _argmaxScratch = 0;
            ulong ssmScratch = _ssmDeltaScratch;
            _ssmDeltaScratch = 0;
            _ssmDeltaScratchBytes = 0;
            if (state is not null)
            {
                if (dp4aScratch != 0) Attempt("dp4a scratch", () => GpuTransferHelper.FreeDevice(state, dp4aScratch));
                if (argmaxScratch != 0) Attempt("argmax scratch", () => GpuTransferHelper.FreeDevice(state, argmaxScratch));
                if (ssmScratch != 0) Attempt("SSM delta scratch", () => GpuTransferHelper.FreeDevice(state, ssmScratch));
                Attempt("W8A8 caches", () => FreeW8A8Cache(state));
            }
            Attempt("FP8 input-scale cache", FreeAllFp8InputScales);
            if (state is not null) Attempt("transfer caches", () => GpuTransferHelper.EvictAll(state));
            if (_streamingCache is not null) Attempt("streaming pinned staging", _streamingCache.UnregisterPinnedSources);

            // The explicit-state frees above can enqueue stream-ordered releases. Drain again before destroying
            // modules/executors/streams, independently attempting both streams even if one reports an error.
            if (_stream is not null) Attempt("post-free compute-stream drain", _stream.Synchronize);
            if (_uploadStream is not null) Attempt("post-free upload-stream drain", _uploadStream.Synchronize);

            if (_kernels is not null)
                Attempt("kernel modules", () => { lock (_cudnnSdpaLock) _kernels.Dispose(); });

            CudnnSdpa? sdpa = null;
            Attempt("cuDNN SDPA detach", () =>
            {
                lock (_cudnnSdpaLock)
                {
                    sdpa = _cudnnSdpa;
                    _cudnnSdpa = null;
                }
            });
            if (sdpa is not null)
            {
                Attempt("cuDNN SDPA session", () =>
                {
                    sdpa.Dispose();
                    Interlocked.Increment(ref _cudnnSdpaDisposedSessionCount);
                });
            }

            Fp8GemmExecutor? fp8;
            Int8GemmExecutor? int8;
            LtGemmExecutor? lt;
            TensorCoreGemm? tensorCore;
            lock (_nativeExecutorLock)
            {
                fp8 = _fp8Executor;
                _fp8Executor = null;
                int8 = _int8Executor;
                _int8Executor = null;
                lt = _ltGemmExecutor;
                _ltGemmExecutor = null;
                tensorCore = _tensorCoreGemm;
                _tensorCoreGemm = null;
            }
            if (fp8 is not null) Attempt("FP8 GEMM executor", fp8.Dispose);
            if (int8 is not null) Attempt("INT8 GEMM executor", int8.Dispose);
            if (lt is not null) Attempt("cuBLASLt GEMM executor", lt.Dispose);
            if (tensorCore is not null) Attempt("tensor-core GEMM executor", tensorCore.Dispose);
            CudnnConv? conv = Interlocked.Exchange(ref _cudnnConv, null);
            if (conv is not null) Attempt("cuDNN convolution", conv.Dispose);

            nint cublas = Interlocked.Exchange(ref _cublasHandle, 0);
            if (cublas != 0) Attempt("cuBLAS handle", () => CublasApi.cublasDestroy(cublas).ThrowOnCublasError());

            if (state is not null)
                Attempt("tensor finalizer-cleanup bucket", () => Tensor.RetireFinalizerGpuCleanup(state.Key));

            if (_context is not null && Interlocked.Exchange(ref _mempoolPolicyAcquired, 0) != 0)
                Attempt("device mempool policy", () => DeviceMempoolPolicy.Release(_context.DeviceOrdinal));

            // Destroy side stream first, then compute stream, then release the primary-context retain.
            if (_uploadStream is not null) Attempt("upload stream", _uploadStream.Dispose);
            if (_stream is not null) Attempt("compute stream", _stream.Dispose);
            if (_context is not null) Attempt("CUDA context", _context.Dispose);
        }
        finally
        {
            // CompleteRetire is managed-only and must happen even if a future cleanup edit escapes Attempt.
            if (state is not null)
            {
                try { GpuTransferHelper.CompleteRetire(state); }
                catch (Exception error)
                {
                    (failures ??= []).Add(new InvalidOperationException("CUDA cleanup failed to retire transfer state.", error));
                }
            }
            _cleanupFailure = failures is null ? null
                : new AggregateException("One or more CUDA backend resources failed to release.", failures);
            Volatile.Write(ref _lifecycleState, LifecycleCleaned);
            _cleanupCompleted.Set();
        }

        return _cleanupFailure;
    }

    ~CudaBackend()
    {
        if (Interlocked.CompareExchange(ref _lifecycleState, LifecycleClaimed, LifecycleActive) != LifecycleActive)
            return;
        try
        {
            BackendAbandonmentReaper.Enqueue(this);
        }
        catch
        {
            // A finalizer must never terminate the process. No native call is legal here; an enqueue OOM is the
            // sole case where the process cannot preserve the abandoned instance for managed-worker cleanup.
            Interlocked.Increment(ref _abandonedCleanupFailedCount);
        }
    }

    #endregion

    /// <summary>Everything the nvfp4 dequant kernel needs about one resident weight beyond its packed bytes.</summary>
    /// <param name="BlockScaleDevice">Device copy of the swizzled E4M3 scales; 0 means the weight is not nvfp4.</param>
    /// <param name="PaddedCols">Stored last-dim length of the scale tensor, which is the swizzle's stride.</param>
    private readonly record struct Nvfp4WeightScales(ulong BlockScaleDevice, float ScaleFactor, float GlobalScale, int PaddedCols);
}
