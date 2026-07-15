using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.Cuda;

/// <summary>CUDA GPU backend implementing IBackend. Routes operations to cuBLAS SGEMM for matmul and PTX kernels for element-wise/normalization ops. Uses activation caching to keep intermediate results on GPU between ops — lazy sync to CPU on DataPointer access.</summary>
public sealed class CudaBackend : IBackend
{
    private readonly CudaContext _context;
    private readonly CudaStream _stream;
    /// <summary>Side stream used by <see cref="_streamingCache"/> for asynchronous
    /// weight uploads that overlap with compute on <see cref="_stream"/>. Created
    /// non-blocking so it doesn't serialize with the compute stream — synchronization
    /// between the two is explicit via <c>cuEventRecord</c> / <c>cuStreamWaitEvent</c>
    /// inside the streaming cache.</summary>
    private readonly CudaStream _uploadStream;
    private readonly CudaStreamingWeightCache _streamingCache;
    private readonly CudaKernels? _kernels;
    private nint _cublasHandle;
    private Fp8GemmExecutor? _fp8Executor;
    private LtGemmExecutor? _ltGemmExecutor;
    private TensorCoreGemm? _tensorCoreGemm;
    private CudnnSdpa? _cudnnSdpa;
    private bool _cudnnSdpaDead;   // set if cuDNN INIT throws once — never retry, fall back for the session

    /// <summary>Per-head-dim failure/backoff state — replaces a plain permanent-dead set. A structural
    /// failure (e.g. D=256 on a build whose fused engine tops out at 128 — <see cref="CudnnStatusException.IsPermanent"/>)
    /// disables that dim forever, same as before; a transient failure (e.g. the host-RAM allocation error
    /// that motivated this — see the plan notes referenced from <see cref="TryCudnnSdpa"/>) gets bounded
    /// backoff instead, since the resource pressure that caused it is typically external to this process
    /// and does clear up. <see cref="ConcurrentDictionary{TKey,TValue}"/> (not a plain <c>HashSet</c>) because
    /// this backend is a process-wide DI singleton and nothing enforces <c>InferenceQueue.MaxConcurrency</c>
    /// stays at its default of 1 — a plain <c>HashSet</c> was never actually safe under a higher setting.</summary>
    private sealed class DimFailureState
    {
        public int ConsecutiveFailures;
        public long NextRetryAtTicks;   // Environment.TickCount64; 0 = eligible immediately
        public bool Permanent;
    }
    private readonly ConcurrentDictionary<long, DimFailureState> _cudnnSdpaDimState = new();

    /// <summary>Test-only fault injection for <see cref="TryCudnnSdpa"/>: when set, invoked once per attempt
    /// with the head dim; a non-null return is thrown instead of running the real cuDNN call. Exists so the
    /// classify/retry/backoff behavior can be tested deterministically — a real host-RAM-starvation failure
    /// can't be reproduced on demand in a fast unit test. Null in production.</summary>
    internal Func<long, CudnnStatusException?>? TestCudnnSdpaFaultInjector { get; set; }

    /// <summary>Per-dim diagnostic snapshot for <see cref="TryCudnnSdpa"/>'s classify/retry/backoff state —
    /// programmatically queryable observability surface (deliberately not wired into <c>/ready</c>: a
    /// degraded-but-correct fallback to the materialized attention path is not a "can't serve traffic"
    /// condition, matching this engine's existing health-check philosophy).</summary>
    public IReadOnlyDictionary<long, (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt)> CudnnSdpaDimDiagnostics =>
        _cudnnSdpaDimState.ToDictionary(
            kv => kv.Key,
            kv => (kv.Value.ConsecutiveFailures, kv.Value.Permanent,
                   kv.Value.Permanent ? (DateTimeOffset?)null : DateTimeOffset.UtcNow.AddMilliseconds(kv.Value.NextRetryAtTicks - Environment.TickCount64)));

    /// <summary>True if head dim <paramref name="d"/> is currently eligible for a cuDNN SDPA attempt — no
    /// failure on record (the overwhelmingly common case: a single dictionary miss, same O(1) cost as the
    /// plain <c>HashSet.Contains</c> this replaces), not permanently disabled, and any backoff window from a
    /// prior transient failure has elapsed.</summary>
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

    /// <summary>True once the cuDNN fused-attention fast path has successfully executed at least once this
    /// session. Diagnostic surface (tests / deploy logs) to confirm the path actually engaged vs fell back.</summary>
    public bool CudnnSdpaEngaged { get; private set; }

    /// <summary>True once the cuDNN convolution fast path has successfully executed at least once this
    /// session — same diagnostic role as <see cref="CudnnSdpaEngaged"/>.</summary>
    public bool CudnnConvEngaged { get; private set; }
    private readonly string? _ptxDir;
    private bool _disposed;

    /// <summary>The device this backend targets.</summary>
    public DeviceKind Device { get; }

    /// <summary>Capabilities of this CUDA backend.</summary>
    public BackendCapabilities Capabilities { get; }

    /// <summary>The CUDA context used by this backend.</summary>
    public CudaContext Context => _context;

    /// <summary>The default compute stream.</summary>
    public CudaStream Stream => _stream;

    /// <summary>The upload stream for asynchronous weight transfers. Exposed for
    /// diagnostics and tests; production callers should use <see cref="StreamingCache"/>.</summary>
    public CudaStream UploadStream => _uploadStream;

    /// <inheritdoc/>
    public IStreamingWeightCache? StreamingCache => _streamingCache;

    /// <summary>Opt-in: page-lock weight host sources before streaming uploads so async H2D truly overlaps with compute (see <see cref="CudaStreamingWeightCache.PinUploadSource"/>). Defaults to <c>false</c>. Beneficial for block-swap workloads that re-upload weights across steps.</summary>
    public bool EnablePinnedWeightUploads
    {
        get => _streamingCache.PinUploadSource;
        set => _streamingCache.PinUploadSource = value;
    }

    /// <summary>The cuBLAS handle for GEMM operations.</summary>
    public nint CublasHandle => _cublasHandle;

    /// <summary>Opt-in flag for the native cuBLASLt FP8 GEMM path on Ada+ (SM 8.9+) GPUs. Defaults to <c>false</c> — on Ampere and below the path is unsupported and the existing cast-to-F16 fallback is correct. The native path is gated on this flag because it has not been end-to-end validated on Ada hardware in CI; flip on after benchmarking against the F16 fallback.</summary>
    public bool EnableNativeFp8Gemm { get; set; }

    /// <summary>Low-VRAM lever. When <c>true</c> (default) the per-weight fp8/quant→F16 GEMM cast is computed
    /// once and kept resident for every preloaded/uploaded weight — fastest, but it keeps BOTH the fp8 weight
    /// (~1 byte/param) and its F16 cast (~2 bytes/param) on the device, ≈3× the fp8 footprint. Set <c>false</c>
    /// to force the transient path (one weight cast at a time, freed per GEMM): ~3× less weight VRAM at the cost
    /// of re-casting each step. Needed to fit large fp8 DiTs (e.g. AuraFlow 6.8B) on a single 24 GB card.</summary>
    public bool CacheWeightCasts { get; set; } = true;

    /// <summary>Lazily-initialized FP8 GEMM executor. Exposed for diagnostic and benchmarking callers; production GEMM dispatch goes through <see cref="MatMul"/> / <see cref="Linear"/>.</summary>
    public Fp8GemmExecutor Fp8Executor
    {
        get
        {
            _fp8Executor ??= new Fp8GemmExecutor(_context.ComputeCapabilityMajor, _context.ComputeCapabilityMinor);
            return _fp8Executor;
        }
    }

    /// <summary>Opt-in flag for fusing the Linear bias add into the cuBLASLt GEMM epilogue (works on every targeted SM, including the RTX 3060). Defaults to <c>false</c> until benchmarked on hardware; when on, a Linear with bias runs as a single <c>cublasLtMatmul</c> instead of <c>cublasGemmEx</c> + a separate <c>BiasAdd</c> launch. The result is numerically equivalent to the unfused path.</summary>
    public bool EnableEpilogueFusion { get; set; }

    /// <summary>Opt-in: run mixed bf16/f32 GEMMs at F32 compute precision (cast the bf16 weight UP to F32 rather
    /// than truncating the F32 activation DOWN to bf16). Default <c>false</c> preserves the bf16 Tensor-Core fast
    /// path. Turn ON for precision-sensitive models whose bf16 weights stay resident to fit VRAM but whose
    /// activations must keep full F32 mantissa across many layers/steps (e.g. ACE-Step's 3.5B DiT on a 12 GB 3060).</summary>
    public bool HighPrecisionGemm { get; set; }

    /// <summary>Route the fp8 GEMM activation cast through F16 (10-bit) instead of BF16 (7-bit). Safe for GELU-FFN models
    /// (Wan); needed for deep fp8 DiTs where BF16's coarser mantissa compounds a per-step bias into divergence.</summary>
    public bool EnableFp8F16Gemm { get; set; }

    /// <summary>Compute the fp8 GEMM in F32 (max precision; slow, high memory) — decisive test for whether F16/BF16 compute error is compounding.</summary>
    public bool EnableFp8F32Gemm { get; set; }

    /// <summary>Lazily-initialized general-precision cuBLASLt GEMM executor used by the epilogue-fusion path.</summary>
    public LtGemmExecutor LtGemm
    {
        get
        {
            _ltGemmExecutor ??= new LtGemmExecutor();
            return _ltGemmExecutor;
        }
    }

    /// <summary>Opt-in flag for the hand-written tensor-core HGEMM in the F16 Linear path. Defaults to <c>false</c>. The kernel is validated bit-exact against cuBLAS (<c>TensorCoreGemmTests</c>) but is the unoptimized one-warp-per-tile baseline, so it is opt-in pending a perf comparison against cuBLAS on the target GPU. Only dispatches when operands and output are F16 and dimensions are aligned (M%16, N%8, K%16 == 0); otherwise falls through to cuBLAS.</summary>
    public bool EnableTensorCoreGemm { get; set; }

    /// <summary>Use the fused dense BF16/F16 decode GEMV kernel for small-M (≤8) F32-activation matmuls instead of
    /// cuBLAS GemmEx (which is inefficient at M=1). On by default — it's faster and at least as accurate as the
    /// cuBLAS BF16 path (activations stay F32). Set <c>HARTSY_BF16_GEMV=0</c> to fall back to cuBLAS for A/B.</summary>
    public bool EnableBf16Gemv { get; set; } = Environment.GetEnvironmentVariable("HARTSY_BF16_GEMV") != "0";

    /// <summary>Lazily-initialized tensor-core HGEMM launcher. Requires PTX directory and SM 8.0+.</summary>
    public TensorCoreGemm TensorCoreGemm
    {
        get
        {
            _tensorCoreGemm ??= new TensorCoreGemm(
                _ptxDir ?? throw new InvalidOperationException("TensorCoreGemm requires a PTX directory; construct CudaBackend with ptxDir."),
                _context.ComputeCapabilityMajor);
            return _tensorCoreGemm;
        }
    }

    /// <summary>Reads a <c>HARTSY_*</c> boolean env var (set "1" to enable), mirroring <see cref="Profiling.NvtxRange.ProfileEnabled"/>. Unset or any non-"1" value is treated as off.</summary>
    private static bool EnvFlag(string name) => Environment.GetEnvironmentVariable(name) == "1";

    /// <summary>TF32 tensor-core math for F32-operand GEMMs on Ampere+ (SM ≥ 8.0) — the same default PyTorch
    /// uses for inference. Plain-F32 GEMMs have no tensor-core path on consumer Ampere (a 3060 runs them at a
    /// fraction of tensor-core throughput), leaving F32 pipelines compute-bound far below the hardware. TF32
    /// keeps F32 range with a 10-bit mantissa. Opt out with <c>HARTSY_NO_TF32=1</c>.</summary>
    private readonly bool _allowTf32;

    /// <summary>HARTSY_GEMM_F16=1: use F16-mantissa (COMPUTE_32F_FAST_16F) tensor-core math for F32 GEMMs — faster
    /// than TF32 on Ada, F32 accumulate kept. Opt-in per parity-checked model (Oasis interactive DiT).</summary>
    private readonly bool _gemmFast16;

    /// <summary>HARTSY_SDPA_F16=1: force the F16 SDPA path on for ALL callers (not just allowF16 ones) — testing/override.</summary>
    private readonly bool _sdpaF16ForceOn;
    /// <summary>HARTSY_SDPA_NO_F16=1: global kill-switch for the F16 SDPA path even when a caller passes allowF16.</summary>
    private readonly bool _sdpaF16Disabled;
    /// <summary>Routes MHA (D∈{64,128,256}, no mask or a [B,1,Sq,Skv]-broadcast additive F32 mask) through cuDNN's
    /// fused flash-attention engine instead of the materialized cuBLAS QKᵀ→softmax→PV path (~34× on the Krea2
    /// self-attention shape). Standard-profile default ON; HARTSY_SDPA_CUDNN=0 disables. Missing cuDNN or engine
    /// rejections fall back to the materialized paths automatically.</summary>
    private readonly bool _sdpaCudnn;

    /// <summary>Routes F16/BF16 NCHW convolutions through cuDNN conv-forward engines instead of the
    /// im2col→cuBLAS GEMM path. Standard-profile default ON; HARTSY_CONV_CUDNN=0 disables. Failures
    /// self-disable for the session and fall back to im2col.</summary>
    private readonly bool _convCudnn;

    /// <summary>Route the audio 1D convs (vocoders/codecs/VITS) through cuDNN (mapped to 2D). DEFAULT OFF: the
    /// path is written but not yet verified stable on a working cuDNN install (on the dev box cuDNN isn't on the
    /// default library path, and forcing it caused a hang) — opt in with HARTSY_AUDIO_CONV_CUDNN=1 to verify.</summary>
    private readonly bool _audioConvCudnn;

    /// <summary>Compute type for a GEMM whose operands resolved to <paramref name="gemmType"/>: FAST_TF32 for
    /// F32 operands when allowed (and <see cref="HighPrecisionGemm"/> — an explicit full-precision request —
    /// is off), otherwise plain 32F accumulate (mixed F16-in/F32-acc stays 32F).</summary>
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
        _context = new CudaContext(deviceOrdinal);
        // Must use blocking stream (CU_STREAM_DEFAULT) because GpuTransferHelper uses synchronous
        // cuMemcpyHtoD/DtoH which operate on the NULL stream. A non-blocking stream does NOT
        // synchronize with the NULL stream, causing race conditions where kernels read incomplete
        // data from in-progress H2D transfers. Fix: switch to cuMemcpyHtoDAsync on this stream.
        _stream = new CudaStream(nonBlocking: false);
        Profiling.NvtxRange.ProfileSyncStream = _stream.Handle;   // enables HARTSY_PROFILE_SYNC per-op GPU-time attribution
        // Upload stream is non-blocking so its in-flight work doesn't gate the compute
        // stream's NULL-stream "wait for everything" semantics — without that, prefetched
        // uploads would force compute to wait, defeating overlap. The streaming cache uses
        // explicit cuEventRecord/cuStreamWaitEvent for the parts that *do* need to sync.
        _uploadStream = new CudaStream(nonBlocking: true);
        _streamingCache = new CudaStreamingWeightCache(_context, _stream.Handle, _uploadStream.Handle);
        _ptxDir = ptxDir;
        Device = DeviceKind.Cuda(deviceOrdinal);

        // Keep freed activation buffers warm in the stream-ordered pool. The default device mempool's
        // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD is 0 (the streaming weight cache sets it so, to avoid an OOM during weight
        // streaming) — but the compute stream churns activations through this SAME pool, so a 0 threshold returns every
        // freed activation to the driver and re-acquires it on the next alloc, stalling the compute stream (the
        // dominant non-kernel GPU-stream cost on big-activation diffusion: Krea2 1024² was ~13 s of pure alloc/free
        // round-trips). Raise it to "hold everything" so per-op activation reuse is instant; the OOM path
        // (AllocateAsync → SyncStreamsAndReleasePool) and the streaming cache's explicit cuMemPoolTrimTo still return
        // memory on demand. Standard-profile default ON (HARTSY_MEMPOOL_KEEP=0 to disable): the whole fleet —
        // including the block-swap streaming video models — is benchmark-verified under it, and the OOM-retry +
        // explicit-trim paths cover the held-pool pressure case the old default-off guarded against.
        bool mempoolKeep = EnvSwitch.IsEnabled("HARTSY_MEMPOOL_KEEP", defaultOn: true);
        if (mempoolKeep)
        {
            try
            {
                CudaDriverApi.cuDeviceGetDefaultMemPool(out nint computePool, _context.DeviceOrdinal).ThrowOnError();
                unsafe
                {
                    ulong keepAll = ulong.MaxValue;
                    CudaDriverApi.cuMemPoolSetAttribute(computePool, CudaDriverApi.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &keepAll).ThrowOnError();
                }
            }
            catch (Exception ex)
            {
                HartsyInference.Core.Logging.Logs.Warning($"[Cuda] could not raise mempool release threshold: {ex.Message}");
            }
        }

        // Perf-flag wiring, two tiers (docs/PERFORMANCE.md). STANDARD PROFILE features default ON via
        // tri-state EnvSwitch (unset → documented default, "0"/"false" is the kill-switch) so every install
        // reproduces the published benchmark times with zero configuration. EXPERIMENTAL switches keep the
        // strict opt-in EnvFlag form ("1" only) for A/B benchmarking without recompiling.
        // cuBLASLt bias-epilogue GEMM: promoted to the standard profile 2026-07-09 — every biased Linear
        // otherwise pays a separate BiasAdd kernel + an output-sized HBM round-trip (~700/step on the SDXL
        // UNet, measured −0.16 s/gen). Falls back to GemmEx+BiasAdd when Lt is unavailable or shapes
        // don't qualify; HARTSY_EPILOGUE_FUSION=0 is the kill-switch.
        EnableEpilogueFusion = EnvSwitch.IsEnabled("HARTSY_EPILOGUE_FUSION", defaultOn: true);
        EnableTensorCoreGemm = EnvFlag("HARTSY_TENSORCORE_GEMM");
        // fp8 tensor-core GEMM (activation-quant e4m3) requires SM 8.9+ (Ada); older parts default to the
        // F16-cast path. Verified quality-clean fleet-wide in the standard Swarm config.
        bool fp8TensorCores = _context.ComputeCapabilityMajor > 8
            || (_context.ComputeCapabilityMajor == 8 && _context.ComputeCapabilityMinor >= 9);
        EnableNativeFp8Gemm = EnvSwitch.IsEnabled("HARTSY_FP8_NATIVE", defaultOn: fp8TensorCores);
        HighPrecisionGemm = EnvFlag("HARTSY_HIGH_PRECISION_GEMM");
        EnableFp8F16Gemm = EnvFlag("HARTSY_FP8_F16");
        EnableFp8F32Gemm = EnvFlag("HARTSY_FP8_F32");
        _allowTf32 = _context.ComputeCapabilityMajor >= 8 && !EnvFlag("HARTSY_NO_TF32");
        _gemmFast16 = _context.ComputeCapabilityMajor >= 8 && EnvFlag("HARTSY_GEMM_F16");
        // F16 SDPA is gated PER-CALL via the allowF16 arg (callers with bounded/RMS-normed scores like Wan pass true);
        // safe by default because unbounded-score archs (Z-Image fp8) don't pass it. Env: force-on all callers, or kill.
        _sdpaF16ForceOn = EnvFlag("HARTSY_SDPA_F16");
        _sdpaF16Disabled = EnvFlag("HARTSY_SDPA_NO_F16");
        // cuDNN fused flash attention: default ON — resolution failures self-disable per session (and engine
        // rejections per head-dim), falling back to the materialized paths, so machines without cuDNN lose
        // speed, never correctness.
        // Probe cuDNN ONCE up front: locate/provision a CUDA-major-matched build, guard against a mismatch (which
        // would hang mid-inference), and log a single clear line so a "why is it slow?" report is one log check.
        // Every cuDNN fast path is AND-gated on availability, so a missing/mismatched cuDNN cleanly stays on the
        // im2col+cuBLAS / custom-flash fallbacks instead of throwing per-op or hanging.
        CudnnRuntime.LogStatus();
        _sdpaCudnn = CudnnRuntime.Available && EnvSwitch.IsEnabled("HARTSY_SDPA_CUDNN", defaultOn: true);
        // cuDNN convolution forward: default ON — replaces im2col→GEMM for F16/BF16 NCHW convs (the SDXL
        // UNet/VAE cost). Same self-disable-on-failure contract as the fused SDPA path.
        _convCudnn = CudnnRuntime.Available && EnvSwitch.IsEnabled("HARTSY_CONV_CUDNN", defaultOn: true);
        // Audio conv1d cuDNN (1D→2D, H=1, F32 output via TF32 tensor cores): default ON. The Oobleck/EnCodec
        // vocoder decoders are conv-bound — routing their forward convs through cuDNN is ~1.6-2× on ACE-Step
        // end-to-end (bigger on longer audio, where the VAE decode dominates) with corr 0.9999 vs the direct
        // kernel (same-seed A/B, verified coherent across ACE-Step/MusicGen/AudioGen). Same session-sticky
        // self-disable-on-failure contract as the 2D path, so shapes cuDNN can't serve fall back safely.
        _audioConvCudnn = CudnnRuntime.Available && EnvSwitch.IsEnabled("HARTSY_AUDIO_CONV_CUDNN", defaultOn: true);
        // Each result dir self-documents the config it ran under: log the resolved flag set once.
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] perf flags: SdpaCudnn={_sdpaCudnn} ConvCudnn={_convCudnn} NativeFp8Gemm={EnableNativeFp8Gemm} MempoolKeep={mempoolKeep} " +
            $"EpilogueFusion={EnableEpilogueFusion} TensorCoreGemm={EnableTensorCoreGemm} " +
            $"HighPrecisionGemm={HighPrecisionGemm} CacheWeightCasts={CacheWeightCasts} " +
            $"AutoPromoteWeights={GpuTransferHelper.AutoPromoteWeights} Tf32Gemm={_allowTf32}.");

        // Initialize cuBLAS
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

        // Register this backend's transfer state (context + stream + streaming cache) keyed by the CUDA
        // context, so a second backend on another GPU gets its OWN caches/stream instead of silently
        // retargeting this one's (the multi-GPU CUDA 700 poison). Transient allocations route through the
        // stream-ordered pool on the same compute stream, so they reuse the memory their cuMemFreeAsync
        // frees return instead of stranding it in the pool — without this the GPU fills with pool-locked
        // bytes and every op OOM-retries with a full drain+trim (the Ideogram-4 ~100s/step thrash).
        // Persistent weights/workspaces stay on the sync allocator.
        GpuTransferHelper.Register(_context, _stream.Handle, _streamingCache);
        CudaMemory.SetComputeStream(_context, _stream.Handle);

        // Load PTX kernels if directory provided
        if (ptxDir != null && Directory.Exists(ptxDir))
        {
            _kernels = new CudaKernels(ptxDir);
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
    }

    /// <summary>Largest single im2col workspace Conv2D will allocate; larger convs run as output-row bands
    /// (bit-identical GEMMs, output pointer offset per band). Bounds the 512-ch 3×3 @1024² VAE conv (9.2 GB
    /// naive) so full-res decode fits next to resident model weights — 1 GB verified live for the Flux.2 VAE
    /// beside Ideogram 4's 18.6 GB resident DiTs (2 GB still OOM'd there); band size costs no GEMM efficiency
    /// (M stays ≥ tens of thousands of rows). Override via HARTSY_IM2COL_BAND_MB (also lets tests force
    /// banding on small shapes).</summary>
    private static readonly long Im2ColBandCapBytes =
        long.TryParse(Environment.GetEnvironmentVariable("HARTSY_IM2COL_BAND_MB"), out long capMb) && capMb > 0
            ? capMb << 20
            : 1L << 30;

    #region Linear Algebra

    /// <summary>Matrix multiply via cuBLAS GemmEx: output = a @ b. Supports mixed F32/F16/F8 dtypes.</summary>
    public unsafe void MatMul(Tensor output, Tensor a, Tensor b)
    {
        using NvtxRange _nvtx = NvtxRange.Push("MatMul");
        _context.EnsureCurrent();
        EnsureKernels();

        int M = (int)a.Shape[0];
        int K = (int)a.Shape[1];
        int N = (int)b.Shape[1];

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

            CublasApi.cublasGemmEx(_cublasHandle, CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N, N, M, K, &alpha, bPtr, gemmType, N,
                aPtr, gemmType, K,
                &beta,
                pC, cType, N,
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

    /// <summary>Linear layer via cuBLAS GemmEx with transpose: output = input × weight^T + bias. Supports mixed F32/F16/F8 dtypes.
    /// For a quantized weight the dequantized F16 cast is cached per preloaded weight (fast, but the cast occupies
    /// F16-sized VRAM).</summary>
    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
        => LinearImpl(output, input, weight, bias, cacheWeightCast: true);

    /// <summary>Fused quantized matmul (the low-VRAM path): same GEMM as <see cref="Linear"/> but the dequantized
    /// F16 weight is a transient buffer freed immediately, never cached. The quantized weight bytes stay resident
    /// (preloaded), so a Q4/Q5/Q6/Q8 model keeps its on-device footprint at the quant size instead of expanding to
    /// F16. Trades a per-call dequant (memory-bound, cheap) for the VRAM saving.</summary>
    public void QuantizedMatMul(Tensor output, Tensor input, Tensor quantWeight, Tensor? bias)
        => LinearImpl(output, input, quantWeight, bias, cacheWeightCast: false);

    private unsafe void LinearImpl(Tensor output, Tensor input, Tensor weight, Tensor? bias, bool cacheWeightCast)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Linear");
        _context.EnsureCurrent();
        EnsureKernels();

        int N = (int)weight.Shape[0]; // outDim
        int K = (int)weight.Shape[1]; // inDim
        int M = (int)(input.ElementCount / K); // batch*seqLen

        ulong pInput = 0, pWeight = 0, pBias = 0, pOutput = 0, pInputCast = 0, pWeightCast = 0, pBiasCast = 0;
        ulong pInputFp8 = 0, pFp8Scratch = 0;
        bool cachedOutput = false;
        try
        {
            pInput = GpuTransferHelper.CopyToDevice(input);
            pWeight = GpuTransferHelper.CopyToDevice(weight);
            if (bias is not null)
            {
                pBias = GpuTransferHelper.CopyToDevice(bias);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outBytes);

            // Fused quantized GEMV — the LLM decode hot path. When M is small (single-token or small-batch
            // decode) and the weight is Q4_K with F32 activations, read the quantized bytes ONCE and dequant
            // inline with F32 accumulate. This replaces "dequant whole weight to F16 (a full extra HBM pass +
            // F16-sized intermediate) then cuBLAS GEMM at M=1", cutting weight traffic ~4× and skipping the
            // temp — the dominant decode cost. K is always a multiple of 256 for Q4_K. Larger M (prefill)
            // falls through to the cuBLAS GEMM, which is efficient at M≥ a few hundred.
            if (M <= 8 && input.DType == DType.F32 && output.DType == DType.F32)
            {
                if (weight.DType == DType.Q4_K && K % 256 == 0 && EnvFlag("HARTSY_DP4A_ON"))
                {
                    // dp4a path (opt-in): quantize the activation to int8 (Q8_1) once, then int8 GEMV via __dp4a.
                    // Numerically exact vs the float kernel, but NOT faster here — the vectorized float GEMV is
                    // memory-latency-bound, not ALU-bound, so cutting ALU doesn't help and the extra quantize
                    // kernel offsets it. Kept behind a flag for GPUs/shapes where ALU is the limiter.
                    int kblocks = M * (K / 32);
                    ulong pXq = GpuTransferHelper.AllocateDevice((nuint)((long)M * K));
                    ulong pXd = GpuTransferHelper.AllocateDevice((nuint)((long)kblocks * sizeof(float)));
                    ulong pXs = GpuTransferHelper.AllocateDevice((nuint)((long)kblocks * sizeof(float)));
                    try
                    {
                        _kernels!.LaunchQuantizeActivationQ8_1(pXq, pXd, pXs, pInput, M, K, _stream.Handle);
                        _kernels!.LaunchMulMatVecQ4KQ8_1(pOutput, pXq, pXd, pXs, pWeight, pBias, N, K, M, _stream.Handle);
                    }
                    finally
                    {
                        GpuTransferHelper.FreeDevice(pXq);
                        GpuTransferHelper.FreeDevice(pXd);
                        GpuTransferHelper.FreeDevice(pXs);
                    }
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q4_K && K % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ4KF32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q6_K && K % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ6KF32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q8_0 && K % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ8_0F32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Q5_0: llama.cpp's fallback quant (in Q4_K_M-style mixed schemes) for any tensor whose K isn't
                // a multiple of 256 — very common on odd hidden sizes (e.g. qwen2.5-0.5b's 896). Without this
                // branch those tensors missed every fused GEMV path and fell back to the ~10-20× slower
                // dequant-to-F16-then-cuBLAS route (measured: qwen2.5-0.5b decode was ~2.7× slower than its own
                // Q8_0 quant of the identical model, because nearly every projection is K=896 → Q5_0).
                if (weight.DType == DType.Q5_0 && K % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ5_0F32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Q4_0: common legacy/baseline GGUF quant. Previously missed every fused GEMV path and fell
                // back to the ~10-20× slower dequant-to-F16-then-cuBLAS route.
                if (weight.DType == DType.Q4_0 && K % 32 == 0)
                {
                    _kernels!.LaunchMulMatVecQ4_0F32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (weight.DType == DType.Q5_K && K % 256 == 0)
                {
                    _kernels!.LaunchMulMatVecQ5KF32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                // Dense 16-bit-float weights (BF16/F16 checkpoints, e.g. Orpheus and most audio LMs). cuBLAS
                // GemmEx is inefficient at M=1; the fused GEMV reads each weight row once with an F32 accumulate
                // (activation stays F32 — at least as accurate as the cuBLAS BF16 cast). On by default; set
                // HARTSY_BF16_GEMV=0 to fall back to cuBLAS.
                if (EnableBf16Gemv && _kernels!.HasFloatGemv && weight.DType == DType.BF16)
                {
                    _kernels!.LaunchMulMatVecBf16F32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
                    GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                    cachedOutput = true;
                    return;
                }
                if (EnableBf16Gemv && _kernels!.HasFloatGemv && weight.DType == DType.F16)
                {
                    _kernels!.LaunchMulMatVecF16F32(pOutput, pInput, pWeight, pBias, N, K, M, _stream.Handle);
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
            // needs 16-byte leading dims (lda/ldb = K fp8 bytes; ldc = N · output element bytes); non-conforming
            // shapes fall through to the cast-to-F16 path below.
            if (EnableNativeFp8Gemm && weight.DType.IsFp8 && Fp8Executor.IsSupported
                && (input.DType.IsFp8 || input.DType == DType.F32 || input.DType == DType.F16)
                && (output.DType == DType.F16 || output.DType == DType.F32)
                && K % 16 == 0
                && (N * output.DType.SizeInBytes) % 16 == 0)
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

                Fp8Executor.Run(weight: pWeight, input: inputFp8Ptr, outPtr: pOutput, m: M, n: N, k: K,
                    weightScale: alpha, stream: _stream.Handle,
                    inputScaleDev: inputScaleDev, outF32: output.DType == DType.F32);

                if (bias is not null)
                {
                    int totalElementsFp8 = M * N;
                    ulong biasPtr = pBias;
                    if (output.DType != bias!.DType)
                    {
                        pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                        CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                        biasPtr = pBiasCast;
                    }
                    if (output.DType == DType.F32)
                        _kernels!.LaunchBiasAdd(pOutput, biasPtr, N, 1, totalElementsFp8, _stream.Handle);
                    else
                        _kernels!.LaunchBiasAddF16(pOutput, biasPtr, N, 1, totalElementsFp8, _stream.Handle);
                }
                GpuTransferHelper.CacheActivation(output, pOutput, outBytes);
                cachedOutput = true;
                return;
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
                    long headroom = Math.Max(2L << 30, totalBytes / 8);
                    if (freeBytes > 0 && freeBytes - (long)castBytes < headroom)
                    {
                        weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);
                    }
                    else
                    {
                        weightPtr = GpuTransferHelper.AllocateDevice(castBytes);
                        CastOnGpu(weightPtr, pWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
                        GpuTransferHelper.CacheWeightCast(weight, weightPtr, castBytes);
                    }
                }
            }
            else
            {
                weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);
            }

            int gemmType = CublasDataType(gemmDtype);
            int outputType = CublasDataType(output.DType);

            // Bias prep shared by the fused-epilogue path and the separate-add fallback:
            // both want one bias value per output channel N, in the output dtype.
            ulong biasDevicePtr = 0;
            if (bias is not null)
            {
                biasDevicePtr = pBias;
                if (output.DType != bias!.DType)
                {
                    pBiasCast = CudaMemory.Allocate((nuint)(bias.ElementCount * output.DType.SizeInBytes));
                    CastOnGpu(pBiasCast, pBias, bias.DType, output.DType, (int)bias.ElementCount);
                    biasDevicePtr = pBiasCast;
                }
            }

            // Tensor-core HGEMM path (opt-in, validation-pending): F16 operands+output,
            // aligned dims. Produces the GEMM only; bias is added by the block below.
            bool ltBiasFused = false;
            if (EnableTensorCoreGemm
                && gemmDtype == DType.F16 && output.DType == DType.F16
                && TensorCoreGemm.IsSupported && Cuda.TensorCoreGemm.IsAligned(M, N, K))
            {
                TensorCoreGemm.Run(a: inputPtr, b: weightPtr, c: pOutput, m: M, n: N, k: K, alpha: alpha, stream: _stream.Handle);
            }
            // Fused path: fold the bias into the cuBLASLt epilogue, saving a BiasAdd launch
            // plus an output-sized HBM round-trip. Only worthwhile when there is a bias to fuse.
            else if (EnableEpilogueFusion && bias is not null && LtGemm.IsSupported)
            {
                LtGemm.Run(
                    weight: weightPtr, input: inputPtr, outPtr: pOutput,
                    m: M, n: N, k: K, alpha: alpha,
                    abType: gemmType, dType: outputType,
                    biasPtr: biasDevicePtr, epilogue: CublasLtApi.CUBLASLT_EPILOGUE_BIAS,
                    stream: _stream.Handle);
                ltBiasFused = true;
            }
            else
            {
                // cuBLAS col-major: C_cm = op(A) × op(B) where op(A)=weight^T [N,K], op(B)=input [K,M]
                // Row-major interpretation: output[M,N] = input[M,K] × weight^T[K,N].
                //
                // cublasGemmEx supports Ctype ∈ {Atype, F32} only. When the output tensor is a 16-bit type
                // that differs from the operand gemmDtype — e.g. BF16 operands (fp8/quant weight × F32
                // activation → BF16, chosen so the F32→16-bit cast can't overflow F16's 65504 in a SwiGLU
                // MLP) but an F16 output tensor — there is NO BF16→F16 kernel and cuBLAS returns
                // CUBLAS_STATUS_NOT_SUPPORTED. Run the GEMM into a temp of gemmDtype, then cast to the real
                // output. (F32 output is always compatible with 16-bit operands, so it skips the temp.)
                bool outNeedsCast = output.DType != gemmDtype && output.DType != DType.F32;
                ulong gemmOut = pOutput;
                int gemmOutType = outputType;
                ulong pGemmTemp = 0;
                try
                {
                    if (outNeedsCast)
                    {
                        pGemmTemp = GpuTransferHelper.AllocateDevice((nuint)((long)M * N * gemmDtype.SizeInBytes));
                        gemmOut = pGemmTemp;
                        gemmOutType = gemmType;
                    }
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        weightPtr, gemmType, K,
                        inputPtr, gemmType, K,
                        &beta,
                        gemmOut, gemmOutType, N,
                        Compute32F(gemmType), CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                    if (outNeedsCast)
                        CastOnGpu(pOutput, pGemmTemp, gemmDtype, output.DType, M * N);
                }
                finally
                {
                    if (pGemmTemp != 0) GpuTransferHelper.FreeDevice(pGemmTemp);
                }
            }

            // Bias add for every GEMM path except the cuBLASLt epilogue, which already fused it.
            if (bias is not null && !ltBiasFused)
            {
                int totalElements = M * N;
                if (output.DType == DType.F16)
                    _kernels!.LaunchBiasAddF16(pOutput, biasDevicePtr, N, 1, totalElements, _stream.Handle);
                else if (output.DType == DType.BF16)
                    _kernels!.LaunchBiasAddBf16(pOutput, biasDevicePtr, N, 1, totalElements, _stream.Handle);
                else
                    _kernels!.LaunchBiasAdd(pOutput, biasDevicePtr, N, 1, totalElements, _stream.Handle);
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
        _context.EnsureCurrent();
        EnsureKernels();

        long batchSize = a.Shape[0];
        int M = (int)a.Shape[1];
        int K = (int)a.Shape[2];

        bool bIs2D = b.Shape.Rank == 2;
        int N = bIs2D ? (int)b.Shape[1] : (int)b.Shape[2];

        long strideA = M * K;
        long strideB = bIs2D ? 0 : K * N;
        long strideC = M * N;

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
                N, M, K,
                &alpha,
                bPtr, gemmType, N, strideB,
                aPtr, gemmType, K, strideA,
                &beta,
                pC, cType, N, strideC,
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
        using NvtxRange _nvtx = NvtxRange.Push("Conv2D");
        _context.EnsureCurrent();
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
        if (_convCudnn && !_cudnnConvDead
            && input.DType == weight.DType && output.DType == input.DType
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

    /// <summary>Attempts the cuDNN conv-forward route for <see cref="Conv2D"/>. Returns false (after
    /// disabling the route for the session) on any cuDNN failure so the caller falls through to the
    /// im2col path — a rejection costs one warning, never a session kill. Bias is added by the same
    /// per-channel kernel as the im2col path, so the two routes differ only by GEMM-class accumulation
    /// order.</summary>
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
                batch, inCh, inH, inW, outCh, kH, kW, outH, outW, strideH, strideW, padH, padW, dataType);

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

    public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GroupNorm");
        _context.EnsureCurrent();
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
                _kernels!.LaunchGroupNormF16(
                    pOut, pIn, wPtr, bPtr,
                    batch, channels, spatial, groups, eps,
                    _stream.Handle);
            }
            else if (input.DType == DType.BF16)
            {
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
        _context.EnsureCurrent();
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
                _kernels!.LaunchLayerNormF16(
                    pOut, pIn, wPtr, bPtr,
                    normDim, totalRows, eps,
                    _stream.Handle);
            }
            else if (input.DType == DType.BF16)
            {
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

    public unsafe void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RmsNorm");
        _context.EnsureCurrent();
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
        // F32 activation, or F16 activation with F32 scale/shift (the DiT F16 recipe: activation halved, tiny
        // per-channel params kept F32 for precision). scale/shift must stay F32 in both cases.
        bool f16 = input.DType == DType.F16 && output.DType == DType.F16;
        if ((!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            || scale.DType != DType.F32 || (shift is not null && shift.DType != DType.F32))
            throw new NotSupportedException("CUDA AffineBroadcastLastDim supports F32, or F16 activation with F32 scale/shift.");
        _context.EnsureCurrent();
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

    public void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate)
    {
        using NvtxRange _nvtx = NvtxRange.Push("GatedResidual");
        // F32, or F16 activations (residual/value/output) with an F32 gate — the DiT F16 recipe.
        bool f16 = residual.DType == DType.F16 && value.DType == DType.F16 && output.DType == DType.F16;
        if ((!f16 && (output.DType != DType.F32 || residual.DType != DType.F32 || value.DType != DType.F32)) || gate.DType != DType.F32)
            throw new NotSupportedException("CUDA GatedResidualLastDim supports F32, or F16 activations with an F32 gate.");
        _context.EnsureCurrent();
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

    public void ModulationSplit4(Tensor scaleMsa, Tensor gateMsa, Tensor scaleMlp, Tensor gateMlp, Tensor proj)
    {
        if (proj.DType != DType.F32)
            throw new NotSupportedException("CUDA ModulationSplit4 supports F32 only.");
        _context.EnsureCurrent();
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

    public unsafe void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked,
        int patch, bool innerChannelFastest)
    {
        using NvtxRange _nvtx = NvtxRange.Push("UnpatchifyTokens");
        if (output.DType != DType.F32 || tokens.DType != DType.F32)
            throw new NotSupportedException("CUDA UnpatchifyTokens supports F32 only.");
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(tokens);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchDitUnpatchify(pOut, pIn, channels, hPacked, wPacked, patch, innerChannelFastest, _stream.Handle);
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CUDA CfgEulerStep supports F32 only.");
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pZ = 0, pPos = 0, pNeg = 0;
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
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pPos);
            GpuTransferHelper.FreeDevice(pNeg);
        }
    }

    // ── Step-graph capture (CUDA-graph replay of a fixed per-step op sequence; see IBackend docs) ─────────
    private CudaGraph? _stepGraph;
    private bool _stepGraphCapturing;

    public bool SupportsF16Activations => true;

    public bool StepGraphSupported => true;

    public bool StepGraphReady => _stepGraph?.IsReady == true && !_stepGraphCapturing;

    /// <summary>Owner token for the single step-graph slot (see IBackend.StepGraphOwner).</summary>
    public object? StepGraphOwner { get; set; }

    public void StepGraphBegin()
    {
        _context.EnsureCurrent();
        _stepGraph ??= new CudaGraph(_stream.Handle, autoFreeAllocationsOnRelaunch: true);
        _stepGraph.Reset();
        lock (CudaMemory.CaptureAllocs)
        {
            CudaMemory.CaptureAllocs.Clear();
            CudaMemory.CaptureAllocBytes = 0;
            CudaMemory.CaptureFreeBytes = 0;
            CudaMemory.CaptureAllocCount = 0;
            CudaMemory.CaptureFreeCount = 0;
            CudaMemory.TrackCaptureWindow = true;
        }
        _stepGraph.BeginCapture();
        _stepGraphCapturing = true;
    }

    public void StepGraphEndAndLaunch()
    {
        if (!_stepGraphCapturing || _stepGraph is null)
            return;
        _stepGraphCapturing = false;
        CudaMemory.TrackCaptureWindow = false;
        long outstanding;
        int outstandingCount;
        lock (CudaMemory.CaptureAllocs)
        {
            outstanding = CudaMemory.CaptureAllocBytes - CudaMemory.CaptureFreeBytes;
            outstandingCount = CudaMemory.CaptureAllocs.Count;
        }
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] step-graph capture window: allocs {CudaMemory.CaptureAllocCount} ({CudaMemory.CaptureAllocBytes >> 20} MB), " +
            $"frees {CudaMemory.CaptureFreeCount} ({CudaMemory.CaptureFreeBytes >> 20} MB), " +
            $"OUTSTANDING {outstandingCount} allocs / {outstanding >> 20} MB");
        _stepGraph.EndCaptureAndInstantiate();
        _stepGraph.Launch();
    }

    public void StepGraphLaunch()
    {
        _context.EnsureCurrent();
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

    /// <summary>Device copy into <paramref name="dst"/>'s EXISTING buffer (address-preserving — the captured-graph
    /// boundary refresh). First call materializes a device buffer for dst; subsequent calls reuse it. Host src is
    /// uploaded; device src is DtoD'd, both stream-ordered on the compute stream.</summary>
    public unsafe void CopyInto(Tensor dst, Tensor src)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CopyInto");
        _context.EnsureCurrent();
        EnsureKernels();
        ulong pSrc = 0;
        ulong pDst = GpuTransferHelper.CopyToDevice(dst);
        try
        {
            pSrc = GpuTransferHelper.CopyToDevice(src);
            nuint bytes = GpuTransferHelper.ByteSize(dst);
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
        // F32, or F16 q/k with F32 cos/sin (DiT F16 recipe — cos/sin are tiny and precision-sensitive).
        bool f16 = q.DType == DType.F16;
        if ((!f16 && q.DType != DType.F32) || k.DType != q.DType || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRope supports F32, or F16 q/k with F32 cos/sin.");
        _context.EnsureCurrent();
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

    /// <summary>Oasis head split: frame-major qkv[token,3·dim] → out[b,heads,seq,headDim] (device port of the host loop).</summary>
    public void OasisSplitHeads(Tensor output, Tensor qkv, int frames, int sp, int heads, int headDim, int part, bool temporal)
    {
        if (output.DType != DType.F32 || qkv.DType != DType.F32) throw new NotSupportedException("CUDA OasisSplitHeads supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32) throw new NotSupportedException("CUDA OasisRopeInterleaved supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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
        _context.EnsureCurrent(); EnsureKernels();
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
        if (output.DType != DType.F32 || input.DType != DType.F32 || mod.DType != DType.F32) throw new NotSupportedException("CUDA OasisAdaLn supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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
        _context.EnsureCurrent(); EnsureKernels();
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

    /// <summary>TripoSR triplane grid-sample (device). <paramref name="planes"/> is uploaded once (weight-cache
    /// resident via the decoder's PreloadWeights); the result <paramref name="outputF"/> stays device-resident so
    /// the NeRF MLP Linears hit the activation cache — no host round-trip per point.</summary>
    public unsafe void TriplaneGridSample(Tensor outputF, Tensor planes, Tensor? coords, long chunkStart,
        int count, int channels, int planeH, int planeW, float radius, int gridRes)
    {
        if (outputF.DType != DType.F32 || planes.DType != DType.F32)
            throw new NotSupportedException("CUDA TriplaneGridSample supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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

    /// <summary>2D transposed convolution (device gather kernel). Weight <c>[Cin,Cout,kH,kW]</c>. Overrides the
    /// CPU scatter-add default — the TripoSR/YOLO/Demucs upsample was running on the host.</summary>
    public unsafe void ConvTranspose2d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int strideH, int strideW, int padH, int padW)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32 || weight.DType != DType.F32)
            throw new NotSupportedException("CUDA ConvTranspose2d supports F32 only.");
        int n = (int)input.Shape[0], cIn = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int cOut = (int)output.Shape[1], oH = (int)output.Shape[2], oW = (int)output.Shape[3];
        int kH = (int)weight.Shape[2], kW = (int)weight.Shape[3];
        _context.EnsureCurrent(); EnsureKernels();
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
        if (output.DType != DType.F32 || proj.DType != DType.F32)
            throw new NotSupportedException("CUDA GegluErf supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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

    /// <summary>DIAMOND pixel quantize to 256 levels: out = floor((clamp(v,-1,1)+1)·127.5)/127.5 − 1 (device).</summary>
    public void PixelQuantize(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32) throw new NotSupportedException("CUDA PixelQuantize supports F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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

    /// <summary>Wan-Video interleaved in-place RoPE (shared cos/sin) — keeps q/k GPU-resident through RoPE so the
    /// whole RmsNorm→RoPE→transpose→SDPA chain never leaves the device. In-place: x's device buffer is rotated and
    /// re-cached to the same pointer (no reallocation, no host round-trip).</summary>
    public unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2SplitRope");
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA Ltx2SplitRope supports F32 only.");
        _context.EnsureCurrent();
        EnsureKernels();
        ulong pX = 0, pCos = 0, pSin = 0;
        try
        {
            pX = GpuTransferHelper.CopyToDevice(x);       // in-place target (cached activation)
            pCos = GpuTransferHelper.CopyToDevice(cos);
            pSin = GpuTransferHelper.CopyToDevice(sin);
            _kernels!.LaunchLtx2SplitRope(pX, pCos, pSin, seqLen, numHeads, headDim, _stream.Handle);
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
        _context.EnsureCurrent();
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

    /// <summary>Per-head interleaved RoPE (Matrix-Game 3.0 sigma_theta): cos/sin are <c>[heads, S, headDim]</c>. Ported
    /// off the host loop that dominated the MG3 backbone (~1.9 s of a 2.05 s forward → ~0.15 s).</summary>
    public unsafe void WanRopeInterleavedPerHead(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleavedPerHead");
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA WanRopeInterleavedPerHead supports F32 x/cos/sin.");
        _context.EnsureCurrent();
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
        _context.EnsureCurrent(); EnsureKernels();
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
        _context.EnsureCurrent(); EnsureKernels();
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
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32) throw new NotSupportedException("CUDA Mg3RopeBatched F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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
        if (outT.DType != DType.F32 || hidden.DType != DType.F32 || mouseWin.DType != DType.F32) throw new NotSupportedException("CUDA Mg3MouseMlpConcat F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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
        if (kOut.DType != DType.F32 || vOut.DType != DType.F32 || kv.DType != DType.F32) throw new NotSupportedException("CUDA Mg3KvExpand F32 only.");
        _context.EnsureCurrent(); EnsureKernels();
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

    /// <summary>GPU temporal frame extract: output[B,C,H,W] = src[B,C,Tsrc,H,W][:, :, ti, :, :]. Keeps CausalConv3d
    /// frame slicing on-device (no D2H/host-slice).</summary>
    public unsafe void ExtractVaeFrame(Tensor output, Tensor src, int ti)
    {
        _context.EnsureCurrent();
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
            _kernels!.LaunchWanVaeExtractFrame(pOut, pSrc, ti, b, c, tsrc, frameHW, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pSrc);
        }
    }

    /// <summary>GPU temporal frame write (in-place): output[B,C,Tout,H,W][:, :, to, :, :] = acc[B,C,H,W] + bias[c].
    /// Writes one slot of <paramref name="output"/>'s device buffer; used to accumulate CausalConv3d output frames
    /// without ever leaving the device.</summary>
    public unsafe void WriteVaeFrame(Tensor output, Tensor acc, Tensor? bias, int to)
    {
        _context.EnsureCurrent();
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
            _kernels!.LaunchWanVaeWriteFrame(pOut, pAcc, pBias, to, b, c, tout, frameHW, _stream.Handle);
            GpuTransferHelper.CacheActivation(output, pOut, GpuTransferHelper.ByteSize(output));   // in-place re-assert
        }
        finally
        {
            // pOut is output's cached buffer — do NOT free. Free acc/bias inputs (skipped if cached).
            GpuTransferHelper.FreeDevice(pAcc);
            if (pBias != 0) GpuTransferHelper.FreeDevice(pBias);
        }
    }

    /// <summary>GPU build of the frame-major padded input for batched CausalConv3d (transpose + temporal
    /// zero/replicate pad + cache prepend + optional spatial edge-replicate pad), so the whole conv layer runs as
    /// kt batched Conv2D calls instead of tout·kt tiny ones.</summary>
    public unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0)
    {
        _context.EnsureCurrent();
        EnsureKernels();
        int paddedT = (int)padded.Shape[0], cIn = (int)padded.Shape[1];
        int H = (int)input.Shape[3], W = (int)input.Shape[4];
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
            _kernels!.LaunchWanVaeBuildPadded(pOut, pIn, pCache, paddedT, cIn, Tin, cacheLen, zeroPad, H, W, padH, padW, replicateFirst, _stream.Handle);
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

    /// <summary>GPU fill of the conv output with per-channel bias (init for the temporal accumulate).</summary>
    public unsafe void FillBias(Tensor output, Tensor? bias)
    {
        _context.EnsureCurrent();
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
            _kernels!.LaunchWanVaeFillBias(pOut, pBias, cOut, tout, HW, _stream.Handle);
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
        _context.EnsureCurrent();
        EnsureKernels();
        int cOut = (int)output.Shape[1], tout = (int)output.Shape[2];
        int HW = (int)(output.ElementCount / ((long)cOut * tout));
        ulong pOut = 0, pConv = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);   // in-place accumulate target
            pConv = GpuTransferHelper.CopyToDevice(convDt);
            _kernels!.LaunchWanVaeAccumulateTap(pOut, pConv, dt, strideT, cOut, tout, HW, _stream.Handle);
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
        _context.EnsureCurrent();
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
            _kernels!.LaunchWanVaeRmsNormChannel(pOut, pIn, pGamma, c, spatial, eps, sqrtC, numPos, _stream.Handle);
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

    /// <summary>GPU image-output conversion (CHW F32 [-1,1] → HWC u8): converts on-device so only the 3 MB byte
    /// image crosses PCIe instead of the 12 MB float planes + a 12M-element host loop (~140 ms → ~30 ms).</summary>
    public unsafe void ChwF32ToHwcU8(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
            _kernels!.LaunchWanVaeUnpatchify(pOut, pIn, b, c, t, h, w, p, numOut, _stream.Handle);
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>In-place rotary embedding on a single GPU-resident tensor <c>[B, L, numHeads, headDim]</c>.
    /// Used by grouped-query attention where Q and K differ in head count (the paired <see cref="ApplyRope"/>
    /// would mis-stride K). cos/sin are <c>[B, L, headDim]</c>.</summary>
    public void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRopeSingle supports F32 only.");
        _context.EnsureCurrent();
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

    /// <summary>In-place interleaved (GPT-J) rotary embedding on a single GPU-resident tensor
    /// <c>[B, L, numHeads, headDim]</c> (adjacent pairs rotated by frequency i). cos/sin are <c>[B, L, headDim]</c>.
    /// Without this override the shared <see cref="IBackend"/> CPU fallback would drag q/k off the device every layer
    /// (Sesame CSM / HeartMuLa use interleaved RoPE) — the whole RmsNorm→RoPE→SDPA chain stays resident.</summary>
    public unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleaved");
        if (Environment.GetEnvironmentVariable("HM_ROPE_CPU") == "1")   // TEMP perf-repro gate: old CPU-fallback path
        {
            int b = (int)x.Shape[0], sl = (int)x.Shape[1], nh = (int)x.Shape[2], hd = (int)x.Shape[3], hf = hd / 2;
            float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int s = 0; s < sl; s++)
                {
                    long fb = ((long)bi * sl + s) * hd;
                    for (int h = 0; h < nh; h++)
                    {
                        float* v = xp + (((long)bi * sl + s) * nh + h) * hd;
                        for (int i = 0; i < hf; i++)
                        {
                            float xe = v[2 * i], xo = v[2 * i + 1];
                            v[2 * i] = xe * cp[fb + i] - xo * sp[fb + i];
                            v[2 * i + 1] = xo * cp[fb + i] + xe * sp[fb + i];
                        }
                    }
                }
            return;
        }
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA ApplyRopeInterleaved supports F32 only.");
        _context.EnsureCurrent();
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
            _kernels!.LaunchRopeInterleaved(pX, pCos, pSin, numHeads, headDim, totalVecs, _stream.Handle);

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

    public void SliceLastDim(Tensor output, Tensor input, int offset)
    {
        using NvtxRange _nvtx = NvtxRange.Push("SliceLastDim");
        bool f16 = output.DType == DType.F16 && input.DType == DType.F16;
        if (!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            throw new NotSupportedException("CUDA SliceLastDim supports F32 or F16 (both sides same dtype).");
        _context.EnsureCurrent();
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
        if (output.DType != DType.F32 || input.DType != DType.F32 || rowMask.DType != DType.F32)
            throw new NotSupportedException("CUDA MaskRows supports F32 only.");
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>Fused adaLN modulation: out = (1+scale)·LayerNormNoAffine(in) + shift (F32 or F16 activation, F32
    /// scale/shift). Replaces LayerNormNoAffine + AddScalar(+1) + AffineBroadcast for the DiT NormModulate.</summary>
    public void LayerNormModulate(Tensor output, Tensor input, Tensor scale, Tensor shift, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("LayerNormModulate");
        bool f16 = input.DType == DType.F16 && output.DType == DType.F16;
        if (!f16 && (output.DType != DType.F32 || input.DType != DType.F32))
            throw new NotSupportedException("CUDA LayerNormModulate supports F32 or F16 activations.");
        if (scale.DType != DType.F32 || shift.DType != DType.F32)
            throw new NotSupportedException("CUDA LayerNormModulate requires F32 scale/shift.");
        _context.EnsureCurrent();
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

    /// <summary>Fused QKV split + per-head QK-RMSNorm (F32 or F16 activation, F32 weights). Writes q/k/v each
    /// [., W] laid [token, head, d]. Replaces SliceLastDim×3 + RmsNorm×2 per attention stream.</summary>
    public void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvSplitNorm");
        bool f16 = qkv.DType == DType.F16;
        if (!f16 && qkv.DType != DType.F32)
            throw new NotSupportedException("CUDA QkvSplitNorm supports F32 or F16 activations.");
        _context.EnsureCurrent();
        EnsureKernels();
        int headDim = (int)qWeight.Shape[qWeight.Shape.Rank - 1];
        int w = (int)qkv.Shape[qkv.Shape.Rank - 1] / 3;
        int heads = w / headDim;
        int tokens = (int)(qkv.ElementCount / (3L * w));
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

    /// <summary>FourierEmbedder over device coords [count,3] → dst [count, 3·(2·bands+1)] (F32).</summary>
    public void FourierEmbed(Tensor dst, Tensor coords, int count, int bands)
    {
        using NvtxRange _nvtx = NvtxRange.Push("FourierEmbed");
        if (dst.DType != DType.F32 || coords.DType != DType.F32)
            throw new NotSupportedException("CUDA FourierEmbed supports F32 only.");
        _context.EnsureCurrent();
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

    public void IndexAddRows(Tensor h, Tensor table, Tensor indices)
    {
        if (h.DType != DType.F32 || table.DType != DType.F32)
            throw new NotSupportedException("CUDA IndexAddRows supports F32 only.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException("CUDA IndexAddRows requires I32 indices.");
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    public void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
        if (input.DType != output.DType || (output.DType != DType.F32 && output.DType != DType.F16))
            throw new NotSupportedException("CUDA SliceRows supports F32 or F16 (matching input/output dtype).");
        _context.EnsureCurrent();
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
            if (output.DType == DType.F16)
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

    public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps)
    {
        if (input.DType != DType.F32 || gamma.DType != DType.F32 || beta.DType != DType.F32)
            throw new NotSupportedException($"CUDA AdaInstanceNorm1d supports F32 only — got input {input.DType}, gamma {gamma.DType}, beta {beta.DType}.");
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>GPU cast FP16 → FP32 via PTX kernel.</summary>
    /// <summary>Dequantizes a GGUF-quantized weight tensor (Q4_K / Q5_K / Q6_K / Q8_0 / fp8) to a host-readable
    /// F32 tensor using the SAME on-GPU <see cref="CastOnGpu"/> path that <c>Linear</c> uses per GEMM. Test/tooling
    /// hook for validating the GPU dequant kernels against a reference — not a hot path.</summary>
    public Tensor DequantizeToF32(Tensor quant)
    {
        _context.EnsureCurrent();
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

    public void CastToF32(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("CastToF32");
        _context.EnsureCurrent();
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

    /// <summary>GPU cast FP32 â BF16 via PTX kernel. Routes through F32 when input
    /// dtype is anything else (F16, etc.) by chaining the F16âF32 cast first.</summary>
    public void CastToBf16(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        EnsureKernels();

        long B = query.Shape[0];
        long H = query.Shape[1];
        long Sq = query.Shape[2];
        long D = query.Shape[3];
        long Skv = key.Shape[2];

        long totalHeads = B * H;

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
        if (CudnnMaskCompatible(mask, B, Sq, Skv) && query.DType == DType.F16 && key.DType == DType.F16
            && value.DType == DType.F16 && output.DType == DType.F16
            && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(D) && CudnnSdpa.ShapeSupported(D)
            && TryCudnnSdpa(output, query, key, value, mask, scale))
        {
            return;
        }

        if (query.DType == DType.F32)
        {
            // cuDNN fused flash-attention (HARTSY_SDPA_CUDNN): a single fused kernel — no materialized
            // [heads,Sq,Skv] score matrix — via cuDNN's runtime-compiled attention engine. ~34× over the
            // materialized cuBLAS path at Krea2 shape. MHA only, D∈{64,128}. Safe for RMS-normed-Q/K archs
            // (bounded scores) since we run fp16 I/O; callers gate the same way as the F16 path (allowF16).
            // Masked callers: the additive F32 mask is added to the fp32 scores INSIDE the engine (bias
            // score-modifier), so the mask never rounds through F16 — only Q/K/V do, same as unmasked.
            // Self-disables for the session if cuDNN init/exec ever throws, falling back to the paths below.
            if (CudnnMaskCompatible(mask, B, Sq, Skv)
                && _sdpaCudnn && !_cudnnSdpaDead && CudnnSdpaDimEligible(D) && CudnnSdpa.ShapeSupported(D)
                && (allowF16 || _sdpaF16ForceOn) && !_sdpaF16Disabled
                && TryCudnnSdpa(output, query, key, value, mask, scale))
            {
                return;
            }
        }

        if (mask is null && query.DType == DType.F32)
        {

            // Fused FlashAttention-2 (TF32 tensor cores, F32 accum, no materialized score matrix). Opt-in while
            // validating (HARTSY_SDPA_V2); MHA only (Hq==Hkv here — single B×H×S×D layout), D∈{64,128}.
            if (EnvFlag("HARTSY_SDPA_V2") && (D == 64 || D == 128))
            {
                FlashAttentionV2(output, query, key, value, scale);
                return;
            }
            if (EnvFlag("HARTSY_SDPA_FORCE_FLASH"))
            {
                FlashAttention(output, query, key, value, (int)Skv, kvGroup: 1, causal: false, qOffset: 0, scale);
                return;
            }
            ulong scoreBytesEst = (ulong)totalHeads * (ulong)Sq * (ulong)Skv * sizeof(float);
            (nuint freeBytes, _) = _context.GetMemoryInfo();
            if (EnvFlag("HARTSY_SDPA_FORCE_TILED") || scoreBytesEst > (ulong)freeBytes / 2)
            {
                SdpaTiledF32NoMask(output, query, key, value, scale);
                return;
            }
        }

        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0, scoresBuf = 0;
        ulong pQCast = 0, pKCast = 0, pVCast = 0, pOutCast = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
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

            nuint scoresBytes = (nuint)(totalHeads * Sq * Skv * elemSize);
            scoresBuf = CudaMemory.Allocate(scoresBytes);

            float alpha = scale;
            float beta = 0.0f;

            long strideQ = Sq * D;
            long strideK = Skv * D;
            long strideScores = Sq * Skv;

            // QK^T per head
            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong qPtr = opQ + (ulong)(bh * strideQ * elemSize);
                ulong kPtr = opK + (ulong)(bh * strideK * elemSize);
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);

                if (isF16)
                {
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        (int)Skv, (int)Sq, (int)D,
                        &alpha,
                        kPtr, CublasApi.CUDA_R_16F, (int)D,
                        qPtr, CublasApi.CUDA_R_16F, (int)D,
                        &beta,
                        sPtr, CublasApi.CUDA_R_16F, (int)Skv,
                        CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
                else
                {
                    // F32 QK^T via TF32 tensor cores (~8x over FP32 CUDA cores). TF32 keeps the F32
                    // exponent range, so raw pre-softmax scores can't overflow the way F16 would.
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        (int)Skv, (int)Sq, (int)D,
                        &alpha,
                        kPtr, CublasApi.CUDA_R_32F, (int)D,
                        qPtr, CublasApi.CUDA_R_32F, (int)D,
                        &beta,
                        sPtr, CublasApi.CUDA_R_32F, (int)Skv,
                        CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
            }

            // Apply mask if present
            if (mask is not null)
            {
                long maskElements = mask.ElementCount;
                long scoreElements = totalHeads * Sq * Skv;

                if (maskElements == Sq * Skv)
                {
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                        if (isF16)
                            _kernels!.LaunchAddF16(sPtr, sPtr, pMask, (int)(Sq * Skv), _stream.Handle);
                        else
                            _kernels!.LaunchAdd(sPtr, sPtr, pMask, (int)(Sq * Skv), _stream.Handle);
                    }
                }
                else if (maskElements == scoreElements)
                {
                    if (isF16)
                        _kernels!.LaunchAddF16(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                    else
                        _kernels!.LaunchAdd(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                }
                else if (maskElements == B * Sq * Skv && B > 1)
                {
                    // [B, 1, Sq, Skv] mask: per-batch, broadcast over heads. Add each batch's [Sq,Skv] block to all H
                    // head-blocks for that batch. (B=1 is already caught by the Sq*Skv branch above.) One add per (b,h).
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        long b = bh / H;
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                        ulong mPtr = pMask + (ulong)(b * Sq * Skv * elemSize);
                        if (isF16)
                            _kernels!.LaunchAddF16(sPtr, sPtr, mPtr, (int)(Sq * Skv), _stream.Handle);
                        else
                            _kernels!.LaunchAdd(sPtr, sPtr, mPtr, (int)(Sq * Skv), _stream.Handle);
                    }
                }
            }

            // Softmax
            if (isF16)
                _kernels!.LaunchSoftmaxF16(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);
            else
                _kernels!.LaunchSoftmax(scoresBuf, (int)Skv, (int)(totalHeads * Sq), _stream.Handle);

            // attn_weights @ V
            long strideV = Skv * D;
            long strideOut = Sq * D;
            float one = 1.0f;
            float zero = 0.0f;

            for (long bh = 0; bh < totalHeads; bh++)
            {
                ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                ulong vPtr = opV + (ulong)(bh * strideV * elemSize);
                ulong oPtr = opOut + (ulong)(bh * strideOut * elemSize);

                if (isF16)
                {
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        (int)D, (int)Sq, (int)Skv,
                        &one,
                        vPtr, CublasApi.CUDA_R_16F, (int)D,
                        sPtr, CublasApi.CUDA_R_16F, (int)Skv,
                        &zero,
                        oPtr, CublasApi.CUDA_R_16F, (int)D,
                        CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
                else
                {
                    // attn_weights @ V via TF32 tensor cores (~8x over FP32 CUDA cores).
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        (int)D, (int)Sq, (int)Skv,
                        &one,
                        vPtr, CublasApi.CUDA_R_32F, (int)D,
                        sPtr, CublasApi.CUDA_R_32F, (int)Skv,
                        &zero,
                        oPtr, CublasApi.CUDA_R_32F, (int)D,
                        CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }
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
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
            if (pQCast != 0) CudaMemory.FreeAsync(pQCast, _stream.Handle);
            if (pKCast != 0) CudaMemory.FreeAsync(pKCast, _stream.Handle);
            if (pVCast != 0) CudaMemory.FreeAsync(pVCast, _stream.Handle);
            if (pOutCast != 0) CudaMemory.FreeAsync(pOutCast, _stream.Handle);
        }
    }

    /// <summary>cuDNN fused flash-attention fast path for the plain F32/no-mask/MHA case. Casts Q/K/V to F16,
    /// runs cuDNN's fused attention (fp16 I/O, fp32 accum — no materialized score matrix), casts the F16 result
    /// back to the F32 output. Returns false (and permanently disables cuDNN for the session) on any failure so
    /// the caller falls through to the existing materialized/tiled paths. Q/out [B,H,Sq,D], K/V [B,H,Skv,D].</summary>
    /// <summary>Whether a caller's attention mask can ride the cuDNN fused engine as an additive fp32 bias:
    /// no mask at all, or an F32 additive mask broadcastable as [B,1,Sq,Skv] / [1,1,Sq,Skv] over heads.
    /// Per-head ([B,H,Sq,Skv]) or non-F32 masks fall back to the materialized path.</summary>
    private static bool CudnnMaskCompatible(Tensor? mask, long B, long Sq, long Skv)
    {
        if (mask is null) return true;
        if (mask.DType != DType.F32) return false;
        long n = mask.ElementCount;
        return n == Sq * Skv || n == B * Sq * Skv;
    }

    private unsafe bool TryCudnnSdpa(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        long B = query.Shape[0], H = query.Shape[1], Sq = query.Shape[2], D = query.Shape[3];
        long Skv = key.Shape[2];
        ulong pQ = 0, pK = 0, pV = 0, pMask = 0, pOut = 0, qF16 = 0, kF16 = 0, vF16 = 0, oF16 = 0;
        bool cachedOutput = false;
        try
        {
            _cudnnSdpa ??= new CudnnSdpa(_stream.Handle);

            // Checked right after real (or already-cached) engine construction, so an injected failure is
            // classified the same way a real post-init failure (e.g. a BuildPlan/Execute failure) would be
            // — the catch below branches on whether _cudnnSdpa is null, i.e. whether construction itself
            // ever succeeded, and a fault injected before that point would be indistinguishable from a
            // genuine init failure (permanent, session-wide), defeating the point of testing per-dim
            // retry/backoff/classification.
            CudnnStatusException? injected = TestCudnnSdpaFaultInjector?.Invoke(D);
            if (injected is not null) throw injected;

            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            long biasB = 1;
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
                biasB = mask.ElementCount / (Sq * Skv);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (query.DType == DType.F16)
            {
                // Native F16 Q/K/V/output — the engine's fp16 I/O dtype already; execute directly, zero casts.
                _cudnnSdpa.Execute(pQ, pK, pV, pOut, B, H, Sq, Skv, D, scale, pMask, biasB);
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

                _cudnnSdpa.Execute(qF16, kF16, vF16, oF16, B, H, Sq, Skv, D, scale, pMask, biasB);

                _kernels!.LaunchCastF16ToF32(pOut, oF16, (int)output.ElementCount, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (_cudnnSdpaDimState.TryRemove(D, out DimFailureState? recovered))
            {
                HartsyInference.Core.Logging.Logs.Info(
                    $"[cuDNN SDPA] D={D} recovered after {recovered.ConsecutiveFailures} prior failure(s) — fused path re-engaged.");
            }
            if (!CudnnSdpaEngaged)
            {
                CudnnSdpaEngaged = true;
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN SDPA] fused flash-attention engaged (D={D}, cuDNN {CudnnApi.cudnnGetVersion()})");
            }
            return true;
        }
        catch (Exception ex)
        {
            // Init failure (lib missing) ⇒ session-dead — unconditional, this is never worth retrying (the
            // library either loads or it doesn't).
            if (_cudnnSdpa is null)
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
            bool permanent = ex is not CudnnStatusException cse || cse.IsPermanent;
            if (permanent)
            {
                _cudnnSdpaDimState[D] = new DimFailureState { Permanent = true };
                HartsyInference.Core.Logging.Logs.Warning(
                    $"[cuDNN SDPA] D={D} permanently disabled (structural failure) — this is the steady state for the rest of the process: {ex.Message}");
            }
            else
            {
                DimFailureState state = _cudnnSdpaDimState.AddOrUpdate(D,
                    _ => new DimFailureState { ConsecutiveFailures = 1 },
                    (_, existing) => { existing.ConsecutiveFailures++; return existing; });
                long backoffMs = CudnnSdpaBackoffMs(state.ConsecutiveFailures);
                state.NextRetryAtTicks = Environment.TickCount64 + backoffMs;
                long? ramMb = HostMemoryInfo.AvailableBytes() is { } bytes ? bytes / (1024 * 1024) : null;
                HartsyInference.Core.Logging.Logs.Warning(
                    $"[cuDNN SDPA] D={D} transient failure #{state.ConsecutiveFailures} " +
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

    /// <summary>Fused FlashAttention-2 (TF32 tensor cores, F32 accumulate) — no-mask MHA, D∈{64,128}. Never
    /// materializes the [heads,Sq,Skv] score matrix. Q/out [B,H,Sq,D], K/V [B,H,Skv,D].</summary>
    private unsafe void FlashAttentionV2(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
        _context.EnsureCurrent();
        EnsureKernels();
        long B = query.Shape[0], H = query.Shape[1], Sq = query.Shape[2], D = query.Shape[3];
        long Skv = key.Shape[2];
        ulong pQ = 0, pK = 0, pV = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);
            _kernels!.LaunchFlashAttentionV2Tf32(pOut, pQ, pK, pV, (int)B, (int)H, (int)Sq, (int)Skv, (int)D, scale, _stream.Handle);
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

    /// <summary>Query-tiled scaled-dot-product attention for the plain F32/no-mask/MHA case (Wan-Video full-res
    /// self-attention). Tiles the query axis into <c>Br</c>-row chunks so only a <c>[totalHeads, Br, Skv]</c> score
    /// buffer is materialized per tile — never the full <c>[totalHeads, Sq, Skv]</c> matrix (24×14040²×4B ≈ 19 GB).
    /// Each tile reuses the same TF32 tensor-core QK^T / softmax / scores·V ops as <see cref="ScaledDotProductAttention"/>,
    /// so results are numerically identical to the plain path (and much faster than the online-softmax flash kernel,
    /// which re-reads all K/V per query row). <c>Br</c> is sized to a quarter of free VRAM (override: <c>HARTSY_SDPA_TILE</c>).</summary>
    private unsafe void SdpaTiledF32NoMask(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
        _context.EnsureCurrent();
        EnsureKernels();

        long B = query.Shape[0], H = query.Shape[1], Sq = query.Shape[2], D = query.Shape[3];
        long Skv = key.Shape[2];
        long totalHeads = B * H;

        ulong pQ = 0, pK = 0, pV = 0, pOut = 0, scoresBuf = 0;
        bool cachedOutput = false;
        try
        {
            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            // Query-tile height: fit [totalHeads, Br, Skv] into ~a quarter of free VRAM (leaves room for Q/K/V/out
            // and the model weights). Env override HARTSY_SDPA_TILE forces a fixed Br (benchmarking).
            (nuint freeBytes, _) = _context.GetMemoryInfo();
            long perRow = totalHeads * Skv * sizeof(float);            // bytes for one query row across all heads
            long Br = (long)((ulong)freeBytes / 4) / Math.Max(1, perRow);
            if (Br < 1) Br = 1;
            if (Br > Sq) Br = Sq;
            if (int.TryParse(Environment.GetEnvironmentVariable("HARTSY_SDPA_TILE"), out int envBr) && envBr > 0)
                Br = Math.Min(envBr, Sq);

            scoresBuf = CudaMemory.Allocate((nuint)(totalHeads * Br * Skv * sizeof(float)));

            long strideQ = Sq * D, strideK = Skv * D, strideV = Skv * D, strideOut = Sq * D;
            float alpha = scale, beta = 0f, one = 1f, zero = 0f;

            for (long q0 = 0; q0 < Sq; q0 += Br)
            {
                long curBr = Math.Min(Br, Sq - q0);
                long tileStride = curBr * Skv;   // per-head stride within the (packed) tile score buffer

                // QK^T per head: scores[curBr, Skv] = scale · Q[q0:q0+curBr] · Kᵀ  (TF32 tensor cores)
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    ulong qPtr = pQ + (ulong)((bh * strideQ + q0 * D) * sizeof(float));
                    ulong kPtr = pK + (ulong)(bh * strideK * sizeof(float));
                    ulong sPtr = scoresBuf + (ulong)(bh * tileStride * sizeof(float));
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        (int)Skv, (int)curBr, (int)D,
                        &alpha,
                        kPtr, CublasApi.CUDA_R_32F, (int)D,
                        qPtr, CublasApi.CUDA_R_32F, (int)D,
                        &beta,
                        sPtr, CublasApi.CUDA_R_32F, (int)Skv,
                        CublasApi.CUBLAS_COMPUTE_32F_FAST_TF32, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
                }

                // Row-softmax over Skv for the (totalHeads·curBr) packed rows.
                _kernels!.LaunchSoftmax(scoresBuf, (int)Skv, (int)(totalHeads * curBr), _stream.Handle);

                // scores·V per head → out[q0:q0+curBr]  (TF32 tensor cores)
                for (long bh = 0; bh < totalHeads; bh++)
                {
                    ulong sPtr = scoresBuf + (ulong)(bh * tileStride * sizeof(float));
                    ulong vPtr = pV + (ulong)(bh * strideV * sizeof(float));
                    ulong oPtr = pOut + (ulong)((bh * strideOut + q0 * D) * sizeof(float));
                    CublasApi.cublasGemmEx(
                        _cublasHandle,
                        CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_N,
                        (int)D, (int)curBr, (int)Skv,
                        &one,
                        vPtr, CublasApi.CUDA_R_32F, (int)D,
                        sPtr, CublasApi.CUDA_R_32F, (int)Skv,
                        &zero,
                        oPtr, CublasApi.CUDA_R_32F, (int)D,
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
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
        }
    }

    #endregion

    #region Activations

    public void Gelu(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Gelu");
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchGeluF16(pOut, pIn, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchGelu(pOut, pIn, (int)input.ElementCount, _stream.Handle);

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
        if (input.DType != DType.F32 && !(input.DType == DType.F16 && output.DType == DType.F16))
            throw new NotSupportedException($"CUDA Sigmoid supports F32 or F16 — got {input.DType}.");
        _context.EnsureCurrent();
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
        if (input.DType != DType.F32)
            throw new NotSupportedException($"CUDA Tanh currently supports F32 only — got {input.DType}.");
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        if (output.DType != DType.F32)
            throw new NotSupportedException($"CUDA Conv1d writes F32 output — got output {output.DType}.");
        _context.EnsureCurrent();
        EnsureKernels();

        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)output.Shape[1], tOut = (int)output.Shape[2], kernel = (int)weight.Shape[2];

        // cuDNN fast path: symmetric-padded, non-grouped 1D conv → 2D (H=1) so cuDNN's tensor-core (TF32 on Ampere)
        // implicit-GEMM/Winograd engines run it — faster than the direct kernel on the wide vocoder convs. Output
        // stays F32. Falls back to the direct kernel on group/asymmetric-pad or any cuDNN failure (session-sticky).
        if (_audioConvCudnn && !_cudnnConvDead && groups == 1 && padLeft == padRight
            && TryCudnnConv1d(output, input, weight, bias, batch, cIn, cOut, tIn, tOut, kernel, stride, padLeft, dilation))
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

    /// <summary>cuDNN 1D convolution (mapped to 2D, H=1): F32 output via TF32 tensor cores. Weights/inputs are cast
    /// to F32; bias is added F32 after. Returns false (session-sticky) on any cuDNN failure so the caller falls back
    /// to the direct kernel.</summary>
    private unsafe bool TryCudnnConv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        int batch, int cIn, int cOut, int tIn, int tOut, int kernel, int stride, int pad, int dilation)
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
            // 1D → 2D: N,C,H=1,W=tIn ; filter K,C,R=1,S=kernel ; strideW=stride, padW=pad, dilationW=dilation.
            _cudnnConv.Execute(inF32, wF32, pOut,
                batch, cIn, 1, tIn, cOut, 1, kernel, 1, tOut, 1, stride, 0, pad, CudnnApi.CUDNN_DATA_FLOAT, 1, dilation);
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
            HartsyInference.Core.Logging.Logs.Warning($"[cuDNN conv] audio conv1d disabled for the session (falling back to direct kernel): {ex.Message}");
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
        _context.EnsureCurrent();
        EnsureKernels();

        // ConvTranspose1d weight is [C_in, C_out/groups, K].
        int batch = (int)input.Shape[0], cIn = (int)input.Shape[1], tIn = (int)input.Shape[2];
        int cOut = (int)output.Shape[1], tOut = (int)output.Shape[2], kernel = (int)weight.Shape[2];

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

    public void Silu(Tensor output, Tensor input)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Silu");
        _context.EnsureCurrent();
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

    #endregion

    #region Element-wise

    public void Add(Tensor output, Tensor a, Tensor b)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    public void Clamp(Tensor output, Tensor input, float min, float max)
    {
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (input.DType == DType.F16)
                _kernels!.LaunchClampF16(pOut, pIn, min, max, (int)input.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchClamp(pOut, pIn, min, max, (int)input.ElementCount, _stream.Handle);

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

    /// <summary>Resolves the dtype a single operand will end up at after fp8 → F16 fallback. Kept for callers (e.g. Conv2D's im2col elemSize) that need the per-operand answer without the joint-dtype rule; new code should prefer the two-operand overload.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private DType ResolveGemmDtype(DType dtype)
    {
        if (dtype.IsFp8) return DType.F16; // Ampere fallback: cast F8→F16 for GEMM
        return dtype;
    }

    /// <summary>Maps a HartsyInference dtype to its cuBLAS data-type constant for use with
    /// <c>cublasGemmEx</c> / <c>cublasGemmStridedBatchedEx</c>. Handles F16, BF16, and F32;
    /// throws on anything else (FP8 should be cast to F16/BF16 via <see cref="CastIfNeeded"/>
    /// before reaching cuBLAS).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CublasDataType(DType dtype)
    {
        if (dtype == DType.F16) return CublasApi.CUDA_R_16F;
        if (dtype == DType.BF16) return CublasApi.CUDA_R_16BF;
        if (dtype == DType.F32) return CublasApi.CUDA_R_32F;
        throw new NotSupportedException($"cuBLAS GEMM does not support dtype {dtype}.");
    }

    /// <summary>Resolves the GEMM compute dtype for an op with two operands. Prefer F16 over F32 whenever either operand is F16 (or fp8, which casts to F16). cublasGemmEx COMPUTE_32F does not support F32×F32→F16, so when an F16 activation feeds an F16 output and the weight happens to be F32, the F32 weight must cast down to F16 — F16 also gets Tensor Core acceleration on Ampere+.</summary>
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

    /// <summary>Ensures a GPU buffer holding a tensor of <paramref name="srcDtype"/> is
    /// available in <paramref name="dstDtype"/>. Returns the existing pointer if no cast
    /// is needed, or allocates + casts and writes the new dptr to <paramref name="castOut"/>
    /// (which the caller is responsible for freeing with <c>cuMemFreeAsync</c>). Hides the
    /// F8 special case so the four GEMM call sites all look the same.</summary>
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

    /// <summary>Casts GPU data between dtypes using PTX kernels. Handles F8↔F16, F16↔F32 conversions, and GGUF quantized → F16/F32 dequant (Q8_0 / Q4_K / Q5_K / Q6_K).</summary>
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

    /// <summary>Dispatches the right per-DType GGUF dequant kernel. Element count must respect the block size of the source dtype (32 for Q8_0, 256 for Q*_K).</summary>
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

    /// <summary>Test hook for the native-fp8 activation quantization kernels: computes the per-tensor e4m3 dequant
    /// scale (<c>amax/448</c>) into <paramref name="scaleOut"/> (1-element F32) and the quantized bytes into
    /// <paramref name="fp8Out"/> (same element count as <paramref name="input"/>, F8E4M3). Runs on any CUDA GPU —
    /// the kernels are plain compute (only the GEMM needs Ada) — so the Ampere CI box can validate them.</summary>
    internal void Fp8QuantizeActivationForTest(Tensor fp8Out, Tensor scaleOut, Tensor input)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>Synchronizes the default compute stream. Only needed at pipeline boundaries or before explicit D2H.
    /// Per-op sync removed — CUDA guarantees sequential execution on a single blocking stream.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sync()
    {
        _context.EnsureCurrent();
        CudaDriverApi.cuStreamSynchronize(_stream.Handle).ThrowOnError();
        // HARTSY_PROFILE_EACH=1: dump the accumulated per-op profile at each Sync (end of a generation) — the Swarm
        // ShutdownServer path does not reliably dispose the backend, so this is the reliable per-gen dump hook.
        if (Environment.GetEnvironmentVariable("HARTSY_PROFILE_EACH") == "1")
            Profiling.NvtxRange.DumpProfile(Environment.GetEnvironmentVariable("HARTSY_PROFILE_OUT") ?? "/tmp/hartsy_profile.txt");
    }

    #endregion

    #region Transpose / Permute

    /// <summary>Batched 2D transpose: [B, D1, D2] -> [B, D2, D1] via PTX kernel.</summary>
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>GQA K/V head repeat (block pattern): [B, Hkv, L, D] → [B, Hkv*groupSize, L, D]. GPU-resident
    /// (device-to-device gather), so the decode loop never leaves the device for the grouped-query expand.</summary>
    public void RepeatKvHeads(Tensor output, Tensor input, int kvHeads, int groupSize)
    {
        _context.EnsureCurrent();
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

            if (input.DType == DType.F16)
                _kernels!.LaunchRepeatKvF16(pOut, pIn, kvHeads, groupSize, seqLen, headDim, output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchRepeatKv(pOut, pIn, kvHeads, groupSize, seqLen, headDim, output.ElementCount, _stream.Handle);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            GpuTransferHelper.FreeDevice(pIn);
        }
    }

    /// <summary>FlashAttention (online-softmax, GQA-aware, no materialized score matrix). Requires F32 and a
    /// power-of-two head dimension (true for the 64/128 head dims we run); falls back to the base CPU reference
    /// otherwise.</summary>
    public unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        // The kernel pads the block to the next power of two >= D, so any D up to 1024 works (Phi-3 D=96 etc.).
        bool kernelOk = d > 0 && d <= 1024;
        if (query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32 || output.DType != DType.F32 || !kernelOk
            || (sink is not null && sink.DType != DType.F32) || (alibiSlopes is not null && alibiSlopes.DType != DType.F32))
        {
            AttentionReference.FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink, slidingWindow, alibiSlopes);
            return;
        }

        _context.EnsureCurrent();
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
            bool plain = pSink == 0 && pAlibi == 0 && softcap <= 0f && slidingWindow <= 0;
            bool forceSplit = EnvFlag("HARTSY_FLASH_SPLIT_FORCE");
            // Occupancy-limited = the LLM decode case (tq=1, few heads → e.g. 16 blocks on 28 SMs). Splitting the
            // key axis there fills the GPU and is a large decode win (attention was ~30% of decode; split-K ≈ +38%
            // end-to-end on Qwen3). The split kernel is numerically exact vs monolithic (online-softmax merge). The
            // old kvLen≥1024 floor never engaged for decode (kvLen<300); gate on occupancy instead, floor 128.
            bool occLimited = baseBlocks < 2 * _context.MultiprocessorCount;
            int splits = 1;
            if (!EnvFlag("HARTSY_FLASH_SPLIT_OFF")
                && plain && (forceSplit || occLimited) && kvLen >= (forceSplit ? 8 : 128))
            {
                int target = forceSplit ? 4 * baseBlocks : 2 * _context.MultiprocessorCount;
                int g = (target + baseBlocks - 1) / baseBlocks;
                int minChunk = forceSplit ? 1 : (occLimited ? 32 : 256);
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
                        kvGroup <= 0 ? 1 : kvGroup, causal, qOffset, scale, splits, chunk, _stream.Handle);
                    _kernels!.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, tq, d, splits, _stream.Handle);
                }
                finally
                {
                    GpuTransferHelper.FreeDevice(pM); GpuTransferHelper.FreeDevice(pL); GpuTransferHelper.FreeDevice(pAcc);
                }
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
        }
    }

    /// <summary>In-place KV-cache append (device-to-device strided write); keeps the fixed-capacity cache buffer
    /// GPU-resident with no reallocation.</summary>
    public unsafe void KvCacheAppend(Tensor buffer, Tensor newKv, int offset)
    {
        if (buffer.DType != DType.F32 || newKv.DType != DType.F32)
        {
            ((IBackend)this).KvCacheAppend(buffer, newKv, offset);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();
        int h = (int)buffer.Shape[1], maxSeq = (int)buffer.Shape[2], d = (int)buffer.Shape[3], tNew = (int)newKv.Shape[2];
        ulong pBuf = 0, pNew = 0;
        try
        {
            pBuf = GpuTransferHelper.CopyToDevice(buffer);
            pNew = GpuTransferHelper.CopyToDevice(newKv);
            _kernels!.LaunchKvAppend(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle);
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

    /// <summary>Extracts a contiguous time-range from a KV-shaped tensor (device-to-device strided read); used
    /// by the paged KV cache to split multi-token appends across page boundaries and to gather a
    /// partially-filled page's occupied prefix.</summary>
    public unsafe void SliceTimeRange(Tensor output, Tensor input, int start, int len)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            ((IBackend)this).SliceTimeRange(output, input, start, len);
            return;
        }
        _context.EnsureCurrent();
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

    private static unsafe ulong UploadInts(ReadOnlySpan<int> values)
    {
        nuint bytes = (nuint)((long)values.Length * sizeof(int));
        ulong dptr = GpuTransferHelper.AllocateDevice(bytes);
        fixed (int* p = values) CudaDriverApi.cuMemcpyHtoD(dptr, (nint)p, bytes).ThrowOnError();
        return dptr;
    }

    // ── Device-side decode position (autoregressive step-graph replay) ──────────────────────────────────────
    public bool GraphDecodeSupported => true;

    /// <summary>Persistent 2-int device buffer {kvLen, qOffset} the graph-decode kernels read each step. Allocated
    /// once (outside capture) via the synchronous persistent allocator so it survives graph AUTO_FREE_ON_LAUNCH.</summary>
    public ulong AllocDevicePos()
    {
        _context.EnsureCurrent();
        return CudaMemory.AllocatePersistent((nuint)(2 * sizeof(int)));
    }

    /// <summary>Refreshes the device position buffer for the next step. Must be called OUTSIDE a capture region
    /// (the captured graph reads this buffer's address; only its contents change per step).</summary>
    public unsafe void WriteDevicePos(ulong handle, int kvLen, int qOffset)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        int* v = stackalloc int[2]; v[0] = kvLen; v[1] = qOffset;
        // HtoDAsync from pageable host memory stages synchronously (host buffer consumed before return), so the
        // stackalloc is safe; the device write is stream-ordered before the subsequent kernels/graph launch.
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)v, (nuint)(2 * sizeof(int)), _stream.Handle).ThrowOnError();
    }

    public void FreeDevicePos(ulong handle)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        CudaMemory.Free(handle);
    }

    /// <summary>KV-cache append with the write slot read from <paramref name="devicePos"/> (graph-replayable).</summary>
    public unsafe void KvCacheAppendDev(Tensor buffer, Tensor newKv, int offset, ulong devicePos)
    {
        if (devicePos == 0 || buffer.DType != DType.F32 || newKv.DType != DType.F32)
        {
            KvCacheAppend(buffer, newKv, offset);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();
        // Flatten batch into heads: buffer [B,H,maxSeq,D] is contiguous == [B*H, maxSeq, D], so B*H heads appends
        // every batch element at the same slot (cond+uncond share the position). B=1 → unchanged for all callers.
        int h = (int)(buffer.Shape[0] * buffer.Shape[1]), maxSeq = (int)buffer.Shape[2], d = (int)buffer.Shape[3], tNew = (int)newKv.Shape[2];
        ulong pBuf = 0, pNew = 0;
        try
        {
            pBuf = GpuTransferHelper.CopyToDevice(buffer);
            pNew = GpuTransferHelper.CopyToDevice(newKv);
            _kernels!.LaunchKvAppend(pBuf, pNew, h, maxSeq, tNew, d, offset, _stream.Handle, devicePos);
            buffer._gpuSyncCallback = null;
            buffer._gpuDisposeCallback = null;
            GpuTransferHelper.CacheActivation(buffer, pBuf, GpuTransferHelper.ByteSize(buffer));
        }
        finally
        {
            GpuTransferHelper.FreeDevice(pNew);
        }
    }

    /// <summary>Self-attention FlashAttention with kvLen/qOffset read from <paramref name="devicePos"/>. Uses the
    /// split-K decode path with a FIXED split count (chunk sized to the cache CAPACITY, not the current kvLen), so
    /// the grid is position-independent and graph-capturable while still filling the GPU at long context: splits
    /// whose chunk starts past the current kvLen early-exit in the kernel, so active parallelism grows with position.</summary>
    public unsafe void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos)
    {
        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        bool kernelOk = d > 0 && d <= 1024;
        if (devicePos == 0 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32 || output.DType != DType.F32 || !kernelOk)
        {
            FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();

        // Fixed split count from CAPACITY (lk) so the grid never changes across steps. Target ~4× SM occupancy;
        // chunk = ceil(capacity / splits) is constant, so early steps leave later splits empty (they early-exit).
        // Split-K is gated to b==1: the split/combine kernels have a latent batch>1 bug (they were only exercised by
        // the b=1 LLM decode), and for CFG-batched decode b=2 already yields 2·heads blocks — enough to fill the GPU
        // with the monolithic kernel — so batching keeps the (correct) monolithic path with no occupancy loss.
        int baseBlocks = b * hq * tq;
        int splits = 1;
        if (b == 1 && lk >= 128)
        {
            int target = 4 * _context.MultiprocessorCount;
            int g = (target + baseBlocks - 1) / baseBlocks;
            int maxG = Math.Max(1, lk / 64);   // keep chunks ≥ 64 keys
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
                        grp, causal, qOffset, scale, splits, chunk, _stream.Handle, devicePos);
                    _kernels!.LaunchFlashAttentionCombine(pOut, pM, pL, pAcc, b, hq, tq, d, splits, _stream.Handle);
                }
                finally
                {
                    GpuTransferHelper.FreeDevice(pM); GpuTransferHelper.FreeDevice(pL); GpuTransferHelper.FreeDevice(pAcc);
                }
            }
            else
            {
                _kernels!.LaunchFlashAttention(pOut, pQ, pK, pV, b, hq, tq, d, hkv, lk, kvLen, grp,
                    causal, qOffset, scale, 0f, 0, 0, 0, _stream.Handle, devicePos);
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

    /// <summary>DtoH a resident device tensor WITHOUT freeing its buffer (a normal DataPointer read syncs then frees,
    /// which would break a captured graph's fixed output-buffer address). Stream-syncs first so the pending launch
    /// has produced the data.</summary>
    public unsafe void ReadResidentInto(Tensor src, float[] dst)
    {
        _context.EnsureCurrent();
        ulong p = GpuTransferHelper.CopyToDevice(src);   // resident activation → cached ptr (no re-upload, no free)
        _stream.Synchronize();
        fixed (float* d = dst)
            CudaDriverApi.cuMemcpyDtoH((nint)d, p, (nuint)((long)dst.Length * sizeof(float))).ThrowOnError();
    }

    // ── LLM decode-graph: RoPE / embed / argmax device state ────────────────────────────────────────────────
    // See IBackend's matching doc comments for the overall design. This backend implements all of it.

    public ulong AllocDeviceTokenId()
    {
        _context.EnsureCurrent();
        return CudaMemory.AllocatePersistent((nuint)sizeof(int));
    }

    public unsafe void WriteDeviceTokenId(ulong handle, int tokenId)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        int v = tokenId;
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)(&v), (nuint)sizeof(int), _stream.Handle).ThrowOnError();
    }

    public void FreeDeviceTokenId(ulong handle)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        CudaMemory.Free(handle);
    }

    /// <summary>D2H sync read of the 1-int device token-id buffer. Blocking (cuMemcpyDtoH) — the caller does this
    /// once per decode step, after LaunchGraph, so the sync waits on exactly the work that step's graph replay
    /// queued (not the whole stream's backlog).</summary>
    public unsafe int ReadDeviceTokenId(ulong handle)
    {
        if (handle == 0) throw new NotSupportedException("ReadDeviceTokenId called with an unallocated buffer.");
        _context.EnsureCurrent();
        int v;
        CudaDriverApi.cuMemcpyDtoH((nint)(&v), handle, (nuint)sizeof(int)).ThrowOnError();
        return v;
    }

    /// <summary>Builds the table once and registers it as a permanent (weight-cache) resident tensor pair —
    /// stable across graph capture/replay and never evicted, exactly like a model weight. Same math as
    /// GenericTransformer.BuildRope, just for every position 0..maxPos-1 up front instead of one small
    /// per-step tensor.</summary>
    public unsafe (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, RopeScaling scaling)
    {
        int rdim = rotaryDim > 0 && rotaryDim < headDim ? rotaryDim : headDim;
        int half = rdim / 2;
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(rdim, theta, scaling, maxPos);
        Tensor cos = new(new TensorShape(maxPos, headDim), DType.F32);
        Tensor sin = new(new TensorShape(maxPos, headDim), DType.F32);
        float* pc = (float*)cos.DataPointer;
        float* ps = (float*)sin.DataPointer;
        for (int pos = 0; pos < maxPos; pos++)
        {
            long baseOff = (long)pos * headDim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale);
                float s = (float)(Math.Sin(angle) * mscale);
                pc[baseOff + i] = c; pc[baseOff + i + half] = c;
                ps[baseOff + i] = s; ps[baseOff + i + half] = s;
            }
        }
        PreloadWeights([cos, sin]);
        return (cos, sin);
    }

    public void RopeApplyDecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos)
    {
        if (devicePos == 0 || x.DType != DType.F32 || cosTable.DType != DType.F32 || sinTable.DType != DType.F32)
        {
            ((IBackend)this).RopeApplyDecodeStep(x, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        EnsureKernels();
        int c = (int)input.Shape[input.Shape.Rank - 1];
        int rows = (int)(input.ElementCount / c);
        ulong pIn = GpuTransferHelper.CopyToDevice(input);
        _kernels!.LaunchArgMaxLastDim(outputTokenId, pIn, rows, c, _stream.Handle);
    }

    // ── Graph-capture decode: repetition penalty ────────────────────────────────────────────────────────────

    public ulong AllocDeviceHistory(int capacity)
    {
        _context.EnsureCurrent();
        return CudaMemory.AllocatePersistent((nuint)(Math.Max(1, capacity) * sizeof(int)));
    }

    public void FreeDeviceHistory(ulong handle)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        CudaMemory.Free(handle);
    }

    public ulong AllocDeviceCounter()
    {
        _context.EnsureCurrent();
        return CudaMemory.AllocatePersistent((nuint)sizeof(int));
    }

    public void FreeDeviceCounter(ulong handle)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        CudaMemory.Free(handle);
    }

    public unsafe void WriteDeviceCounter(ulong handle, int value)
    {
        if (handle == 0) return;
        _context.EnsureCurrent();
        int v = value;
        CudaDriverApi.cuMemcpyHtoDAsync(handle, (nint)(&v), (nuint)sizeof(int), _stream.Handle).ThrowOnError();
    }

    public void AppendTokenHistoryStep(ulong history, ulong historyCount, ulong tokenId)
    {
        if (history == 0 || historyCount == 0 || tokenId == 0) return;
        _context.EnsureCurrent();
        EnsureKernels();
        _kernels!.LaunchHistoryAppend(history, historyCount, tokenId, _stream.Handle);
    }

    public void ApplyRepetitionPenaltyStep(Tensor logits, ulong history, ulong historyCount, float penalty)
    {
        if (history == 0 || historyCount == 0 || logits.DType != DType.F32)
        {
            throw new NotSupportedException("ApplyRepetitionPenaltyStep requires F32 logits and valid device history buffers.");
        }
        _context.EnsureCurrent();
        EnsureKernels();
        int vocabSize = (int)logits.Shape[logits.Shape.Rank - 1];
        ulong pLogits = GpuTransferHelper.CopyToDevice(logits);
        _kernels!.LaunchRepetitionPenalty(pLogits, history, historyCount, penalty, vocabSize, _stream.Handle);
        GpuTransferHelper.CacheActivation(logits, pLogits, GpuTransferHelper.ByteSize(logits));
    }

    /// <summary>Backend-agnostic graph capture (see IBackend). Replays require repeated launches (a decode
    /// loop), so the graph is instantiated with auto-free-on-relaunch — validated on hardware
    /// (docs/Research/CUDA_GRAPH_FINDINGS.md): per-op stream-ordered activation allocations captured inside
    /// recordWork are freed before each relaunch and reuse the same virtual addresses, so pointers cached at
    /// capture time (e.g. the graph-decode session's cosTable/embedTable/devicePos) stay valid across replays.</summary>
    public object? CaptureGraph(Action recordWork)
    {
        _context.EnsureCurrent();
        CudaGraph graph = new(_stream.Handle, autoFreeAllocationsOnRelaunch: true);
        graph.Capture(recordWork);
        return graph;
    }

    public void LaunchGraph(object graphHandle)
    {
        _context.EnsureCurrent();
        ((CudaGraph)graphHandle).Launch();
    }

    public void DisposeGraph(object graphHandle) => ((CudaGraph)graphHandle).Dispose();

    private static unsafe ulong UploadFloats(ReadOnlySpan<float> values)
    {
        nuint bytes = (nuint)((long)values.Length * sizeof(float));
        ulong dptr = GpuTransferHelper.AllocateDevice(bytes);
        fixed (float* p = values) CudaDriverApi.cuMemcpyHtoD(dptr, (nint)p, bytes).ThrowOnError();
        return dptr;
    }

    /// <summary>MoE row-gather: output[m] = input[rowIndices[m]] (collect an expert's routed tokens).</summary>
    public unsafe void GatherRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            ((IBackend)this).GatherRows(output, input, rowIndices);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();
        int k = (int)input.Shape[input.Shape.Rank - 1];
        ulong total = (ulong)rowIndices.Length * (ulong)k;
        ulong pIn = 0, pIdx = 0, pOut = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = UploadInts(rowIndices);
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
        }
    }

    /// <summary>Per-row argmax over the last dim: indices[r] = argmax_c input[r,c]. The reduction runs on-device
    /// (one block per row); only the resulting indices (one int per row) sync back, not the full logit rows.</summary>
    public unsafe void ArgMaxLastDim(Tensor indices, Tensor input)
    {
        if (input.DType != DType.F32 || indices.DType != DType.I32)
        {
            ((IBackend)this).ArgMaxLastDim(indices, input);
            return;
        }
        _context.EnsureCurrent();
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
            _kernels!.LaunchArgMaxLastDim(pOut, pIn, rows, c, _stream.Handle);
            GpuTransferHelper.CacheActivation(indices, pOut, outBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
        }
    }

    /// <summary>MoE weighted scatter-add (in place): output[rowIndices[m]] += scales[m]·input[m]. Output must be a
    /// resident, already-accumulating activation (pre-zeroed, then one call per expert).</summary>
    public unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            ((IBackend)this).ScatterAddWeightedRows(output, input, rowIndices, scales);
            return;
        }
        _context.EnsureCurrent();
        EnsureKernels();
        int k = (int)input.Shape[input.Shape.Rank - 1];
        ulong total = (ulong)rowIndices.Length * (ulong)k;
        ulong pIn = 0, pIdx = 0, pScale = 0, pOut = 0;
        try
        {
            pOut = GpuTransferHelper.CopyToDevice(output);   // resident accumulator (cache hit)
            pIn = GpuTransferHelper.CopyToDevice(input);
            pIdx = UploadInts(rowIndices);
            pScale = UploadFloats(scales);
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

    /// <summary>GEGLU activation: output[i] = input[i] * GELU(input[i + outputElements]) via PTX kernel.</summary>
    public void GeGlu(Tensor output, Tensor input)
    {
        _context.EnsureCurrent();
        EnsureKernels();

        ulong pOut = 0, pIn = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            int innerDim = (int)output.Shape[output.Shape.Rank - 1];
            if (input.DType == DType.F16)
                _kernels!.LaunchGeGluF16(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);
            else
                _kernels!.LaunchGeGlu(pOut, pIn, innerDim, (int)output.ElementCount, _stream.Handle);

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
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
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

    /// <summary>Splits <paramref name="input"/> into <paramref name="outputs"/> along <paramref name="dim"/>. Delegates to the CPU kernel — Split is pure memcpy and rare in our pipelines (one call per VAE encode), so the GPU round-trip is acceptable. TODO: GPU-native Split via cuMemcpyDtoDAsync.</summary>
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        HartsyInference.Cpu.Kernels.ElementWiseKernels.Split(outputs, input, dim);
    }

    #endregion

    #region Sampling

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
        _context.EnsureCurrent();
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
        throw new NotImplementedException("CUDA UpsampleBilinear2D not yet implemented");
    }

    #endregion

    #region Data Movement

    /// <summary>Copies tensor data between host and device or device to device.</summary>
    public unsafe void CopyTo(Tensor destination, Tensor source)
    {
        _context.EnsureCurrent();
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

    /// <summary>Fills a tensor with a constant float value. Works on CPU tensors directly.</summary>
    public unsafe void Fill(Tensor tensor, float value)
    {
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
    public void PreloadWeights(IEnumerable<Tensor> weights)
    {
        _context.EnsureCurrent();
        foreach (Tensor weight in weights)
        {
            GpuTransferHelper.PreloadWeight(weight);
        }
    }

    /// <summary>Frees specific weight tensors from GPU to reclaim VRAM (e.g., UNet weights before VAE decode).</summary>
    public void FreeWeights(IEnumerable<Tensor> weights)
    {
        _context.EnsureCurrent();
        GpuTransferHelper.FreeWeights(weights);
    }

    public void FreeActivations()
    {
        _context.EnsureCurrent();
        // A captured step graph bakes activation-pool device pointers (fixed latent / velocity buffers) —
        // freeing activations under it leaves the graph pointing at freed memory, and the next replay is a
        // context-poisoning CUDA 700. Reset the graph slot here so cross-generation graphs (Chroma) survive
        // ONLY as long as their buffers do; owners detect the external reset and re-warm.
        StepGraphInvalidateForActivationFree();
        GpuTransferHelper.FreeActivations();
    }

    public void FreeActivations(bool trimPool)
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        GpuTransferHelper.TrimPool();
    }

    public void FreeAllDeviceMemory()
    {
        _context.EnsureCurrent();
        long f0 = (long)_context.GetMemoryInfo().freeBytes;
        StepGraphInvalidateForActivationFree();
        // EvictAll clears weights + casts + activations (syncing the stream first); the trim then returns the
        // stream-ordered pool's reservations so cuMemGetInfo/persistent allocs see the memory as actually free.
        GpuTransferHelper.EvictAll();
        GpuTransferHelper.TrimPool();
        long f1 = (long)_context.GetMemoryInfo().freeBytes;
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] FreeAllDeviceMemory: free {f0 >> 20} MB → {f1 >> 20} MB");
    }

    public long FreeMemoryBytes()
    {
        _context.EnsureCurrent();
        return (long)_context.GetMemoryInfo().freeBytes;
    }

    /// <summary>Frees all preloaded weight memory from GPU and clears the cache.</summary>
    public void FreePreloadedWeights()
    {
        _context.EnsureCurrent();
        GpuTransferHelper.FreeAllCached();
    }

    /// <summary>Evicts all cached GPU weight buffers. Call between pipeline stages to free VRAM.</summary>
    public void EvictGpuCache()
    {
        _context.EnsureCurrent();
        GpuTransferHelper.EvictAll();
    }

    /// <summary>Returns GPU cache stats: (cachedBytes, hits, misses).</summary>
    public (long cachedBytes, long hits, long misses) GetGpuCacheStats()
    {
        return GpuTransferHelper.GetStats();
    }

    /// <summary>Number of lazy device-to-host syncs fired since the last <see cref="ResetD2hSyncCount"/>.
    /// Each is a full GPU stall plus a copy; a GPU-resident denoise loop should fire none. Residency metric.</summary>
    public long GetD2hSyncCount() => GpuTransferHelper.GetSyncCount();

    /// <summary>Resets the device-to-host sync counter (call before a region you want to measure for residency).</summary>
    public void ResetD2hSyncCount() => GpuTransferHelper.ResetSyncCount();

    /// <summary>Foundation check for graph-based decode: captures a Scale kernel over a stable persistent
    /// buffer into a CUDA graph, replays it, then changes the input buffer's content and replays the SAME
    /// graph again. A working capture/replay returns (input0·3, input1·3) = (6, 15) — proving replay reads
    /// live buffer content (stable pointers) and the async-pool memory model is capture-compatible.</summary>
    public unsafe (float first, float second) GraphSmokeTest()
    {
        _context.EnsureCurrent();
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
        _context.EnsureCurrent();
        return CudaMemory.GetMemInfo();
    }

    #endregion

    #region Disposal

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;

            if (NvtxRange.ProfileEnabled)
                NvtxRange.DumpProfile(Environment.GetEnvironmentVariable("HARTSY_PROFILE_OUT") ?? "/tmp/hartsy_profile.txt");

            // Bind context on the disposing thread so cuMemFree / cublasDestroy /
            // cuStreamDestroy don't hit CUDA_ERROR_INVALID_CONTEXT. Disposal can run
            // on the finalizer thread or a different worker than constructed us.
            _context.EnsureCurrent();

            GpuTransferHelper.EvictAll();
            _streamingCache.UnregisterPinnedSources();
            _kernels?.Dispose();

            if (_fp8Executor is not null)
            {
                _fp8Executor.Dispose();
                _fp8Executor = null;
            }

            if (_ltGemmExecutor is not null)
            {
                _ltGemmExecutor.Dispose();
                _ltGemmExecutor = null;
            }

            if (_tensorCoreGemm is not null)
            {
                _tensorCoreGemm.Dispose();
                _tensorCoreGemm = null;
            }

            if (_cudnnSdpa is not null)
            {
                _cudnnSdpa.Dispose();
                _cudnnSdpa = null;
            }

            if (_cudnnConv is not null)
            {
                _cudnnConv.Dispose();
                _cudnnConv = null;
            }

            if (_stepGraph is not null)
            {
                _stepGraph.Dispose();
                _stepGraph = null;
            }

            if (_cublasHandle != 0)
            {
                CublasApi.cublasDestroy(_cublasHandle);
                _cublasHandle = 0;
            }

            // Drop this backend's per-context registrations before tearing the context down, so a later
            // backend (or a same-device re-construction) never resolves to freed stream handles.
            bool lastOwner = GpuTransferHelper.Unregister(_context);
            CudaMemory.RemoveComputeStream(_context);
            // Discard (don't run) finalizer cleanups still queued under this context's handle: EvictAll above
            // already freed everything they could reference, and CUDA reuses primary-context handles — a later
            // same-device backend would otherwise drain these stale callbacks during ITS construction. Only when
            // this backend was the handle's last owner: a live same-device sibling shares the handle (and queue).
            if (lastOwner)
            {
                HartsyInference.Core.Tensors.Tensor.DiscardPendingFinalizerGpuCleanup(_context.Handle);
            }

            // Order: upload stream first (no other code holds events on it after
            // EvictAll above), then compute stream, then context.
            _uploadStream.Dispose();
            _stream.Dispose();
            _context.Dispose();
        }
        GC.SuppressFinalize(this);
    }

    ~CudaBackend()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_cublasHandle != 0)
            {
                CublasApi.cublasDestroy(_cublasHandle);
                _cublasHandle = 0;
            }
        }
    }

    #endregion
}
