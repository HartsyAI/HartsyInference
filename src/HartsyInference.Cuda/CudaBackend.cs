using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
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
    /// <summary>This backend's transfer state — its identity in <see cref="GpuTransferHelper"/>, bound to the
    /// calling thread by <see cref="EnterOp"/> at every op entry. Two same-device backends share a primary context
    /// but never a state.</summary>
    private readonly GpuTransferHelper.State _transferState;
    private readonly CudaKernels? _kernels;
    private nint _cublasHandle;
    private Fp8GemmExecutor? _fp8Executor;
    private LtGemmExecutor? _ltGemmExecutor;
    private TensorCoreGemm? _tensorCoreGemm;
    private CudnnSdpa? _cudnnSdpa;
    private bool _cudnnSdpaDead;   // set if cuDNN INIT throws once — never retry, fall back for the session

    /// <summary>Per-head-dim failure/backoff state — replaces a plain permanent-dead set.</summary>
    /// <remarks>A structural failure (e.g. D=256 on a build whose fused engine tops out at 128 — <see
    /// cref="CudnnStatusException.IsPermanent"/>) disables that dim forever, same as before; a transient failure
    /// (e.g. the host-RAM allocation error that motivated this — see the plan notes referenced from <see
    /// cref="TryCudnnSdpa"/>) gets bounded backoff instead, since the resource pressure that caused it is typically
    /// external to this process and does clear up. <see cref="ConcurrentDictionary{TKey,TValue}"/> (not a plain
    /// <c>HashSet</c>) because this backend is a process-wide DI singleton and nothing enforces
    /// <c>InferenceQueue.MaxConcurrency</c> stays at its default of 1 — a plain <c>HashSet</c> was never actually
    /// safe under a higher setting.</remarks>
    private sealed class DimFailureState
    {
        public int ConsecutiveFailures;
        public long NextRetryAtTicks;   // Environment.TickCount64; 0 = eligible immediately
        public bool Permanent;
    }
    private readonly ConcurrentDictionary<long, DimFailureState> _cudnnSdpaDimState = new();

    /// <summary>Test-only fault hook for <see cref="TryCudnnSdpa"/>: a non-null return from this is thrown instead of the real call.</summary>
    /// <remarks>Exists so the classify/retry/backoff behavior can be tested deterministically — a real
    /// host-RAM-starvation failure can't be reproduced on demand in a fast unit test. Null in production.</remarks>
    internal Func<long, CudnnStatusException?>? TestCudnnSdpaFaultInjector { get; set; }

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

    /// <summary>True once the cuDNN fused-attention fast path has run at least once this session (confirms engagement vs fallback).</summary>
    public bool CudnnSdpaEngaged { get; private set; }

    /// <summary>True once the cuDNN convolution fast path has run at least once — same diagnostic role as <see cref="CudnnSdpaEngaged"/>.</summary>
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
            _fp8Executor ??= new Fp8GemmExecutor(_context.ComputeCapabilityMajor, _context.ComputeCapabilityMinor);
            return _fp8Executor;
        }
    }

    /// <summary>Opt-in flag for fusing the Linear bias add into the cuBLASLt GEMM epilogue. Defaults to <c>false</c> pending benchmarks.</summary>
    /// <remarks>Works on every targeted SM, including the RTX 3060. When on, a Linear with bias runs as a single
    /// <c>cublasLtMatmul</c> instead of <c>cublasGemmEx</c> + a separate <c>BiasAdd</c> launch. The result is
    /// numerically equivalent to the unfused path.</remarks>
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
            _int8Executor ??= new Int8GemmExecutor();
            return _int8Executor;
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

    /// <summary>Sets (or replaces) the SmoothQuant per-input-channel scale s[K] for <paramref name="weight"/>: X_hat = X/s, W_hat = W*s.</summary>
    /// <remarks>Product-preserving pre-quantization — migrates activation outlier difficulty into the weight (see
    /// src/HartsyInference.Cuda/Kernels/dequant/w8a8.cu's invScale param). Must be called BEFORE the weight's first W8A8 use to take
    /// effect on the initial quantization; calling it after evicts the already-cached quantized weight so the NEXT
    /// use re-quantizes smoothed (safe, just a re-pay of the one-time quant cost). <paramref name="s"/> length must
    /// equal the weight's K (in-dim).</remarks>
    public unsafe void SetW8A8SmoothingScale(Core.Tensors.Tensor weight, ReadOnlySpan<float> s)
    {
        int k = (int)weight.Shape[1];
        if (s.Length != k)
            throw new ArgumentException($"SmoothQuant scale length {s.Length} != weight K={k}.", nameof(s));
        float[] invScale = new float[k];
        for (int i = 0; i < k; i++) invScale[i] = s[i] != 0f ? 1f / s[i] : 0f;

        ulong dev = GpuTransferHelper.AllocateDevice((nuint)(k * sizeof(float)));
        fixed (float* p = invScale)
            CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)p, (nuint)(k * sizeof(float)), _stream.Handle).ThrowOnError();
        _stream.Synchronize();

        if (_w8a8SmoothInvScaleDevice.TryGetValue(weight, out ulong old)) GpuTransferHelper.FreeDevice(old);
        _w8a8SmoothInvScaleDevice[weight] = dev;
        _w8a8SmoothScaleHost[weight] = s.ToArray();

        if (_w8a8WeightCache.TryGetValue(weight, out ulong cachedQuant))
        {
            GpuTransferHelper.FreeDevice(cachedQuant);
            _w8a8WeightCache.Remove(weight);
        }
    }

    /// <summary>Frees every SmoothQuant device scale buffer (mirrors FreeW8A8Cache's scope/callers).</summary>
    private void FreeW8A8SmoothScaleCache()
    {
        foreach (ulong ptr in _w8a8SmoothInvScaleDevice.Values)
            GpuTransferHelper.FreeDevice(ptr);
        _w8a8SmoothInvScaleDevice.Clear();
        _w8a8SmoothScaleHost.Clear();
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
            _ltGemmExecutor ??= new LtGemmExecutor();
            return _ltGemmExecutor;
        }
    }

    /// <summary>Opt-in flag for the hand-written tensor-core HGEMM in the F16 Linear path. Defaults to <c>false</c>.</summary>
    /// <remarks>The kernel is validated bit-exact against cuBLAS (<c>TensorCoreGemmTests</c>) but is the unoptimized
    /// one-warp-per-tile baseline, so it is opt-in pending a perf comparison against cuBLAS on the target GPU. Only
    /// dispatches when operands and output are F16 and dimensions are aligned (M%16, N%8, K%16 == 0); otherwise
    /// falls through to cuBLAS.</remarks>
    public bool EnableTensorCoreGemm { get; set; }

    /// <summary>Fused BF16/F16 decode GEMV for small-m (≤8) F32-activation matmuls; replaces cuBLAS GemmEx (slow at m=1). On by default.</summary>
    /// <remarks>Faster and at least as accurate as the cuBLAS BF16 path (activations stay F32). Set
    /// <c>HARTSY_BF16_GEMV=0</c> to fall back to cuBLAS for A/B.</remarks>
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

    /// <summary>Reads a <c>HARTSY_*</c> boolean env var; unset or any non-"1" value is treated as off.</summary>
    /// <remarks>Mirrors <see cref="Profiling.NvtxRange.ProfileEnabled"/>.</remarks>
    private static bool EnvFlag(string name) => Environment.GetEnvironmentVariable(name) == "1";

    /// <summary>SageAttention INT8 dispatch preference — DEFAULT ON since 2026-07-23 (kill switch <c>HARTSY_SAGE_ATTN=0</c>).</summary>
    /// <remarks>E2E gates behind the flip: Wan-14B +4.7%/step, Hunyuan-720p +10.2%/step at 0.15–0.5% pixel drift
    /// (eyeball-clean, deterministic); micro 1.18× vs cuDNN-F16-fused at 16k²/D=128 and 3–8× vs the F32 paths;
    /// crossover predictions validated on 5 models (2026-07-22 records). The per-site shape gates keep every
    /// measured-loss shape (small seq, masked, D∉{64,128}) on its previous-best path — image models are untouched
    /// by construction (F16-native ≥8192-Skv gate; D=256 excluded).</remarks>
    private static readonly bool UseSageAttn = Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN") != "0";

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
        _context = new CudaContext(deviceOrdinal);
        // Must use blocking stream (CU_STREAM_DEFAULT) because GpuTransferHelper uses synchronous
        // cuMemcpyHtoD/DtoH which operate on the NULL stream. A non-blocking stream does NOT
        // synchronize with the NULL stream, causing race conditions where kernels read incomplete
        // data from in-progress H2D transfers. Fix: switch to cuMemcpyHtoDAsync on this stream.
        _stream = new CudaStream(nonBlocking: false);
        // HARTSY_PROFILE_SYNC's per-op GPU-time attribution resolves ITS stream from the ambient backend State
        // (see NvtxRange.Dispose) — no registration needed here.
        // Upload stream is non-blocking so its in-flight work doesn't gate the compute
        // stream's NULL-stream "wait for everything" semantics — without that, prefetched
        // uploads would force compute to wait, defeating overlap. The streaming cache uses
        // explicit cuEventRecord/cuStreamWaitEvent for the parts that *do* need to sync.
        _uploadStream = new CudaStream(nonBlocking: true);
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
        bool mempoolKeep = EnvSwitch.IsEnabled("HARTSY_MEMPOOL_KEEP", defaultOn: true);
        DeviceMempoolPolicy.Acquire(_context.DeviceOrdinal, mempoolKeep);

        // Perf-flag wiring, two tiers (docs/PERFORMANCE.md). STANDARD PROFILE features default ON via
        // tri-state EnvSwitch (unset → documented default, "0"/"false" is the kill-switch) so every install
        // reproduces the published benchmark times with zero configuration. EXPERIMENTAL switches keep the
        // strict opt-in EnvFlag form ("1" only) for A/B benchmarking without recompiling.
        // cuBLASLt bias-epilogue GEMM: promoted to the standard profile 2026-07-09 — every biased Linear
        // otherwise pays a separate BiasAdd kernel + an output-sized HBM round-trip (~700/step on the SDXL
        // UNet, measured −0.16 s/gen). Falls back to GemmEx+BiasAdd when Lt is unavailable or shapes
        // don't qualify; HARTSY_EPILOGUE_FUSION=0 is the kill-switch.
        EnableEpilogueFusion = EnvSwitch.IsEnabled("HARTSY_EPILOGUE_FUSION", defaultOn: true);
        // dp4a int8-activation decode GEMV: promoted to the standard profile 2026-07-22 after the
        // full Q4_K/Q6_K/Q8_0 kernel set measured +13-27% end-to-end decode on both benchmark models
        // with ground-truth-bounded numerics (Dp4aGemvGroundTruthTests, LLM_DECODE_PERF_GRIND.md).
        EnableDp4aGemv = EnvSwitch.IsEnabled("HARTSY_DP4A_ON", defaultOn: true);
        EnableTensorCoreGemm = EnvFlag("HARTSY_TENSORCORE_GEMM");
        // fp8 tensor-core GEMM (activation-quant e4m3) requires SM 8.9+ (Ada); older parts default to the
        // F16-cast path. Verified quality-clean fleet-wide in the standard Swarm config.
        bool fp8TensorCores = _context.ComputeCapabilityMajor > 8
            || (_context.ComputeCapabilityMajor == 8 && _context.ComputeCapabilityMinor >= 9);
        EnableNativeFp8Gemm = EnvSwitch.IsEnabled("HARTSY_FP8_NATIVE", defaultOn: fp8TensorCores);
        EnableW8A8 = EnvSwitch.IsEnabled("HARTSY_W8A8", defaultOn: false);
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

    /// <summary>This backend's transfer state, for the multi-backend isolation tests.</summary>
    internal GpuTransferHelper.State TransferState => _transferState;

    /// <summary>Op-entry guard: binds this backend as the calling thread's ambient transfer state, binds the CUDA
    /// context, and drains THIS backend's finalizer-queued GPU cleanups. Replaces the bare
    /// <c>_context.EnsureCurrent()</c> at every op entry — context identity alone cannot name the owning backend
    /// when two backends share a device's primary context, and the cleanup buckets are keyed per backend.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnterOp()
    {
        GpuTransferHelper.SetAmbient(_transferState);
        _context.EnsureCurrent();
        Tensor.DrainPendingFinalizerGpuCleanup(_transferState.Key);
    }

    /// <summary>Largest im2col workspace Conv2D will allocate; larger convs run as output-row bands (bit-identical GEMMs, per-band offset).</summary>
    /// <remarks>Bounds the 512-ch 3×3 @1024² VAE conv (9.2 GB naive) so full-res decode fits next to resident model
    /// weights — 1 GB verified live for the Flux.2 VAE beside Ideogram 4's 18.6 GB resident DiTs (2 GB still OOM'd
    /// there); band size costs no GEMM efficiency (m stays ≥ tens of thousands of rows). Override via
    /// HARTSY_IM2COL_BAND_MB (also lets tests force banding on small shapes).</remarks>
    private static readonly long Im2ColBandCapBytes =
        long.TryParse(Environment.GetEnvironmentVariable("HARTSY_IM2COL_BAND_MB"), out long capMb) && capMb > 0
            ? capMb << 20
            : 1L << 30;

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
            CudaDriverApi.cuMemcpyHtoDAsync(dev, (nint)dst, totalBytes, _stream.Handle).ThrowOnError();
            // Host buffer is stack-scoped (fixed byte[]) — drain the async copy before it goes out of scope.
            _stream.Synchronize();
            return dev;
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
    private void FreeW8A8Cache()
    {
        foreach (ulong ptr in _w8a8WeightCache.Values)
            GpuTransferHelper.FreeDevice(ptr);
        _w8a8WeightCache.Clear();
        FreeW8A8SmoothScaleCache();
    }

    /// <summary>Linear layer via cuBLAS GemmEx with transpose: output = input × weight^T + bias.</summary>
    /// <remarks>Supports mixed F32/F16/F8 dtypes. For a quantized weight the dequantized F16 cast is cached per
    /// preloaded weight (fast, but the cast occupies F16-sized VRAM).</remarks>
    public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
        => LinearImpl(output, input, weight, bias, cacheWeightCast: true);

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

    private unsafe void LinearImpl(Tensor output, Tensor input, Tensor weight, Tensor? bias, bool cacheWeightCast)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Linear");
        EnterOp();
        EnsureKernels();

        int n = (int)weight.Shape[0]; // outDim
        int k = (int)weight.Shape[1]; // inDim
        int m = (int)(input.ElementCount / k); // batch*seqLen

        // W8A8 eligibility decided BEFORE any device work: the int8 weight cache replaces the F16 weight on
        // device entirely, so the F16 upload is skipped — and the cache-miss host quant must read
        // weight.DataPointer BEFORE this call touches the transfer caches (a mid-forward DataPointer read on a
        // device-cached tensor trips the lazy-sync consume and the outer finally would double-free pWeight —
        // bisected 2026-07-23, W8A8ReproTemp).
        bool w8a8 = EnableW8A8 && _kernels!.HasW8A8Kernels && Int8Gemm.IsSupported && m >= 32
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
            if (m <= 8 && input.DType == DType.F32 && output.DType == DType.F32)
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
                    ulong gemvBias = bias is null
                        ? 0
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
                && (output.DType == DType.F16 || output.DType == DType.F32)
                && k % 16 == 0
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

                Fp8Executor.Run(weight: pWeight, input: inputFp8Ptr, outPtr: pOutput, m: m, n: n, k: k,
                    weightScale: alpha, stream: _stream.Handle,
                    inputScaleDev: inputScaleDev, outF32: output.DType == DType.F32);

                if (bias is not null)
                {
                    int totalElementsFp8 = m * n;
                    ulong biasPtr = pBias;
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
                    long headroom = weight.DType.IsQuantized
                        ? Math.Max(4L << 30, totalBytes / 3)
                        : Math.Max(2L << 30, totalBytes / 8);
                    if (freeBytes > 0 && freeBytes - (long)castBytes < headroom)
                    {
                        System.Threading.Interlocked.Increment(ref _castTransientGated);
                        weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);
                    }
                    else
                    {
                        System.Threading.Interlocked.Increment(ref _castCachedNew);
                        weightPtr = GpuTransferHelper.AllocateDevice(castBytes);
                        CastOnGpu(weightPtr, pWeight, weight.DType, gemmDtype, (int)weight.ElementCount);
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
                weightPtr = CastIfNeeded(pWeight, weight.DType, gemmDtype, (int)weight.ElementCount, out pWeightCast);
            }

            int gemmType = CublasDataType(gemmDtype);
            int outputType = CublasDataType(output.DType);

            // Bias prep shared by the fused-epilogue path and the separate-add fallback:
            // both want one bias value per output channel n, in the output dtype.
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
                && TensorCoreGemm.IsSupported && Cuda.TensorCoreGemm.IsAligned(m, n, k))
            {
                TensorCoreGemm.Run(a: inputPtr, b: weightPtr, c: pOutput, m: m, n: n, k: k, alpha: alpha, stream: _stream.Handle);
            }
            // Fused path: fold the bias into the cuBLASLt epilogue, saving a BiasAdd launch
            // plus an output-sized HBM round-trip. Only worthwhile when there is a bias to fuse.
            else if (EnableEpilogueFusion && bias is not null && LtGemm.IsSupported)
            {
                LtGemm.Run(
                    weight: weightPtr, input: inputPtr, outPtr: pOutput,
                    m: m, n: n, k: k, alpha: alpha,
                    abType: gemmType, dType: outputType,
                    biasPtr: biasDevicePtr, epilogue: CublasLtApi.CUBLASLT_EPILOGUE_BIAS,
                    stream: _stream.Handle);
                ltBiasFused = true;
            }
            else
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
                bool outNeedsCast = output.DType != gemmDtype && output.DType != DType.F32;
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

    private static readonly bool _quantAtProducer =
        Environment.GetEnvironmentVariable("HARTSY_QUANT_AT_PRODUCER") != "0";

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

    public unsafe void UnpatchifyTokens(Tensor output, Tensor tokens, int channels, int hPacked, int wPacked,
        int patch, bool innerChannelFastest)
    {
        using NvtxRange _nvtx = NvtxRange.Push("UnpatchifyTokens");
        if (output.DType != DType.F32 || tokens.DType != DType.F32)
            throw new NotSupportedException("CUDA UnpatchifyTokens supports F32 only.");
        EnterOp();
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
        if (z.DType != DType.F32 || pos.DType != DType.F32 || neg.DType != DType.F32)
            throw new NotSupportedException("CUDA CfgEulerStep supports F32 only.");
        EnterOp();
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

    /// <summary>True once the optional stepcache.ptx module is compiled/shipped; the step-cache stays disabled on CUDA without it.</summary>
    /// <remarks>Built via src/HartsyInference.Cuda/Kernels/dit/build.sh.</remarks>
    public bool SupportsDeviceStepCacheGate => _kernels is not null && _kernels.HasStepCacheKernels;

    /// <summary>Marks a tensor's device activation as surviving <see cref="FreeActivations()"/> (see IBackend doc).</summary>
    public void PinActivation(Tensor tensor) => GpuTransferHelper.PinActivation(tensor);

    /// <summary>Removes a <see cref="PinActivation"/> mark.</summary>
    public void UnpinActivation(Tensor tensor) => GpuTransferHelper.UnpinActivation(tensor);

    public bool FlashDecodeSupported => true;

    public bool StepGraphSupported => true;

    public bool StepGraphReady => _stepGraph?.IsReady == true && !_stepGraphCapturing;

    /// <summary>Owner token for the single step-graph slot (see IBackend.StepGraphOwner).</summary>
    public object? StepGraphOwner { get; set; }

    public void StepGraphBegin()
    {
        EnterOp();
        _stepGraph ??= new CudaGraph(_stream.Handle, autoFreeAllocationsOnRelaunch: true);
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

    /// <summary>Wan-Video interleaved in-place RoPE (shared cos/sin) — keeps q/k resident so RmsNorm→RoPE→SDPA never leaves the device.</summary>
    /// <remarks>In-place: x's device buffer is rotated and re-cached to the same pointer (no reallocation, no host round-trip).</remarks>
    public unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        using NvtxRange _nvtx = NvtxRange.Push("Ltx2SplitRope");
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("CUDA Ltx2SplitRope supports F32 only.");
        EnterOp();
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

    /// <summary>GPU temporal frame extract: output[B,C,H,W] = src[B,C,Tsrc,H,W][:,:,ti,:,:]. Keeps CausalConv3d slicing on-device (no D2H).</summary>
    /// <summary>Resolves the wan_vae kernel dtype: F32 → false, BF16 → true (bias, when present, must be F32 — the BF16 kernels read it as float).</summary>
    private static bool RequireVaeFrameDtype(DType a, DType b, Tensor? bias = null)
    {
        if (a != b || (a != DType.F32 && a != DType.BF16))
            throw new NotSupportedException($"CUDA VAE frame ops support matching F32 or BF16 tensors, got {a}/{b}.");
        if (a == DType.BF16 && bias is not null && bias.DType != DType.F32)
            throw new NotSupportedException($"CUDA VAE frame ops require an F32 bias on the BF16 path, got {bias.DType}.");
        return a == DType.BF16;
    }

    public unsafe void ExtractVaeFrame(Tensor output, Tensor src, int ti)
    {
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
        bool replicateFirst = false, int padH = 0, int padW = 0)
    {
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
                bf16: RequireVaeFrameDtype(padded.DType, input.DType));
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

    /// <summary>GPU MAGViT channel→space shuffle (SeedVR2 upsampler) — replaces the host per-element loop
    /// that forced a multi-GB D2H+H2D round trip per upsampler.</summary>
    public unsafe void SeedVr2PixelShuffle(Tensor output, Tensor input, int spatialRatio, int temporalRatio)
    {
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

    /// <summary>In-place interleaved (GPT-J) RoPE on a GPU-resident <c>[B,L,numHeads,headDim]</c> tensor (pairs rotated by frequency i).</summary>
    /// <remarks>cos/sin are <c>[B, L, headDim]</c>. Without this override the shared <see cref="IBackend"/> CPU
    /// fallback would drag q/k off the device every layer (Sesame CSM / HeartMuLa use interleaved RoPE) — the whole
    /// RmsNorm→RoPE→SDPA chain stays resident.</remarks>
    public unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        using NvtxRange _nvtx = NvtxRange.Push("RopeInterleaved");
        if (Environment.GetEnvironmentVariable("HM_ROPE_CPU") == "1")   // TEMP perf-repro gate: old CPU-fallback path
        {
            int b = (int)x.Shape[0], sl = (int)x.Shape[1], nh = (int)x.Shape[2], hd = (int)x.Shape[3], hf = hd / 2;
            int rdim = rotaryDim <= 0 || rotaryDim > hd ? hd : rotaryDim;
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
                            if (2 * i >= rdim) break;
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
            ((IBackend)this).QkvRopeScatterDecodeStep(qOut, kCache, vCache, qkv, cosTable, sinTable, hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
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
            ((IBackend)this).QkRopeScatterVDecodeStep(qOut, kCache, vCache, qk, v, cosTable, sinTable, hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
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
            ((IBackend)this).RopeScatterKvDecodeStep(qOut, kCache, vCache, q, k, v, cosTable, sinTable, hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
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
            ((IBackend)this).QkvNormRopeScatterDecodeStep(qOut, kCache, vCache, qkv, qNorm, kNorm, eps,
                cosTable, sinTable, hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
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
            ((IBackend)this).QkNormRopeScatterVDecodeStep(qOut, kCache, vCache, qk, v, qNorm, kNorm, eps,
                cosTable, sinTable, hq, hkv, headDim, rotaryDim, interleaved, devicePos);
            return;
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
        if (residOut.DType != DType.F32 || a.DType != DType.F32 || b.DType != DType.F32 || weight.DType != DType.F32)
        {
            ((IBackend)this).AddRmsNorm(residOut, normOut, a, b, weight, eps);
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
            ((IBackend)this).NormAddRmsNormEmitQ8(residOut, normOut, a, b, w1, w2, eps);
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
            ((IBackend)this).RmsNormAdd(output, a, b, weight, eps);
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
        if (output.DType != DType.F32 || gateUp.DType != DType.F32)
        {
            ((IBackend)this).GluActivate(output, gateUp, ff, gelu);
            return;
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

    /// <summary>Fused QKV split + per-head QK-RMSNorm (F32 or F16 activation, F32 weights); writes q/k/v each [., W] laid [token, head, d].</summary>
    /// <remarks>Replaces SliceLastDim×3 + RmsNorm×2 per attention stream.</remarks>
    public void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps)
    {
        using NvtxRange _nvtx = NvtxRange.Push("QkvSplitNorm");
        bool f16 = qkv.DType == DType.F16;
        if (!f16 && qkv.DType != DType.F32)
            throw new NotSupportedException("CUDA QkvSplitNorm supports F32 or F16 activations.");
        EnterOp();
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

    public void SliceRows(Tensor output, Tensor input, int rowOffset)
    {
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
        // gate high (HARTSY_SAGE_F16_MIN_SKV, default 12288) until the crossover is measured per-arch.
        if (UseSageAttn && mask is null && (d == 64 || d == 128)
            && query.DType == DType.F16 && key.DType == DType.F16
            && value.DType == DType.F16 && output.DType == DType.F16
            && skv >= SageF16MinSkv())
        {
            EnsureKernels();
            if (_kernels!.HasSageAttentionKernels && _kernels.HasSageV1)
            {
                SageAttentionInt8(output, query, key, value, scale);
                return;
            }
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
            // This is deliberately NOT gated on `allowF16`: that parameter DEFAULTS to false, so 33 of the ~80
            // call sites in this repo are nominally "F16 not allowed" while running Sage happily today. Gating
            // here would silently disable Sage for all of them — a broad perf regression to fix a hazard that
            // has bitten exactly one model. The cuDNN branch below can gate on it precisely because its callers
            // opt IN explicitly.
            //
            // Diagnostic fingerprint, if a future model renders black/NaN: exactly ONE bad element per token in
            // the SDPA output (a single overflowing (head, dim) column of V, spread over all query rows by the
            // softmax). Check max|V| per block against 65504. Lens hit this at block 45 with max|V| growing
            // 18940 → 31281 → 71583 across forwards; its fix (LensTransformerBlock) scales V by 1/256 before
            // SDPA and back after — exact, since attention is linear in V, and power-of-two so exponent-only.
            // A model-agnostic fix belongs inside SageAttentionInt8, not here: a blanket V damp would push small
            // values toward F16 subnormals, so it needs its own range analysis.
            // ──────────────────────────────────────────────────────────────────────────────────────────────
            // SageAttention preference (opt-in, HARTSY_SAGE_ATTN=1): for no-mask F32 calls the INT8 flash
            // path beats the cuDNN-F16-cast branch below at large seq (110.6 vs 130.4 ms at 16384²/D=128,
            // 2026-07-22 BDN A/B) — and unlike it, keeps F32-fidelity accumulation. Gate on Skv ≥ 2048:
            // below that the quant prologue outweighs the win (small-seq shapes measured 0.93× vs cuDNN).
            if (UseSageAttn && mask is null && (d == 64 || d == 128) && skv >= 2048)
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
            // Sq%32==0 — its WMMA Q-tile loads are unguarded). Default ON (HARTSY_SAGE_ATTN=0 kills);
            // no-mask MHA, D∈{64,128}, Skv≥1024 (tiny-seq shapes keep the F32 fallback — prologue-bound).
            // Falls through when the PTX isn't built.
            if (UseSageAttn && (d == 64 || d == 128) && skv >= 1024)
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
            if (EnvFlag("HARTSY_SDPA_V2") && (d == 64 || d == 128))
            {
                FlashAttentionV2(output, query, key, value, scale);
                return;
            }
            if (EnvFlag("HARTSY_SDPA_FORCE_FLASH"))
            {
                FlashAttention(output, query, key, value, (int)skv, kvGroup: 1, causal: false, qOffset: 0, scale);
                return;
            }
            ulong scoreBytesEst = (ulong)totalHeads * (ulong)sq * (ulong)skv * sizeof(float);
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
                long maskElements = mask.ElementCount;
                long scoreElements = totalHeads * sq * skv;

                if (maskElements == sq * skv)
                {
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                        if (isF16)
                            _kernels!.LaunchAddF16(sPtr, sPtr, pMask, (int)(sq * skv), _stream.Handle);
                        else
                            _kernels!.LaunchAdd(sPtr, sPtr, pMask, (int)(sq * skv), _stream.Handle);
                    }
                }
                else if (maskElements == scoreElements)
                {
                    if (isF16)
                        _kernels!.LaunchAddF16(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                    else
                        _kernels!.LaunchAdd(scoresBuf, scoresBuf, pMask, (int)scoreElements, _stream.Handle);
                }
                else if (maskElements == b * sq * skv && b > 1)
                {
                    // [B, 1, Sq, Skv] mask: per-batch, broadcast over heads. Add each batch's [Sq,Skv] block to all H
                    // head-blocks for that batch. (B=1 is already caught by the Sq*Skv branch above.) One add per (b,h).
                    for (long bh = 0; bh < totalHeads; bh++)
                    {
                        long batchIdx = bh / h;
                        ulong sPtr = scoresBuf + (ulong)(bh * strideScores * elemSize);
                        ulong mPtr = pMask + (ulong)(batchIdx * sq * skv * elemSize);
                        if (isF16)
                            _kernels!.LaunchAddF16(sPtr, sPtr, mPtr, (int)(sq * skv), _stream.Handle);
                        else
                            _kernels!.LaunchAdd(sPtr, sPtr, mPtr, (int)(sq * skv), _stream.Handle);
                    }
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
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
            if (pQCast != 0) CudaMemory.FreeAsync(pQCast, _stream.Handle);
            if (pKCast != 0) CudaMemory.FreeAsync(pKCast, _stream.Handle);
            if (pVCast != 0) CudaMemory.FreeAsync(pVCast, _stream.Handle);
            if (pOutCast != 0) CudaMemory.FreeAsync(pOutCast, _stream.Handle);
        }
    }

    /// <summary>Whether a mask can ride the cuDNN fused engine as an additive fp32 bias (no mask, or F32 broadcastable over heads).</summary>
    /// <remarks>Per-head ([B,H,Sq,Skv]) or non-F32 masks fall back to the materialized path.</remarks>
    private static bool CudnnMaskCompatible(Tensor? mask, long b, long sq, long skv)
    {
        if (mask is null) return true;
        if (mask.DType != DType.F32) return false;
        long n = mask.ElementCount;
        return n == sq * skv || n == b * sq * skv;
    }

    /// <summary>cuDNN fused flash-attention fast path for the plain F32/no-mask/MHA case (F16 execute, F32 output).</summary>
    /// <remarks>Casts Q/K/V to F16, runs cuDNN's fused attention, casts the F16 result back to F32. Returns false (and
    /// permanently disables cuDNN for the session) on any failure so the caller falls through to the materialized/tiled
    /// paths. Q/out [B,H,Sq,D], K/V [B,H,Skv,D].</remarks>
    private unsafe bool TryCudnnSdpa(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale)
    {
        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
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
            CudnnStatusException? injected = TestCudnnSdpaFaultInjector?.Invoke(d);
            if (injected is not null) throw injected;

            pQ = GpuTransferHelper.CopyToDevice(query);
            pK = GpuTransferHelper.CopyToDevice(key);
            pV = GpuTransferHelper.CopyToDevice(value);
            long biasB = 1;
            if (mask is not null)
            {
                pMask = GpuTransferHelper.CopyToDevice(mask);
                biasB = mask.ElementCount / (sq * skv);
            }
            nuint outBytes = GpuTransferHelper.ByteSize(output);
            pOut = GpuTransferHelper.AllocateDevice(outBytes);

            if (query.DType == DType.F16)
            {
                // Native F16 Q/K/V/output — the engine's fp16 I/O dtype already; execute directly, zero casts.
                _cudnnSdpa.Execute(pQ, pK, pV, pOut, b, h, sq, skv, d, scale, pMask, biasB);
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

                _cudnnSdpa.Execute(qF16, kF16, vF16, oF16, b, h, sq, skv, d, scale, pMask, biasB);

                _kernels!.LaunchCastF16ToF32(pOut, oF16, (int)output.ElementCount, _stream.Handle);
            }
            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
            if (_cudnnSdpaDimState.TryRemove(d, out DimFailureState? recovered))
            {
                HartsyInference.Core.Logging.Logs.Info(
                    $"[cuDNN SDPA] D={d} recovered after {recovered.ConsecutiveFailures} prior failure(s) — fused path re-engaged.");
            }
            if (!CudnnSdpaEngaged)
            {
                CudnnSdpaEngaged = true;
                HartsyInference.Core.Logging.Logs.Info($"[cuDNN SDPA] fused flash-attention engaged (D={d}, cuDNN {CudnnApi.cudnnGetVersion()})");
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
                _cudnnSdpaDimState[d] = new DimFailureState { Permanent = true };
                HartsyInference.Core.Logging.Logs.Warning(
                    $"[cuDNN SDPA] D={d} permanently disabled (structural failure) — this is the steady " +
                    $"state for the rest of the process: {ex.Message}");
            }
            else
            {
                DimFailureState state = _cudnnSdpaDimState.AddOrUpdate(d,
                    _ => new DimFailureState { ConsecutiveFailures = 1 },
                    (_, existing) => { existing.ConsecutiveFailures++; return existing; });
                long backoffMs = CudnnSdpaBackoffMs(state.ConsecutiveFailures);
                state.NextRetryAtTicks = Environment.TickCount64 + backoffMs;
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

    /// <summary>Fused FlashAttention-2 (TF32 tensor cores, F32 accumulate) — no-mask MHA, D∈{64,128}; never materializes the score matrix.</summary>
    private unsafe void FlashAttentionV2(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
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

    /// <summary>Min Skv (override: HARTSY_SAGE_F16_MIN_SKV) above which native-F16 SageAttention ingest is preferred over cuDNN.</summary>
    private static int SageF16MinSkv()
    {
        string? v = Environment.GetEnvironmentVariable("HARTSY_SAGE_F16_MIN_SKV");
        return int.TryParse(v, out int n) && n > 0 ? n : 8192;   // measured: 1.11x at 8192, 1.15x at 12288, parity at 4096 (3060)
    }

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

            _kernels!.LaunchSageKMean(pKmean, pK, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            // log2-domain softmax: fold log2(e) into the Q row scales so the flash kernels exponentiate via
            // native exp2 (one fewer multiply per score element; max/corr/l all commute with the constant).
            const float Log2E = 1.4426950408889634f;
            _kernels!.LaunchSageQuantQ(pQ8, pQs, pQ, (int)b, (int)h, (int)sq, (int)d, scale * Log2E, _stream.Handle, srcF16: f16);
            _kernels!.LaunchSageQuantK(pK8, pKs, pK, pKmean, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            if (_kernels!.UseSageV1)
            {
                // v1 prologue: one-shot [B,H,Skv,D]→[B,H,D,skvPad] F16 transpose (cp.async-able staging).
                long skvPad = (skv + 7L) & ~7L;
                if (skvPad >= 2048 && (skvPad & (skvPad - 1)) == 0) skvPad += 8;   // anti-aliasing pad (see sage_skv_pad)
                pVt16 = GpuTransferHelper.AllocateDevice((nuint)((long)b * h * d * skvPad * 2));
                _kernels!.LaunchSageVF16T(pVt16, pV, (int)b, (int)h, (int)skv, (int)d, _stream.Handle, srcF16: f16);
            }
            _kernels!.LaunchSageAttnInt8(pOut, pQ8, pQs, pK8, pKs, pV, pVt16, (int)b, (int)h, (int)sq, (int)skv, (int)d, _stream.Handle, f16Io: f16);

            GpuTransferHelper.CacheActivation(output, pOut, outBytes);
            cachedOutput = true;
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
    private unsafe void SdpaTiledF32NoMask(Tensor output, Tensor query, Tensor key, Tensor value, float scale)
    {
        EnterOp();
        EnsureKernels();

        long b = query.Shape[0], h = query.Shape[1], sq = query.Shape[2], d = query.Shape[3];
        long skv = key.Shape[2];
        long totalHeads = b * h;

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
            long perRow = totalHeads * skv * sizeof(float);            // bytes for one query row across all heads
            long Br = (long)((ulong)freeBytes / 4) / Math.Max(1, perRow);
            if (Br < 1) Br = 1;
            if (Br > sq) Br = sq;
            if (int.TryParse(Environment.GetEnvironmentVariable("HARTSY_SDPA_TILE"), out int envBr) && envBr > 0)
                Br = Math.Min(envBr, sq);

            scoresBuf = CudaMemory.Allocate((nuint)(totalHeads * Br * skv * sizeof(float)));

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
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOut);
            if (scoresBuf != 0) CudaMemory.FreeAsync(scoresBuf, _stream.Handle);
        }
    }

    #endregion

    #region Activations

    public void Gelu(Tensor output, Tensor input)
    {
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

    /// <summary>cuDNN transposed 1D convolution via convolution-backward-data (mapped to 2D, H=1): F32 output via
    /// TF32 tensor cores; bias is added F32 after. Returns false on a geometry mismatch (without disabling the
    /// route) or on any cuDNN failure (session-sticky) so the caller falls back to the direct kernel.</summary>
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
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"MaxPool2D requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        EnterOp();
        EnsureKernels();

        int n = (int)input.Shape[0], c = (int)input.Shape[1], iH = (int)input.Shape[2], iW = (int)input.Shape[3];
        int oH = (int)output.Shape[2], oW = (int)output.Shape[3];

        ulong pOut = 0, pIn = 0, inCast = 0;
        bool cachedOutput = false;
        try
        {
            pIn = GpuTransferHelper.CopyToDevice(input);
            ulong inTyped = CastIfNeeded(pIn, input.DType, output.DType, (int)input.ElementCount, out inCast);
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
        if (coords != 2 && coords != 4)
            throw new ArgumentException($"DeformableAttention coords must be 2 or 4, got {coords}.");
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
            return results[1] > 0f ? results[0] / results[1] : 0f;
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

    /// <summary>Maps a HartsyInference dtype to its cuBLAS data-type constant for <c>cublasGemmEx</c> / <c>cublasGemmStridedBatchedEx</c>.</summary>
    /// <remarks>Handles F16, BF16, F32; throws otherwise (FP8 casts to F16/BF16 via <see cref="CastIfNeeded"/> before cuBLAS).</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        if (Environment.GetEnvironmentVariable("HARTSY_PROFILE_EACH") == "1")
            Profiling.NvtxRange.DumpProfile(Environment.GetEnvironmentVariable("HARTSY_PROFILE_OUT") ?? "/tmp/hartsy_profile.txt");
    }

    #endregion

    #region Transpose / Permute

    /// <summary>Batched 2D transpose: [B, D1, D2] -> [B, D2, D1] via PTX kernel.</summary>
    public void Transpose2D(Tensor output, Tensor input, int d1, int d2)
    {
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

    /// <summary>FlashAttention (online-softmax, GQA-aware, no materialized score matrix).</summary>
    /// <remarks>Requires F32 and a power-of-two head dimension (true for the 64/128 head dims we run); falls back
    /// to the base CPU reference otherwise.</remarks>
    public unsafe void FlashAttention(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, float softcap = 0f, Tensor? sink = null, int slidingWindow = 0, Tensor? alibiSlopes = null)
    {
        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        // The kernel pads the block to the next power of two >= D, so any D up to 1024 works (Phi-3 D=96 etc.).
        bool kernelOk = d > 0 && d <= 1024;
        // F16-storage KV cache (halved VRAM): key/value are the FixedKvCache buffers, which under kvDtype=F16
        // are __half-typed; Q/out/sink/alibi are unaffected (still F32). v1 scope: monolithic kernel only —
        // the split-K path below is forced off for this case (LaunchFlashAttentionF16Kv has no split variant).
        bool f16Kv = key.DType == DType.F16 && value.DType == DType.F16;
        bool kvOk = f16Kv || (key.DType == DType.F32 && value.DType == DType.F32);
        if (query.DType != DType.F32 || !kvOk || output.DType != DType.F32 || !kernelOk
            || (sink is not null && sink.DType != DType.F32) || (alibiSlopes is not null && alibiSlopes.DType != DType.F32))
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
            // F16 KV forces the monolithic path — LaunchFlashAttentionSplit has no F16-KV variant (v1 scope).
            bool splitEligible = pSink == 0 && pAlibi == 0 && !f16Kv;
            bool forceSplit = EnvFlag("HARTSY_FLASH_SPLIT_FORCE");
            // Occupancy-limited = the LLM decode case (tq=1, few heads → e.g. 16 blocks on 28 SMs). Splitting the
            // key axis there fills the GPU and is a large decode win (attention was ~30% of decode; split-K ≈ +38%
            // end-to-end on Qwen3). The split kernel is numerically exact vs monolithic (online-softmax merge). The
            // old kvLen≥1024 floor never engaged for decode (kvLen<300); gate on occupancy instead, floor 128.
            bool occLimited = baseBlocks < 2 * _context.MultiprocessorCount;
            // Severely under-occupied launches (fewer blocks than SMs — gemma3-1b: 4, qwen2.5-0.5b: 14) are
            // worth splitting even at short kvLen; the 128 floor only applies to moderately-occupied shapes.
            int engageLen = baseBlocks <= _context.MultiprocessorCount ? 32 : 128;
            int splits = 1;
            if (!EnvFlag("HARTSY_FLASH_SPLIT_OFF")
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
                        softcap: softcap, slidingWindow: slidingWindow);
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
            ((IBackend)this).KvCacheAppend(buffer, newKv, offset);
            return;
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
            ((IBackend)this).SliceTimeRange(output, input, start, len);
            return;
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
        fixed (T* p = values) CudaDriverApi.cuMemcpyHtoD(dptr, (nint)p, bytes).ThrowOnError();
        return dptr;
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
        if (devicePos == 0 || buffer.DType != DType.F32 || newKv.DType != DType.F32)
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

    /// <summary>Self-attention FlashAttention; kvLen/qOffset read from <paramref name="devicePos"/>, split-K decode, FIXED split count.</summary>
    /// <remarks>Chunk sized to the cache CAPACITY, not the current kvLen, so the grid is position-independent and
    /// graph-capturable while still filling the GPU at long context: splits whose chunk starts past the current
    /// kvLen early-exit in the kernel, so active parallelism grows with position.</remarks>
    public unsafe void FlashAttentionDev(Tensor output, Tensor query, Tensor key, Tensor value, int kvLen, int kvGroup, bool causal, int qOffset, float scale, ulong devicePos,
        float softcap = 0f, int slidingWindow = 0)
    {
        int b = (int)query.Shape[0], hq = (int)query.Shape[1], tq = (int)query.Shape[2], d = (int)query.Shape[3];
        int hkv = (int)key.Shape[1], lk = (int)key.Shape[2];
        bool kernelOk = d > 0 && d <= 1024;
        // key/value != F32 also catches F16-storage KV (v1 scope: graph-decode has no F16-KV variant) — falls
        // through to the eager FlashAttention above, which DOES handle F16 KV (monolithic kernel only, split
        // forced off there too). Intentional, not a gap: same "disable the fast path, not extend it" precedent
        // as Phase 6 disabling CUDA-graph decode when LLM layers are staged across devices.
        if (devicePos == 0 || query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32 || output.DType != DType.F32 || !kernelOk)
        {
            FlashAttention(output, query, key, value, kvLen, kvGroup, causal, qOffset, scale, softcap, sink: null, slidingWindow, alibiSlopes: null);
            return;
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
                        grp, causal, qOffset, scale, splits, chunk, _stream.Handle, devicePos, softcap, slidingWindow);
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
                    causal, qOffset, scale, softcap, 0, slidingWindow, 0, _stream.Handle, devicePos);
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
            ((IBackend)this).RopeApplyDecodeStep(x, cosTable, sinTable, rotaryDim, interleaved, devicePos);
            return;
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
        if (Environment.GetEnvironmentVariable("HARTSY_GRAPH_ARENA") != "0")
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
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            ((IBackend)this).GatherRows(output, input, rowIndices);
            return;
        }
        EnterOp();
        EnsureKernels();
        int k = (int)input.Shape[input.Shape.Rank - 1];
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
        }
    }

    /// <summary>Per-row argmax over the last dim: indices[r] = argmax_c input[r,c]. On-device; only indices sync back, not full logit rows.</summary>
    public unsafe void ArgMaxLastDim(Tensor indices, Tensor input)
    {
        if (input.DType != DType.F32 || indices.DType != DType.I32)
        {
            ((IBackend)this).ArgMaxLastDim(indices, input);
            return;
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

    /// <summary>MoE weighted scatter-add (in place): output[rowIndices[m]] += scales[m]·input[m].</summary>
    /// <remarks>Output must be a resident, already-accumulating activation (pre-zeroed, then one call per expert).</remarks>
    public unsafe void ScatterAddWeightedRows(Tensor output, Tensor input, ReadOnlySpan<int> rowIndices, ReadOnlySpan<float> scales)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
        {
            ((IBackend)this).ScatterAddWeightedRows(output, input, rowIndices, scales);
            return;
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

    /// <summary>GEGLU activation: output[i] = input[i] * GELU(input[i + outputElements]) via PTX kernel.</summary>
    public void GeGlu(Tensor output, Tensor input)
    {
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

    /// <summary>Splits <paramref name="input"/> into <paramref name="outputs"/> along <paramref name="dim"/>. Delegates to the CPU kernel.</summary>
    /// <remarks>Split is pure memcpy and rare in our pipelines (one call per VAE encode), so the GPU round-trip is
    /// acceptable. TODO: GPU-native Split via cuMemcpyDtoDAsync.</remarks>
    public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        HartsyInference.Cpu.Kernels.ElementWiseKernels.Split(outputs, input, dim);
    }

    #endregion

    #region Sampling

    public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW)
    {
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

    /// <summary>This backend's cached device pointer for <paramref name="tensor"/> (weight or activation shadow),
    /// or 0 when none. Read-only peek for the peer-copy boundary; the owning backend must be quiescent (the
    /// per-device gate / stage sequencing guarantees the producing stage finished issuing work).</summary>
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

    /// <summary>Cross-backend boundary copy. When the source lives on another CUDA device and P2P is available,
    /// the device copy moves directly (event-ordered against the source's stream, no host round-trip and no
    /// eviction of the source backend's resident copy). Otherwise the source's device data is staged into the
    /// DESTINATION's host buffer — deliberately not the default interface path, which would fire the source
    /// tensor's demote hooks and evict it from the source backend.</summary>
    public unsafe void CopyFromPeer(Tensor destination, Tensor source, IBackend sourceBackend)
    {
        nuint byteSize = (nuint)(source.ElementCount * source.DType.SizeInBytes);
        if ((nuint)(destination.ElementCount * destination.DType.SizeInBytes) != byteSize)
        {
            throw new ArgumentException(
                $"CopyFromPeer size mismatch: source {byteSize} bytes vs destination " +
                $"{destination.ElementCount * destination.DType.SizeInBytes} bytes.");
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
        GpuTransferHelper.FreeWeights(weights);
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

    public void FreeAllDeviceMemory()
    {
        EnterOp();
        long freeBefore = (long)_context.GetMemoryInfo().freeBytes;
        StepGraphInvalidateForActivationFree();
        // Drop the cuDNN sessions too: their execution-plan + workspace caches held ~4.5 GB after a
        // Z-Image session (measured 2026-07-23 — enough to trip Ideogram's ≥20 GB guard after a model
        // switch). Both instances lazily recreate on next use; the only cost is one plan re-search.
        _cudnnSdpa?.Dispose();
        _cudnnSdpa = null;
        _cudnnConv?.Dispose();
        _cudnnConv = null;
        // EvictAll clears weights + casts + activations (syncing the stream first); the trim then returns the
        // stream-ordered pool's reservations so cuMemGetInfo/persistent allocs see the memory as actually free.
        FreeW8A8Cache();
        GpuTransferHelper.EvictAll();
        GpuTransferHelper.TrimPool();
        long freeAfter = (long)_context.GetMemoryInfo().freeBytes;
        HartsyInference.Core.Logging.Logs.Info(
            $"[Cuda] FreeAllDeviceMemory: free {freeBefore >> 20} MB → {freeAfter >> 20} MB");
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
        FreeW8A8Cache();
        GpuTransferHelper.FreeAllCached();
    }

    /// <summary>Evicts all cached GPU weight buffers. Call between pipeline stages to free VRAM.</summary>
    public void EvictGpuCache()
    {
        EnterOp();
        GpuTransferHelper.EvictAll();
    }

    /// <summary>Returns GPU cache stats: (cachedBytes, hits, misses).</summary>
    public (long cachedBytes, long hits, long misses) GetGpuCacheStats()
    {
        return GpuTransferHelper.GetStats();
    }

    /// <summary>Number of lazy D2H syncs since <see cref="ResetD2hSyncCount"/> — each is a full GPU stall plus a copy.</summary>
    public long GetD2hSyncCount() => GpuTransferHelper.GetSyncCount();

    /// <summary>Resets the device-to-host sync counter (call before a region you want to measure for residency).</summary>
    public void ResetD2hSyncCount() => GpuTransferHelper.ResetSyncCount();

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
            EnterOp();

            if (_dp4aScratch != 0) { GpuTransferHelper.FreeDevice(_dp4aScratch); _dp4aScratch = 0; _dp4aScratchBytes = 0; }
            if (_argmaxScratch != 0) { GpuTransferHelper.FreeDevice(_argmaxScratch); _argmaxScratch = 0; }
            if (_ssmDeltaScratch != 0) { GpuTransferHelper.FreeDevice(_ssmDeltaScratch); _ssmDeltaScratch = 0; _ssmDeltaScratchBytes = 0; }
            FreeW8A8Cache();
            GpuTransferHelper.EvictAll();
            _streamingCache.UnregisterPinnedSources();
            _kernels?.Dispose();

            if (_fp8Executor is not null)
            {
                _fp8Executor.Dispose();
                _fp8Executor = null;
            }

            if (_int8Executor is not null)
            {
                _int8Executor.Dispose();
                _int8Executor = null;
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

            // Drop this backend's registration before tearing the context down, so nothing later resolves to
            // freed stream handles. State keys are per-backend and never reused, so this backend's
            // finalizer-cleanup bucket is discarded unconditionally (EvictAll above already freed everything
            // those callbacks could reference) — a live same-device sibling has its own key, bucket, and
            // stream binding, all untouched by this teardown.
            GpuTransferHelper.Unregister(_transferState);
            HartsyInference.Core.Tensors.Tensor.DiscardPendingFinalizerGpuCleanup(_transferState.Key);
            DeviceMempoolPolicy.Release(_context.DeviceOrdinal);

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
