using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Serialization;
using static HartsyInference.Cuda.CudnnApi;
using static HartsyInference.Cuda.Tests.CudnnProbeApi;

namespace HartsyInference.Cuda.Tests;

/// <summary>cuDNN backend constants <see cref="CudnnApi"/> does not declare. Values and line numbers are from the cuDNN
/// 9.20.0 <c>cudnn_graph.h</c> (nvidia-cudnn wheel) unless a line says otherwise.</summary>
internal static class CudnnProbeApi
{
    // cudnnBackendDescriptorType_t (:903, :906)
    internal const int CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = 41;
    internal const int CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR = 44;

    // cudnnBackendAttributeType_t (:833)
    internal const int CUDNN_TYPE_DOUBLE = 5;

    // cudnnDataType_t (:226, :227, :234)
    internal const int CUDNN_DATA_INT8 = 3;
    internal const int CUDNN_DATA_INT32 = 4;
    internal const int CUDNN_DATA_BOOLEAN = 11;

    // cudnnPointwiseMode_t (:381, :387, :406, :425-427, :430, :434, :436)
    internal const int CUDNN_POINTWISE_DIV = 6;
    internal const int CUDNN_POINTWISE_SUB = 9;
    internal const int CUDNN_POINTWISE_TANH_FWD = 101;
    internal const int CUDNN_POINTWISE_CMP_GT = 302;
    internal const int CUDNN_POINTWISE_CMP_GE = 303;
    internal const int CUDNN_POINTWISE_CMP_LT = 304;
    internal const int CUDNN_POINTWISE_LOGICAL_AND = 400;
    internal const int CUDNN_POINTWISE_GEN_INDEX = 501;
    internal const int CUDNN_POINTWISE_BINARY_SELECT = 601;

    // cudnnBackendAttributeName_t
    internal const int CUDNN_ATTR_POINTWISE_AXIS = 9;                              // :484
    internal const int CUDNN_ATTR_OPERATION_POINTWISE_TDESC = 758;                 // :550
    internal const int CUDNN_ATTR_TENSOR_IS_BY_VALUE = 908;                        // :590
    internal const int CUDNN_ATTR_MATMUL_PADDING_VALUE = 1503;                     // :621
    internal const int CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = 1525;    // :628
    internal const int CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = 1526;    // :629
    internal const int CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = 1527;    // :630
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC = 2800;                 // :775
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC = 2801;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC = 2802;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC = 2803;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = 2804;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = 2805;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC = 2806;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = 2809;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = 2810;        // :785, the last SDPA_FWD entry in 9.20
    // Not in the 9.20 header: values from the cuDNN 9.24 graph-library docs as quoted in the attention research report
    // §3.2; the enumerator spellings follow the 2800-range prefix and are unconfirmed.
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH = 2811;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID = 2812;
    internal const int CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID = 2813;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC = 3000;       // :810
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC = 3001;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC = 3002;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC = 3003;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC = 3004;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC = 3005;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC = 3006;
    internal const int CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE = 3007; // :817
    internal const int CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC = 3102;             // :821
    internal const int CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC = 3105;              // :824

    // cudnnBackendHeurMode_t (:990, :991); A and FALLBACK are in CudnnApi
    internal const int CUDNN_HEUR_MODE_INSTANT = 0;
    internal const int CUDNN_HEUR_MODE_B = 1;
}

internal enum ProbeDtype { F16, BF16 }
internal enum ProbeLayout { HeadMajor, TokenMajor }
internal enum ProbeCausal { None, TopLeft, BottomRight }
internal enum ProbeMaskForm { Band, LegacyChain }
internal enum ProbePadding { None, Composite, Unified }
internal enum ProbeBoundType { Int32, Float }
internal enum ProbeBiasType { F32, F16, BF16 }

/// <summary>What the docs predict. Only Required (refusal fails), Refused (acceptance fails) and KnownIssue (wrong output
/// is recorded, not failed) change the verdict; Supported and Unknown are labels.</summary>
internal enum ProbeExpect { Required, Supported, Unknown, Refused, KnownIssue }

/// <summary>Additive bias of dims {B,H,Sq,Skv}. <see cref="SqStrideZero"/> stores one row and declares the Sq axis with stride 0.</summary>
internal sealed record ProbeBias(long B, long H, long Sq, ProbeBiasType Type = ProbeBiasType.F32, bool SqStrideZero = false);

/// <summary>One SDPA graph to build, plan with every heuristic mode, execute and check.</summary>
internal sealed record ProbeSpec
{
    public required string Name { get; init; }
    public ProbeDtype Dtype { get; init; } = ProbeDtype.F16;
    public ProbeLayout Layout { get; init; } = ProbeLayout.HeadMajor;
    public long B { get; init; } = 1;
    public long Hq { get; init; }
    public long Hkv { get; init; }                 // 0 means Hq
    public long Sq { get; init; }
    public long Skv { get; init; }
    public long D { get; init; }
    public long KvCapacity { get; init; }          // stored K/V rows per head (a FixedKvCache-shaped buffer); 0 means Skv
    public bool KvHeadStrideZero { get; init; }    // one stored K/V head declared with Hq heads and head stride 0 (P2b)
    public bool Unified { get; init; }             // one SDPA_FWD op instead of the matmul/softmax chain
    public ProbeBias? Bias { get; init; }
    public ProbeCausal Causal { get; init; }
    public ProbeMaskForm MaskForm { get; init; }
    public long? ShiftRightBound { get; init; }
    public long Window { get; init; }              // left bound; 0 = none
    public ProbeBoundType BoundType { get; init; }
    public float Fill { get; init; } = float.NegativeInfinity;
    public float Softcap { get; init; }
    public bool Alibi { get; init; }
    public bool Sink { get; init; }
    public bool Stats { get; init; }
    public bool BlockMask { get; init; }
    public ProbePadding Padding { get; init; }
    public int[]? SeqQ { get; init; }
    public int[]? SeqKv { get; init; }
    public bool ScaleByValue { get; init; }
    public float[]? Scales { get; init; }
    public int Seed { get; init; } = 1;
    public ProbeExpect Expect { get; init; } = ProbeExpect.Supported;
    public string? Note { get; init; }
    [JsonIgnore] public bool KeepOutput { get; init; }

    [JsonIgnore] public long KvHeads => KvHeadStrideZero ? Hq : Hkv == 0 ? Hq : Hkv;
    [JsonIgnore] public long KvPhysicalHeads => KvHeadStrideZero ? 1 : KvHeads;
    [JsonIgnore] public long KvRows => KvCapacity > 0 ? KvCapacity : Skv;
    [JsonIgnore] public float[] ScaleList => Scales ?? [1f / MathF.Sqrt(D)];
}

internal sealed record ProbeModeResult(
    string Mode, float Scale, string Verdict, bool Finalized, bool Executed,
    double RelL2, double MaxAbs, double Cosine, long NanCount, double StatsRelL2, double StatsRelL2Log2,
    double PlanBuildMs, double HeuristicMs, double FinalizeMs, double ExecMedianMs,
    int ConfigsReturned, int ConfigsTried, long EngineGlobalIndex, bool RuntimeCompiled, int[] NumericalNotes,
    long WorkspaceBytes, string? Error);

internal sealed record ProbeSpecResult(ProbeSpec Spec, string Outcome, double Tolerance, int RowsCompared, double ReferenceMs,
    IReadOnlyList<ProbeModeResult> Modes)
{
    [JsonIgnore] public ushort[]? Output { get; init; }
}

internal sealed record CudnnProbeEnvironment(string Device, int Sm, int ExpectedSm, long CudnnVersion, long CudartVersion,
    int DriverVersion, string ExpectedLibDir, IReadOnlyList<string> CudnnFiles, string? CudaVisibleDevices,
    string? CudaDeviceOrder, IReadOnlyList<string> Problems);

/// <summary>Test-only cuDNN SDPA graph builder for the Phase 0 capability probes. It owns its cuDNN handle, stream,
/// device buffers and descriptors and never goes through <see cref="CudaBackend"/>, whose cuDNN failure state is keyed
/// on head dim alone. Each spec is built once, planned with every heuristic mode, executed, and compared against a
/// float64 host reference on the same quantized inputs.</summary>
internal sealed unsafe class CudnnSdpaProbe : IDisposable
{
    internal const string GpuEnvVar = "HARTSY_CUDNN_PROBE_GPU";
    internal const string SmEnvVar = "HARTSY_CUDNN_PROBE_SM";
    internal const string VersionEnvVar = "HARTSY_CUDNN_PROBE_VERSION";
    internal const string LibDirEnvVar = "HARTSY_CUDNN_PROBE_LIBDIR";
    internal const string RefMacsEnvVar = "HARTSY_CUDNN_PROBE_REF_MACS";
    internal const long ExpectedVersion = 92400;
    internal const string VerdictOk = "ok", VerdictWrong = "wrong", VerdictExecError = "exec-error";
    internal const string VerdictRefused = "refused", VerdictWorkspaceCap = "workspace-cap";
    internal const double F16Tolerance = 1e-3;    // research report §3.1.6; the audit measured 2.9e-4
    internal const double Bf16Tolerance = 8e-3;
    private const double StatsTolerance = 2e-3;
    private const double DefaultRefMacs = 4e9;
    private const int MaxConfigs = 32;
    private const int TimedRuns = 10;
    private const long WorkspaceCap = 2L << 30;
    private const int SlotScale = 0, SlotFill = 1, SlotCap = 2, SlotWindow = 3, SlotShift = 4, SlotCount = 5, SlotBytes = 16;

    private static readonly (string Name, int Mode)[] _modes =
    [
        ("A", CUDNN_HEUR_MODE_A), ("B", CUDNN_HEUR_MODE_B), ("FALLBACK", CUDNN_HEUR_MODE_FALLBACK), ("INSTANT", CUDNN_HEUR_MODE_INSTANT),
    ];

    private readonly CudaContext _ctx;
    private readonly nint _stream;
    private readonly nint _handle;
    private readonly byte* _slots;
    private readonly nint _library;
    private readonly delegate* unmanaged<byte*, nuint, void> _lastError;
    private bool _disposed;

    public CudnnProbeEnvironment Env { get; }

    public CudnnSdpaProbe(CudaContext ctx, CudnnProbeEnvironment env)
    {
        if (env.Problems.Count > 0)
            throw new InvalidOperationException("cuDNN probe environment rejected: " + string.Join("; ", env.Problems));
        _ctx = ctx;
        Env = env;
        ctx.EnsureCurrent();
        CudaDriverApi.cuStreamCreate(out _stream, 0).ThrowOnError();
        int st = cudnnCreate(out _handle);
        if (st == CUDNN_STATUS_SUCCESS)
            st = cudnnSetStream(_handle, _stream);
        if (st != CUDNN_STATUS_SUCCESS)
        {
            if (_handle != 0) cudnnDestroy(_handle);
            CudaDriverApi.cuStreamDestroy(_stream);
            throw new InvalidOperationException($"cudnnCreate/cudnnSetStream failed: {ErrorString(st)}");
        }
        _slots = (byte*)NativeMemory.AlignedAlloc(SlotCount * SlotBytes, SlotBytes);
        NativeMemory.Clear(_slots, SlotCount * SlotBytes);
        string? dispatcher = env.CudnnFiles.FirstOrDefault(f => Path.GetFileName(f).StartsWith("libcudnn.so", StringComparison.Ordinal));
        if (dispatcher is not null && NativeLibrary.TryLoad(dispatcher, out _library)
            && NativeLibrary.TryGetExport(_library, "cudnnGetLastErrorString", out nint export))
            _lastError = (delegate* unmanaged<byte*, nuint, void>)export;
    }

    /// <summary>Reads the device, SM, cuDNN version and every mapped libcudnn* file and lists what does not match what the
    /// run asked for. Context-free cuDNN queries only: a build for another CUDA major hangs on its first handle op.</summary>
    public static CudnnProbeEnvironment Inspect(CudaContext ctx)
    {
        ctx.EnsureCurrent();
        List<string> problems = new();
        string? gpu = Environment.GetEnvironmentVariable(GpuEnvVar);
        if (string.IsNullOrWhiteSpace(gpu))
            problems.Add($"{GpuEnvVar} is not set; run tests/cudnn-probes.sh");
        else if (!ctx.DeviceName.Contains(gpu, StringComparison.OrdinalIgnoreCase))
            problems.Add($"device 0 is '{ctx.DeviceName}', not '{gpu}': pin it with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES");
        int expectedSm = ExpectedSm(gpu);
        if (expectedSm == 0)
            problems.Add($"no expected SM for '{gpu}': set {SmEnvVar}");
        else if (ctx.Sm != expectedSm)
            problems.Add($"device SM is {ctx.Sm}, expected {expectedSm}");
        long wantVersion = long.TryParse(Environment.GetEnvironmentVariable(VersionEnvVar), NumberStyles.Integer,
            CultureInfo.InvariantCulture, out long v) ? v : ExpectedVersion;
        long version = 0, cudart = 0;
        try
        {
            version = (long)cudnnGetVersion();
            cudart = (long)cudnnGetCudartVersion();
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            problems.Add($"cuDNN did not load: {ex.Message}");
        }
        if (version != 0 && version != wantVersion)
            problems.Add($"cuDNN version is {version}, expected {wantVersion}");
        int driver = ctx.QueryDriverVersion();
        if (cudart != 0 && cudart / 1000 != driver / 1000)
            problems.Add($"cuDNN is built for CUDA {cudart / 1000} but the driver is CUDA {driver / 1000}");
        string libDir = ExpectedLibDir();
        string[] files = LoadedCudnnFiles();
        problems.AddRange(CheckLibraries(files, libDir));
        return new CudnnProbeEnvironment(ctx.DeviceName, ctx.Sm, expectedSm, version, cudart, driver, libDir, files,
            Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"), Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER"), problems);
    }

    /// <summary>Re-checks the mapped cuDNN files: engine sub-libraries load lazily during planning.</summary>
    public IReadOnlyList<string> LibraryProblems() => CheckLibraries(LoadedCudnnFiles(), Env.ExpectedLibDir);

    internal static string[] LoadedCudnnFiles()
    {
        const string maps = "/proc/self/maps";
        if (!File.Exists(maps))
            return [];
        SortedSet<string> found = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(maps))
        {
            int slash = line.IndexOf('/');
            if (slash < 0) continue;
            string path = line[slash..].Trim();
            if (Path.GetFileName(path).StartsWith("libcudnn", StringComparison.Ordinal))
                found.Add(path);
        }
        return [.. found];
    }

    private static List<string> CheckLibraries(IReadOnlyList<string> files, string expectedDir)
    {
        List<string> problems = new();
        if (files.Count == 0)
            problems.Add("no libcudnn* is mapped in /proc/self/maps, so the library directory cannot be shown");
        foreach (string file in files)
        {
            if (!string.Equals(Path.GetDirectoryName(file), expectedDir, StringComparison.Ordinal))
                problems.Add($"{file} was loaded from outside {expectedDir}");
        }
        return problems;
    }

    private static string ExpectedLibDir()
    {
        string dir = Environment.GetEnvironmentVariable(LibDirEnvVar) is { Length: > 0 } custom
            ? custom
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".local", "lib", "cuda13");
        dir = Path.TrimEndingDirectorySeparator(Path.GetFullPath(dir));
        try { return Directory.ResolveLinkTarget(dir, returnFinalTarget: true)?.FullName ?? dir; }
        catch (IOException) { return dir; }
    }

    private static int ExpectedSm(string? gpu)
    {
        string? raw = Environment.GetEnvironmentVariable(SmEnvVar)?.Replace(".", "", StringComparison.Ordinal);
        if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int sm))
            return sm;
        if (gpu is null) return 0;
        return gpu.Contains("3060", StringComparison.Ordinal) ? 86 : gpu.Contains("4090", StringComparison.Ordinal) ? 89 : 0;
    }

    /// <summary>cuDNN's status text plus its last-error detail (cudnnGetLastErrorString) when the export resolved.</summary>
    private string Describe(int status, string what)
    {
        string text = $"{what}: {ErrorString(status)} ({status})";
        if (_lastError == null)
            return text;
        byte[] buffer = new byte[2048];
        fixed (byte* p = buffer)
            _lastError(p, (nuint)buffer.Length);
        int end = Array.IndexOf(buffer, (byte)0);
        string detail = Encoding.ASCII.GetString(buffer, 0, end < 0 ? buffer.Length : end).Trim();
        return detail.Length == 0 ? text : $"{text} | {detail}";
    }

    // ── run one spec ────────────────────────────────────────────────────

    public ProbeSpecResult Run(ProbeSpec spec)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Validate(spec);
        _ctx.EnsureCurrent();
        HostInputs host = HostInputs.Create(spec);
        int[] rows = ReferenceRows(spec, host.SeqQ);
        float[] scales = spec.ScaleList;
        double refMs = 0;
        Lazy<Reference[]> refs = new(() =>
        {
            long start = Stopwatch.GetTimestamp();
            Reference[] built = [.. scales.Select(s => Reference.Compute(spec, host, s, rows))];
            refMs = Ms(Stopwatch.GetTimestamp() - start);
            return built;
        });
        double tolerance = spec.Dtype == ProbeDtype.BF16 ? Bf16Tolerance : F16Tolerance;
        WriteSlots(spec);
        List<ProbeModeResult> modes = new();
        ushort[]? kept = null;
        using DeviceBuffers dev = DeviceBuffers.Upload(spec, host);
        using GraphBuilder g = new(this);
        nint graph = 0;
        string? graphError = null;
        try
        {
            graph = spec.Unified ? BuildUnified(g, spec, dev) : BuildComposite(g, spec, dev);
        }
        catch (ProbeRefusedException ex)
        {
            graphError = $"graph: {ex.Message}";
        }
        foreach ((string name, int mode) in _modes)
        {
            if (graph == 0)
                modes.Add(Result(name, scales[0], VerdictRefused, new PlanAttempt { Error = graphError }, false, Metrics.None, double.NaN, null));
            else
                modes.AddRange(RunMode(spec, g, graph, dev, name, mode, scales, refs, rows, tolerance, ref kept));
        }
        bool bad = modes.Exists(m => m.Verdict is VerdictWrong or VerdictExecError);
        bool ok = modes.Exists(m => m.Verdict == VerdictOk);
        string outcome = bad ? "WRONG" : ok ? "supported" : "unsupported";
        return new ProbeSpecResult(spec, outcome, tolerance, rows.Length, refMs, modes) { Output = kept };
    }

    private List<ProbeModeResult> RunMode(ProbeSpec spec, GraphBuilder g, nint graph, DeviceBuffers dev, string name, int mode,
        float[] scales, Lazy<Reference[]> refs, int[] rows, double tolerance, ref ushort[]? kept)
    {
        List<ProbeModeResult> results = new();
        PlanAttempt a = BuildPlan(graph, mode);
        if (a.Plan == 0)
        {
            results.Add(Result(name, scales[0], VerdictRefused, a, false, Metrics.None, double.NaN, null));
            return results;
        }
        ulong workspace = 0;
        try
        {
            if (a.WorkspaceBytes > WorkspaceCap)
            {
                results.Add(Result(name, scales[0], VerdictWorkspaceCap, a, false, Metrics.None, double.NaN,
                    $"workspace {a.WorkspaceBytes} bytes is over the probe's {WorkspaceCap}-byte cap"));
                return results;
            }
            if (a.WorkspaceBytes > 0)
                CudaDriverApi.cuMemAlloc(out workspace, (nuint)a.WorkspaceBytes).ThrowOnError();
            for (int si = 0; si < scales.Length; si++)
            {
                SetScale(spec, dev, scales[si]);
                dev.Poison();
                int st = Execute(g, a.Plan, workspace);
                if (st != CUDNN_STATUS_SUCCESS)
                {
                    results.Add(Result(name, scales[si], VerdictExecError, a, false, Metrics.None, double.NaN, Describe(st, "execute")));
                    continue;
                }
                CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
                ushort[] output = dev.ReadOutput();
                float[]? stats = spec.Stats ? dev.ReadStats() : null;
                Metrics m = Compare(spec, output, stats, refs.Value[si], rows);
                bool wrong = m.Nan > 0 || !(m.RelL2 <= tolerance)
                    || (spec.Stats && !(Math.Min(m.StatsRelL2, m.StatsRelL2Log2) <= StatsTolerance));
                double execMs = si == 0 ? TimeExecutions(g, a.Plan, workspace) : double.NaN;
                if (!wrong && si == 0 && spec.KeepOutput)
                    kept ??= output;
                results.Add(Result(name, scales[si], wrong ? VerdictWrong : VerdictOk, a, true, m, execMs, null));
            }
            return results;
        }
        finally
        {
            if (workspace != 0) CudaDriverApi.cuMemFree(workspace);
            cudnnBackendDestroyDescriptor(a.Plan);
        }
    }

    private static ProbeModeResult Result(string mode, float scale, string verdict, PlanAttempt a, bool executed, Metrics m,
        double execMs, string? error) =>
        new(mode, scale, verdict, a.Plan != 0, executed, m.RelL2, m.MaxAbs, m.Cosine, m.Nan, m.StatsRelL2, m.StatsRelL2Log2,
            Ms(a.HeuristicTicks + a.FinalizeTicks), Ms(a.HeuristicTicks), Ms(a.FinalizeTicks), execMs,
            a.ConfigsReturned, a.ConfigsTried, a.EngineGlobalIndex, a.RuntimeCompiled, a.NumericalNotes, a.WorkspaceBytes,
            error ?? a.Error);

    private static void Validate(ProbeSpec s)
    {
        if (s.B <= 0 || s.Hq <= 0 || s.Sq <= 0 || s.Skv <= 0 || s.D <= 0 || s.Hq % s.KvHeads != 0 || s.KvRows < s.Skv)
            throw new ArgumentException($"{s.Name}: bad shape");
        if (s.Padding == ProbePadding.None ? s.SeqQ is not null || s.SeqKv is not null
                : s.SeqQ?.Length != s.B || s.SeqKv?.Length != s.B)
            throw new ArgumentException($"{s.Name}: SeqQ/SeqKv must be given (one per batch) exactly when padding is on");
        if (s.Bias is ProbeBias b && (b.B is not 1 && b.B != s.B || b.H is not 1 && b.H != s.Hq || b.Sq is not 1 && b.Sq != s.Sq))
            throw new ArgumentException($"{s.Name}: bias dims must be 1 or the matching score dim");
        if (s.Unified && (s.Softcap > 0 || s.Alibi || s.Sink || s.Window > 0 || s.MaskForm == ProbeMaskForm.LegacyChain
                || s.Causal == ProbeCausal.BottomRight || s.ShiftRightBound is not null || s.Padding == ProbePadding.Composite))
            throw new ArgumentException($"{s.Name}: the unified builder covers top-left band causal, bias, stats, seq-len padding and block mask only");
        if (!s.Unified && (s.Padding == ProbePadding.Unified || s.BlockMask))
            throw new ArgumentException($"{s.Name}: seq-len padding descriptors and block masks need the unified op");
        if (s.MaskForm == ProbeMaskForm.LegacyChain && (s.Causal != ProbeCausal.TopLeft || s.ShiftRightBound is not null))
            throw new ArgumentException($"{s.Name}: the legacy chain is top-left causal only");
        if (s.BlockMask && s.B != 1)
            throw new ArgumentException($"{s.Name}: block-mask probes use B = 1");
    }

    private void WriteSlots(ProbeSpec s)
    {
        *(float*)Slot(SlotFill) = s.Fill;
        *(float*)Slot(SlotCap) = s.Softcap;
        if (s.BoundType == ProbeBoundType.Int32)
        {
            *(int*)Slot(SlotWindow) = checked((int)s.Window);
            *(int*)Slot(SlotShift) = checked((int)(s.ShiftRightBound ?? 0));
        }
        else
        {
            *(float*)Slot(SlotWindow) = s.Window;
            *(float*)Slot(SlotShift) = s.ShiftRightBound ?? 0;
        }
    }

    private void SetScale(ProbeSpec spec, DeviceBuffers dev, float scale)
    {
        if (spec.ScaleByValue)
            *(float*)Slot(SlotScale) = scale;
        else
            CudaDriverApi.cuMemcpyHtoD(dev.Scale, (nint)(&scale), sizeof(float)).ThrowOnError();
    }

    private byte* Slot(int index) => _slots + index * SlotBytes;

    // ── graphs ──────────────────────────────────────────────────────────

    /// <summary>The frontend's composite form (research report §2, C1): bmm1 → scale → [softcap] → [bias] → [ALiBi] →
    /// [padding select] → [causal band or legacy chain] → [window band] → softmax [stats, sink] → bmm2.</summary>
    private nint BuildComposite(GraphBuilder g, ProbeSpec s, DeviceBuffers dev)
    {
        int io = IoType(s.Dtype);
        long b = s.B, hq = s.Hq, hk = s.KvHeads, sq = s.Sq, skv = s.Skv, d = s.D;
        long[] qs = QStrides(s), ks = KvStrides(s);
        nint tQ = g.Device([b, hq, sq, d], qs, io, dev.Q);
        nint tKt = g.Device([b, hk, d, skv], [ks[0], ks[1], 1, ks[2]], io, dev.K);   // Kᵀ: dims and strides 2↔3 swapped
        nint tV = g.Device([b, hk, skv, d], ks, io, dev.V);
        nint tO = g.Device([b, hq, sq, d], qs, io, dev.O);
        long[] sDim = [b, hq, sq, skv];
        bool padded = s.Padding == ProbePadding.Composite;
        bool bottomRight = s.Causal == ProbeCausal.BottomRight;
        nint seqQ = padded || bottomRight ? g.Device([b, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_INT32, dev.SeqQ) : 0;
        nint seqKv = padded || bottomRight ? g.Device([b, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_INT32, dev.SeqKv) : 0;
        List<nint> ops = new();
        nint cur = g.Virtual(sDim, CUDNN_DATA_FLOAT);
        ops.Add(g.Matmul(tQ, tKt, cur, padded ? seqQ : 0, padded ? seqKv : 0, 0, zeroPadding: padded));
        nint scale = s.ScaleByValue
            ? g.ByValue(CUDNN_DATA_FLOAT, Slot(SlotScale))
            : g.Device([1, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_FLOAT, dev.Scale);
        cur = Apply(g, ops, CUDNN_POINTWISE_MUL, cur, scale, sDim);
        if (s.Softcap > 0)
        {
            nint cap = g.ByValue(CUDNN_DATA_FLOAT, Slot(SlotCap));
            cur = Apply(g, ops, CUDNN_POINTWISE_DIV, cur, cap, sDim);
            cur = Apply(g, ops, CUDNN_POINTWISE_TANH_FWD, cur, 0, sDim);
            cur = Apply(g, ops, CUDNN_POINTWISE_MUL, cur, cap, sDim);
        }
        if (s.Bias is ProbeBias bias)
            cur = Apply(g, ops, CUDNN_POINTWISE_ADD, cur, g.Device(BiasDims(s, bias), BiasStrides(s, bias), BiasType(bias.Type), dev.Bias), sDim);
        if (s.Alibi)
        {
            (nint row, nint col) = Indices(g, ops, cur, sDim);
            nint rel = g.Virtual(sDim, CUDNN_DATA_INT32);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_SUB, CUDNN_DATA_INT32, col, row, rel));
            nint slopes = g.Device([1, hq, 1, 1], [hq, 1, 1, 1], CUDNN_DATA_FLOAT, dev.Slopes);
            nint alibi = g.Virtual(sDim, CUDNN_DATA_FLOAT);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_MUL, CUDNN_DATA_FLOAT, rel, slopes, alibi));
            cur = Apply(g, ops, CUDNN_POINTWISE_ADD, cur, alibi, sDim);
        }
        nint fill = g.ByValue(CUDNN_DATA_FLOAT, Slot(SlotFill));
        if (padded)
        {
            (nint row, nint col) = Indices(g, ops, cur, sDim);
            nint rowOk = g.Virtual(sDim, CUDNN_DATA_BOOLEAN);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_CMP_LT, CUDNN_DATA_BOOLEAN, row, seqQ, rowOk));
            nint colOk = g.Virtual(sDim, CUDNN_DATA_BOOLEAN);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_CMP_LT, CUDNN_DATA_BOOLEAN, col, seqKv, colOk));
            nint keep = g.Virtual(sDim, CUDNN_DATA_BOOLEAN);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_LOGICAL_AND, CUDNN_DATA_BOOLEAN, rowOk, colOk, keep));
            nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_BINARY_SELECT, CUDNN_DATA_FLOAT, cur, fill, y, keep));
            cur = y;
        }
        if (s.Causal != ProbeCausal.None && s.MaskForm == ProbeMaskForm.LegacyChain)
        {
            (nint row, nint col) = Indices(g, ops, cur, sDim);
            nint keep = g.Virtual(sDim, CUDNN_DATA_BOOLEAN);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_CMP_GE, CUDNN_DATA_BOOLEAN, row, col, keep));
            nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT);
            ops.Add(g.Pointwise(CUDNN_POINTWISE_BINARY_SELECT, CUDNN_DATA_FLOAT, cur, fill, y, keep));
            cur = y;
        }
        else if (s.Causal != ProbeCausal.None)
        {
            nint shift = s.ShiftRightBound is null ? 0 : g.ByValue(BoundType(s), Slot(SlotShift));
            nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT);
            ops.Add(g.Band(cur, y, fill, CUDNN_POINTWISE_CMP_GE, 0, shift, bottomRight ? seqQ : 0, bottomRight ? seqKv : 0));
            cur = y;
        }
        if (s.Window > 0)
        {
            nint left = g.ByValue(BoundType(s), Slot(SlotWindow));
            nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT);
            ops.Add(g.Band(cur, y, fill, CUDNN_POINTWISE_CMP_GT, left, 0, bottomRight ? seqQ : 0, bottomRight ? seqKv : 0));
            cur = y;
        }
        nint p = g.Virtual(sDim, io);   // P takes the I/O dtype before bmm2, as the frontend does
        nint stats = s.Stats ? g.Device([b, hq, sq, 1], [hq * sq, sq, 1, 1], CUDNN_DATA_FLOAT, dev.Stats) : 0;
        nint sink = s.Sink ? g.Device([1, hq, 1, 1], [hq, 1, 1, 1], CUDNN_DATA_FLOAT, dev.Sink) : 0;
        ops.Add(g.Softmax(cur, p, stats, sink));
        ops.Add(g.Matmul(p, tV, tO, padded ? seqQ : 0, 0, padded ? seqKv : 0, zeroPadding: false));
        return g.OpGraph(ops, "operation graph");
    }

    /// <summary>The unified form (research report §2, C2): one SDPA_FWD op, K not transposed, with an optional
    /// subgraph holding the bias add and the causal band.</summary>
    private nint BuildUnified(GraphBuilder g, ProbeSpec s, DeviceBuffers dev)
    {
        int io = IoType(s.Dtype);
        long b = s.B, hq = s.Hq, hk = s.KvHeads, sq = s.Sq, skv = s.Skv, d = s.D;
        long[] qs = QStrides(s), ks = KvStrides(s);
        nint tQ = g.Device([b, hq, sq, d], qs, io, dev.Q);
        nint tK = g.Device([b, hk, skv, d], ks, io, dev.K);
        nint tV = g.Device([b, hk, skv, d], ks, io, dev.V);
        nint tO = g.Device([b, hq, sq, d], qs, io, dev.O);
        nint scale = s.ScaleByValue
            ? g.ByValue(CUDNN_DATA_FLOAT, Slot(SlotScale))
            : g.Device([1, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_FLOAT, dev.Scale);
        nint stats = s.Stats ? g.Device([b, hq, sq, 1], [hq * sq, sq, 1, 1], CUDNN_DATA_FLOAT, dev.Stats) : 0;
        bool padded = s.Padding == ProbePadding.Unified;
        nint seqQ = padded ? g.Device([b, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_INT32, dev.SeqQ) : 0;
        nint seqKv = padded ? g.Device([b, 1, 1, 1], [1, 1, 1, 1], CUDNN_DATA_INT32, dev.SeqKv) : 0;
        nint mask = 0;
        if (s.BlockMask)
        {
            long[] maskDims = [1, hq, Tiles(sq), Tiles(skv)];
            mask = g.Device(maskDims, Dense(maskDims), CUDNN_DATA_INT8, dev.BlockMask);
        }
        nint sub = 0;
        long inUid = 0, outUid = 0;
        if (s.Bias is not null || s.Causal != ProbeCausal.None)
        {
            long[] sDim = [b, hq, sq, skv];
            List<nint> subOps = new();
            nint cur = g.Virtual(sDim, CUDNN_DATA_FLOAT, out inUid);
            outUid = inUid;
            if (s.Bias is ProbeBias bias)
            {
                nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT, out outUid);
                nint tBias = g.Device(BiasDims(s, bias), BiasStrides(s, bias), BiasType(bias.Type), dev.Bias);
                subOps.Add(g.Pointwise(CUDNN_POINTWISE_ADD, CUDNN_DATA_FLOAT, cur, tBias, y));
                cur = y;
            }
            if (s.Causal != ProbeCausal.None)
            {
                nint y = g.Virtual(sDim, CUDNN_DATA_FLOAT, out outUid);
                subOps.Add(g.Band(cur, y, g.ByValue(CUDNN_DATA_FLOAT, Slot(SlotFill)), CUDNN_POINTWISE_CMP_GE, 0, 0, 0, 0));
            }
            sub = g.OpGraph(subOps, "SDPA subgraph");
        }
        nint op = g.SdpaFwd(tQ, tK, tV, tO, scale, stats, seqQ, seqKv, mask, sub, inUid, outUid);
        return g.OpGraph([op], "operation graph");
    }

    private static nint Apply(GraphBuilder g, List<nint> ops, int mode, nint x, nint b, long[] dims)
    {
        nint y = g.Virtual(dims, CUDNN_DATA_FLOAT);
        ops.Add(g.Pointwise(mode, CUDNN_DATA_FLOAT, x, b, y));
        return y;
    }

    private static (nint Row, nint Col) Indices(GraphBuilder g, List<nint> ops, nint scores, long[] dims)
    {
        nint row = g.Virtual(dims, CUDNN_DATA_INT32);
        ops.Add(g.Pointwise(CUDNN_POINTWISE_GEN_INDEX, CUDNN_DATA_FLOAT, scores, 0, row, axis: 2));
        nint col = g.Virtual(dims, CUDNN_DATA_INT32);
        ops.Add(g.Pointwise(CUDNN_POINTWISE_GEN_INDEX, CUDNN_DATA_FLOAT, scores, 0, col, axis: 3));
        return (row, col);
    }

    // ── planning and execution ─────────────────────────────────────────

    private sealed class PlanAttempt
    {
        public nint Plan;
        public long WorkspaceBytes;
        public long HeuristicTicks, FinalizeTicks;
        public int ConfigsReturned, ConfigsTried;
        public long EngineGlobalIndex = -1;
        public bool RuntimeCompiled;
        public int[] NumericalNotes = [];
        public string? Error;
    }

    /// <summary>One heuristic mode: query, then finalize configs in order until one yields a plan. Never throws; every
    /// failure lands in <see cref="PlanAttempt.Error"/> with cuDNN's own text.</summary>
    private PlanAttempt BuildPlan(nint graph, int mode)
    {
        PlanAttempt a = new();
        long start = Stopwatch.GetTimestamp();
        nint heur = 0;
        nint[] cfgs = new nint[MaxConfigs];
        int created = 0;
        try
        {
            int st = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, out heur);
            if (st == CUDNN_STATUS_SUCCESS)
            {
                void* gp = (void*)graph;
                int m = mode;
                st = cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &gp);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendFinalize(heur);
            }
            if (st != CUDNN_STATUS_SUCCESS)
            {
                a.HeuristicTicks = Stopwatch.GetTimestamp() - start;
                a.Error = Describe(st, "heuristic");
                return a;
            }
            for (; created < MaxConfigs; created++)
            {
                st = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, out cfgs[created]);
                if (st != CUDNN_STATUS_SUCCESS)
                {
                    a.Error = Describe(st, "engine config create");
                    return a;
                }
            }
            long returned;
            fixed (nint* cp = cfgs)
                st = cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, MaxConfigs, out returned, cp);
            a.HeuristicTicks = Stopwatch.GetTimestamp() - start;
            if (st != CUDNN_STATUS_SUCCESS)
            {
                a.Error = Describe(st, "heuristic results");
                return a;
            }
            a.ConfigsReturned = (int)Math.Min(returned, MaxConfigs);
            long finalizeStart = Stopwatch.GetTimestamp();
            string? lastError = null;
            for (int i = 0; i < a.ConfigsReturned && a.Plan == 0; i++)
            {
                a.ConfigsTried++;
                st = FinalizePlan(cfgs[i], out nint plan, out long ws);
                if (st == CUDNN_STATUS_SUCCESS)
                {
                    a.Plan = plan;
                    a.WorkspaceBytes = ws;
                    DescribeEngine(cfgs[i], a);
                }
                else
                {
                    lastError = Describe(st, $"plan finalize, config {i}");
                }
            }
            a.FinalizeTicks = Stopwatch.GetTimestamp() - finalizeStart;
            if (a.Plan == 0)
                a.Error = a.ConfigsReturned == 0 ? "heuristic returned no engine configs" : lastError;
            return a;
        }
        finally
        {
            for (int i = 0; i < created; i++)
                cudnnBackendDestroyDescriptor(cfgs[i]);
            if (heur != 0)
                cudnnBackendDestroyDescriptor(heur);
        }
    }

    private int FinalizePlan(nint cfg, out nint plan, out long workspace)
    {
        workspace = 0;
        int st = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, out plan);
        if (st != CUDNN_STATUS_SUCCESS)
        {
            plan = 0;
            return st;
        }
        void* hp = (void*)_handle, cp = (void*)cfg;
        long ws = 0;
        st = cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
        if (st == CUDNN_STATUS_SUCCESS)
            st = cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        if (st == CUDNN_STATUS_SUCCESS)
            st = cudnnBackendFinalize(plan);
        if (st == CUDNN_STATUS_SUCCESS)
            st = cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, out _, &ws);
        if (st != CUDNN_STATUS_SUCCESS)
        {
            cudnnBackendDestroyDescriptor(plan);
            plan = 0;
            return st;
        }
        workspace = ws;
        return CUDNN_STATUS_SUCCESS;
    }

    private static void DescribeEngine(nint cfg, PlanAttempt a)
    {
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, out nint engine) != CUDNN_STATUS_SUCCESS)
            return;
        try
        {
            nint target = engine;
            if (cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, out long n, &target)
                    != CUDNN_STATUS_SUCCESS || n < 1)
                return;
            long index = -1;
            if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, out _, &index) == CUDNN_STATUS_SUCCESS)
                a.EngineGlobalIndex = index;
            int[] notes = new int[16];
            fixed (int* np = notes)
            {
                if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, notes.Length,
                        out long behaviorCount, np) == CUDNN_STATUS_SUCCESS)
                {
                    for (int i = 0; i < behaviorCount && i < notes.Length; i++)
                        a.RuntimeCompiled |= notes[i] == CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
                }
                if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, notes.Length,
                        out long numericalCount, np) == CUDNN_STATUS_SUCCESS)
                    a.NumericalNotes = notes[..(int)Math.Min(numericalCount, notes.Length)];
            }
        }
        finally
        {
            cudnnBackendDestroyDescriptor(engine);
        }
    }

    private int Execute(GraphBuilder g, nint plan, ulong workspace)
    {
        int n = g.Bindings.Count;
        long[] uids = new long[n];
        nint[] pointers = new nint[n];
        for (int i = 0; i < n; i++)
            (uids[i], pointers[i]) = g.Bindings[i];
        int st = cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, out nint vp);
        if (st != CUDNN_STATUS_SUCCESS)
            return st;
        try
        {
            void* ws = (void*)workspace;
            fixed (long* up = uids)
            fixed (nint* pp = pointers)
            {
                st = cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, n, up);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, n, pp);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendFinalize(vp);
                if (st == CUDNN_STATUS_SUCCESS)
                    st = cudnnBackendExecute(_handle, plan, vp);
            }
            return st;
        }
        finally
        {
            cudnnBackendDestroyDescriptor(vp);
        }
    }

    /// <summary>Median wall time of <see cref="TimedRuns"/> executions, each with its variant pack and a stream sync
    /// (the driver bindings have no event-timing call).</summary>
    private double TimeExecutions(GraphBuilder g, nint plan, ulong workspace)
    {
        double[] ms = new double[TimedRuns];
        for (int i = 0; i < TimedRuns; i++)
        {
            long start = Stopwatch.GetTimestamp();
            if (Execute(g, plan, workspace) != CUDNN_STATUS_SUCCESS)
                return double.NaN;
            CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
            ms[i] = Ms(Stopwatch.GetTimestamp() - start);
        }
        Array.Sort(ms);
        return ms[TimedRuns / 2];
    }

    // ── reference and comparison ───────────────────────────────────────

    private static int[] ReferenceRows(ProbeSpec s, int[] seqQ)
    {
        double budget = double.TryParse(Environment.GetEnvironmentVariable(RefMacsEnvVar), NumberStyles.Float,
            CultureInfo.InvariantCulture, out double v) && v > 0 ? v : DefaultRefMacs;
        double macs = 2.0 * s.B * s.Hq * s.Sq * s.Skv * s.D;
        long stride = Math.Max(1, (long)Math.Ceiling(macs / budget));
        SortedSet<int> rows = new();
        for (long i = 0; i < s.Sq; i += stride)
            rows.Add((int)i);
        rows.Add((int)s.Sq - 1);
        foreach (int q in seqQ)
            if (q > 0) rows.Add(q - 1);
        return [.. rows];
    }

    private readonly record struct Metrics(double RelL2, double MaxAbs, double Cosine, long Nan, double StatsRelL2, double StatsRelL2Log2)
    {
        public static Metrics None => new(double.NaN, double.NaN, double.NaN, 0, double.NaN, double.NaN);
    }

    private static Metrics Compare(ProbeSpec s, ushort[] output, float[]? stats, Reference r, int[] rows)
    {
        long[] qs = QStrides(s);
        long nRows = rows.Length, d = s.D;
        double errSq = 0, refSq = 0, outSq = 0, dot = 0, maxAbs = 0, lseErr = 0, lseRef = 0, lse2Err = 0, lse2Ref = 0;
        long nan = 0, n = 0;
        for (long bi = 0; bi < s.B; bi++)
            for (long h = 0; h < s.Hq; h++)
                for (long ri = 0; ri < nRows; ri++)
                {
                    long item = (bi * s.Hq + h) * nRows + ri;
                    if (double.IsNaN(r.Lse[item]))
                        continue;   // past this batch's query length
                    long i = rows[ri], at = bi * qs[0] + h * qs[1] + i * qs[2];
                    for (long e = 0; e < d; e++)
                    {
                        double got = FromBits(output[at + e], s.Dtype), want = r.Out[item * d + e];
                        n++;
                        if (!double.IsFinite(got))
                        {
                            nan++;
                            continue;
                        }
                        double diff = got - want;
                        errSq += diff * diff;
                        refSq += want * want;
                        outSq += got * got;
                        dot += got * want;
                        maxAbs = Math.Max(maxAbs, Math.Abs(diff));
                    }
                    if (stats is null)
                        continue;
                    double lse = r.Lse[item], lse2 = lse / Math.Log(2), stat = stats[(bi * s.Hq + h) * s.Sq + i];
                    lseErr += (stat - lse) * (stat - lse);
                    lseRef += lse * lse;
                    lse2Err += (stat - lse2) * (stat - lse2);
                    lse2Ref += lse2 * lse2;
                }
        // Relative L2 with an absolute floor on the denominator, so a near-zero reference cannot inflate it.
        double relL2 = Math.Sqrt(errSq) / Math.Max(Math.Sqrt(refSq), 1e-6 * Math.Sqrt(Math.Max(n, 1)));
        double cos = outSq > 0 && refSq > 0 ? dot / Math.Sqrt(outSq * refSq) : double.NaN;
        double statsRel = stats is null ? double.NaN : Math.Sqrt(lseErr) / Math.Max(Math.Sqrt(lseRef), 1e-6);
        double statsRel2 = stats is null ? double.NaN : Math.Sqrt(lse2Err) / Math.Max(Math.Sqrt(lse2Ref), 1e-6);
        return new Metrics(relL2, maxAbs, cos, nan, statsRel, statsRel2);
    }

    private sealed class Reference
    {
        public double[] Out = [];   // [b,hq,rows,d]
        public double[] Lse = [];   // [b,hq,rows]; NaN marks a row past the batch's query length

        /// <summary>float64 softmax(scale·QKᵀ [softcap] + bias + ALiBi, masked) · V, plus the natural-log LSE. Masks follow
        /// the frontend: right bound keeps j ≤ i + diag, left bound keeps j &gt; i + diag − W, diag = Skv − Sq for
        /// bottom-right and the shift otherwise; a sink adds exp(sink) to the denominator.</summary>
        public static Reference Compute(ProbeSpec s, HostInputs x, float scale, int[] rows)
        {
            long nRows = rows.Length, hq = s.Hq, d = s.D, skv = s.Skv, sq = s.Sq, kvRows = s.KvRows, hp = s.KvPhysicalHeads;
            long group = s.Hq / s.KvHeads, window = s.Window, shift = s.ShiftRightBound ?? 0, tq = Tiles(sq), tk = Tiles(skv);
            bool stride0 = s.KvHeadStrideZero, causal = s.Causal != ProbeCausal.None, bottomRight = s.Causal == ProbeCausal.BottomRight;
            double cap = s.Softcap;
            long bB = s.Bias?.B ?? 1, bH = s.Bias?.H ?? 1, bS = s.Bias is null ? 1 : BiasRows(s.Bias);
            float[] q = x.Q, k = x.K, v = x.V;
            float[]? bias = x.Bias, sink = x.Sink, slopes = x.Slopes;
            byte[]? mask = x.BlockMask;
            int[] seqQ = x.SeqQ, seqKv = x.SeqKv;
            Reference r = new() { Out = new double[s.B * hq * nRows * d], Lse = new double[s.B * hq * nRows] };
            Parallel.For(0L, s.B * hq * nRows, () => new double[skv], (item, _, scores) =>
            {
                long ri = item % nRows, h = item / nRows % hq, bi = item / (nRows * hq);
                long i = rows[ri], sQ = seqQ[bi], sK = seqKv[bi];
                Span<double> o = r.Out.AsSpan((int)(item * d), (int)d);
                if (i >= sQ)
                {
                    o.Fill(double.NaN);
                    r.Lse[item] = double.NaN;
                    return scores;
                }
                long kh = stride0 ? 0 : h / group;
                long diag = bottomRight ? sK - sQ : shift;
                long qBase = ((bi * hq + h) * sq + i) * d;
                double max = double.NegativeInfinity;
                for (long j = 0; j < skv; j++)
                {
                    bool visible = j < sK
                        && (!causal || j <= i + diag)
                        && (window <= 0 || j > i + diag - window)
                        && (mask is null || mask[(h * tq + i / 128) * tk + j / 128] != 0);
                    if (!visible)
                    {
                        scores[j] = double.NegativeInfinity;
                        continue;
                    }
                    long kBase = ((bi * hp + kh) * kvRows + j) * d;
                    double dot = 0;
                    for (long e = 0; e < d; e++)
                        dot += (double)q[qBase + e] * k[kBase + e];
                    double sc = dot * scale;
                    if (cap > 0)
                        sc = cap * Math.Tanh(sc / cap);
                    if (bias is not null)
                        sc += bias[(((bB == 1 ? 0 : bi) * bH + (bH == 1 ? 0 : h)) * bS + (bS == 1 ? 0 : i)) * skv + j];
                    if (slopes is not null)
                        sc += slopes[h] * (double)(j - i);
                    scores[j] = sc;
                    if (sc > max) max = sc;
                }
                double sinkLogit = sink is null ? double.NegativeInfinity : sink[h];
                double m = Math.Max(max, sinkLogit);
                double l = sink is null ? 0 : Math.Exp(sinkLogit - m);
                for (long j = 0; j < skv; j++)
                {
                    double p = double.IsNegativeInfinity(scores[j]) ? 0 : Math.Exp(scores[j] - m);
                    scores[j] = p;
                    l += p;
                }
                o.Clear();
                for (long j = 0; j < skv; j++)
                {
                    double p = scores[j];
                    if (p == 0.0) continue;
                    long vBase = ((bi * hp + kh) * kvRows + j) * d;
                    for (int e = 0; e < d; e++)
                        o[e] += p * v[vBase + e];
                }
                for (int e = 0; e < d; e++)
                    o[e] /= l;
                r.Lse[item] = m + Math.Log(l);
                return scores;
            }, _ => { });
            return r;
        }
    }

    // ── host inputs and device buffers ─────────────────────────────────

    private sealed class HostInputs
    {
        public float[] Q = [], K = [], V = [];   // logical [b,h,rows,d], quantized to the I/O dtype
        public float[]? Bias, Sink, Slopes;
        public byte[]? BlockMask;
        public int[] SeqQ = [], SeqKv = [];

        public static HostInputs Create(ProbeSpec s)
        {
            Random rng = new(s.Seed);
            HostInputs x = new()
            {
                SeqQ = s.SeqQ ?? Enumerable.Repeat((int)s.Sq, (int)s.B).ToArray(),
                SeqKv = s.SeqKv ?? Enumerable.Repeat((int)s.Skv, (int)s.B).ToArray(),
            };
            x.Q = Values(rng, s.B, s.Hq, s.Sq, s.D, x.SeqQ, s.Dtype);
            x.K = Values(rng, s.B, s.KvPhysicalHeads, s.KvRows, s.D, x.SeqKv, s.Dtype);
            x.V = Values(rng, s.B, s.KvPhysicalHeads, s.KvRows, s.D, x.SeqKv, s.Dtype);
            if (s.Bias is ProbeBias bias)
            {
                x.Bias = new float[bias.B * bias.H * BiasRows(bias) * s.Skv];
                for (long i = 0; i < x.Bias.Length; i++)
                    x.Bias[i] = QuantizeBias(Gauss(rng), bias.Type);
            }
            if (s.Sink)
            {
                x.Sink = new float[s.Hq];
                for (int h = 0; h < s.Hq; h++)
                    x.Sink[h] = Gauss(rng);
            }
            if (s.Alibi)
            {
                x.Slopes = new float[s.Hq];
                for (int h = 0; h < s.Hq; h++)
                    x.Slopes[h] = (float)Math.Pow(2, -8.0 * (h + 1) / s.Hq);
            }
            if (s.BlockMask)
            {
                long tq = Tiles(s.Sq), tk = Tiles(s.Skv);
                x.BlockMask = new byte[s.Hq * tq * tk];
                for (long h = 0; h < s.Hq; h++)
                    for (long ti = 0; ti < tq; ti++)
                        for (long tj = 0; tj < tk; tj++)
                            // The diagonal tile stays on, so no query row is fully masked.
                            x.BlockMask[(h * tq + ti) * tk + tj] = (byte)(ti == tj || (ti + 2 * tj + h) % 3 != 0 ? 1 : 0);
            }
            return x;
        }

        private static float[] Values(Random rng, long b, long h, long rows, long d, int[] valid, ProbeDtype dt)
        {
            float[] t = new float[b * h * rows * d];
            long n = 0;
            for (long bi = 0; bi < b; bi++)
                for (long hi = 0; hi < h; hi++)
                    for (long r = 0; r < rows; r++)
                        for (long e = 0; e < d; e++)
                            // Rows past the batch's length hold large finite junk, so a kernel that reads them shows in the error.
                            t[n++] = Quantize(r < valid[bi] ? Gauss(rng) : (r + e) % 2 == 0 ? 1000f : -1000f, dt);
            return t;
        }
    }

    private sealed class DeviceBuffers : IDisposable
    {
        private readonly List<ulong> _owned = new();
        public ulong Q, K, V, O, Scale, Bias, Stats, Sink, Slopes, SeqQ, SeqKv, BlockMask;
        public long OutputElements, StatsElements;

        public static DeviceBuffers Upload(ProbeSpec s, HostInputs x)
        {
            DeviceBuffers dev = new();
            try
            {
                dev.Q = dev.Copy(Image(x.Q, s.B, s.Hq, s.Sq, s.D, QStrides(s), s.Dtype));
                dev.K = dev.Copy(Image(x.K, s.B, s.KvPhysicalHeads, s.KvRows, s.D, KvStrides(s), s.Dtype));
                dev.V = dev.Copy(Image(x.V, s.B, s.KvPhysicalHeads, s.KvRows, s.D, KvStrides(s), s.Dtype));
                dev.OutputElements = s.B * s.Hq * s.Sq * s.D;
                dev.O = dev.Alloc(dev.OutputElements * sizeof(ushort));
                dev.Scale = dev.Alloc(sizeof(float));
                dev.SeqQ = dev.Copy(x.SeqQ);
                dev.SeqKv = dev.Copy(x.SeqKv);
                if (x.Bias is not null && s.Bias is ProbeBias bias)
                    dev.Bias = bias.Type == ProbeBiasType.F32 ? dev.Copy(x.Bias)
                        : dev.Copy(Bits(x.Bias, bias.Type == ProbeBiasType.BF16 ? ProbeDtype.BF16 : ProbeDtype.F16));
                if (s.Stats)
                {
                    dev.StatsElements = s.B * s.Hq * s.Sq;
                    dev.Stats = dev.Alloc(dev.StatsElements * sizeof(float));
                }
                if (x.Sink is not null) dev.Sink = dev.Copy(x.Sink);
                if (x.Slopes is not null) dev.Slopes = dev.Copy(x.Slopes);
                if (x.BlockMask is not null) dev.BlockMask = dev.Copy(x.BlockMask);
                return dev;
            }
            catch
            {
                dev.Dispose();
                throw;
            }
        }

        public ulong Alloc(long bytes)
        {
            CudaDriverApi.cuMemAlloc(out ulong p, (nuint)bytes).ThrowOnError();
            _owned.Add(p);
            return p;
        }

        public ulong Copy<T>(T[] data) where T : unmanaged
        {
            long bytes = (long)data.Length * sizeof(T);
            ulong p = Alloc(bytes);
            fixed (T* src = data)
                CudaDriverApi.cuMemcpyHtoD(p, (nint)src, (nuint)bytes).ThrowOnError();
            return p;
        }

        /// <summary>0xFF bytes are NaN in F16, BF16 and F32, so an engine that skips a write cannot pass.</summary>
        public void Poison()
        {
            CudaDriverApi.cuMemsetD8(O, 0xFF, (nuint)(OutputElements * sizeof(ushort))).ThrowOnError();
            if (Stats != 0)
                CudaDriverApi.cuMemsetD8(Stats, 0xFF, (nuint)(StatsElements * sizeof(float))).ThrowOnError();
        }

        public ushort[] ReadOutput()
        {
            ushort[] host = new ushort[OutputElements];
            fixed (ushort* p = host)
                CudaDriverApi.cuMemcpyDtoH((nint)p, O, (nuint)(OutputElements * sizeof(ushort))).ThrowOnError();
            return host;
        }

        public float[] ReadStats()
        {
            float[] host = new float[StatsElements];
            fixed (float* p = host)
                CudaDriverApi.cuMemcpyDtoH((nint)p, Stats, (nuint)(StatsElements * sizeof(float))).ThrowOnError();
            return host;
        }

        public void Dispose()
        {
            foreach (ulong p in _owned)
                CudaDriverApi.cuMemFree(p);
            _owned.Clear();
        }
    }

    // ── descriptor builder ─────────────────────────────────────────────

    private sealed class ProbeRefusedException(string message) : Exception(message);

    /// <summary>Creates and finalizes backend descriptors for one graph and remembers every (UID, pointer) the variant
    /// pack needs. Any refused attribute or finalize throws <see cref="ProbeRefusedException"/> carrying cuDNN's text.</summary>
    private sealed class GraphBuilder : IDisposable
    {
        private readonly CudnnSdpaProbe _owner;
        private readonly List<nint> _owned = new();
        private long _nextUid = 1;

        public GraphBuilder(CudnnSdpaProbe owner) => _owner = owner;

        /// <summary>Variant-pack bindings: device pointers, or host pointers for by-value scalars.</summary>
        public List<(long Uid, nint Pointer)> Bindings { get; } = new();

        public nint Device(long[] dims, long[] strides, int dtype, ulong pointer)
        {
            nint t = Tensor(dims, strides, dtype, isVirtual: false, byValue: false, out long uid);
            Bindings.Add((uid, (nint)pointer));
            return t;
        }

        public nint ByValue(int dtype, byte* host)
        {
            nint t = Tensor([1, 1, 1, 1], [1, 1, 1, 1], dtype, isVirtual: false, byValue: true, out long uid);
            Bindings.Add((uid, (nint)host));
            return t;
        }

        public nint Virtual(long[] dims, int dtype) => Tensor(dims, Dense(dims), dtype, isVirtual: true, byValue: false, out _);

        public nint Virtual(long[] dims, int dtype, out long uid) => Tensor(dims, Dense(dims), dtype, isVirtual: true, byValue: false, out uid);

        public nint Matmul(nint a, nint b, nint c, nint mOverride, nint nOverride, nint kOverride, bool zeroPadding)
        {
            nint mm = Create(CUDNN_BACKEND_MATMUL_DESCRIPTOR, "matmul desc");
            int comp = CUDNN_DATA_FLOAT;
            Set(mm, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp, "matmul comp type");
            if (zeroPadding)
            {
                // The header does not give this attribute's type: DOUBLE first, FLOAT if refused.
                double pad = 0;
                if (cudnnBackendSetAttribute(mm, CUDNN_ATTR_MATMUL_PADDING_VALUE, CUDNN_TYPE_DOUBLE, 1, &pad) != CUDNN_STATUS_SUCCESS)
                {
                    float padF = 0;
                    Set(mm, CUDNN_ATTR_MATMUL_PADDING_VALUE, CUDNN_TYPE_FLOAT, 1, &padF, "matmul padding value");
                }
            }
            Complete(mm, "matmul desc");
            nint op = Create(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, "matmul op");
            SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_ADESC, a, "matmul A");
            SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_BDESC, b, "matmul B");
            SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_CDESC, c, "matmul C");
            SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_DESC, mm, "matmul desc");
            if (mOverride != 0) SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC, mOverride, "matmul M override");
            if (nOverride != 0) SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC, nOverride, "matmul N override");
            if (kOverride != 0) SetDesc(op, CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC, kOverride, "matmul K override");
            Complete(op, "matmul op");
            return op;
        }

        public nint Pointwise(int mode, int precision, nint x, nint b, nint y, nint t = 0, long axis = -1)
        {
            nint pw = Create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, "pointwise desc");
            int m = mode, prec = precision;
            Set(pw, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &m, $"pointwise mode {mode}");
            Set(pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &prec, $"pointwise {mode} precision");
            if (axis >= 0)
            {
                long ax = axis;
                Set(pw, CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, 1, &ax, "pointwise axis");
            }
            Complete(pw, $"pointwise desc {mode}");
            nint op = Create(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, "pointwise op");
            SetDesc(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pw, "pointwise desc");
            SetDesc(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x, "pointwise X");
            if (b != 0) SetDesc(op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b, "pointwise B");
            SetDesc(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y, "pointwise Y");
            if (t != 0) SetDesc(op, CUDNN_ATTR_OPERATION_POINTWISE_TDESC, t, "pointwise T");
            Complete(op, $"pointwise op {mode}");
            return op;
        }

        public nint Band(nint x, nint y, nint fill, int comparison, nint leftBound, nint shiftRight, nint seqQ, nint seqKv)
        {
            nint op = Create(CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR, "band op");
            SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, x, "band X");
            SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, y, "band Y");
            SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, fill, "band fill");
            int cmp = comparison;
            Set(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &cmp, "band comparison");
            if (leftBound != 0) SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC, leftBound, "band left bound");
            if (shiftRight != 0) SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC, shiftRight, "band right shift");
            if (seqQ != 0) SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC, seqQ, "band seq_len q");
            if (seqKv != 0) SetDesc(op, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC, seqKv, "band seq_len kv");
            Complete(op, "band op");
            return op;
        }

        public nint Softmax(nint x, nint y, nint stats, nint sink)
        {
            nint op = Create(CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR, "softmax op");
            SetDesc(op, CUDNN_ATTR_OPERATION_SOFTMAX_XDESC, x, "softmax X");
            SetDesc(op, CUDNN_ATTR_OPERATION_SOFTMAX_YDESC, y, "softmax Y");
            if (stats != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC, stats, "softmax stats");
            if (sink != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC, sink, "softmax sink");
            Complete(op, "softmax op");
            return op;
        }

        public nint SdpaFwd(nint q, nint k, nint v, nint o, nint scale, nint stats, nint seqQ, nint seqKv, nint blockMask,
            nint subgraph, long inUid, long outUid)
        {
            nint op = Create(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR, "SDPA_FWD op");
            SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, q, "SDPA_FWD Q");
            SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, k, "SDPA_FWD K");
            SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, v, "SDPA_FWD V");
            SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, o, "SDPA_FWD O");
            SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, scale, "SDPA_FWD scale");
            if (stats != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC, stats, "SDPA_FWD stats");
            if (seqQ != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC, seqQ, "SDPA_FWD seq_len q");
            if (seqKv != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, seqKv, "SDPA_FWD seq_len kv");
            if (blockMask != 0) SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC, blockMask, "SDPA_FWD block mask");
            if (subgraph != 0)
            {
                long inId = inUid, outId = outUid;
                SetDesc(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH, subgraph, "SDPA_FWD subgraph");
                Set(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID, CUDNN_TYPE_INT64, 1, &inId, "SDPA_FWD subgraph input uid");
                Set(op, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID, CUDNN_TYPE_INT64, 1, &outId, "SDPA_FWD subgraph output uid");
            }
            Complete(op, "SDPA_FWD op");
            return op;
        }

        public nint OpGraph(List<nint> ops, string what)
        {
            nint graph = Create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, what);
            void* hp = (void*)_owner._handle;
            Set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp, $"{what} handle");
            nint[] array = [.. ops];
            fixed (nint* op = array)
                Set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, array.Length, op, $"{what} ops");
            Complete(graph, what);
            return graph;
        }

        private nint Tensor(long[] dims, long[] strides, int dtype, bool isVirtual, bool byValue, out long uid)
        {
            nint t = Create(CUDNN_BACKEND_TENSOR_DESCRIPTOR, "tensor");
            uid = _nextUid++;
            int type = dtype;
            long id = uid, alignment = 16;
            byte virt = (byte)(isVirtual ? 1 : 0), byVal = 1;
            Set(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type, "tensor data type");
            fixed (long* dp = dims)
                Set(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims.Length, dp, "tensor dims");
            fixed (long* sp = strides)
                Set(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides.Length, sp, "tensor strides");
            Set(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id, "tensor uid");
            Set(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment, "tensor alignment");
            Set(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &virt, "tensor virtual");
            if (byValue)
                Set(t, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &byVal, "tensor by-value");
            Complete(t, "tensor");
            return t;
        }

        private nint Create(int type, string what)
        {
            int st = cudnnBackendCreateDescriptor(type, out nint desc);
            if (st != CUDNN_STATUS_SUCCESS)
                throw new ProbeRefusedException(_owner.Describe(st, $"{what} create"));
            _owned.Add(desc);
            return desc;
        }

        private void Set(nint desc, int attribute, int type, long count, void* value, string what)
        {
            int st = cudnnBackendSetAttribute(desc, attribute, type, count, value);
            if (st != CUDNN_STATUS_SUCCESS)
                throw new ProbeRefusedException(_owner.Describe(st, what));
        }

        private void SetDesc(nint desc, int attribute, nint value, string what)
        {
            void* p = (void*)value;
            Set(desc, attribute, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &p, what);
        }

        private void Complete(nint desc, string what)
        {
            int st = cudnnBackendFinalize(desc);
            if (st != CUDNN_STATUS_SUCCESS)
                throw new ProbeRefusedException(_owner.Describe(st, $"{what} finalize"));
        }

        public void Dispose()
        {
            for (int i = _owned.Count - 1; i >= 0; i--)
                cudnnBackendDestroyDescriptor(_owned[i]);
            _owned.Clear();
        }
    }

    // ── layout and dtype helpers ───────────────────────────────────────

    private static long[] QStrides(ProbeSpec s) => s.Layout == ProbeLayout.TokenMajor
        ? [s.Sq * s.Hq * s.D, s.D, s.Hq * s.D, 1]
        : [s.Hq * s.Sq * s.D, s.Sq * s.D, s.D, 1];

    private static long[] KvStrides(ProbeSpec s)
    {
        long hp = s.KvPhysicalHeads, rows = s.KvRows, d = s.D;
        long[] strides = s.Layout == ProbeLayout.TokenMajor ? [rows * hp * d, d, hp * d, 1] : [hp * rows * d, rows * d, d, 1];
        if (s.KvHeadStrideZero)
            strides[1] = 0;
        return strides;
    }

    private static long BiasRows(ProbeBias b) => b.SqStrideZero ? 1 : b.Sq;

    private static long[] BiasDims(ProbeSpec s, ProbeBias b) => [b.B, b.H, b.Sq, s.Skv];

    private static long[] BiasStrides(ProbeSpec s, ProbeBias b)
    {
        long rows = BiasRows(b);
        return [b.H * rows * s.Skv, rows * s.Skv, b.SqStrideZero ? 0 : s.Skv, 1];
    }

    private static long[] Dense(long[] dims)
    {
        long[] strides = new long[dims.Length];
        long stride = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= dims[i];
        }
        return strides;
    }

    private static long Tiles(long n) => (n + 127) / 128;

    private static int IoType(ProbeDtype t) => t == ProbeDtype.BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_HALF;

    private static int BiasType(ProbeBiasType t) => t switch
    {
        ProbeBiasType.F16 => CUDNN_DATA_HALF,
        ProbeBiasType.BF16 => CUDNN_DATA_BFLOAT16,
        _ => CUDNN_DATA_FLOAT,
    };

    private static int BoundType(ProbeSpec s) => s.BoundType == ProbeBoundType.Int32 ? CUDNN_DATA_INT32 : CUDNN_DATA_FLOAT;

    private static ushort[] Image(float[] logical, long b, long h, long rows, long d, long[] strides, ProbeDtype dt)
    {
        ushort[] image = new ushort[logical.Length];
        long n = 0;
        for (long bi = 0; bi < b; bi++)
            for (long hi = 0; hi < h; hi++)
                for (long r = 0; r < rows; r++)
                {
                    long at = bi * strides[0] + hi * strides[1] + r * strides[2];
                    for (long e = 0; e < d; e++)
                        image[at + e] = ToBits(logical[n++], dt);
                }
        return image;
    }

    private static ushort[] Bits(float[] values, ProbeDtype dt)
    {
        ushort[] bits = new ushort[values.Length];
        for (long i = 0; i < values.Length; i++)
            bits[i] = ToBits(values[i], dt);
        return bits;
    }

    private static float Quantize(float v, ProbeDtype dt) => FromBits(ToBits(v, dt), dt);

    private static float QuantizeBias(float v, ProbeBiasType t) => t switch
    {
        ProbeBiasType.F16 => Quantize(v, ProbeDtype.F16),
        ProbeBiasType.BF16 => Quantize(v, ProbeDtype.BF16),
        _ => v,
    };

    private static ushort ToBits(float v, ProbeDtype dt)
    {
        if (dt == ProbeDtype.F16)
            return BitConverter.HalfToUInt16Bits((Half)v);
        if (float.IsNaN(v))
            return 0x7FC0;
        uint u = BitConverter.SingleToUInt32Bits(v);
        return (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
    }

    private static float FromBits(ushort u, ProbeDtype dt) => dt == ProbeDtype.F16
        ? (float)BitConverter.UInt16BitsToHalf(u)
        : BitConverter.UInt32BitsToSingle((uint)u << 16);

    private static float Gauss(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _ctx.EnsureCurrent();
        cudnnDestroy(_handle);
        CudaDriverApi.cuStreamDestroy(_stream);
        NativeMemory.AlignedFree(_slots);
        if (_library != 0)
            NativeLibrary.Free(_library);
    }
}
