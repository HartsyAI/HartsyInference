using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.Engine;

/// <summary>Resolves the <c>--backend</c> selector (including <c>auto</c>) into a concrete <see cref="IBackend"/>,
/// centralizing the PTX/SPIR-V directory conventions that were copy-pasted across the sample CLIs.</summary>
public static class BackendFactory
{
    /// <summary>Subdirectory beside the executable holding compiled PTX kernels for the CUDA backend.</summary>
    public const string PtxDirName = "Ptx";

    /// <summary>Subdirectory beside the executable holding compiled SPIR-V kernels for the Vulkan backend.</summary>
    public const string SpirvDirName = "Spirv";

    /// <summary>Valid selector tokens accepted on the command line; device selectors also accept a <c>:{ordinal}</c> suffix.</summary>
    public static IReadOnlyList<string> ValidSelectors { get; } = new[] { "auto", "cpu", "cuda", "vulkan" };

    /// <summary>Explicit kernel-directory override; wins over auto-detection. For hosts that deploy the compiled
    /// kernels somewhere other than beside the engine assemblies.</summary>
    public static string? KernelDirOverride { get; set; }

    /// <summary>Directory holding compiled kernels, resolved relative to THIS assembly rather than the entry
    /// application: when the engine is hosted inside another app (a SwarmUI extension loaded into its own
    /// <c>AssemblyLoadContext</c>) <see cref="AppContext.BaseDirectory"/> is the host's bin dir, not the engine's,
    /// and the kernels ship beside the engine DLLs. Falls back to <see cref="AppContext.BaseDirectory"/> when the
    /// assembly location is unavailable (single-file publish) or the resolved directory does not exist.</summary>
    public static string KernelDir(string subdir)
    {
        if (!string.IsNullOrWhiteSpace(KernelDirOverride))
        {
            return Path.Combine(KernelDirOverride, subdir);
        }
        string asmDir = Path.GetDirectoryName(typeof(BackendFactory).Assembly.Location) ?? "";
        string candidate = Path.Combine(string.IsNullOrEmpty(asmDir) ? AppContext.BaseDirectory : asmDir, subdir);
        if (!Directory.Exists(candidate))
        {
            string fallback = Path.Combine(AppContext.BaseDirectory, subdir);
            if (Directory.Exists(fallback))
            {
                return fallback;
            }
        }
        return candidate;
    }

    /// <summary>Constructs a CUDA backend on <paramref name="ordinal"/> with the resolved PTX directory. Single place,
    /// so every caller (facade, TextService, recipes) agrees on where kernels live.</summary>
    public static IBackend CreateCuda(int ordinal) => new CudaBackend(ordinal, KernelDir(PtxDirName));

    /// <summary>Constructs a Vulkan backend with the resolved SPIR-V directory; a null <paramref name="ordinal"/> lets
    /// the engine rank the devices and take the best.</summary>
    /// <remarks>Null is not a synonym for 0. A raw index pins whatever the loader happens to list first, which on a
    /// machine that also exposes a software rasterizer (Mesa's lavapipe, standard on Linux CI images) is routinely not
    /// the GPU; the null path scores devices by type and by the capabilities the kernels need. So a caller who did not
    /// name a device has to pass null, or a host with its real GPU at index 1 quietly runs on the rasterizer at 0.</remarks>
    public static IBackend CreateVulkan(int? ordinal = null) => new VulkanBackend(ordinal, KernelDir(SpirvDirName));

    /// <summary>Constructs the backend named by <paramref name="selector"/>, mapping <c>auto</c> via <see cref="Resolve"/>
    /// and honoring a <c>cuda:1</c>-style device-ordinal suffix.</summary>
    public static IBackend Create(string selector)
    {
        string chosen = Resolve(selector);
        int ordinal = ParseOrdinal(selector);
        return chosen switch
        {
            "cuda" => CreateCuda(ValidateCudaOrdinal(ordinal, selector)),
            "vulkan" => CreateVulkan(HasExplicitOrdinal(selector) ? ordinal : null),
            "cpu" => new CpuBackend(),
            _ => throw new ArgumentException($"Unknown backend '{selector}'. Valid: {string.Join(", ", ValidSelectors)}."),
        };
    }

    /// <summary>Maps <c>auto</c> to the best available backend and reduces an explicit selector to its bare kind,
    /// with the <c>:{ordinal}</c> suffix stripped (see <see cref="ParseOrdinal"/>).
    ///
    /// <para>Order is CUDA, then Vulkan, then CPU. Vulkan is the path for every GPU CUDA does not serve (AMD, Intel,
    /// and NVIDIA cards whose CUDA toolkit is absent), so resolving straight to CPU when CUDA is missing sent those
    /// machines to the slowest backend they own while a working GPU sat idle. CPU stays the answer only when no GPU
    /// is present at all: a software-rasterizer Vulkan device does not count as one (see
    /// <see cref="VulkanContext.GetDeviceCount"/>).</para>
    ///
    /// <para>An ordinal on <c>auto</c> is read by whichever backend wins, and the two APIs enumerate independently:
    /// <c>auto:2</c> means CUDA device 2 on a machine that resolves to CUDA and Vulkan device 2 on one that resolves
    /// to Vulkan, which need not be the same card, or exist. The ambiguity predates the Vulkan step (an ordinal has
    /// always been written before the backend was known) but only became reachable through <c>auto</c> with it, so a
    /// caller that cares which physical device it gets should name the backend rather than leave it to
    /// <c>auto</c>.</para>
    ///
    /// <para>With no ordinal written at all, nothing is pinned and each API picks for itself: CUDA's ordinal 0 is its
    /// own fastest-first choice, and Vulkan ranks the devices (see <see cref="CreateVulkan"/>) rather than taking raw
    /// index 0. That difference matters on exactly the machines this fallthrough was added for, where the loader lists
    /// a software rasterizer alongside the real card.</para></summary>
    public static string Resolve(string selector)
    {
        string s = Kind(selector);
        if (s != "auto")
            return s;
        if (CudaContext.IsAvailable() && CudaContext.GetDeviceCount() > 0)
        {
            return "cuda";
        }
        return VulkanContext.IsAvailable() ? "vulkan" : "cpu";
    }

    /// <summary>Why <see cref="Resolve"/> last passed over CUDA; <c>null</c> when CUDA was usable.
    /// Only meaningful after a Resolve/IsAvailable call.</summary>
    public static string? CudaUnavailableReason => CudaContext.LastUnavailableReason;

    /// <summary>Why <see cref="Resolve"/> last passed over Vulkan; <c>null</c> when a Vulkan GPU was found.
    /// Only meaningful after a Resolve/IsAvailable call.</summary>
    public static string? VulkanUnavailableReason => VulkanContext.LastUnavailableReason;

    // One lock per API, not one shared: the two probes touch independent hardware and cache independent answers,
    // so a slow or wedged CUDA probe has no business blocking a caller asking about Vulkan.
    // Cached per device rather than once overall, because the answer is per device: one box can hold a card that
    // computes and a card that does not, and a single bool hands the first caller's verdict to every later one.
    private static readonly object _cudaProbeLock = new();
    private static readonly Dictionary<int, (bool Ok, string? Reason)> _cudaProbes = [];
    private static string? _probeReason;
    private static readonly object _vulkanProbeLock = new();
    private static readonly Dictionary<int, (bool Ok, string? Reason)> _vulkanProbes = [];
    private static string? _vulkanProbeReason;

    /// <summary>Probe-cache key for "no device named, let the engine rank them", which is a different request from raw
    /// index 0 and must not share its cached answer. A sentinel rather than a nullable key because <see cref="Dictionary{TKey, TValue}"/>
    /// rejects a null key even when TKey is a nullable value type; ordinals are non-negative, so this cannot collide.</summary>
    private const int UnspecifiedDevice = -1;

    /// <summary>Why <see cref="ProbeCuda"/> last failed; <c>null</c> when it passed or has not run.</summary>
    public static string? CudaProbeFailureReason => _probeReason;

    /// <summary>Why <see cref="ProbeVulkan"/> last failed; <c>null</c> when it passed or has not run.</summary>
    public static string? VulkanProbeFailureReason => _vulkanProbeReason;

    /// <summary>Actually runs something on the GPU and checks the answer, rather than asking whether one exists.
    ///
    /// <para><see cref="CudaContext.IsAvailable"/> answers a different question: can the libraries be loaded and
    /// does the driver see a device. Everything between that and a working backend is untested by it — kernels
    /// compiled for another architecture, a PTX directory that did not ship, a card with no free memory, a
    /// driver and toolkit that disagree. Each of those says yes to availability and then throws on the first
    /// real operation, which is somewhere in the middle of a user's request rather than at startup.</para>
    ///
    /// <para>So this constructs the backend, multiplies two small matrices on it, and compares the result to the
    /// CPU's. A wrong answer counts as a failure: a GPU that computes is worse than one that throws, because
    /// nothing downstream will notice.</para>
    ///
    /// <para>Cached after the first call — it costs a context creation and a kernel load, which is a few hundred
    /// milliseconds, and the answer cannot change without a restart. Opt-in: <see cref="Resolve"/> is unchanged
    /// and stays cheap, and a host that wants the stronger check calls <see cref="ResolveProbed"/>.</para></summary>
    /// <param name="ordinal">Device to probe.</param>
    /// <returns>True when a real matmul ran on the GPU and agreed with the CPU.</returns>
    public static bool ProbeCuda(int ordinal = 0)
    {
        lock (_cudaProbeLock)
        {
            if (!_cudaProbes.TryGetValue(ordinal, out (bool Ok, string? Reason) cached))
            {
                bool ok = RunCudaProbe(ordinal, out string? reason);
                cached = (ok, reason);
                _cudaProbes[ordinal] = cached;
            }
            _probeReason = cached.Reason;
            return cached.Ok;
        }
    }

    private static bool RunCudaProbe(int ordinal, out string? reason)
    {
        if (!CudaContext.IsAvailable())
        {
            reason = CudaContext.LastUnavailableReason ?? "CUDA is not available.";
            return false;
        }
        return RunMatMulProbe(() => CreateCuda(ordinal), out reason);
    }

    private static bool RunVulkanProbe(int? ordinal, out string? reason)
    {
        if (!VulkanContext.IsAvailable())
        {
            reason = VulkanContext.LastUnavailableReason ?? "Vulkan is not available.";
            return false;
        }
        return RunMatMulProbe(() => CreateVulkan(ordinal), out reason);
    }

    /// <summary>Builds a backend, multiplies two small matrices on it and checks the answer against the CPU's.</summary>
    private static bool RunMatMulProbe(Func<IBackend> create, out string? reason)
    {
        // Small enough to cost nothing, large enough that a broken kernel cannot accidentally agree: 32x32x32
        // is 32768 multiply-adds against values that are not all the same.
        const int n = 32;
        Tensor a = new(new TensorShape(n, n), DType.F32);
        Tensor b = new(new TensorShape(n, n), DType.F32);
        Tensor gpu = new(new TensorShape(n, n), DType.F32);
        Tensor cpu = new(new TensorShape(n, n), DType.F32);
        IBackend? device = null;
        try
        {
            unsafe
            {
                float* ap = (float*)a.DataPointer;
                float* bp = (float*)b.DataPointer;
                for (int i = 0; i < n * n; i++)
                {
                    ap[i] = MathF.Sin(i * 0.37f);
                    bp[i] = MathF.Cos(i * 0.11f);
                }
            }
            using (CpuBackend reference = new())
            {
                reference.MatMul(cpu, a, b);
            }
            device = create();
            device.MatMul(gpu, a, b);
            device.Sync();

            float worst = 0f;
            unsafe
            {
                float* g = (float*)gpu.DataPointer;
                float* c = (float*)cpu.DataPointer;
                for (int i = 0; i < n * n; i++)
                {
                    worst = MathF.Max(worst, MathF.Abs(g[i] - c[i]));
                }
            }
            // Loose: cuBLAS accumulates in a different order and may use tensor cores. A real failure here is
            // not a last-place difference, it is a zeroed or garbage buffer. Vulkan's own F32 matmul tests hold
            // themselves to 1e-4, so this threshold constrains neither backend's honest rounding.
            if (worst > 1e-2f)
            {
                reason = $"the GPU computed a 32x32 matmul that differs from the CPU by {worst:E2} — kernels are "
                    + "loading but producing wrong answers.";
                return false;
            }
            reason = null;
            return true;
        }
        catch (Exception ex)
        {
            reason = $"a test matmul on the GPU failed: {ex.GetType().Name}: {ex.Message}";
            return false;
        }
        finally
        {
            device?.Dispose();
            a.Dispose();
            b.Dispose();
            gpu.Dispose();
            cpu.Dispose();
        }
    }

    /// <summary>The Vulkan twin of <see cref="ProbeCuda"/>: builds a Vulkan backend and checks that it computes,
    /// rather than asking whether the loader can see a device.
    ///
    /// <para>Worth as much here as it is on CUDA, for different reasons. A Vulkan device can enumerate and then
    /// fail to meet the engine's own requirements (FP16, the subgroup ops the kernels are written against, a
    /// compute queue), which surfaces as a throw from device creation rather than a missing device. And the SPIR-V
    /// directory can be absent exactly as the PTX one can.</para></summary>
    /// <param name="ordinal">Device to probe; null probes whichever device <see cref="CreateVulkan"/> would rank best.</param>
    /// <returns>True when a real matmul ran on the GPU and agreed with the CPU.</returns>
    /// <remarks>Pass what will actually be built. Probing raw index 0 and then running on the ranked-best device
    /// tests the wrong card, and on a box whose index 0 is a software rasterizer it is the one test that passes
    /// regardless of whether the real GPU works.</remarks>
    public static bool ProbeVulkan(int? ordinal = null)
    {
        lock (_vulkanProbeLock)
        {
            int key = ordinal ?? UnspecifiedDevice;
            if (!_vulkanProbes.TryGetValue(key, out (bool Ok, string? Reason) cached))
            {
                bool ok = RunVulkanProbe(ordinal, out string? reason);
                cached = (ok, reason);
                _vulkanProbes[key] = cached;
            }
            _vulkanProbeReason = cached.Reason;
            return cached.Ok;
        }
    }

    /// <summary>Like <see cref="Resolve"/>, but each GPU candidate has to prove it computes before it is picked:
    /// CUDA if <see cref="ProbeCuda"/> passes, else Vulkan if <see cref="ProbeVulkan"/> passes, else CPU.
    ///
    /// <para>For a host deciding once at startup what device to build its engine on. Costs the probes the first
    /// time and nothing after; the payoff is that a machine which would have failed on its first request fails
    /// here instead, with a reason, and runs on the CPU rather than not at all.</para>
    ///
    /// <para>The Vulkan step is what makes this more than a stricter CUDA check. A machine whose CUDA install is
    /// broken rather than absent (driver present, toolkit skewed, kernels built for another architecture) has
    /// its GPU rejected here, and on most such machines that same card answers on Vulkan. Falling to CPU without
    /// asking would leave it idle.</para></summary>
    public static string ResolveProbed(string selector)
    {
        string s = Kind(selector);
        if (s != "auto")
        {
            return s;
        }
        int ordinal = ParseOrdinal(selector);
        if (ProbeCuda(ordinal))
        {
            return "cuda";
        }
        // Null rather than the parsed 0 when the caller named no device, so the probe builds the device Create will.
        return ProbeVulkan(HasExplicitOrdinal(selector) ? ordinal : null) ? "vulkan" : "cpu";
    }

    /// <summary>Whether <paramref name="selector"/> names a device outright (<c>vulkan:1</c>) instead of leaving the
    /// choice to the engine (<c>vulkan</c>, <c>auto</c>).</summary>
    /// <remarks>Not the same question as <c>ParseOrdinal(selector) == 0</c>, which cannot tell <c>vulkan:0</c> from
    /// <c>vulkan</c>. Vulkan honors the distinction: the first pins the loader's raw index 0, the second ranks the
    /// devices (see <see cref="CreateVulkan"/>).</remarks>
    public static bool HasExplicitOrdinal(string? selector) => (selector ?? "").Trim().Contains(':');

    /// <summary>The bare backend kind of <paramref name="selector"/> with any <c>:{ordinal}</c> suffix removed; <c>auto</c> stays <c>auto</c>.</summary>
    public static string Kind(string? selector)
    {
        string s = (selector ?? "auto").Trim().ToLowerInvariant();
        int colon = s.IndexOf(':');
        return colon < 0 ? s : s[..colon];
    }

    /// <summary>The device ordinal encoded in <paramref name="selector"/> (<c>cuda:1</c> → 1); 0 when no suffix is present.</summary>
    /// <remarks>Syntax-only — does not touch the driver, so it is safe on a CPU-only host. Throws on a malformed or negative suffix.</remarks>
    public static int ParseOrdinal(string? selector)
    {
        if (!TryParseOrdinal(selector, out int ordinal, out string? error))
        {
            throw new ArgumentException(error, nameof(selector));
        }
        return ordinal;
    }

    /// <summary>Whether <paramref name="selector"/> is a syntactically valid selector, including a <c>cuda:1</c>-style ordinal suffix.</summary>
    /// <remarks>Never queries the driver, so a REPL/API validation call succeeds on a machine with no GPU; the device count is
    /// checked later, in <see cref="Create"/>.</remarks>
    public static bool IsValidSelector(string? selector)
    {
        if (string.IsNullOrWhiteSpace(selector))
        {
            return false;
        }
        return ValidSelectors.Contains(Kind(selector), StringComparer.OrdinalIgnoreCase)
            && TryParseOrdinal(selector, out _, out _);
    }

    /// <summary>Non-throwing core of <see cref="ParseOrdinal"/>; <paramref name="error"/> carries the message the throwing form uses.</summary>
    private static bool TryParseOrdinal(string? selector, out int ordinal, out string? error)
    {
        ordinal = 0;
        error = null;
        string s = (selector ?? "auto").Trim();
        int colon = s.IndexOf(':');
        if (colon < 0)
        {
            return true;
        }
        string suffix = s[(colon + 1)..];
        if (!int.TryParse(suffix, NumberStyles.Integer, CultureInfo.InvariantCulture, out ordinal) || ordinal < 0)
        {
            ordinal = 0;
            error = $"Backend selector '{selector}' has an invalid device ordinal '{suffix}' — expected a non-negative integer, e.g. 'cuda:1'.";
            return false;
        }
        // 'cpu:0' would read as a device selection the CPU backend cannot honor; reject rather than silently ignore.
        // 'auto:1' IS meaningful — a host with separate "backend" and "GPU id" settings (SwarmUI's ComputeBackend +
        // GPU_ID) leaves the first on 'auto', and the ordinal has to survive until Resolve picks a device backend.
        string kind = Kind(s);
        if (!IsDeviceKind(kind) && kind != "auto")
        {
            ordinal = 0;
            error = $"Backend selector '{selector}' cannot carry a device ordinal — only 'cuda', 'vulkan' and 'auto' select a device.";
            return false;
        }
        return true;
    }

    /// <summary>Composes a selector carrying <paramref name="ordinal"/>, so a host with separate "backend" and "GPU id" settings can join them.</summary>
    /// <remarks>Ordinal 0 composes to the bare kind, so the common case round-trips to exactly the string callers passed before device selection
    /// existed. A non-zero ordinal on a device-less kind throws, matching <see cref="ParseOrdinal"/>'s rejection of <c>cpu:1</c> — the same request
    /// must not succeed silently through one door and fail through another.</remarks>
    public static string WithOrdinal(string? selector, int ordinal)
    {
        if (ordinal < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "Device ordinal must be non-negative.");
        }
        string kind = Kind(selector);
        if (ordinal == 0)
        {
            return kind;
        }
        if (!IsDeviceKind(kind) && kind != "auto")
        {
            throw new ArgumentException(
                $"Backend '{kind}' cannot run on device {ordinal} — only 'cuda', 'vulkan' and 'auto' select a device.",
                nameof(selector));
        }
        return $"{kind}:{ordinal}";
    }

    /// <summary>Human-readable description of what <paramref name="selector"/> resolves to, for banners and <c>--verbose</c>.</summary>
    /// <remarks>A selector <see cref="Create"/> would reject is echoed back verbatim rather than laundered into a plausible-looking
    /// resolution — a banner must not show something the factory will refuse to build.</remarks>
    public static string Describe(string selector)
    {
        if (!IsValidSelector(selector))
        {
            return (selector ?? "auto").Trim();
        }
        string resolved = Resolve(selector);
        int ordinal = ParseOrdinal(selector);
        string withDevice = IsDeviceKind(resolved) && ordinal != 0 ? $"{resolved}:{ordinal}" : resolved;
        bool auto = string.Equals(Kind(selector), "auto", StringComparison.OrdinalIgnoreCase);
        return auto ? $"auto → {withDevice}" : withDevice;
    }

    /// <summary>Reduces <paramref name="selector"/> to the concrete device it names: <c>auto</c> is resolved,
    /// a device kind is given its explicit ordinal, and <c>cpu</c> stays bare.</summary>
    /// <remarks>For callers that use a selector as a KEY — a cache slot, a gate ordinal — where two spellings of one
    /// device must not read as two devices, and where <c>auto</c> must not survive to be classified as "not a
    /// device" by a later check while <see cref="Create"/> builds a GPU backend from it.
    ///
    /// <para>A layer-split composite (<c>cuda:0+cuda:1</c>) and anything <see cref="IsValidSelector"/> rejects come
    /// back unchanged: the first is a list of selectors rather than one, and the second has no concrete device to
    /// name. Both are the caller's to handle.</para>
    ///
    /// <para>This keys the REQUEST, not the hardware. It reports <c>vulkan:0</c> for a bare <c>vulkan</c> because that
    /// is what the string says, while the backend built from it runs on whichever device ranked best. A caller asking
    /// "are these two engines on one physical GPU" wants <see cref="IBackend.DeviceKey"/> from the built backend
    /// instead; this answers only "did these two callers ask for the same thing".</para></remarks>
    public static string CanonicalDeviceKey(string? selector)
    {
        string key = (selector ?? "").Trim().ToLowerInvariant();
        if (key.Length == 0 || key.Contains('+') || !IsValidSelector(key))
        {
            return key;
        }
        string kind = Kind(key);
        string resolved = kind == "auto" ? Resolve(key) : kind;
        return IsDeviceKind(resolved) ? $"{resolved}:{ParseOrdinal(key)}" : resolved;
    }

    /// <summary>Whether <paramref name="kind"/> names a backend that runs on a selectable device.</summary>
    /// <remarks>Public so callers stop spelling the test as <c>StartsWith("cuda")</c>: a device selector that is not
    /// CPU is not therefore CUDA, and that assumption silently routed non-CUDA requests onto CUDA.</remarks>
    public static bool IsDeviceKind(string kind) => kind == "cuda" || kind == "vulkan";

    /// <summary>Throws when <paramref name="selector"/> cannot resolve to a constructible backend on this machine,
    /// without constructing one — syntax plus a driver device-count query only. Lets a host surface a bad
    /// configuration (missing driver, out-of-range GPU id) at startup instead of on the first generation.</summary>
    /// <remarks>Stricter than <see cref="Create"/>'s lazy path in one way: an explicit <c>cuda</c>/<c>cuda:0</c> or
    /// <c>vulkan</c> selector fails here when that API has no device at all, instead of deferring to a driver error
    /// mid-generation. <c>auto</c> always validates, since it falls back to CPU by design.
    ///
    /// <para>Only Vulkan's device presence is checked, not its ordinal: <see cref="VulkanContext.GetDeviceCount"/>
    /// counts GPUs, while a <c>vulkan:{n}</c> ordinal indexes the loader's raw device list, software rasterizers
    /// included. Bounding one by the other would reject valid ordinals. The raw list's own bound is enforced where
    /// it is known, at device creation.</para></remarks>
    public static void Validate(string selector)
    {
        if (!IsValidSelector(selector))
        {
            if (!TryParseOrdinal(selector, out _, out string? ordinalError) && ordinalError is not null)
            {
                throw new ArgumentException(ordinalError, nameof(selector));
            }
            throw new ArgumentException(
                $"Unknown backend '{selector}'. Valid: {string.Join(", ", ValidSelectors)}.", nameof(selector));
        }
        if (Kind(selector) == "vulkan")
        {
            if (!VulkanContext.IsAvailable())
            {
                throw new ArgumentException(
                    $"Backend selector '{selector}' requires Vulkan, but no Vulkan GPU is available on this machine: "
                    + $"{VulkanContext.LastUnavailableReason ?? "no reason reported"}",
                    nameof(selector));
            }
            return;
        }
        if (Kind(selector) != "cuda")
        {
            return;
        }
        int ordinal = ParseOrdinal(selector);
        int count = CudaContext.IsAvailable() ? CudaContext.GetDeviceCount() : 0;
        if (count == 0)
        {
            throw new ArgumentException(
                $"Backend selector '{selector}' requires CUDA, but no CUDA driver/device is available on this machine.",
                nameof(selector));
        }
        if (ordinal >= count)
        {
            throw new ArgumentException(
                $"Backend selector '{selector}' requests CUDA device {ordinal}, but this machine has {count} CUDA " +
                $"device(s) (valid ordinals 0..{count - 1}). " +
                "Note CUDA's default enumeration is fastest-first and need not match nvidia-smi's PCI order.",
                nameof(selector));
        }
    }

    /// <summary>Fails with a message naming both the requested ordinal and what the machine actually has.</summary>
    /// <remarks>Ordinal 0 is skipped deliberately: it is the pre-existing default path, and the only case the check would
    /// catch there is "no CUDA devices at all", which <see cref="CudaBackend"/>'s own driver error already reports. Leaving
    /// it alone means adding device selection cannot change behavior for a caller who never asked for a device.</remarks>
    private static int ValidateCudaOrdinal(int ordinal, string selector)
    {
        if (ordinal == 0)
        {
            return 0;
        }
        int count = CudaContext.IsAvailable() ? CudaContext.GetDeviceCount() : 0;
        if (ordinal >= count)
        {
            throw new ArgumentException(
                $"Backend selector '{selector}' requests CUDA device {ordinal}, but this machine has {count} CUDA " +
                $"device(s){(count > 0 ? $" (valid ordinals 0..{count - 1})" : "")}. " +
                "Note CUDA's default enumeration is fastest-first and need not match nvidia-smi's PCI order.",
                nameof(selector));
        }
        return ordinal;
    }
}
