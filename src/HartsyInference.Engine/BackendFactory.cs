using System.Globalization;
using HartsyInference.Core.Backends;
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

    /// <summary>Constructs a Vulkan backend on <paramref name="ordinal"/> with the resolved SPIR-V directory.</summary>
    public static IBackend CreateVulkan(int ordinal) => new VulkanBackend(ordinal, KernelDir(SpirvDirName));

    /// <summary>Constructs the backend named by <paramref name="selector"/>, mapping <c>auto</c> via <see cref="Resolve"/>
    /// and honoring a <c>cuda:1</c>-style device-ordinal suffix.</summary>
    public static IBackend Create(string selector)
    {
        string chosen = Resolve(selector);
        int ordinal = ParseOrdinal(selector);
        return chosen switch
        {
            "cuda" => CreateCuda(ValidateCudaOrdinal(ordinal, selector)),
            "vulkan" => CreateVulkan(ordinal),
            "cpu" => new CpuBackend(),
            _ => throw new ArgumentException($"Unknown backend '{selector}'. Valid: {string.Join(", ", ValidSelectors)}."),
        };
    }

    /// <summary>Maps <c>auto</c> to the best available backend (CUDA when a device is present, else CPU) and reduces an
    /// explicit selector to its bare kind — the <c>:{ordinal}</c> suffix is stripped (see <see cref="ParseOrdinal"/>).</summary>
    public static string Resolve(string selector)
    {
        string s = Kind(selector);
        if (s != "auto")
            return s;
        return CudaContext.IsAvailable() && CudaContext.GetDeviceCount() > 0 ? "cuda" : "cpu";
    }

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

    /// <summary>Whether <paramref name="kind"/> names a backend that runs on a selectable device.</summary>
    private static bool IsDeviceKind(string kind) => kind == "cuda" || kind == "vulkan";

    /// <summary>Throws when <paramref name="selector"/> cannot resolve to a constructible backend on this machine,
    /// without constructing one — syntax plus a driver device-count query only. Lets a host surface a bad
    /// configuration (missing driver, out-of-range GPU id) at startup instead of on the first generation.</summary>
    /// <remarks>Stricter than <see cref="Create"/>'s lazy path in one way: an explicit <c>cuda</c>/<c>cuda:0</c>
    /// selector fails here when no CUDA device exists at all, instead of deferring to a driver error mid-generation.
    /// Vulkan has no cheap availability probe, so only its syntax is checked. <c>auto</c> always validates — it
    /// falls back to CPU by design.</remarks>
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
