using System.Runtime.InteropServices;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Locates, driver-guards and (on first use) auto-provisions cuBLAS + cuBLASLt, so a machine with only the NVIDIA driver
/// can run the CUDA backend without installing the CUDA Toolkit.</summary>
/// <remarks>
/// <para>Search order, per CUDA major from the driver's down to 11: <c>paths.cublasDir</c>, the other resolver probe dirs
/// (<c>paths.cudaLibDir</c>, the per-user cache, a <c>cublas/</c> folder beside the assembly), then the OS loader. A cuBLAS built
/// for a newer CUDA major than the driver is never chosen: it loads, then fails at <c>cublasCreate</c>.</para>
/// <para>Mirrors <see cref="CudnnRuntime"/>: <see cref="EnsureProbed"/> only looks, <see cref="EnsureAvailable"/> may download.
/// Probes such as <see cref="CudaContext.IsAvailable"/> stay download-free; the fetch runs on the first
/// <see cref="CudaBackend"/> construction or from <see cref="CudaContext.EnsureProvisioned"/> during automatic backend selection.</para>
/// </remarks>
public static class CublasRuntime
{
    /// <summary>One pinned NVIDIA redist build: version and the archive's sha256 and size per platform.</summary>
    internal readonly record struct RedistBuild(int CudaMajor, string Version, string WindowsSha256, long WindowsBytes, string LinuxSha256, long LinuxBytes);

    /// <summary>Default builds, from NVIDIA's <c>redistrib_13.5.0.json</c> and <c>redistrib_12.9.2.json</c> manifests.</summary>
    internal static readonly RedistBuild[] DefaultBuilds =
    [
        new(13, "13.10.0.20",
            "468d90ff5680fd08fd9130e6e443c6e0c934f2f98b6ae4189960b9d9039e8969", 426_404_439,
            "6da2dd456733bc45235e344a6e6aab3e8f43fb775a835c5dc432f4a1fc9f93d2", 888_132_896),
        new(12, "12.9.2.10",
            "62782c354e1536118caab8fe5fd295ac5fd9076e4814aba0bdaf032fe2b833ae", 549_731_131,
            "c80a724aafa1ee49110b0d3385316ea54d418b03e796e91e8f0b47d52f8022fd", 944_339_140),
    ];

    /// <summary>CUDA majors the engine knows a cuBLAS soname for, newest first.</summary>
    internal static readonly int[] KnownMajors = [13, 12, 11];

    private static readonly object _lock = new();
    private static int _probed;
    private static volatile bool _available;
    private static volatile bool _fetchPending;
    private static int _driverMajor;
    private static int[] _majors = KnownMajors;

    /// <summary>True when a cuBLAS no newer than the driver loaded.</summary>
    public static bool Available => _available;

    /// <summary>True when no cuBLAS was found and the first CUDA backend construction will download one.</summary>
    public static bool FetchPending => _fetchPending;

    /// <summary>Why cuBLAS is or is not usable, for logs and <see cref="CudaContext.LastUnavailableReason"/>.</summary>
    public static string Reason { get; private set; } = "not yet probed";

    /// <summary>The CUDA major of the loaded cuBLAS, or 0.</summary>
    public static int ResolvedMajor { get; private set; }

    /// <summary>The file the loaded cuBLAS came from, or null when the OS loader found it by name.</summary>
    internal static string? LoadedPath { get; private set; }

    /// <summary>The folder the loaded cuBLAS came from; cuBLASLt is looked for there first.</summary>
    public static string? LibDir => LoadedPath is null ? null : Path.GetDirectoryName(LoadedPath);

    /// <summary>The CUDA majors whose cuBLAS may load under a driver of <paramref name="driverMajor"/>, newest first. An unknown
    /// driver (0) allows every known major.</summary>
    public static int[] PickMajors(int driverMajor) =>
        driverMajor <= 0 ? KnownMajors : KnownMajors.Where(m => m <= driverMajor).ToArray();

    /// <summary>The versioned file name of cuBLAS (or cuBLASLt) for a CUDA major on this OS.</summary>
    public static string Soname(int major, bool lt = false, bool? windows = null)
    {
        bool win = windows ?? RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string name = lt ? "cublasLt" : "cublas";
        return win ? $"{name}64_{major}.dll" : $"lib{name}.so.{major}";
    }

    /// <summary>The sonames the resolver tries for <c>[LibraryImport("cublas")]</c> / <c>("cublasLt")</c>, newest allowed first.</summary>
    internal static string[] Sonames(bool lt)
    {
        EnsureProbed();
        return _majors.Select(m => Soname(m, lt)).ToArray();
    }

    /// <summary>Looks for an installed cuBLAS once (thread-safe). Never downloads and never throws.</summary>
    public static void EnsureProbed()
    {
        if (Volatile.Read(ref _probed) != 0) return;
        lock (_lock)
        {
            if (_probed != 0) return;
            try { Locate(); }
            catch (Exception ex) { _available = false; Reason = $"cuBLAS probe error: {ex.Message}"; }
            Volatile.Write(ref _probed, 1);
        }
    }

    /// <summary>True when cuBLAS can serve a backend now. When none is installed and <c>paths.cublasAutofetch</c> is on, the
    /// first call downloads the matching redist (once per process, successful or not) and loads it.</summary>
    public static bool EnsureAvailable()
    {
        EnsureProbed();
        if (!_fetchPending) return Available;
        lock (_lock)
        {
            if (!_fetchPending) return Available;
            _fetchPending = false;
            CudaRedistSpec? spec = FetchSpec(_driverMajor);
            string? fetched = spec is null ? null : CudaRedistProvisioner.FetchInto(spec);
            Locate();
            _fetchPending = false;
            if (!Available && fetched is null)
                Reason = $"cuBLAS not found and the download failed (driver is CUDA {_driverMajor}); see the warning above. " +
                         $"Install the CUDA {_driverMajor} Toolkit, or set paths.cublasDir to a folder holding {Soname(Math.Min(_driverMajor, 13))}.";
        }
        return Available;
    }

    /// <summary>Test hook: forgets the probe so the next call searches again.</summary>
    internal static void ResetForTests()
    {
        lock (_lock)
        {
            _probed = 0;
            _available = false;
            _fetchPending = false;
            _majors = KnownMajors;
            LoadedPath = null;
            ResolvedMajor = 0;
            Reason = "not yet probed";
        }
    }

    /// <summary>The archive to fetch for a driver of <paramref name="driverMajor"/>, or null when there is no public build.
    /// A driver newer than CUDA 13 gets the CUDA 13 build, which it runs.</summary>
    internal static CudaRedistSpec? FetchSpec(int driverMajor)
    {
        bool win = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string? url = EngineKnobs.CublasUrl.Value;
        string? versionOverride = EngineKnobs.CublasVersion.Value;
        int major = Math.Min(driverMajor, 13);
        if (!string.IsNullOrEmpty(versionOverride) && int.TryParse(versionOverride.Split('.')[0], out int overrideMajor))
            major = Math.Min(overrideMajor, major);
        RedistBuild? build = DefaultBuilds.Cast<RedistBuild?>().FirstOrDefault(b => b!.Value.CudaMajor == major);
        string? sha = null;
        long? bytes = null;
        if (string.IsNullOrEmpty(url))
        {
            string? plat = CudaRedistProvisioner.Platform();
            string? version = string.IsNullOrEmpty(versionOverride) ? build?.Version : versionOverride;
            if (plat is null || version is null)
            {
                Logs.Warning($"[cuBLAS] no public cuBLAS redist is known for CUDA {driverMajor} on this platform. Install the CUDA " +
                             "Toolkit, or point paths.cublasUrl/paths.cublasDir at a matching build.");
                return null;
            }
            url = $"https://developer.download.nvidia.com/compute/cuda/redist/libcublas/{plat}/libcublas-{plat}-{version}-archive.{(win ? "zip" : "tar.xz")}";
            if (string.IsNullOrEmpty(versionOverride) && build is { } pinned)
            {
                sha = win ? pinned.WindowsSha256 : pinned.LinuxSha256;
                bytes = win ? pinned.WindowsBytes : pinned.LinuxBytes;
            }
            else
            {
                Logs.Warning($"[cuBLAS] paths.cublasVersion={versionOverride} has no pinned sha256; the download is not verified.");
            }
        }
        return new CudaRedistSpec(
            Component: "cublas",
            Label: "cuBLAS",
            Url: url,
            CudaMajor: major,
            ExpectedSha256: sha,
            ExpectedBytes: bytes,
            LibraryPatterns: [win ? "cublas*.dll" : "libcublas*.so*"],
            Sentinel: Soname(major, windows: win),
            ApproxSize: win ? "about 0.4 GB" : "about 0.9 GB",
            KnobHint: "paths.cublasAutofetch");
    }

    /// <summary>Finds and loads the newest allowed cuBLAS. Arms the fetch when none exists and the driver can use a public build.</summary>
    private static void Locate()
    {
        CudaLibraryResolver.Register();
        _driverMajor = CudnnRuntime.DriverCudaMajor();
        _majors = PickMajors(_driverMajor);
        foreach (int major in _majors)
        {
            if (CudaLibraryResolver.TryLoadFirst(out _, out string? loadedFrom, Soname(major)))
            {
                _available = true;
                ResolvedMajor = major;
                LoadedPath = loadedFrom;
                Reason = $"cuBLAS for CUDA {major} from {loadedFrom ?? "the OS library path"} (driver is CUDA {_driverMajor}).";
                return;
            }
        }
        _available = false;
        ResolvedMajor = 0;
        LoadedPath = null;
        string newer = _driverMajor > 0 && _driverMajor < KnownMajors[0]
            ? $" A cuBLAS for a newer CUDA than the driver's {_driverMajor} is never loaded."
            : string.Empty;
        bool fetchable = _driverMajor >= 12 || (_driverMajor > 0 && !string.IsNullOrEmpty(EngineKnobs.CublasUrl.Value));
        if (EngineKnobs.CublasAutofetch.Value && fetchable)
        {
            _fetchPending = true;
            Reason = $"cuBLAS is not installed; the CUDA {Math.Min(_driverMajor, 13)} redist downloads on the first CUDA backend " +
                     "construction (paths.cublasAutofetch=false disables).";
            return;
        }
        string hint = EngineKnobs.CublasAutofetch.Value ? string.Empty : "Set paths.cublasAutofetch=true to download it, ";
        Reason = $"cuBLAS could not be loaded (searched paths.cublasDir, paths.cudaLibDir, " +
                 $"{CudaRedistProvisioner.CacheLibDir("cublas", Math.Max(_driverMajor, 11))}, {CudaRedistProvisioner.BundledDir("cublas")} " +
                 $"and the OS path; driver is CUDA {_driverMajor}).{newer} {hint}install the CUDA Toolkit, or set paths.cublasDir.";
    }
}
