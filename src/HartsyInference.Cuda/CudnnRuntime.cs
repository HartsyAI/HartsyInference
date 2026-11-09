using System.Runtime.InteropServices;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Locates, version-guards, and (on first use) auto-provisions the cuDNN runtime so the conv/SDPA fast paths engage across OS / CUDA-version / GPU combos WITHOUT the user hand-installing it. <para>cuDNN is a ~1 GB, <b>CUDA-major-specific</b>, OS-specific set of libraries, so it is NOT shipped inside the DLL folder (that would be ~6 GB across CUDA 11/12/13 × Windows/Linux). Instead the matching build is resolved from, in order: <c>paths.cudnnDir</c> → the per-user engine cache → a <c>cudnn/</c> folder beside the assembly (an optional bundled deploy) → the OS default (a system install). If none is found and <c>paths.cudnnAutofetch</c> is on, the matching redist is downloaded once, the first time a model asks for a cuDNN path (<see cref="EnsureAvailable"/>), not at engine start. A cuDNN built for a different CUDA major than the running driver is <b>REJECTED</b> — that exact mismatch (a CUDA-12 cuDNN on a CUDA-13 runtime) previously hung the engine mid-inference.</para></summary>
public static class CudnnRuntime
{
    private static volatile bool _available;

    /// <summary>True when a version-matched cuDNN loaded and passed the CUDA-major guard.</summary>
    public static bool Available => _available;

    /// <summary>True when the loaded runtime exposes the public softmax operation used by the custom SDPA graph.</summary>
    internal static bool SupportsSdpa => Available && IsSdpaVersionSupported(Version);

    internal static bool IsSdpaVersionSupported(long version) => version >= 92100;

    /// <summary>Human-readable status for the diagnostic (why cuDNN is / isn't active).</summary>
    public static string Reason { get; private set; } = "not yet probed";

    /// <summary>Loaded cuDNN version (e.g. 90600) when <see cref="Available"/>, else 0.</summary>
    public static long Version { get; private set; }

    /// <summary>Directory the matching cuDNN was resolved from (cache/bundle/system/env); added to the loader search so <c>[LibraryImport("cudnn")]</c> resolves the same file. Read by <see cref="CudaLibraryResolver"/>.</summary>
    public static string? LibDir { get; private set; }

    private static int _probed;
    private static volatile bool _fetchPending;
    private static volatile bool _fetchInProgress;
    private static int _cudaMajor;
    private static readonly object _lock = new();

    /// <summary>Latest CUDA version the DRIVER supports (13020 → CUDA 13.2); 0 if unknown. Picks the cuDNN build.</summary>
    public static int DriverCudaMajor()
    {
        try { return CudaDriverApi.cuDriverGetVersion(out int v) == 0 ? v / 1000 : 0; }
        catch { return 0; }
    }

    /// <summary>Deterministic per-user cache dir the matching cuDNN is provisioned into (and searched from).</summary>
    public static string CacheLibDir(int cudaMajor) => CudaRedistProvisioner.CacheLibDir("cudnn", cudaMajor);

    /// <summary>The optional bundled cuDNN folder beside the engine assembly (a self-contained deploy may drop the matching libs here).</summary>
    public static string BundledDir() => CudaRedistProvisioner.BundledDir("cudnn");

    /// <summary>Looks for an existing cuDNN once (thread-safe) and loads it if found. Never downloads and never throws. When none is
    /// found and auto-fetch is on, the fetch is left pending for <see cref="EnsureAvailable"/>.</summary>
    public static void EnsureProbed()
    {
        if (Volatile.Read(ref _probed) != 0) return;
        lock (_lock)
        {
            if (_probed != 0) return;
            try { Locate(); }
            catch (Exception ex) { _available = false; Reason = $"cuDNN probe error: {ex.Message}"; }
            Volatile.Write(ref _probed, 1);
        }
    }

    /// <summary>True when cuDNN can serve a call now. The first call that needs cuDNN and finds none installed downloads the matching
    /// redist (once per process, when <c>paths.cudnnAutofetch</c> is on), then loads it. A failed download leaves cuDNN off for the
    /// session, so the caller's fallback serves every call after it.</summary>
    public static bool EnsureAvailable()
    {
        EnsureProbed();
        // A download another thread is running holds the lock: wait on it instead of reporting cuDNN missing.
        if (!_fetchPending && !_fetchInProgress) return Available;
        lock (_lock)
        {
            if (!_fetchPending) return Available;
            _fetchInProgress = true;
            _fetchPending = false;   // one attempt per process, successful or not
            try
            {
                string? fetched = TryFetch(_cudaMajor);
                LibDir = fetched;
                // Load either way: a fetched build sits in the cache, which the loader searches first, and a failed fetch still
                // leaves the OS library path, where a matching system install works without the download.
                if (Load(_cudaMajor) && fetched is null)
                    Reason = $"cuDNN not found and auto-fetch failed (driver is CUDA {_cudaMajor}); see the warning above. " +
                             $"Install cuDNN 9 for CUDA {_cudaMajor} or set paths.cudnnDir.";
            }
            finally
            {
                _fetchInProgress = false;
            }
        }
        return Available;
    }

    /// <summary>Emits the status line once, at backend init, so a "why is it slow?" report is one log check. Never downloads.</summary>
    public static void LogStatus()
    {
        EnsureProbed();
        if (Available)
        {
            Logs.Info($"[cuDNN] {Reason}");
        }
        else if (_fetchPending || _fetchInProgress)
        {
            Logs.Info($"[cuDNN] not installed; the matching redist for CUDA {_cudaMajor} downloads on first use by a conv or " +
                      "attention model (paths.cudnnAutofetch=false to disable). Until then those paths use the fallbacks.");
        }
        else
        {
            Logs.Warning($"[cuDNN] NOT ACTIVE — {Reason} Convolutions fall back to im2col+cuBLAS and attention to the " +
                         $"custom flash kernel (correct, but slower on conv-heavy models: image UNet/VAE + audio vocoders).");
        }
    }

    /// <summary>Finds the cuDNN on disk. Sets the pending fetch when there is none and auto-fetch is on; otherwise loads what it found.</summary>
    private static void Locate()
    {
        int cudaMajor = DriverCudaMajor();
        _cudaMajor = cudaMajor;
        LibDir = ResolveExistingDir(cudaMajor);
        bool fetchable = cudaMajor >= 12 || !string.IsNullOrEmpty(EngineKnobs.CudnnUrl.Value);
        if (LibDir is null && EngineKnobs.CudnnAutofetch.Value && fetchable)
        {
            // The loader gets the first look: an OS-path or resolver-probe-dir install is usable without a download.
            // Only a missing library leaves the fetch pending; a CUDA-major mismatch stays rejected.
            if (Load(cudaMajor)) _fetchPending = true;
            return;
        }
        Load(cudaMajor);
    }

    /// <summary>Loads the cuDNN in <see cref="LibDir"/> (or the OS path) and applies the CUDA-major guard. Sets <see cref="Available"/>.
    /// Returns true when no cuDNN library could be found at all.</summary>
    private static bool Load(int cudaMajor)
    {
        // Trigger a load via a CONTEXT-FREE version query (compile-time constant — cannot hang even on a mismatched
        // build, unlike a conv/handle op). A DllNotFound here means the loader couldn't find cuDNN anywhere.
        long cudartVer;
        try { cudartVer = (long)CudnnApi.cudnnGetCudartVersion(); }
        catch (Exception ex)
        {
            _available = false;
            LibDir = null;
            string fetchHint = EngineKnobs.CudnnAutofetch.Value ? string.Empty : "Set paths.cudnnAutofetch=true to download it, ";
            Reason = $"cuDNN not found (searched paths.cudnnDir, {CacheLibDir(cudaMajor)}, {BundledDir()}, and the OS path; " +
                     $"driver is CUDA {cudaMajor}). {ex.GetType().Name}. " +
                     $"{fetchHint}drop the matching cuDNN 9 libs in the cache dir, or install cuDNN system-wide.";
            return ex is DllNotFoundException;
        }

        int cudnnCudaMajor = (int)(cudartVer / 1000);
        if (cudaMajor > 0 && cudnnCudaMajor > 0 && cudnnCudaMajor != cudaMajor)
        {
            _available = false;
            Reason = $"cuDNN REJECTED: built for CUDA {cudnnCudaMajor} but the driver runtime is CUDA {cudaMajor} — a mismatch " +
                     $"hangs the engine mid-inference, so the fast paths stay off. Install cuDNN 9 for CUDA {cudaMajor} " +
                     $"(or set paths.cudnnDir to a matching build).";
            return false;
        }

        Version = (long)CudnnApi.cudnnGetVersion();
        _available = true;
        string attentionStatus = SupportsSdpa ? "convolution + fused-attention fast paths ENABLED"
            : "convolution ENABLED; fused attention requires cuDNN 9.21 or newer";
        Reason = $"cuDNN {Version / 10000}.{Version / 100 % 100}.{Version % 100} (built for CUDA {cudnnCudaMajor}) " +
                 $"from {(LibDir ?? "the OS library path")} — {attentionStatus}.";
        return false;
    }

    /// <summary>The first search dir (env / cache / bundled) that actually holds the cuDNN dispatcher, or null.</summary>
    private static string? ResolveExistingDir(int cudaMajor)
    {
        string soname = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cudnn64_9.dll" : "libcudnn.so.9";
        foreach (string? dir in new[] { EngineKnobs.CudnnDir.Value, CacheLibDir(cudaMajor), BundledDir() })
        {
            if (!string.IsNullOrEmpty(dir) && File.Exists(Path.Combine(dir, soname)))
            {
                return dir;
            }
        }
        return null;   // fall through to the OS default (system install) in the loader
    }

    /// <summary>Best-effort one-time download of the matching cuDNN redist into the cache. Uses <c>paths.cudnnUrl</c> when set (a direct .tar.xz / .zip), else NVIDIA's public redist for this CUDA major + OS. Returns the cache lib dir on success, else null (the diagnostic then tells the user how to install manually).</summary>
    private static string? TryFetch(int cudaMajor)
    {
        bool win = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string? url = EngineKnobs.CudnnUrl.Value;
        if (string.IsNullOrEmpty(url))
        {
            string? plat = CudaRedistProvisioner.Platform();
            // NVIDIA publishes cuDNN 9.21 redist archives for CUDA 12/13 only — the CUDA 11 URL 404s.
            if (cudaMajor < 12 || plat is null)
            {
                Logs.Warning($"[cuDNN] no public cuDNN 9.21 redist exists for CUDA {cudaMajor} on this platform — install " +
                    "cuDNN 9 manually or point paths.cudnnUrl/paths.cudnnDir at a compatible build.");
                return null;
            }
            // NVIDIA redist. Version is pinned but overridable; the redist layout is stable per major.
            // The custom SDPA graph uses the public softmax operation added in cuDNN 9.21, so older
            // CUDA-13 redistributables can run convolution but cannot construct this attention graph.
            string ver = EngineKnobs.CudnnVersion.Value ?? "9.21.0.82";
            string ext = win ? "zip" : "tar.xz";
            url = $"https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/{plat}/" +
                  $"cudnn-{plat}-{ver}_cuda{cudaMajor}-archive.{ext}";
        }
        return CudaRedistProvisioner.FetchInto(new CudaRedistSpec(
            Component: "cudnn",
            Label: "cuDNN",
            Url: url,
            CudaMajor: cudaMajor,
            ExpectedSha256: null,
            ExpectedBytes: null,
            LibraryPatterns: [win ? "cudnn*.dll" : "libcudnn*.so*"],
            Sentinel: win ? "cudnn64_9.dll" : "libcudnn.so.9",
            ApproxSize: "about 1 GB",
            KnobHint: "paths.cudnnAutofetch"));
    }
}
