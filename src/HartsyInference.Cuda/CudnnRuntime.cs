using System.Diagnostics;
using System.Runtime.InteropServices;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Cuda;

/// <summary>Locates, version-guards, and (optionally) auto-provisions the cuDNN runtime so the conv/SDPA fast paths engage across OS / CUDA-version / GPU combos WITHOUT the user hand-installing it — and emits one clear diagnostic line at engine start so a "why is it slow?" report is a single log check. <para>cuDNN is a ~1 GB, <b>CUDA-major-specific</b>, OS-specific set of libraries, so it is NOT shipped inside the DLL folder (that would be ~6 GB across CUDA 11/12/13 × Windows/Linux). Instead the matching build is resolved from, in order: <c>HARTSY_CUDNN_DIR</c> → the per-user engine cache → a <c>cudnn/</c> folder beside the assembly (an optional bundled deploy) → the OS default (a system install). If none is found and auto-fetch is on, the matching redist is downloaded once to the cache. A cuDNN built for a different CUDA major than the running driver is <b>REJECTED</b> — that exact mismatch (a CUDA-12 cuDNN on a CUDA-13 runtime) previously hung the engine mid-inference.</para></summary>
public static class CudnnRuntime
{
    /// <summary>True when a version-matched cuDNN loaded and passed the CUDA-major guard.</summary>
    public static bool Available { get; private set; }

    /// <summary>True when the loaded runtime exposes the public softmax operation used by the custom SDPA graph.</summary>
    internal static bool SupportsSdpa => Available && IsSdpaVersionSupported(Version);

    internal static bool IsSdpaVersionSupported(long version) => version >= 92100;

    /// <summary>Human-readable status for the startup diagnostic (why cuDNN is / isn't active).</summary>
    public static string Reason { get; private set; } = "not yet probed";

    /// <summary>Loaded cuDNN version (e.g. 90600) when <see cref="Available"/>, else 0.</summary>
    public static long Version { get; private set; }

    /// <summary>Directory the matching cuDNN was resolved from (cache/bundle/system/env); added to the loader search so <c>[LibraryImport("cudnn")]</c> resolves the same file. Read by <see cref="CudaLibraryResolver"/>.</summary>
    public static string? LibDir { get; private set; }

    private static int _probed;
    private static readonly object _lock = new();

    /// <summary>Latest CUDA version the DRIVER supports (13020 → CUDA 13.2); 0 if unknown. Picks the cuDNN build.</summary>
    public static int DriverCudaMajor()
    {
        try { return CudaDriverApi.cuDriverGetVersion(out int v) == 0 ? v / 1000 : 0; }
        catch { return 0; }
    }

    /// <summary>Deterministic per-user cache dir the matching cuDNN is provisioned into (and searched from).</summary>
    public static string CacheLibDir(int cudaMajor)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".cache", "hartsyinference", "cudnn", $"cuda{cudaMajor}", "lib");
    }

    /// <summary>The optional bundled cuDNN folder beside the engine assembly (a self-contained deploy may drop the matching libs here).</summary>
    public static string BundledDir() => Path.Combine(AppContext.BaseDirectory, "cudnn");

    /// <summary>Probes cuDNN once (thread-safe): provisions if needed, loads, and applies the CUDA-major guard. Never throws — on any failure <see cref="Available"/> stays false with a <see cref="Reason"/>.</summary>
    public static void EnsureProbed()
    {
        if (Volatile.Read(ref _probed) != 0) return;
        lock (_lock)
        {
            if (_probed != 0) return;
            try { Probe(); }
            catch (Exception ex) { Available = false; Reason = $"cuDNN probe error: {ex.Message}"; }
            Volatile.Write(ref _probed, 1);
        }
    }

    private static void Probe()
    {
        int cudaMajor = DriverCudaMajor();

        // Point the loader at the cache dir first so, if we fetch, the same file is what [LibraryImport] resolves.
        LibDir = ResolveExistingDir(cudaMajor);
        if (LibDir is null && AutofetchEnabled())
        {
            LibDir = TryFetch(cudaMajor);
        }

        // Trigger a load via a CONTEXT-FREE version query (compile-time constant — cannot hang even on a mismatched
        // build, unlike a conv/handle op). A DllNotFound here means the loader couldn't find cuDNN anywhere.
        long cudartVer;
        try { cudartVer = (long)CudnnApi.cudnnGetCudartVersion(); }
        catch (Exception ex)
        {
            Available = false;
            LibDir = null;
            Reason = $"cuDNN not found (searched HARTSY_CUDNN_DIR, {CacheLibDir(cudaMajor)}, {BundledDir()}, and the OS path; " +
                     $"driver is CUDA {cudaMajor}). {ex.GetType().Name}. " +
                     $"Enable with HARTSY_CUDNN_AUTOFETCH=1, drop the matching cuDNN 9 libs in the cache dir, or install cuDNN system-wide.";
            return;
        }

        int cudnnCudaMajor = (int)(cudartVer / 1000);
        if (cudaMajor > 0 && cudnnCudaMajor > 0 && cudnnCudaMajor != cudaMajor)
        {
            Available = false;
            Reason = $"cuDNN REJECTED: built for CUDA {cudnnCudaMajor} but the driver runtime is CUDA {cudaMajor} — a mismatch " +
                     $"hangs the engine mid-inference, so the fast paths stay off. Install cuDNN 9 for CUDA {cudaMajor} " +
                     $"(or set HARTSY_CUDNN_DIR to a matching build).";
            return;
        }

        Version = (long)CudnnApi.cudnnGetVersion();
        Available = true;
        string attentionStatus = SupportsSdpa
            ? "convolution + fused-attention fast paths ENABLED"
            : "convolution ENABLED; fused attention requires cuDNN 9.21 or newer";
        Reason = $"cuDNN {Version / 10000}.{Version / 100 % 100}.{Version % 100} (built for CUDA {cudnnCudaMajor}) " +
                 $"from {(LibDir ?? "the OS library path")} — {attentionStatus}.";
    }

    /// <summary>Emits the startup diagnostic. Called once from the backend init so slow-perf reports are one line.</summary>
    public static void LogStatus()
    {
        EnsureProbed();
        if (Available)
        {
            Logs.Info($"[cuDNN] {Reason}");
        }
        else
        {
            Logs.Warning($"[cuDNN] NOT ACTIVE — {Reason} Convolutions fall back to im2col+cuBLAS and attention to the " +
                         $"custom flash kernel (correct, but slower on conv-heavy models: image UNet/VAE + audio vocoders).");
        }
    }

    private static bool AutofetchEnabled() => EnvSwitch.IsEnabled("HARTSY_CUDNN_AUTOFETCH", defaultOn: false);

    /// <summary>The first search dir (env / cache / bundled) that actually holds the cuDNN dispatcher, or null.</summary>
    private static string? ResolveExistingDir(int cudaMajor)
    {
        string soname = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cudnn64_9.dll" : "libcudnn.so.9";
        foreach (string? dir in new[] { Environment.GetEnvironmentVariable("HARTSY_CUDNN_DIR"), CacheLibDir(cudaMajor), BundledDir() })
        {
            if (!string.IsNullOrEmpty(dir) && File.Exists(Path.Combine(dir, soname)))
            {
                return dir;
            }
        }
        return null;   // fall through to the OS default (system install) in the loader
    }

    /// <summary>Best-effort one-time download of the matching cuDNN redist into the cache. Uses <c>HARTSY_CUDNN_URL</c> when set (a direct .tar.xz / .zip), else NVIDIA's public redist for this CUDA major + OS. Extraction shells to the system <c>tar</c>. Returns the cache lib dir on success, else null (the diagnostic then tells the user how to install manually).</summary>
    private static string? TryFetch(int cudaMajor)
    {
        try
        {
            bool win = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            string plat = win ? "windows-x86_64" : "linux-x86_64";
            string? url = Environment.GetEnvironmentVariable("HARTSY_CUDNN_URL");
            if (string.IsNullOrEmpty(url))
            {
                // NVIDIA publishes cuDNN 9.21 redist archives for CUDA 12/13 only — the CUDA 11 URL 404s.
                if (cudaMajor < 12)
                {
                    Logs.Warning($"[cuDNN] no public cuDNN 9.21 redist exists for CUDA {cudaMajor} — install " +
                        "cuDNN 9 manually or point HARTSY_CUDNN_URL/HARTSY_CUDNN_DIR at a compatible build.");
                    return null;
                }
                // NVIDIA redist. Version is pinned but overridable; the redist layout is stable per major.
                // The custom SDPA graph uses the public softmax operation added in cuDNN 9.21, so older
                // CUDA-13 redistributables can run convolution but cannot construct this attention graph.
                string ver = Environment.GetEnvironmentVariable("HARTSY_CUDNN_VERSION") ?? "9.21.0.82";
                string ext = win ? "zip" : "tar.xz";
                url = $"https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/{plat}/" +
                      $"cudnn-{plat}-{ver}_cuda{cudaMajor}-archive.{ext}";
            }
            string libDir = CacheLibDir(cudaMajor);
            Directory.CreateDirectory(libDir);
            string work = Path.Combine(Path.GetDirectoryName(libDir)!, "download");
            Directory.CreateDirectory(work);
            string archive = Path.Combine(work, Path.GetFileName(new Uri(url).LocalPath));

            Logs.Info($"[cuDNN] not found for CUDA {cudaMajor}; auto-fetching {url} " +
                $"(one-time, ~1 GB → {libDir}; set HARTSY_CUDNN_AUTOFETCH=0 to disable)...");
            using (System.Net.Http.HttpClient http = new() { Timeout = TimeSpan.FromMinutes(30) })
            using (Stream s = http.GetStreamAsync(url).GetAwaiter().GetResult())
            using (FileStream f = File.Create(archive))
            {
                s.CopyTo(f);
            }
            // Extract (tar handles .tar.xz and .zip on modern Windows/Linux) and flatten the archive's lib/*.so.9.
            RunTar($"-xf \"{archive}\" -C \"{work}\"");
            foreach (string lib in Directory.EnumerateFiles(work, win ? "cudnn*.dll" : "libcudnn*.so.*", SearchOption.AllDirectories))
            {
                File.Copy(lib, Path.Combine(libDir, Path.GetFileName(lib)), overwrite: true);
            }
            try { Directory.Delete(work, recursive: true); } catch { /* leave the download dir on cleanup failure */ }
            Logs.Info($"[cuDNN] auto-fetch complete → {libDir}.");
            return File.Exists(Path.Combine(libDir, win ? "cudnn64_9.dll" : "libcudnn.so.9")) ? libDir : null;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[cuDNN] auto-fetch failed ({ex.Message}) — install cuDNN 9 for CUDA {cudaMajor} manually or set HARTSY_CUDNN_DIR.");
            return null;
        }
    }

    private static void RunTar(string args)
    {
        using Process p = Process.Start(new ProcessStartInfo("tar", args) { UseShellExecute = false, RedirectStandardError = true })
            ?? throw new InvalidOperationException("failed to start 'tar'");
        p.WaitForExit();
        if (p.ExitCode != 0) throw new InvalidOperationException($"tar failed: {p.StandardError.ReadToEnd()}");
    }
}
