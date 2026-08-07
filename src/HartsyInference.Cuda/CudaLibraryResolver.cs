using System.Reflection;
using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>Resolves "cuda" and "cublas" library names to platform-specific paths at runtime.</summary>
public static class CudaLibraryResolver
{
    private static int _registered;

    /// <summary>Registers the resolver for the HartsyInference.Cuda assembly. Safe to call multiple times.</summary>
    public static void Register()
    {
        if (Interlocked.CompareExchange(ref _registered, 1, 0) == 0)
        {
            NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, ResolveLibrary);
        }
    }

    /// <summary>Directories probed for CUDA userspace libs (cuBLAS/cuBLASLt/cuDNN/cudart) BEFORE the plain
    /// soname search. The toolkit userspace on dev boxes often lives outside the loader's paths (this
    /// project's documented install is <c>~/.local/lib/cuda13</c>), and <c>LD_LIBRARY_PATH</c> only works
    /// when whoever launched the process remembered to export it — a bare SwarmUI relaunch without it kills
    /// the whole backend (DllNotFoundException: libcublas.so.13). Probing here makes the engine
    /// self-sufficient regardless of launcher environment. <c>HARTSY_CUDA_LIB_DIR</c> overrides/prepends.
    /// libcuda.so.1 is NOT probed here — it's the driver, always in the system loader path.</summary>
    private static readonly string[] ProbeDirs = BuildProbeDirs();

    private static string[] BuildProbeDirs()
    {
        List<string> dirs = new();
        string? overrideDir = Environment.GetEnvironmentVariable("HARTSY_CUDA_LIB_DIR");
        if (!string.IsNullOrEmpty(overrideDir))
        {
            dirs.Add(overrideDir);
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            dirs.Add(Path.Combine(home, ".local", "lib", "cuda13"));
        }
        // cuDNN search dirs (provisioned/bundled): an explicit override, the per-user cache (per CUDA major — the
        // driver isn't queried here to avoid a circular load during static init, so all common majors are listed;
        // File.Exists filters), and a 'cudnn' folder beside the assembly. See CudnnRuntime.
        string? cudnnDir = Environment.GetEnvironmentVariable("HARTSY_CUDNN_DIR");
        if (!string.IsNullOrEmpty(cudnnDir)) dirs.Add(cudnnDir);
        // NCCL commonly lives inside a torch venv (site-packages/nvidia/nccl/lib) rather than the loader
        // path; this override (or a copy/hardlink in any dir above) makes it resolvable without apt.
        string? ncclDir = Environment.GetEnvironmentVariable("HARTSY_NCCL_DIR");
        if (!string.IsNullOrEmpty(ncclDir)) dirs.Add(ncclDir);
        foreach (int major in new[] { 13, 12, 11 }) dirs.Add(CudnnRuntime.CacheLibDir(major));
        dirs.Add(CudnnRuntime.BundledDir());
        return dirs.ToArray();
    }

    /// <summary>Tries each candidate soname in order, first as an absolute path under every probe dir,
    /// then via the default loader search (LD_LIBRARY_PATH / ldconfig / PATH).</summary>
    private static bool TryLoadFirst(out nint handle, params string[] sonames)
    {
        foreach (string soname in sonames)
        {
            foreach (string dir in ProbeDirs)
            {
                string path = Path.Combine(dir, soname);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                    return true;
            }
            if (NativeLibrary.TryLoad(soname, out handle))
                return true;
        }
        handle = 0;
        return false;
    }

    private static nint LoadFirst(params string[] sonames)
    {
        if (TryLoadFirst(out nint handle, sonames))
            return handle;
        // Nothing resolved: force a real Load of the newest name so the caller gets the standard
        // DllNotFoundException (message names the library) instead of a silent 0 handle.
        return NativeLibrary.Load(sonames[0]);
    }

    /// <summary>Availability probe for <see cref="CudaContext.IsAvailable"/>: can cuBLAS be loaded (probe
    /// dirs included)? Keeps the availability check and the actual resolution on the SAME search logic —
    /// a duplicated bare-soname probe here previously reported CUDA unavailable whenever LD_LIBRARY_PATH
    /// was missing, even though the resolver could load everything from the probe dirs.</summary>
    public static bool CublasLoadable()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return TryLoadFirst(out _, "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll");
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return TryLoadFirst(out _, "libcublas.so.13", "libcublas.so.12", "libcublas.so.11");
        return false;
    }

    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        bool windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        bool linux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);

        if (libraryName == "cuda")
        {
            if (windows)
                return NativeLibrary.Load("nvcuda.dll");
            else if (linux)
                return NativeLibrary.Load("libcuda.so.1");
        }

        // NVTX ships with the toolkit but CUDA 12+ leaves it outside the loader path on some distros, so the
        // profiler silently degraded to no-ops. Same probe dirs as the rest of the userspace libs.
        if (libraryName == "libnvToolsExt.so.1" || libraryName == "nvToolsExt64_1")
        {
            if (windows)
                return LoadFirst("nvToolsExt64_1.dll");
            else if (linux)
                return LoadFirst("libnvToolsExt.so.1");
        }

        if (libraryName == "cublas")
        {
            // Try newest first, then older toolkits
            if (windows)
                return LoadFirst("cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll");
            else if (linux)
                return LoadFirst("libcublas.so.13", "libcublas.so.12", "libcublas.so.11");
        }

        // cuDNN (fused flash-attention SDPA fast path, HARTSY_SDPA_CUDNN). Ships only as a versioned soname
        // (libcudnn.so.9 / cudnn64_9.dll) — no unversioned alias — so the bare [LibraryImport("cudnn")] name
        // fails without this case. Default-off, loaded lazily on first cuDNN SDPA call.
        if (libraryName == "cudnn")
        {
            if (windows)
                return LoadFirst("cudnn64_9.dll");
            else if (linux)
                return LoadFirst("libcudnn.so.9");
        }

        // cuBLASLt ships as a SEPARATE library from cuBLAS (libcublasLt.so.N) and, like cuBLAS, only as a
        // versioned soname — there is no unversioned libcublasLt.so — so the bare [LibraryImport("cublasLt")]
        // name fails to load without this case. Exercised by the epilogue-fusion, native-FP8, and general
        // Lt-GEMM paths (all default-off, which is why this was dormant until perf flags got turned on).
        if (libraryName == "cublasLt")
        {
            if (windows)
                return LoadFirst("cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll");
            else if (linux)
                return LoadFirst("libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11");
        }

        // NCCL (collective transport — context/tensor parallelism). Versioned soname only, no unversioned
        // alias; availability is optional by design (NcclApi.IsLoadable gates, the peer-copy transport is
        // the fallback), so resolution failure here surfaces as a caught DllNotFoundException, not a crash.
        if (libraryName == "nccl")
        {
            if (windows)
                return LoadFirst("nccl64_2.dll");
            else if (linux)
                return LoadFirst("libnccl.so.2");
        }

        return 0;
    }
}
