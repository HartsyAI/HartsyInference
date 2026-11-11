using System.Reflection;
using System.Runtime.InteropServices;
using HartsyInference.Core.Configuration;

namespace HartsyInference.Cuda;

/// <summary>Resolves "cuda" and "cublas" library names to platform-specific paths at runtime.</summary>
public static class CudaLibraryResolver
{
    private static readonly object _registerLock = new();
    private static volatile bool _registered;

    /// <summary>Registers the resolver for the HartsyInference.Cuda assembly. Safe to call repeatedly and from several
    /// threads: it returns only once the resolver is set, so no P/Invoke falls back to the default search.</summary>
    public static void Register()
    {
        if (_registered)
        {
            return;
        }
        lock (_registerLock)
        {
            if (_registered)
            {
                return;
            }
            try
            {
                NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, ResolveLibrary);
            }
            finally
            {
                // SetDllImportResolver throws only when a resolver is already set; retrying cannot succeed.
                _registered = true;
            }
        }
    }

    /// <summary>Directories probed for CUDA userspace libs (cuBLAS/cuBLASLt/cuDNN/cudart) BEFORE the plain soname search. The toolkit userspace on dev boxes often lives outside the loader's paths (this project's documented install is <c>~/.local/lib/cuda13</c>), and <c>LD_LIBRARY_PATH</c> only works when whoever launched the process remembered to export it — a bare SwarmUI relaunch without it kills the whole backend (DllNotFoundException: libcublas.so.13). Probing here makes the engine self-sufficient regardless of launcher environment. <c>paths.cudaLibDir</c> overrides/prepends. libcuda.so.1 is NOT probed here — it's the driver, always in the system loader path.</summary>
    private static readonly string[] ProbeDirs = BuildProbeDirs();

    private static string[] BuildProbeDirs()
    {
        List<string> dirs = new();
        string? overrideDir = EngineKnobs.CudaLibDir.Value;
        if (!string.IsNullOrEmpty(overrideDir))
        {
            dirs.Add(overrideDir);
        }
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            dirs.Add(Path.Combine(home, ".local", "lib", "cuda13"));
        }
        // cuBLAS and cuDNN search dirs (provisioned/bundled): explicit overrides, the per-user caches (per CUDA major —
        // the driver isn't queried here to avoid a circular load during static init, so all common majors are listed;
        // File.Exists filters), and 'cublas'/'cudnn' folders beside the assembly. See CublasRuntime and CudnnRuntime.
        string? cublasDir = EngineKnobs.CublasDir.Value;
        if (!string.IsNullOrEmpty(cublasDir)) dirs.Add(cublasDir);
        string? cudnnDir = EngineKnobs.CudnnDir.Value;
        if (!string.IsNullOrEmpty(cudnnDir)) dirs.Add(cudnnDir);
        // NCCL commonly lives inside a torch venv (site-packages/nvidia/nccl/lib) rather than the loader
        // path; this override (or a copy/hardlink in any dir above) makes it resolvable without apt.
        string? ncclDir = EngineKnobs.NcclDir.Value;
        if (!string.IsNullOrEmpty(ncclDir)) dirs.Add(ncclDir);
        foreach (int major in CublasRuntime.KnownMajors) dirs.Add(CudaRedistProvisioner.CacheLibDir("cublas", major));
        foreach (int major in CublasRuntime.KnownMajors) dirs.Add(CudnnRuntime.CacheLibDir(major));
        dirs.Add(CudaRedistProvisioner.BundledDir("cublas"));
        dirs.Add(CudnnRuntime.BundledDir());
        return dirs.ToArray();
    }

    /// <summary>Tries each candidate soname in order, first as an absolute path under every probe dir, then via the default loader search (LD_LIBRARY_PATH / ldconfig / PATH).</summary>
    private static bool TryLoadFirst(out nint handle, params string[] sonames) => TryLoadFirst(out handle, out _, sonames);

    /// <summary>As <see cref="TryLoadFirst(out nint, string[])"/>, also reporting the file loaded (null when the OS loader found it by name).</summary>
    internal static bool TryLoadFirst(out nint handle, out string? loadedFrom, params string[] sonames)
    {
        foreach (string soname in sonames)
        {
            foreach (string dir in ProbeDirs)
            {
                string path = Path.Combine(dir, soname);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                {
                    loadedFrom = path;
                    return true;
                }
            }
            if (NativeLibrary.TryLoad(soname, out handle))
            {
                loadedFrom = null;
                return true;
            }
        }
        handle = 0;
        loadedFrom = null;
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

    /// <summary>Loads cudnn64_9.dll from the first probe dir holding the complete cuDNN set, with that dir first on PATH: the
    /// dispatcher loads its sub-libraries by bare name, not from its own folder, and exits the process when one is missing.</summary>
    private static nint LoadCudnnWindows()
    {
        foreach (string dir in ProbeDirs)
        {
            if (!CudnnRuntime.HasLibraries(dir))
            {
                continue;
            }
            string? path = PathWithDirFirst(Environment.GetEnvironmentVariable("PATH"), dir);
            if (path is not null)
            {
                Environment.SetEnvironmentVariable("PATH", path);
            }
            if (NativeLibrary.TryLoad(Path.Combine(dir, "cudnn64_9.dll"), out nint handle))
            {
                return handle;
            }
        }
        // A system install keeps its folder on PATH, so its sub-libraries resolve beside it.
        return NativeLibrary.Load("cudnn64_9.dll");
    }

    /// <summary><paramref name="path"/> with <paramref name="dir"/> as its first entry, or null when it already is.</summary>
    internal static string? PathWithDirFirst(string? path, string dir)
    {
        if (string.IsNullOrEmpty(path))
        {
            return dir;
        }
        int end = path.IndexOf(Path.PathSeparator);
        string first = (end < 0 ? path : path[..end]).Trim().TrimEnd('\\', '/');
        if (string.Equals(first, dir.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }
        return dir + Path.PathSeparator + path;
    }

    /// <summary>Availability probe for <see cref="CudaContext.IsAvailable"/>: can cuBLAS be loaded (probe dirs included)? Keeps the availability check and the actual resolution on the SAME search logic — a duplicated bare-soname probe here previously reported CUDA unavailable whenever LD_LIBRARY_PATH was missing, even though the resolver could load everything from the probe dirs.</summary>
    public static bool CublasLoadable()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && !RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return false;
        CublasRuntime.EnsureProbed();
        return CublasRuntime.Available;
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

        if (libraryName == "cublas" && (windows || linux))
        {
            // The same file CublasRuntime chose: newest major the driver can run, never a newer one.
            string[] names = CublasRuntime.Sonames(lt: false);
            if (CublasRuntime.LoadedPath is { } chosen && NativeLibrary.TryLoad(chosen, out nint loaded))
                return loaded;
            return LoadFirst(names);
        }

        // cuDNN (fused flash-attention SDPA fast path, numerics.sdpaCudnn). Ships only as a versioned soname
        // (libcudnn.so.9 / cudnn64_9.dll) — no unversioned alias — so the bare [LibraryImport("cudnn")] name
        // fails without this case. Default-off, loaded lazily on first cuDNN SDPA call.
        if (libraryName == "cudnn")
        {
            if (windows)
                return LoadCudnnWindows();
            else if (linux)
                return LoadFirst("libcudnn.so.9");
        }

        // cuBLASLt ships as a SEPARATE library from cuBLAS (libcublasLt.so.N) and, like cuBLAS, only as a
        // versioned soname — there is no unversioned libcublasLt.so — so the bare [LibraryImport("cublasLt")]
        // name fails to load without this case. Exercised by the epilogue-fusion, native-FP8, and general
        // Lt-GEMM paths (all default-off, which is why this was dormant until perf flags got turned on).
        if (libraryName == "cublasLt" && (windows || linux))
        {
            // cuBLASLt must match the loaded cuBLAS: look beside it first, then only that major (any driver-allowed major when
            // no cuBLAS loaded), so an OS-path cuBLAS 12 never pairs with a cached cuBLASLt 13.
            string[] names = CublasRuntime.ResolvedMajor > 0
                ? [CublasRuntime.Soname(CublasRuntime.ResolvedMajor, lt: true)]
                : CublasRuntime.Sonames(lt: true);
            if (CublasRuntime.LibDir is { } dir && CublasRuntime.ResolvedMajor > 0
                && NativeLibrary.TryLoad(Path.Combine(dir, CublasRuntime.Soname(CublasRuntime.ResolvedMajor, lt: true)), out nint loaded))
                return loaded;
            return LoadFirst(names);
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
