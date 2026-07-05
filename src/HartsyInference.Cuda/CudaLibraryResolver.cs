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

    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "cuda")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return NativeLibrary.Load("nvcuda.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return NativeLibrary.Load("libcuda.so.1");
            }
        }

        if (libraryName == "cublas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Try newest first, then older toolkits
                if (NativeLibrary.TryLoad("cublas64_13.dll", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("cublas64_12.dll", out handle))
                    return handle;
                if (NativeLibrary.TryLoad("cublas64_11.dll", out handle))
                    return handle;
                return NativeLibrary.Load("cublas64_13.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcublas.so.13", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("libcublas.so.12", out handle))
                    return handle;
                if (NativeLibrary.TryLoad("libcublas.so.11", out handle))
                    return handle;
                return NativeLibrary.Load("libcublas.so.13");
            }
        }

        // cuDNN (fused flash-attention SDPA fast path, HARTSY_SDPA_CUDNN). Ships only as a versioned soname
        // (libcudnn.so.9 / cudnn64_9.dll) — no unversioned alias — so the bare [LibraryImport("cudnn")] name
        // fails without this case. Default-off, loaded lazily on first cuDNN SDPA call.
        if (libraryName == "cudnn")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("cudnn64_9.dll", out nint handle))
                    return handle;
                return NativeLibrary.Load("cudnn64_9.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcudnn.so.9", out nint handle))
                    return handle;
                return NativeLibrary.Load("libcudnn.so.9");
            }
        }

        // cuBLASLt ships as a SEPARATE library from cuBLAS (libcublasLt.so.N) and, like cuBLAS, only as a
        // versioned soname — there is no unversioned libcublasLt.so — so the bare [LibraryImport("cublasLt")]
        // name fails to load without this case. Exercised by the epilogue-fusion, native-FP8, and general
        // Lt-GEMM paths (all default-off, which is why this was dormant until perf flags got turned on).
        if (libraryName == "cublasLt")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("cublasLt64_13.dll", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("cublasLt64_12.dll", out handle))
                    return handle;
                if (NativeLibrary.TryLoad("cublasLt64_11.dll", out handle))
                    return handle;
                return NativeLibrary.Load("cublasLt64_13.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcublasLt.so.13", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("libcublasLt.so.12", out handle))
                    return handle;
                if (NativeLibrary.TryLoad("libcublasLt.so.11", out handle))
                    return handle;
                return NativeLibrary.Load("libcublasLt.so.13");
            }
        }

        return 0;
    }
}
