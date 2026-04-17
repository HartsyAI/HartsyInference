using System.Reflection;
using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>Resolves "cuda" and "cublas" library names to platform-specific paths at runtime.</summary>
public static class CudaLibraryResolver
{
    private static int _registered;

    /// <summary>Registers the resolver for the SharpInference.Cuda assembly. Safe to call multiple times.</summary>
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
                // Try versioned names first, then unversioned
                if (NativeLibrary.TryLoad("cublas64_12.dll", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("cublas64_11.dll", out handle))
                    return handle;
                return NativeLibrary.Load("cublas64_12.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                if (NativeLibrary.TryLoad("libcublas.so.12", out nint handle))
                    return handle;
                if (NativeLibrary.TryLoad("libcublas.so.11", out handle))
                    return handle;
                return NativeLibrary.Load("libcublas.so.12");
            }
        }

        return 0;
    }
}
