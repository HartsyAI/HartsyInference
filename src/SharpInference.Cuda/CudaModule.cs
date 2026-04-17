using System.Runtime.InteropServices;
using System.Text;

namespace SharpInference.Cuda;

/// <summary>Loads a PTX file into a CUDA module and provides function handle lookup. PTX is JIT-compiled by the driver for the current GPU.</summary>
public sealed class CudaModule : IDisposable
{
    private nint _module;

    /// <summary>The underlying CUDA module handle.</summary>
    public nint Handle
    {
        get
        {
            nint m = _module;
            if (m == 0)
                throw new ObjectDisposedException(nameof(CudaModule));
            return m;
        }
    }

    private CudaModule(nint module)
    {
        _module = module;
    }

    /// <summary>Loads a PTX file from disk and JIT-compiles it for the current device.</summary>
    public static CudaModule LoadFromFile(string ptxPath)
    {
        if (!File.Exists(ptxPath))
            throw new FileNotFoundException($"PTX file not found: {ptxPath}", ptxPath);

        byte[] ptxBytes = File.ReadAllBytes(ptxPath);
        return LoadFromBytes(ptxBytes);
    }

    /// <summary>Loads PTX from a byte array (must be null-terminated UTF-8 or will be null-terminated automatically).</summary>
    public static CudaModule LoadFromBytes(byte[] ptxBytes)
    {
        // Ensure null terminator
        byte[] terminated;
        if (ptxBytes.Length == 0 || ptxBytes[^1] != 0)
        {
            terminated = new byte[ptxBytes.Length + 1];
            Array.Copy(ptxBytes, terminated, ptxBytes.Length);
            terminated[^1] = 0;
        }
        else
        {
            terminated = ptxBytes;
        }

        nint pinned = Marshal.AllocHGlobal(terminated.Length);
        try
        {
            Marshal.Copy(terminated, 0, pinned, terminated.Length);
            CudaDriverApi.cuModuleLoadData(out nint module, pinned).ThrowOnError();
            return new CudaModule(module);
        }
        finally
        {
            Marshal.FreeHGlobal(pinned);
        }
    }

    /// <summary>Loads PTX from a string.</summary>
    public static CudaModule LoadFromString(string ptxSource)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(ptxSource + "\0");
        return LoadFromBytes(bytes);
    }

    /// <summary>Gets a kernel function handle by name from this module.</summary>
    public nint GetFunction(string functionName)
    {
        CudaDriverApi.cuModuleGetFunction(out nint function, Handle, functionName).ThrowOnError();
        return function;
    }

    public void Dispose()
    {
        nint m = Interlocked.Exchange(ref _module, 0);
        if (m != 0)
        {
            CudaDriverApi.cuModuleUnload(m);
        }
        GC.SuppressFinalize(this);
    }

    ~CudaModule()
    {
        nint m = Interlocked.Exchange(ref _module, 0);
        if (m != 0)
        {
            CudaDriverApi.cuModuleUnload(m);
        }
    }
}
