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
    public static unsafe CudaModule LoadFromBytes(byte[] ptxBytes)
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

            // Use cuModuleLoadDataEx with JIT error/info log to get diagnostics on failure
            const int CU_JIT_ERROR_LOG_BUFFER = 5;
            const int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
            const int CU_JIT_INFO_LOG_BUFFER = 3;
            const int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;

            int logSize = 4096;
            nint errorLog = Marshal.AllocHGlobal(logSize);
            nint infoLog = Marshal.AllocHGlobal(logSize);

            try
            {
                // Zero out the buffers
                new Span<byte>((void*)errorLog, logSize).Clear();
                new Span<byte>((void*)infoLog, logSize).Clear();

                int* options = stackalloc int[4];
                options[0] = CU_JIT_INFO_LOG_BUFFER;
                options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
                options[2] = CU_JIT_ERROR_LOG_BUFFER;
                options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;

                nint* optionValues = stackalloc nint[4];
                optionValues[0] = infoLog;
                optionValues[1] = (nint)logSize;
                optionValues[2] = errorLog;
                optionValues[3] = (nint)logSize;

                int result = CudaDriverApi.cuModuleLoadDataEx(
                    out nint module, pinned,
                    4, (nint)options, (nint)optionValues);

                if (result != 0)
                {
                    string errorStr = Marshal.PtrToStringAnsi(errorLog) ?? "(no error log)";
                    string infoStr = Marshal.PtrToStringAnsi(infoLog) ?? "(no info log)";
                    string msg = $"PTX JIT failed. Error log: {errorStr}. Info log: {infoStr}";
                    throw new CudaException(result, msg);
                }

                return new CudaModule(module);
            }
            finally
            {
                Marshal.FreeHGlobal(errorLog);
                Marshal.FreeHGlobal(infoLog);
            }
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
