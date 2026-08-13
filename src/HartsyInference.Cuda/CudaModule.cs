using System.Runtime.InteropServices;
using System.Text;

namespace HartsyInference.Cuda;

/// <summary>Loads a PTX file into a CUDA module and provides function handle lookup. PTX is JIT-compiled by the driver for the current GPU.</summary>
public sealed class CudaModule : IDisposable
{
    private nint _module;
    private static long _liveHandleCount;

    /// <summary>Number of module handles currently owned by managed wrappers. Test-only lifecycle diagnostic.</summary>
    internal static long LiveHandleCountForTests => Interlocked.Read(ref _liveHandleCount);

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
        if (module != 0) Interlocked.Increment(ref _liveHandleCount);
    }

    /// <summary>Loads a PTX file from disk and JIT-compiles it for the current device.</summary>
    /// <param name="maxRegisters">Per-thread register cap for the JIT (0 = let it choose). See
    /// <see cref="LoadFromBytes"/> for why a kernel might need this.</param>
    public static CudaModule LoadFromFile(string ptxPath, int maxRegisters = 0)
    {
        if (!File.Exists(ptxPath))
            throw new FileNotFoundException($"PTX file not found: {ptxPath}", ptxPath);

        byte[] ptxBytes = File.ReadAllBytes(ptxPath);
        return LoadFromBytes(ptxBytes, maxRegisters);
    }

    /// <summary>Loads PTX from a byte array (must be null-terminated UTF-8 or will be null-terminated automatically).</summary>
    /// <param name="maxRegisters">Per-thread register cap for the JIT (0 = let it choose).</param>
    /// <remarks>The cap exists because the driver's PTX JIT does NOT honour a kernel's own
    /// <c>.minnctapersm</c> directive — a <c>__launch_bounds__(threads, blocksPerSM)</c> that ptxas would respect
    /// when compiling to cubin is silently ignored here, and a register-hungry kernel gets all 256 registers and
    /// therefore one block per SM. <c>CU_JIT_MAX_REGISTERS</c> is the only way to ask for the occupancy the
    /// source already asked for. Applies to every kernel in the module, so use it on single-kernel modules.
    /// <para>No caller sets it today — the one kernel that wanted it (the fused int8 mma GEMM) ended up at a tile
    /// whose 90 KB of shared already pins it to one block per SM. Kept because the finding is not rediscoverable
    /// from the source: a launch-bounds directive that looks honoured simply is not, on this path.</para></remarks>
    public static unsafe CudaModule LoadFromBytes(byte[] ptxBytes, int maxRegisters = 0)
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

                const int CU_JIT_MAX_REGISTERS = 0;

                int* options = stackalloc int[5];
                options[0] = CU_JIT_INFO_LOG_BUFFER;
                options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
                options[2] = CU_JIT_ERROR_LOG_BUFFER;
                options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
                options[4] = CU_JIT_MAX_REGISTERS;

                nint* optionValues = stackalloc nint[5];
                optionValues[0] = infoLog;
                optionValues[1] = (nint)logSize;
                optionValues[2] = errorLog;
                optionValues[3] = (nint)logSize;
                optionValues[4] = (nint)maxRegisters;

                int result = CudaDriverApi.cuModuleLoadDataEx(
                    out nint module, pinned,
                    maxRegisters > 0 ? 5u : 4u, (nint)options, (nint)optionValues);

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
        try
        {
            Release(throwOnError: true);
        }
        finally
        {
            GC.SuppressFinalize(this);
        }
    }

    ~CudaModule()
    {
        try { Release(throwOnError: false); }
        catch { /* Finalizers must never surface native-loader or driver failures. */ }
    }

    /// <summary>Releases an adopted module from an owning finalizer without allowing a native error to escape.</summary>
    internal void DisposeNoThrow()
    {
        try { Release(throwOnError: false); }
        catch { /* An owning finalizer cannot safely report a native teardown failure. */ }
        GC.SuppressFinalize(this);
    }

    private void Release(bool throwOnError)
    {
        nint m = Interlocked.Exchange(ref _module, 0);
        if (m != 0)
        {
            Interlocked.Decrement(ref _liveHandleCount);
            int result = CudaDriverApi.cuModuleUnload(m);
            if (throwOnError) result.ThrowOnError();
        }
    }
}
