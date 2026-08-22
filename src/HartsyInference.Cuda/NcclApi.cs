using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>P/Invoke bindings for NCCL (NVIDIA Collective Communications Library, <c>libnccl.so.2</c>). Library-policy exception in the same category as cuBLAS/cuDNN (BSD-3, runtime-resolved, no build-time reference) — the decision and the exact API list live in <c>docs/Research/MULTI_GPU_PARALLELISM.md</c> §4.3. Resolution: <see cref="CudaLibraryResolver"/> probes <c>HARTSY_NCCL_DIR</c> plus the standard CUDA lib dirs, so a torch-venv-bundled copy works without a system install. RCCL (AMD) keeps these symbol names — a library swap targets ROCm without code changes. <para>Streams: NCCL takes <c>cudaStream_t</c>, which is handle-compatible with the driver-API <c>CUstream</c> this codebase uses (both are the same opaque handle under the device's primary context — and <see cref="CudaContext"/> deliberately uses primary contexts, the configuration NCCL expects in a single-process setup).</para></summary>
internal static partial class NcclApi
{
    private const string LibName = "nccl";

    internal const int Success = 0;

    /// <summary>Opaque 128-byte rendezvous blob (<c>ncclUniqueId</c>) — only needed for the multi-process <see cref="ncclCommInitRank"/> path; single-process uses <see cref="ncclCommInitAll"/>.</summary>
    [StructLayout(LayoutKind.Sequential, Size = 128)]
    internal struct NcclUniqueId
    {
    }

    /// <summary>ncclDataType_t (nccl.h values).</summary>
    internal enum DataType
    {
        Int8 = 0,
        Uint8 = 1,
        Int32 = 2,
        Uint32 = 3,
        Int64 = 4,
        Uint64 = 5,
        Float16 = 6,
        Float32 = 7,
        Float64 = 8,
        Bfloat16 = 9,
    }

    /// <summary>ncclRedOp_t (nccl.h values).</summary>
    internal enum RedOp
    {
        Sum = 0,
        Prod = 1,
        Max = 2,
        Min = 3,
        Avg = 4,
    }

    [LibraryImport(LibName)]
    internal static partial int ncclGetVersion(out int version);

    [LibraryImport(LibName)]
    internal static partial int ncclGetUniqueId(out NcclUniqueId uniqueId);

    /// <summary>Single-process init: one communicator per entry of <paramref name="devlist"/> (CUDA ordinals), all created by ONE calling thread. NCCL sets each device current internally via the runtime API, which shares the primary contexts <see cref="CudaContext"/> retains.</summary>
    [LibraryImport(LibName)]
    internal static unsafe partial int ncclCommInitAll(nint* comms, int ndev, int* devlist);

    [LibraryImport(LibName)]
    internal static partial int ncclCommInitRank(out nint comm, int nranks, NcclUniqueId commId, int rank);

    [LibraryImport(LibName)]
    internal static partial int ncclCommDestroy(nint comm);

    [LibraryImport(LibName)]
    internal static partial int ncclCommAbort(nint comm);

    [LibraryImport(LibName)]
    internal static partial nint ncclGetErrorString(int result);

    [LibraryImport(LibName)]
    internal static partial int ncclAllReduce(ulong sendbuff, ulong recvbuff, nuint count, DataType datatype, RedOp op, nint comm, nint stream);

    [LibraryImport(LibName)]
    internal static partial int ncclBroadcast(ulong sendbuff, ulong recvbuff, nuint count, DataType datatype, int root, nint comm, nint stream);

    /// <summary>Gathers each rank's <paramref name="sendbuff"/> (count elements) into every rank's <paramref name="recvbuff"/> (count × nranks elements, rank r's block at offset r·count).</summary>
    [LibraryImport(LibName)]
    internal static partial int ncclAllGather(ulong sendbuff, ulong recvbuff, nuint sendcount, DataType datatype, nint comm, nint stream);

    [LibraryImport(LibName)]
    internal static partial int ncclReduceScatter(ulong sendbuff, ulong recvbuff, nuint recvcount, DataType datatype, RedOp op, nint comm, nint stream);

    /// <summary>Point-to-point. There is NO public ncclAllToAll — grouped Send/Recv inside <see cref="ncclGroupStart"/>/<see cref="ncclGroupEnd"/> IS the all-to-all.</summary>
    [LibraryImport(LibName)]
    internal static partial int ncclSend(ulong sendbuff, nuint count, DataType datatype, int peer, nint comm, nint stream);

    [LibraryImport(LibName)]
    internal static partial int ncclRecv(ulong recvbuff, nuint count, DataType datatype, int peer, nint comm, nint stream);

    [LibraryImport(LibName)]
    internal static partial int ncclGroupStart();

    [LibraryImport(LibName)]
    internal static partial int ncclGroupEnd();

    private static int _loadable = -1;

    /// <summary>Whether libnccl can be loaded on this box (memoized). False means the collective factory falls back to the peer-copy transport — never an exception.</summary>
    internal static bool IsLoadable
    {
        get
        {
            if (_loadable < 0)
            {
                try
                {
                    CudaLibraryResolver.Register();
                    _loadable = ncclGetVersion(out _) == Success ? 1 : 0;
                }
                catch (DllNotFoundException)
                {
                    _loadable = 0;
                }
            }
            return _loadable == 1;
        }
    }

    /// <summary>Loaded NCCL version (e.g. 22105 → 2.21.5), or 0 when unavailable.</summary>
    internal static int Version
    {
        get
        {
            if (!IsLoadable)
            {
                return 0;
            }
            return ncclGetVersion(out int v) == Success ? v : 0;
        }
    }

    internal static string ErrorString(int result) =>
        Marshal.PtrToStringAnsi(ncclGetErrorString(result)) ?? $"nccl error {result}";

    internal static void Check(int result, string operation)
    {
        if (result != Success)
        {
            throw new CudaException(result, $"NCCL {operation} failed: {ErrorString(result)}");
        }
    }
}
