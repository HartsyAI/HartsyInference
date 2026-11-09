using HartsyInference.Core.Backends;
using HartsyInference.Core.Distributed;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>One-rank-per-process NCCL communicator for multi-node jobs: this process owns one local device and its rank's half of every collective. Created by <see cref="JoinAsync"/>, which exchanges the NCCL unique ID through <see cref="UniqueIdRendezvous"/>.</summary>
/// <remarks>Unlike <see cref="NcclComm"/> (every rank in one process), each call takes only this rank's tensors and must be made by every process in the same order. Collectives run on the owning backend's thread. There is no host-staged fallback: a job whose NCCL cannot load fails at join, because a single-process substitute would silently compute a partial result.</remarks>
// TODO(untested): never executed. NcclRankCommTests skips without CUDA, and this code has not run on Linux with libnccl.
// TODO(unsure): ncclCommInitRank relies on BindAmbient making the device current for NCCL; not confirmed on a real device.
// TODO(unsure): ncclCommInitRank blocks until every rank calls it; a rank that dies after rendezvous hangs the others. No timeout or abort yet.
// TODO(unsure): collectives are not checked with ncclCommGetAsyncError; a failed rank surfaces only as a hang.
// TODO(unsure): the finalizer calls ncclCommDestroy on the GC thread; safety with in-flight work is unverified.
// TODO(unsure): FreeDevice(src) runs before Sync(); mirrors NcclComm ordering but is not proven safe for a cudaMalloc-backed src.
// TODO(unsure): nothing enforces that all processes issue collectives in the same order; a mismatch deadlocks.
public sealed class NcclRankComm : IRankCollective
{
    private const int UniqueIdBytes = 128;
    private readonly CudaBackend _backend;
    private nint _comm;
    private int _disposed;

    /// <summary>True when libnccl resolves on this host. Multi-node joins require it.</summary>
    public static bool IsAvailable => NcclApi.IsLoadable;

    /// <inheritdoc/>
    public int Rank { get; }

    /// <inheritdoc/>
    public int WorldSize { get; }

    /// <inheritdoc/>
    public string Transport => "nccl";

    private NcclRankComm(CudaBackend backend, int rank, int worldSize, byte[] uniqueId)
    {
        _backend = backend;
        Rank = rank;
        WorldSize = worldSize;
        _backend.BindAmbient();
        NcclApi.NcclUniqueId id = System.Runtime.InteropServices.MemoryMarshal.Read<NcclApi.NcclUniqueId>(uniqueId);
        NcclApi.Check(NcclApi.ncclCommInitRank(out nint comm, worldSize, id, rank), "CommInitRank");
        _comm = comm;
        Logs.Info($"[Collective] transport=nccl rank={rank} world={worldSize} device={backend.Device.Ordinal}");
    }

    /// <summary>Joins the job as <paramref name="options"/>.<see cref="DistributedRendezvousOptions.Rank"/> on <paramref name="backend"/>'s device. Every rank must call this concurrently.</summary>
    public static async Task<NcclRankComm> JoinAsync(CudaBackend backend, DistributedRendezvousOptions options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        RequireNccl();
        byte[] uniqueId = await UniqueIdRendezvous.ExchangeAsync(options, NewUniqueId, cancellationToken).ConfigureAwait(false);
        return new NcclRankComm(backend, options.Rank, options.WorldSize, uniqueId);
    }

    /// <summary>Generates a NCCL unique ID (rank 0 only) as the 128-byte blob the rendezvous distributes.</summary>
    public static byte[] NewUniqueId()
    {
        RequireNccl();
        NcclApi.Check(NcclApi.ncclGetUniqueId(out NcclApi.NcclUniqueId uniqueId), "GetUniqueId");
        byte[] bytes = new byte[UniqueIdBytes];
        System.Runtime.InteropServices.MemoryMarshal.Write(bytes, in uniqueId);
        return bytes;
    }

    /// <inheritdoc/>
    public void AllReduceSum(Tensor buffer)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(buffer);
        NcclApi.DataType dtype = MapDtype(buffer.DType);
        long count = buffer.ElementCount;
        nuint bytes = (nuint)(count * buffer.DType.SizeInBytes);
        RunCollective(buffer, buffer, bytes, (src, dst) => NcclApi.ncclAllReduce(src, dst, (nuint)count, dtype, NcclApi.RedOp.Sum, _comm, _backend.Stream.Handle), "AllReduce");
    }

    /// <inheritdoc/>
    public void AllGather(Tensor sendBuffer, Tensor recvBuffer)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(sendBuffer);
        ArgumentNullException.ThrowIfNull(recvBuffer);
        long count = sendBuffer.ElementCount;
        if (recvBuffer.ElementCount != count * WorldSize)
        {
            throw new ArgumentException($"AllGather receive buffer must hold sendCount×worldSize elements ({count}×{WorldSize}), got {recvBuffer.ElementCount}.", nameof(recvBuffer));
        }
        if (recvBuffer.DType != sendBuffer.DType)
        {
            throw new ArgumentException($"AllGather buffers must share a dtype (send {sendBuffer.DType}, receive {recvBuffer.DType}).", nameof(recvBuffer));
        }
        NcclApi.DataType dtype = MapDtype(sendBuffer.DType);
        nuint recvBytes = (nuint)(recvBuffer.ElementCount * recvBuffer.DType.SizeInBytes);
        RunCollective(sendBuffer, recvBuffer, recvBytes, (src, dst) => NcclApi.ncclAllGather(src, dst, (nuint)count, dtype, _comm, _backend.Stream.Handle), "AllGather");
    }

    /// <summary>Uploads <paramref name="input"/>, enqueues one collective into a fresh device buffer, and hands that buffer to <paramref name="output"/>. Any failure before the hand-off frees both temporaries.</summary>
    private void RunCollective(Tensor input, Tensor output, nuint outputBytes, Func<ulong, ulong, int> enqueue, string operation)
    {
        _backend.BindAmbient();
        ulong src = GpuTransferHelper.CopyToDevice(input);
        ulong dst = 0;
        bool transferred = false;
        try
        {
            dst = GpuTransferHelper.AllocateDevice(outputBytes);
            NcclApi.Check(enqueue(src, dst), operation);
            _backend.BindAmbient();
            GpuTransferHelper.FreeDevice(src);
            src = 0;
            // TODO(unsure): if CacheActivation throws after registering dst, the cleanup below frees it a second time.
            GpuTransferHelper.CacheActivation(output, dst, outputBytes);
            transferred = true;
            _backend.Sync();
        }
        finally
        {
            if (!transferred)
            {
                ReleaseAfterFailure(src, dst);
            }
        }
    }

    private void ReleaseAfterFailure(ulong src, ulong dst)
    {
        _backend.BindAmbient();
        if (src != 0)
        {
            GpuTransferHelper.FreeDevice(src);
        }
        if (dst != 0)
        {
            GpuTransferHelper.FreeDevice(dst);
        }
    }

    private static void RequireNccl()
    {
        if (!NcclApi.IsLoadable)
        {
            throw new InvalidOperationException("Multi-node collectives require libnccl, which did not resolve on this host (set paths.ncclDir or install libnccl.so.2). Refusing to run a partial job.");
        }
    }

    private static NcclApi.DataType MapDtype(DType dtype)
    {
        if (dtype == DType.F32) return NcclApi.DataType.Float32;
        if (dtype == DType.F16) return NcclApi.DataType.Float16;
        if (dtype == DType.BF16) return NcclApi.DataType.Bfloat16;
        throw new ArgumentException($"Collectives over dtype {dtype} are not supported (F32/F16/BF16 only).");
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed != 0, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        Release();
        GC.SuppressFinalize(this);
    }

    ~NcclRankComm() => Release();

    private void Release()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        nint comm = Interlocked.Exchange(ref _comm, 0);
        if (comm != 0)
        {
            _ = NcclApi.ncclCommDestroy(comm);
        }
    }
}
