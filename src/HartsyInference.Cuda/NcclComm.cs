using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>NCCL-backed <see cref="ICollectiveComm"/>: one communicator per rank via <c>ncclCommInitAll</c> (single-process, one driving thread), collectives issued inside a NCCL group on each rank backend's compute stream, then stream-synced. Requires one DISTINCT CUDA device per rank — NCCL auto-selects the transport per pair (NVLink → PCIe P2P → SHM), so this works on the no-P2P consumer box too. Construct via <see cref="CollectiveComm.Create"/>, which falls back to <see cref="HostStagedComm"/> when NCCL is unavailable.</summary>
public sealed unsafe class NcclComm : ICollectiveComm
{
    private readonly CudaBackend[] _backends;
    private readonly nint[] _comms;
    private bool _disposed;

    internal NcclComm(IReadOnlyList<CudaBackend> backends)
    {
        if (backends.Count < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(backends), backends.Count, "A collective needs at least 2 ranks.");
        }
        _backends = [.. backends];
        int n = _backends.Length;
        int* devs = stackalloc int[n];
        for (int i = 0; i < n; i++)
        {
            devs[i] = _backends[i].Device.Ordinal;
            for (int j = 0; j < i; j++)
            {
                if (devs[j] == devs[i])
                {
                    throw new ArgumentException($"NCCL requires one distinct CUDA device per rank — ordinal {devs[i]} appears twice.", nameof(backends));
                }
            }
        }
        nint[] comms = new nint[n];
        fixed (nint* pComms = comms)
        {
            NcclApi.Check(NcclApi.ncclCommInitAll(pComms, n, devs), "CommInitAll");
        }
        _comms = comms;
    }

    /// <inheritdoc/>
    public int RankCount => _backends.Length;

    /// <inheritdoc/>
    public string Transport => "nccl";

    /// <inheritdoc/>
    public void AllReduceSum(IReadOnlyList<Tensor> buffers)
    {
        ThrowIfDisposed();
        long count = ValidateUniform(buffers, RankCount, nameof(buffers));
        NcclApi.DataType dtype = MapDtype(buffers[0].DType);
        nuint bytes = (nuint)(count * buffers[0].DType.SizeInBytes);
        // Op-shaped like LinearImpl: source = CopyToDevice (transient OR a cache hit — FreeDevice below
        // skips cached pointers), result = a FRESH buffer registered via CacheActivation, so the tensor
        // becomes device-authoritative with the normal D2H-on-host-read binding. Reducing IN PLACE into the
        // source pointer would mutate a possibly-shared cached buffer and leave the host copy silently stale
        // (CopyToDevice's transient path carries no writeback binding).
        ulong[] src = new ulong[RankCount];
        ulong[] dst = new ulong[RankCount];
        for (int i = 0; i < RankCount; i++)
        {
            _backends[i].BindAmbient();
            src[i] = GpuTransferHelper.CopyToDevice(buffers[i]);
            dst[i] = GpuTransferHelper.AllocateDevice(bytes);
        }
        NcclApi.Check(NcclApi.ncclGroupStart(), "GroupStart");
        for (int i = 0; i < RankCount; i++)
        {
            NcclApi.Check(NcclApi.ncclAllReduce(src[i], dst[i], (nuint)count, dtype, NcclApi.RedOp.Sum, _comms[i], _backends[i].Stream.Handle), "AllReduce");
        }
        NcclApi.Check(NcclApi.ncclGroupEnd(), "GroupEnd");
        for (int i = 0; i < RankCount; i++)
        {
            _backends[i].BindAmbient();
            GpuTransferHelper.FreeDevice(src[i]);
            GpuTransferHelper.CacheActivation(buffers[i], dst[i], bytes);
        }
        SyncAll();
    }

    /// <inheritdoc/>
    public void AllGather(IReadOnlyList<Tensor> sendBuffers, IReadOnlyList<Tensor> recvBuffers)
    {
        ThrowIfDisposed();
        long count = ValidateUniform(sendBuffers, RankCount, nameof(sendBuffers));
        long recvCount = ValidateUniform(recvBuffers, RankCount, nameof(recvBuffers));
        if (recvCount != count * RankCount)
        {
            throw new ArgumentException($"AllGather recv buffers must hold sendCount×ranks elements ({count}×{RankCount}), got {recvCount}.", nameof(recvBuffers));
        }
        NcclApi.DataType dtype = MapDtype(sendBuffers[0].DType);
        nuint recvBytes = (nuint)(recvCount * recvBuffers[0].DType.SizeInBytes);
        ulong[] sendPtrs = new ulong[RankCount];
        ulong[] recvPtrs = new ulong[RankCount];
        for (int i = 0; i < RankCount; i++)
        {
            _backends[i].BindAmbient();
            sendPtrs[i] = GpuTransferHelper.CopyToDevice(sendBuffers[i]);
            // Recv is a pure OUTPUT: allocate fresh, never upload its (meaningless) host contents.
            recvPtrs[i] = GpuTransferHelper.AllocateDevice(recvBytes);
        }
        NcclApi.Check(NcclApi.ncclGroupStart(), "GroupStart");
        for (int i = 0; i < RankCount; i++)
        {
            NcclApi.Check(NcclApi.ncclAllGather(sendPtrs[i], recvPtrs[i], (nuint)count, dtype, _comms[i], _backends[i].Stream.Handle), "AllGather");
        }
        NcclApi.Check(NcclApi.ncclGroupEnd(), "GroupEnd");
        for (int i = 0; i < RankCount; i++)
        {
            _backends[i].BindAmbient();
            GpuTransferHelper.FreeDevice(sendPtrs[i]);
            GpuTransferHelper.CacheActivation(recvBuffers[i], recvPtrs[i], recvBytes);
        }
        SyncAll();
    }

    private void SyncAll()
    {
        foreach (CudaBackend backend in _backends)
        {
            backend.Sync();
        }
    }

    private static long ValidateUniform(IReadOnlyList<Tensor> tensors, int ranks, string name)
    {
        if (tensors.Count != ranks)
        {
            throw new ArgumentException($"Expected one tensor per rank ({ranks}), got {tensors.Count}.", name);
        }
        long count = tensors[0].ElementCount;
        DType dtype = tensors[0].DType;
        foreach (Tensor t in tensors)
        {
            if (t.ElementCount != count || t.DType != dtype)
            {
                throw new ArgumentException($"All ranks' tensors must share element count and dtype (expected {count}×{dtype}, got {t.ElementCount}×{t.DType}).", name);
            }
        }
        return count;
    }

    private static NcclApi.DataType MapDtype(DType dtype)
    {
        if (dtype == DType.F32) return NcclApi.DataType.Float32;
        if (dtype == DType.F16) return NcclApi.DataType.Float16;
        if (dtype == DType.BF16) return NcclApi.DataType.Bfloat16;
        throw new ArgumentException($"Collectives over dtype {dtype} are not supported (F32/F16/BF16 only).");
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach (nint comm in _comms)
        {
            if (comm != 0)
            {
                _ = NcclApi.ncclCommDestroy(comm);
            }
        }
    }
}
