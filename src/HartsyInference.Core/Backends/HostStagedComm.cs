using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Universal-fallback <see cref="ICollectiveComm"/>: stages every rank's data through host memory and does the reduction on the CPU. Correct on any backend mix (including no-P2P boxes and CPU backends) and used whenever NCCL is unavailable — slower by design, never wrong. <para>Reading <see cref="Tensor.DataPointer"/> demotes any device copy (the binding fires the D2H sync and unbinds), so the host mutation below is authoritative and the next device consumer re-uploads — the same engine-wide pattern host-side scheduler steps rely on.</para></summary>
public sealed unsafe class HostStagedComm : ICollectiveComm
{
    private readonly int _ranks;

    /// <summary>Creates a host-staged communicator over <paramref name="rankCount"/> ranks (≥ 2).</summary>
    public HostStagedComm(int rankCount)
    {
        if (rankCount < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(rankCount), rankCount, "A collective needs at least 2 ranks.");
        }
        _ranks = rankCount;
    }

    /// <inheritdoc/>
    public int RankCount => _ranks;

    /// <inheritdoc/>
    public string Transport => "host-staged";

    /// <inheritdoc/>
    public void AllReduceSum(IReadOnlyList<Tensor> buffers)
    {
        ValidateRanks(buffers.Count, nameof(buffers));
        long count = ValidateUniform(buffers, nameof(buffers));
        if (buffers[0].DType != DType.F32)
        {
            throw new ArgumentException($"Host-staged AllReduceSum is F32-only (got {buffers[0].DType}) — use the NCCL transport for F16/BF16 reductions.", nameof(buffers));
        }
        float*[] hosts = new float*[_ranks];
        for (int i = 0; i < _ranks; i++)
        {
            hosts[i] = (float*)buffers[i].DataPointer;
        }
        for (long e = 0; e < count; e++)
        {
            float sum = 0f;
            for (int i = 0; i < _ranks; i++)
            {
                sum += hosts[i][e];
            }
            for (int i = 0; i < _ranks; i++)
            {
                hosts[i][e] = sum;
            }
        }
    }

    /// <inheritdoc/>
    public void AllGather(IReadOnlyList<Tensor> sendBuffers, IReadOnlyList<Tensor> recvBuffers)
    {
        ValidateRanks(sendBuffers.Count, nameof(sendBuffers));
        ValidateRanks(recvBuffers.Count, nameof(recvBuffers));
        long count = ValidateUniform(sendBuffers, nameof(sendBuffers));
        long recvCount = ValidateUniform(recvBuffers, nameof(recvBuffers));
        if (recvCount != count * _ranks)
        {
            throw new ArgumentException($"AllGather recv buffers must hold sendCount×ranks elements ({count}×{_ranks}), got {recvCount}.", nameof(recvBuffers));
        }
        nuint blockBytes = (nuint)(count * sendBuffers[0].DType.SizeInBytes);
        for (int src = 0; src < _ranks; src++)
        {
            byte* srcHost = (byte*)sendBuffers[src].DataPointer;
            for (int dst = 0; dst < _ranks; dst++)
            {
                Buffer.MemoryCopy(srcHost, (byte*)recvBuffers[dst].DataPointer + (nuint)src * blockBytes, blockBytes, blockBytes);
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
    }

    private void ValidateRanks(int actual, string name)
    {
        if (actual != _ranks)
        {
            throw new ArgumentException($"Expected one tensor per rank ({_ranks}), got {actual}.", name);
        }
    }

    private static long ValidateUniform(IReadOnlyList<Tensor> tensors, string name)
    {
        long count = tensors[0].ElementCount;
        DType dtype = tensors[0].DType;
        // Host reduction math is F32-only; AllGather is a raw copy and takes any FIXED-WIDTH dtype
        // (block-quantized dtypes have no per-element byte size and are rejected by the size check below).
        foreach (Tensor t in tensors)
        {
            if (t.ElementCount != count || t.DType != dtype)
            {
                throw new ArgumentException($"All ranks' tensors must share element count and dtype (expected {count}×{dtype}, got {t.ElementCount}×{t.DType}).", name);
            }
        }
        if (dtype != DType.F32 && dtype.SizeInBytes == 0)
        {
            throw new ArgumentException($"Collectives over block-quantized dtype {dtype} are not supported.", name);
        }
        return count;
    }
}
