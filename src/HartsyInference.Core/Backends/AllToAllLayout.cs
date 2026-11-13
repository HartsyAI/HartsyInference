using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Validation and offsets for <see cref="ICollectiveComm.AllToAll"/>, shared by every transport so they agree on the layout.
/// Send buffer <c>i</c> is laid out destination-major: its chunk for rank j starts at <see cref="SendOffset"/>. Receive buffer <c>j</c> is
/// laid out source-major: its chunk from rank i starts at <see cref="RecvOffset"/>. A send buffer must not be the receive buffer it
/// exchanges into: the transports zero each receive buffer before copying, so an overlapping send buffer would be wiped.</summary>
public static class AllToAllLayout
{
    /// <summary>Checks the rank count, the counts matrix and every buffer against it. Each buffer needs at least one element, so an
    /// idle rank passes a one-element placeholder.</summary>
    /// <exception cref="ArgumentException">A count is negative, a buffer is too small, or the dtypes differ.</exception>
    public static void Validate(int ranks, IReadOnlyList<Tensor> sendBuffers, IReadOnlyList<Tensor> recvBuffers, long[,] counts)
    {
        ArgumentNullException.ThrowIfNull(sendBuffers);
        ArgumentNullException.ThrowIfNull(recvBuffers);
        ArgumentNullException.ThrowIfNull(counts);
        if (counts.GetLength(0) != ranks || counts.GetLength(1) != ranks)
        {
            throw new ArgumentException($"The all-to-all counts must be {ranks}x{ranks}, got {counts.GetLength(0)}x{counts.GetLength(1)}.", nameof(counts));
        }
        if (sendBuffers.Count != ranks || recvBuffers.Count != ranks)
        {
            throw new ArgumentException($"Expected one send and one receive buffer per rank ({ranks}), got {sendBuffers.Count} and {recvBuffers.Count}.", nameof(sendBuffers));
        }
        DType dtype = sendBuffers[0].DType;
        if (dtype.SizeInBytes == 0)
        {
            throw new ArgumentException($"All-to-all over block-quantized dtype {dtype} is not supported.", nameof(sendBuffers));
        }
        for (int i = 0; i < ranks; i++)
        {
            CheckBuffer(sendBuffers[i], dtype, nameof(sendBuffers));
            CheckBuffer(recvBuffers[i], dtype, nameof(recvBuffers));
        }
        for (int i = 0; i < ranks; i++)
        {
            long sent = 0, received = 0;
            for (int j = 0; j < ranks; j++)
            {
                if (counts[i, j] < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(counts), "All-to-all counts must not be negative.");
                }
                sent += counts[i, j];
                received += counts[j, i];
            }
            if (sent > sendBuffers[i].ElementCount)
            {
                throw new ArgumentException($"Rank {i} sends {sent} elements but its send buffer holds {sendBuffers[i].ElementCount}.", nameof(sendBuffers));
            }
            if (received > recvBuffers[i].ElementCount)
            {
                throw new ArgumentException($"Rank {i} receives {received} elements but its receive buffer holds {recvBuffers[i].ElementCount}.", nameof(recvBuffers));
            }
        }
    }

    /// <summary>Element offset of the chunk rank <paramref name="src"/> sends to <paramref name="dst"/> inside its send buffer.</summary>
    public static long SendOffset(long[,] counts, int src, int dst)
    {
        long offset = 0;
        for (int j = 0; j < dst; j++)
        {
            offset += counts[src, j];
        }
        return offset;
    }

    /// <summary>Element offset of the chunk rank <paramref name="src"/> delivers to <paramref name="dst"/> inside its receive buffer.</summary>
    public static long RecvOffset(long[,] counts, int src, int dst)
    {
        long offset = 0;
        for (int i = 0; i < src; i++)
        {
            offset += counts[i, dst];
        }
        return offset;
    }

    private static void CheckBuffer(Tensor buffer, DType dtype, string name)
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        if (buffer.DType != dtype)
        {
            throw new ArgumentException($"All-to-all buffers must share one dtype (expected {dtype}, got {buffer.DType}).", name);
        }
        if (buffer.ElementCount < 1)
        {
            throw new ArgumentException("All-to-all buffers must hold at least one element.", name);
        }
    }
}
