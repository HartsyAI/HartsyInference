using System.Buffers;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Reads the file bytes behind memory-mapped tensors from several threads at once, so the reads that follow (a device upload's
/// staging, or a CPU kernel's page faults) find them in the page cache.</summary>
/// <remarks>For weights on storage that serves one reader far slower than several, such as a network volume: on a RunPod MooseFS volume a
/// 4 MiB read runs at about 95 MB/s from one thread and 600 to 800 MB/s across 16 to 30. Only tensors with a registered file extent
/// (<see cref="TensorFileExtents"/>) are read; others are skipped. Best effort: a read that fails is dropped, and the real read reports
/// the failure. The bytes read are discarded, so on storage that is already cached this costs one memory copy per tensor.</remarks>
public static class MappedTensorPrefetch
{
    /// <summary>The unit of work one reader takes, so one large tensor still spreads over several readers.</summary>
    internal const int ChunkBytes = 4 << 20;

    private static readonly ArrayPool<byte> Buffers = ArrayPool<byte>.Create(ChunkBytes, 64);

    /// <summary>Reads every registered tensor of <paramref name="tensors"/> with up to <paramref name="threads"/> readers and returns when
    /// they finish.</summary>
    /// <returns>Bytes read; zero when <paramref name="threads"/> is zero or no tensor has a file extent.</returns>
    public static long Read(IReadOnlyList<Tensor> tensors, int threads)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        if (threads == 0 || tensors.Count == 0) return 0;
        List<Chunk> chunks = [];
        foreach (Tensor tensor in tensors)
        {
            if (!TensorFileExtents.TryGet(tensor, out MmapHandle? owner, out long start)) continue;
            long bytes = tensor.DType.ComputeByteCount(tensor.ElementCount);
            for (long at = 0; at < bytes; at += ChunkBytes)
                chunks.Add(new Chunk(owner, start + at, (int)Math.Min(ChunkBytes, bytes - at)));
        }
        if (chunks.Count == 0) return 0;

        long read = 0;
        ParallelOptions options = new() { MaxDegreeOfParallelism = Math.Min(threads, chunks.Count) };
        Parallel.For(0, chunks.Count, options,
            () => Buffers.Rent(ChunkBytes),
            (i, _, buffer) =>
            {
                Chunk chunk = chunks[i];
                try
                {
                    chunk.Owner.ReadAt(chunk.Offset, buffer.AsSpan(0, chunk.Length));
                    Interlocked.Add(ref read, chunk.Length);
                }
                catch (Exception ex) when (ex is IOException or ObjectDisposedException or UnauthorizedAccessException
                    or ArgumentOutOfRangeException or HartsyInferenceException)
                {
                    // A hint only: the upload or kernel that needs these bytes reads them itself and reports the failure.
                }
                return buffer;
            },
            buffer => Buffers.Return(buffer));
        return read;
    }

    private readonly record struct Chunk(MmapHandle Owner, long Offset, int Length);
}
