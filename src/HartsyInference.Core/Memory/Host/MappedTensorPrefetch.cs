using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Reads the file bytes behind memory-mapped tensors from several threads at once, so the reads that follow (a device upload's
/// staging, or a CPU kernel's page faults) find them in the page cache.</summary>
/// <remarks>For weights on storage that serves one reader far slower than several, such as a network volume: on a RunPod MooseFS volume a
/// 4 MiB read runs at about 95 MB/s from one thread and 600 to 800 MB/s across 16 to 30. Only tensors with a registered file extent
/// (<see cref="TensorFileExtents"/>) are read; others are skipped. Best effort: a read that fails is dropped, and the real read reports
/// the failure. The bytes read are discarded, so on storage that is already cached this costs one memory copy per tensor.
/// <para>The readers are dedicated background threads, started on first use and kept for the process, each with its own
/// <see cref="ChunkBytes"/> buffer: blocking reads on thread-pool workers would hold the pool near one reader per core, and a call allocates
/// nothing once the readers and the chunk table have grown to its size. Calls are serialized.</para></remarks>
public static class MappedTensorPrefetch
{
    /// <summary>The unit of work one reader takes, so one large tensor still spreads over several readers.</summary>
    internal const int ChunkBytes = 4 << 20;

    /// <summary>Most readers one call uses.</summary>
    public const int MaxReaders = 64;

    private static readonly object CallGate = new();
    private static readonly SemaphoreSlim Wake = new(0);
    private static readonly ManualResetEventSlim Finished = new(false);
    private static Chunk[] _chunks = new Chunk[256];
    private static int _chunkCount;
    private static int _nextChunk;
    private static int _busyReaders;
    private static long _bytesRead;
    private static int _readerThreads;

    /// <summary>Reads every registered tensor of <paramref name="tensors"/> with up to <paramref name="threads"/> readers (at most
    /// <see cref="MaxReaders"/>) and returns when they finish.</summary>
    /// <returns>Bytes read; zero when <paramref name="threads"/> is zero or no tensor has a file extent.</returns>
    public static long Read(IReadOnlyList<Tensor> tensors, int threads)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        if (threads == 0 || tensors.Count == 0) return 0;
        lock (CallGate)
        {
            _chunkCount = 0;
            for (int t = 0; t < tensors.Count; t++)
            {
                Tensor tensor = tensors[t];
                if (!TensorFileExtents.TryGet(tensor, out MmapHandle? owner, out long start)) continue;
                long bytes = tensor.DType.ComputeByteCount(tensor.ElementCount);
                for (long at = 0; at < bytes; at += ChunkBytes)
                    AddChunk(new Chunk(owner, start + at, (int)Math.Min(ChunkBytes, bytes - at)));
            }
            if (_chunkCount == 0) return 0;

            int readers = Math.Min(Math.Min(threads, MaxReaders), _chunkCount);
            EnsureReaderThreads(readers);
            _nextChunk = 0;
            _bytesRead = 0;
            Volatile.Write(ref _busyReaders, readers);
            Finished.Reset();
            Wake.Release(readers);
            Finished.Wait();
            // Drop the handles so a finished prefetch does not keep a disposed model's mappings reachable.
            Array.Clear(_chunks, 0, _chunkCount);
            return Interlocked.Read(ref _bytesRead);
        }
    }

    private static void AddChunk(Chunk chunk)
    {
        if (_chunkCount == _chunks.Length) Array.Resize(ref _chunks, _chunks.Length * 2);
        _chunks[_chunkCount++] = chunk;
    }

    // Starts reader threads until there are at least count; they live for the process and sleep between calls.
    private static void EnsureReaderThreads(int count)
    {
        while (_readerThreads < count)
        {
            Thread reader = new(ReaderLoop) { IsBackground = true, Name = $"mapped-prefetch-{_readerThreads}" };
            reader.Start();
            _readerThreads++;
        }
    }

    private static void ReaderLoop()
    {
        byte[] buffer = GC.AllocateUninitializedArray<byte>(ChunkBytes);
        while (true)
        {
            Wake.Wait();
            int i;
            while ((i = Interlocked.Increment(ref _nextChunk) - 1) < _chunkCount)
            {
                Chunk chunk = _chunks[i];
                try
                {
                    chunk.Owner.ReadAt(chunk.Offset, buffer.AsSpan(0, chunk.Length));
                    Interlocked.Add(ref _bytesRead, chunk.Length);
                }
#pragma warning disable CA1031 // A hint only: the upload or kernel that needs these bytes reads them itself and reports any failure, and an
                               // exception escaping a background thread would end the process.
                catch (Exception)
#pragma warning restore CA1031
                {
                }
            }
            if (Interlocked.Decrement(ref _busyReaders) == 0) Finished.Set();
        }
    }

    private readonly record struct Chunk(MmapHandle Owner, long Offset, int Length);
}
