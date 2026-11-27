using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Reads the file bytes behind memory-mapped tensors from several threads at once, so the reads that follow (a device upload's
/// staging, or a CPU kernel's page faults) find them in the page cache.</summary>
/// <remarks>For weights on storage that serves one reader far slower than several, such as a network volume: on a RunPod MooseFS volume a
/// 4 MiB read runs at about 95 MB/s from one thread and 600 to 800 MB/s across 16 to 30. Only tensors with a registered file extent
/// (<see cref="TensorFileExtents"/>) are read; others are skipped. Best effort: a read that fails is dropped, and the real read reports
/// the failure. The bytes read are discarded, so on storage that is already cached this costs one memory copy per tensor.
/// <para>The readers take the chunks in list order, so a caller that consumes the tensors in that order can start on each one as soon as
/// it is read (<see cref="Begin"/> and <see cref="Batch.WaitFor"/>) while the readers carry on with the rest.</para>
/// <para>The readers are dedicated background threads, started on first use and kept for the process, each with its own
/// <see cref="ChunkBytes"/> buffer: blocking reads on thread-pool workers would hold the pool near one reader per core, and a call allocates
/// nothing once the readers and the tables have grown to its size. Batches are serialized process-wide, and a read that hangs on stalled
/// storage holds its caller as the upload read would.</para></remarks>
public static class MappedTensorPrefetch
{
    /// <summary>The unit of work one reader takes, so one large tensor still spreads over several readers.</summary>
    internal const int ChunkBytes = 4 << 20;

    /// <summary>Most readers one call uses.</summary>
    public const int MaxReaders = 64;

    private static readonly SemaphoreSlim BatchGate = new(1, 1);
    private static readonly SemaphoreSlim Wake = new(0);
    // Progress and the readers' completion share one monitor. ManualResetEventSlim would be the natural signal, but it creates its kernel event
    // lazily on the first blocking wait, which allocates on whichever caller blocks first; a monitor allocates nothing per wait.
    private static readonly object Progress = new();
    private static Chunk[] _chunks = new Chunk[256];
    private static int[] _tensorChunksLeft = new int[64];
    private static int _chunkCount;
    private static int _tensorCount;
    private static int _nextChunk;
    private static int _busyReaders;
    private static long _bytesRead;
    private static int _readerThreads;
    private static int _generations;     // batches begun, so a stale Batch copy never matches a later one
    private static int _liveGeneration;  // the running batch's generation, 0 when none
    [ThreadStatic] private static bool _threadHoldsBatch;

    /// <summary>Reads every registered tensor of <paramref name="tensors"/> with up to <paramref name="threads"/> readers (at most
    /// <see cref="MaxReaders"/>) and returns when they finish.</summary>
    /// <returns>Bytes read; zero when <paramref name="threads"/> is zero or no tensor has a file extent.</returns>
    public static long Read(IReadOnlyList<Tensor> tensors, int threads)
    {
        using Batch batch = Begin(tensors, threads);
        return batch.Finish();
    }

    /// <summary>Starts reading every registered tensor of <paramref name="tensors"/> with up to <paramref name="threads"/> readers and
    /// returns at once. Wait for tensor <c>i</c> with <see cref="Batch.WaitFor"/>; disposing the batch waits for the rest.</summary>
    /// <remarks>The list must not change until the batch is disposed. Only one batch runs at a time, process-wide, from <see cref="Begin"/>
    /// until its <see cref="Batch.Dispose"/>, including whatever the caller does between waits: another thread's <see cref="Begin"/> waits
    /// for it, so dispose the batch as soon as the last tensor needed has been waited on. Dispose it on the thread that began it.</remarks>
    /// <exception cref="InvalidOperationException">This thread already holds a batch it has not disposed.</exception>
    public static Batch Begin(IReadOnlyList<Tensor> tensors, int threads)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        if (threads == 0 || tensors.Count == 0) return default;
        // The gate is not reentrant, so a nested Begin on this thread would wait on itself forever.
        if (_threadHoldsBatch) throw new InvalidOperationException("This thread already holds a prefetch batch; dispose it before beginning another.");
        BatchGate.Wait();
        bool started = false;
        try
        {
            _chunkCount = 0;
            _tensorCount = tensors.Count;
            if (_tensorChunksLeft.Length < _tensorCount) Array.Resize(ref _tensorChunksLeft, Math.Max(_tensorCount, _tensorChunksLeft.Length * 2));
            for (int t = 0; t < tensors.Count; t++)
            {
                Tensor tensor = tensors[t];
                _tensorChunksLeft[t] = 0;
                if (!TensorFileExtents.TryGet(tensor, out MmapHandle? owner, out long start)) continue;
                long bytes = tensor.DType.ComputeByteCount(tensor.ElementCount);
                for (long at = 0; at < bytes; at += ChunkBytes)
                {
                    AddChunk(new Chunk(owner, start + at, (int)Math.Min(ChunkBytes, bytes - at), t));
                    _tensorChunksLeft[t]++;
                }
            }
            _nextChunk = 0;
            _bytesRead = 0;
            if (_chunkCount > 0)
            {
                int readers = Math.Min(Math.Min(threads, MaxReaders), _chunkCount);
                EnsureReaderThreads(readers);
                Volatile.Write(ref _busyReaders, readers);
                Wake.Release(readers);
            }
            int generation = ++_generations;
            if (generation == 0) generation = ++_generations;
            Volatile.Write(ref _liveGeneration, generation);
            _threadHoldsBatch = true;
            started = true;
            return new Batch(generation);
        }
        finally
        {
            if (!started)
            {
                // Drop the chunks gathered before the failure, as Dispose does, so their mappings are not kept reachable.
                Array.Clear(_chunks, 0, _chunkCount);
                _chunkCount = 0;
                BatchGate.Release();
            }
        }
    }

    /// <summary>A running prefetch; dispose it to wait for its readers and let the next batch start.</summary>
    /// <remarks>Only the first <see cref="Dispose"/> of a batch, through any copy, ends it. A copy kept past that does nothing: its
    /// <see cref="WaitFor"/> returns at once (every read finished before the batch ended) and its <see cref="Dispose"/> cannot touch a later
    /// batch.</remarks>
    public readonly struct Batch : IDisposable
    {
        private readonly int _generation;

        internal Batch(int generation) => _generation = generation;

        private bool IsLive => _generation != 0 && Volatile.Read(ref _liveGeneration) == _generation;

        /// <summary>Returns once tensor <paramref name="index"/> of the list given to <see cref="Begin"/> is read (or its read failed, or it
        /// had no file extent). Returns at once, without checking <paramref name="index"/>, on an empty or ended batch.</summary>
        /// <exception cref="ArgumentOutOfRangeException">The batch is running and <paramref name="index"/> is outside its list.</exception>
        public void WaitFor(int index)
        {
            if (!IsLive) return;
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _tensorCount);
            if (Volatile.Read(ref _tensorChunksLeft[index]) == 0) return;
            lock (Progress)
            {
                while (Volatile.Read(ref _tensorChunksLeft[index]) > 0) Monitor.Wait(Progress);
            }
        }

        /// <summary>Waits for every read and returns the bytes read; zero on an empty or ended batch.</summary>
        public long Finish()
        {
            if (!IsLive) return 0;
            if (_chunkCount > 0) WaitForReaders();
            return Interlocked.Read(ref _bytesRead);
        }

        /// <summary>Waits for every read and ends the batch. Later calls, through this or any copy, do nothing.</summary>
        public void Dispose()
        {
            // Claim the end atomically, so only one Dispose of this batch releases the gate.
            if (_generation == 0 || Interlocked.CompareExchange(ref _liveGeneration, 0, _generation) != _generation) return;
            if (_chunkCount > 0) WaitForReaders();
            // Drop the handles so a finished prefetch does not keep a disposed model's mappings reachable.
            Array.Clear(_chunks, 0, _chunkCount);
            _chunkCount = 0;
            _threadHoldsBatch = false;
            BatchGate.Release();
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
                if (Interlocked.Decrement(ref _tensorChunksLeft[chunk.Tensor]) == 0)
                {
                    lock (Progress) Monitor.PulseAll(Progress);
                }
            }
            if (Interlocked.Decrement(ref _busyReaders) == 0)
            {
                lock (Progress) Monitor.PulseAll(Progress);
            }
        }
    }

    // Blocks until every reader of the running batch has finished its chunks. The count is checked under the monitor the last reader pulses,
    // so the wake-up cannot be missed.
    private static void WaitForReaders()
    {
        lock (Progress)
        {
            while (Volatile.Read(ref _busyReaders) > 0) Monitor.Wait(Progress);
        }
    }

    private readonly record struct Chunk(MmapHandle Owner, long Offset, int Length, int Tensor);
}
