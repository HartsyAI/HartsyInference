using System.Buffers;
using HartsyInference.Core.IO;

namespace HartsyInference.Core.Engram;

/// <summary>
/// Reads rows of a table far larger than host memory (hundreds of millions of rows) from disk by position, keeps the hot ones in
/// a budget-bounded LRU row cache, and hands them out decoded to bf16.
/// </summary>
/// <remarks>
/// <para>A batch of rows is de-duplicated and sorted, then each byte plane's reads are merged: rows that share a 4 KB page become one
/// read, so a batch costs far fewer syscalls than rows. Reads go through <see cref="IWeightByteSource"/>, which is a pread on a file opened
/// with POSIX_FADV_RANDOM on Linux, so the kernel does not read ahead across the table. Nothing is memory-mapped.</para>
/// <para>Rows outside the owned range throw. The store never disposes the byte sources it was given.</para>
/// </remarks>
public sealed class EngramTableStore : IDisposable
{
    /// <summary>The page size reads are merged on.</summary>
    public const int PageBytes = 4096;

    private const int MaxMergedReadBytes = 256 * 1024;
    private const int MaxRowsPerPass = 16384;
    private const int GatherRetries = 3;

    private readonly IEngramRowLayout _layout;
    private readonly IWeightByteSource[] _sources;
    private readonly EngramRowSlice[] _slices;
    private readonly int[] _sliceOffsets;
    private readonly int _packedRowBytes;
    private readonly EngramRowRange _owned;
    private readonly EngramBacking _backing;
    private readonly EngramRowCache _cache;
    private readonly object _cacheLock = new();
    private readonly SemaphoreSlim _ioGate = new(1, 1);
    private readonly List<ByteRange>[] _ranges;

    // Scratch owned by whoever holds _ioGate, reused across loads so a steady-state batch allocates nothing.
    private long[] _uniqueScratch = new long[256];
    private long[] _missingScratch = new long[256];
    private int[] _bufferOffsets = new int[256];
    private byte[] _readBuffer = new byte[64 * 1024];

    private long _prefetchCalls;
    private long _gatherCalls;
    private long _rowsRequested;
    private long _uniqueRows;
    private long _hits;
    private long _misses;
    private long _gatherMisses;
    private long _readRanges;
    private long _bytesRead;
    private long _pagesTouched;
    private bool _disposed;

    /// <summary>Creates a store whose planes all come from one byte source (the official and GGUF layouts).</summary>
    public EngramTableStore(IEngramRowLayout layout, IWeightByteSource source, EngramBacking backing,
        EngramRowRange owned, long budgetBytes)
        : this(layout, [source], backing, owned, budgetBytes)
    {
    }

    /// <summary>Creates a store whose planes come from several byte sources, indexed by <see cref="EngramRowSlice.SourceIndex"/>.</summary>
    public EngramTableStore(IEngramRowLayout layout, IReadOnlyList<IWeightByteSource> sources, EngramBacking backing,
        EngramRowRange owned, long budgetBytes)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(sources);
        if (backing == EngramBacking.Device)
        {
            throw new NotSupportedException(
                "Device-resident Engram rows are not supported yet: they need the model's device buffers, which arrive with the Engram module.");
        }
        if (owned.Start < 0 || owned.Count < 1 || owned.End > layout.Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(owned),
                $"Owned rows [{owned.Start}, {owned.End}) must be non-empty and inside the table's {layout.Rows} rows.");
        }
        _layout = layout;
        _sources = sources.ToArray();
        _packedRowBytes = layout.PackedRowBytes;
        _slices = new EngramRowSlice[layout.SliceCount];
        _sliceOffsets = new int[layout.SliceCount];
        int packed = 0;
        for (int s = 0; s < _slices.Length; s++)
        {
            _slices[s] = layout.GetSlice(s);
            if (_slices[s].SourceIndex < 0 || _slices[s].SourceIndex >= _sources.Length)
            {
                throw new ArgumentException(
                    $"{layout.Name} plane {s} reads from source {_slices[s].SourceIndex} but {_sources.Length} source(s) were given.", nameof(sources));
            }
            _sliceOffsets[s] = packed;
            packed += _slices[s].BytesPerRow;
        }
        if (packed != _packedRowBytes)
            throw new ArgumentException($"{layout.Name} planes total {packed} bytes but its packed row is {_packedRowBytes}.", nameof(layout));
        _owned = owned;
        _backing = backing;
        _ranges = new List<ByteRange>[_sources.Length];
        for (int i = 0; i < _ranges.Length; i++)
            _ranges[i] = new List<ByteRange>();

        long capacity = budgetBytes / _packedRowBytes;
        if (backing == EngramBacking.HostResident)
        {
            if (capacity < owned.Count)
            {
                throw new ArgumentException(
                    $"Host-resident backing needs {owned.Count * _packedRowBytes} bytes for {owned.Count} rows but the budget is {budgetBytes}.", nameof(budgetBytes));
            }
            capacity = owned.Count;
        }
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(budgetBytes), $"The budget must hold at least one {_packedRowBytes}-byte row.");
        MaxBatchRows = (int)Math.Min(capacity, int.MaxValue);
        _cache = new EngramRowCache(MaxBatchRows, _packedRowBytes);
    }

    /// <summary>Where the rows are kept.</summary>
    public EngramBacking Backing => _backing;

    /// <summary>The rows this store serves.</summary>
    public EngramRowRange Owned => _owned;

    /// <summary>Most distinct rows one <see cref="PrefetchAsync"/> or <see cref="Gather"/> call can hold at once (the cache capacity).</summary>
    public int MaxBatchRows { get; }

    /// <summary>A snapshot of the counters.</summary>
    public EngramStoreStats Stats
    {
        get
        {
            lock (_cacheLock)
            {
                return new EngramStoreStats(_prefetchCalls, _gatherCalls, _rowsRequested, _uniqueRows, _hits, _misses,
                    _gatherMisses, _cache.Evictions, _readRanges, _bytesRead, _pagesTouched, _cache.Count,
                    (long)_cache.Count * _packedRowBytes);
            }
        }
    }

    /// <summary>Reads any of <paramref name="rows"/> that are not cached, so a following <see cref="Gather"/> of them does no I/O.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A row is outside the owned range.</exception>
    /// <exception cref="InvalidOperationException">More distinct rows than <see cref="MaxBatchRows"/>.</exception>
    public Task PrefetchAsync(ReadOnlyMemory<long> rows, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        CheckOwned(rows.Span);
        long[] copy = ArrayPool<long>.Shared.Rent(Math.Max(1, rows.Length));
        rows.Span.CopyTo(copy);
        return PrefetchCoreAsync(copy, rows.Length, cancellationToken);
    }

    /// <summary>Loads every owned row into the cache in large sequential passes. Only valid for <see cref="EngramBacking.HostResident"/>.</summary>
    public async Task LoadAllAsync(CancellationToken cancellationToken = default)
    {
        if (_backing != EngramBacking.HostResident)
            throw new InvalidOperationException("LoadAllAsync applies to host-resident backing; storage backing loads on demand.");
        const int chunk = 8192;
        long[] rows = new long[chunk];
        for (long start = _owned.Start; start < _owned.End; start += chunk)
        {
            int count = (int)Math.Min(chunk, _owned.End - start);
            for (int i = 0; i < count; i++)
                rows[i] = start + i;
            await PrefetchAsync(rows.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Decodes <paramref name="rows"/> (repeats allowed) into <paramref name="destBf16"/>, row after row, reading whatever is not cached first.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A row is outside the owned range.</exception>
    public void Gather(ReadOnlySpan<long> rows, Span<ushort> destBf16)
    {
        ThrowIfDisposed();
        int dim = _layout.Dim;
        if (destBf16.Length < (long)rows.Length * dim)
            throw new ArgumentException($"The destination holds {destBf16.Length} values but {rows.Length} rows of {dim} need {(long)rows.Length * dim}.", nameof(destBf16));
        CheckOwned(rows);
        lock (_cacheLock)
        {
            _gatherCalls++;
            _rowsRequested += rows.Length;
        }
        for (int attempt = 0; attempt < GatherRetries; attempt++)
        {
            if (TryDecodeResident(rows, destBf16))
                return;
            LoadSync(rows);
        }
        throw new InvalidOperationException(
            "Rows kept being evicted by concurrent loads while gathering; raise the Engram row budget.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
        _ioGate.Dispose();
    }

    private async Task PrefetchCoreAsync(long[] rows, int count, CancellationToken cancellationToken)
    {
        try
        {
            lock (_cacheLock)
            {
                _prefetchCalls++;
                _rowsRequested += count;
            }
            await _ioGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                int missing = PlanMisses(rows.AsSpan(0, count), gather: false);
                for (int done = 0; done < missing; done += MaxRowsPerPass)
                {
                    int take = Math.Min(MaxRowsPerPass, missing - done);
                    int bytes = PlanReads(done, take);
                    await ReadAllAsync(bytes, cancellationToken).ConfigureAwait(false);
                    Insert(done, take);
                }
            }
            finally
            {
                _ioGate.Release();
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(rows);
        }
    }

    private void LoadSync(ReadOnlySpan<long> rows)
    {
        _ioGate.Wait();
        try
        {
            int missing = PlanMisses(rows, gather: true);
            for (int done = 0; done < missing; done += MaxRowsPerPass)
            {
                int take = Math.Min(MaxRowsPerPass, missing - done);
                int bytes = PlanReads(done, take);
                ReadAllSync(bytes);
                Insert(done, take);
            }
        }
        finally
        {
            _ioGate.Release();
        }
    }

    /// <summary>Sorts and de-duplicates <paramref name="rows"/> into scratch, touches the resident ones and leaves the rest in <c>_missingScratch</c>; returns how many are missing.</summary>
    private int PlanMisses(ReadOnlySpan<long> rows, bool gather)
    {
        if (_uniqueScratch.Length < rows.Length)
            _uniqueScratch = new long[Math.Max(rows.Length, _uniqueScratch.Length * 2)];
        rows.CopyTo(_uniqueScratch);
        Span<long> sorted = _uniqueScratch.AsSpan(0, rows.Length);
        sorted.Sort();
        int unique = 0;
        for (int i = 0; i < sorted.Length; i++)
        {
            if (i == 0 || sorted[i] != sorted[i - 1])
                sorted[unique++] = sorted[i];
        }
        if (unique > MaxBatchRows)
        {
            throw new InvalidOperationException(
                $"{unique} distinct rows do not fit the {MaxBatchRows}-row cache; raise the Engram row budget or split the batch.");
        }
        if (_missingScratch.Length < unique)
            _missingScratch = new long[Math.Max(unique, _missingScratch.Length * 2)];
        int missing = 0;
        lock (_cacheLock)
        {
            for (int i = 0; i < unique; i++)
            {
                long row = sorted[i];
                if (_cache.Touch(row))
                    _hits++;
                else
                    _missingScratch[missing++] = row;
            }
            _uniqueRows += unique;
            _misses += missing;
            if (gather)
                _gatherMisses += missing;
        }
        return missing;
    }

    /// <summary>Merges the reads for <c>_missingScratch[first .. first+count)</c> into per-source range lists and fills <c>_bufferOffsets</c>; returns the bytes the reads total.</summary>
    private int PlanReads(int first, int count)
    {
        int planes = _slices.Length;
        if (_bufferOffsets.Length < count * planes)
            _bufferOffsets = new int[Math.Max(count * planes, _bufferOffsets.Length * 2)];
        foreach (List<ByteRange> list in _ranges)
            list.Clear();

        long cursor = 0;
        long pages = 0;
        for (int source = 0; source < _sources.Length; source++)
        {
            for (int plane = 0; plane < planes; plane++)
            {
                EngramRowSlice slice = _slices[plane];
                if (slice.SourceIndex != source)
                    continue;
                long groupStart = -1, groupEnd = -1;
                for (int i = 0; i < count; i++)
                {
                    long start = slice.BaseOffset + _missingScratch[first + i] * slice.BytesPerRow;
                    long end = start + slice.BytesPerRow;
                    bool joins = groupStart >= 0
                        && start / PageBytes <= (groupEnd - 1) / PageBytes
                        && end - groupStart <= MaxMergedReadBytes;
                    if (!joins)
                    {
                        if (groupStart >= 0)
                        {
                            cursor = FlushGroup(source, groupStart, groupEnd, cursor, ref pages);
                        }
                        groupStart = start;
                    }
                    groupEnd = end;
                    // Offset inside the buffer = where the group will start (cursor) + distance from the group start.
                    _bufferOffsets[i * planes + plane] = checked((int)(cursor + (start - groupStart)));
                }
                if (groupStart >= 0)
                    cursor = FlushGroup(source, groupStart, groupEnd, cursor, ref pages);
            }
        }
        if (cursor > int.MaxValue)
            throw new InvalidOperationException("A single load pass read more than 2 GiB.");
        int total = (int)cursor;
        if (_readBuffer.Length < total)
            _readBuffer = new byte[Math.Max(total, _readBuffer.Length * 2)];
        lock (_cacheLock)
        {
            _pagesTouched += pages;
            _bytesRead += total;
        }
        return total;
    }

    private long FlushGroup(int source, long start, long end, long cursor, ref long pages)
    {
        _ranges[source].Add(new ByteRange(start, checked((int)(end - start))));
        lock (_cacheLock)
            _readRanges++;
        pages += (end - 1) / PageBytes - start / PageBytes + 1;
        return cursor + (end - start);
    }

    private async Task ReadAllAsync(int bytes, CancellationToken cancellationToken)
    {
        if (bytes == 0)
            return;
        int position = 0;
        Task[] reads = new Task[_sources.Length];
        int issued = 0;
        for (int source = 0; source < _sources.Length; source++)
        {
            List<ByteRange> ranges = _ranges[source];
            if (ranges.Count == 0)
                continue;
            int length = 0;
            foreach (ByteRange range in ranges)
                length += range.Length;
            reads[issued++] = _sources[source].ReadBatchAsync(ranges, _readBuffer.AsMemory(position, length), cancellationToken);
            position += length;
        }
        await Task.WhenAll(reads.AsSpan(0, issued).ToArray()).ConfigureAwait(false);
    }

    private void ReadAllSync(int bytes)
    {
        if (bytes == 0)
            return;
        int position = 0;
        for (int source = 0; source < _sources.Length; source++)
        {
            foreach (ByteRange range in _ranges[source])
            {
                _sources[source].ReadAt(range.Offset, _readBuffer.AsSpan(position, range.Length));
                position += range.Length;
            }
        }
    }

    private void Insert(int first, int count)
    {
        int planes = _slices.Length;
        lock (_cacheLock)
        {
            for (int i = 0; i < count; i++)
            {
                Span<byte> slot = _cache.Insert(_missingScratch[first + i]);
                for (int plane = 0; plane < planes; plane++)
                {
                    _readBuffer.AsSpan(_bufferOffsets[i * planes + plane], _slices[plane].BytesPerRow)
                        .CopyTo(slot[_sliceOffsets[plane]..]);
                }
            }
        }
    }

    private bool TryDecodeResident(ReadOnlySpan<long> rows, Span<ushort> dest)
    {
        int dim = _layout.Dim;
        lock (_cacheLock)
        {
            for (int i = 0; i < rows.Length; i++)
            {
                if (!_cache.Contains(rows[i]))
                    return false;
            }
            for (int i = 0; i < rows.Length; i++)
            {
                _cache.Touch(rows[i]);
                _layout.DecodeRow(_cache.Peek(rows[i]), dest.Slice(i * dim, dim));
            }
        }
        return true;
    }

    private void CheckOwned(ReadOnlySpan<long> rows)
    {
        for (int i = 0; i < rows.Length; i++)
        {
            if (!_owned.Contains(rows[i]))
            {
                throw new ArgumentOutOfRangeException(nameof(rows),
                    $"Row {rows[i]} is outside the owned range [{_owned.Start}, {_owned.End}).");
            }
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);
}
