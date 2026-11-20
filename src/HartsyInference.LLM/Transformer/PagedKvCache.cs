using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Block/page-based KV cache for ONE sequence, backed by a shared <see cref="PagedKvPool"/>: unlike <see cref="FixedKvCache"/> (one contiguous per-layer buffer sized to a worst-case max sequence length, hard-restricted to a single sequence per instance), this allocates fixed-size pages from the pool on demand as the sequence grows and returns them on <see cref="Dispose"/>/<see cref="Reset"/>, so VRAM isn't reserved per-sequence for a worst case and a finished sequence's pages are immediately reusable.</summary>
/// <remarks>
/// <para><b>Design: "physically paged, logically contiguous."</b> Each layer keeps a contiguous copy of its keys and of its values beside the pages, and <see cref="KeyPrefix"/>/<see cref="ValuePrefix"/> return it after copying in only the rows that layer wrote since the last call. A write at a position (a decode step, a rollback, a retried round) marks the copy stale from that position on. The copy is as large as the valid prefix, so a pooled sequence holds its keys and values twice; the pool budget counts the pages only. Reading the pages directly from the attention kernels would remove the copy.</para>
/// <para><b>Backend capture:</b> <see cref="IKvCache.KeyPrefix"/>/<see cref="IKvCache.ValuePrefix"/> don't receive an <see cref="IBackend"/> parameter (matching the existing interface), but gathering pages requires issuing backend calls, so this cache captures the <see cref="IBackend"/> passed to the most recent <see cref="AppendStep"/> and reuses it for the gather; every caller in this codebase uses exactly one backend instance per session, so this holds in practice without a wider interface change.</para>
/// </remarks>
public sealed class PagedKvCache : IKvCache
{
    private readonly PagedKvPool _pool;
    private readonly List<int> _blockTable = [];
    private int _currentLength;      // committed position — advances ONLY via AdvanceLength, mirrors FixedKvCache
    private int _physicallyWritten;  // tokens actually present in pages; >= _currentLength, catches up to it via AdvanceLength
    private IBackend? _lastBackend;
    private int _disposed;

    private readonly Tensor?[] _scratchK;
    private readonly Tensor?[] _scratchV;
    // Per layer: rows this layer has written to its pages, and how many leading rows of each contiguous prefix already
    // equal those pages. A prefix call copies only the rows past its watermark; a write at a position lowers it to there.
    private readonly int[] _written;
    // Rows the prefix may be sized ahead to (the caller's token bound, page-rounded); 0 when no bound was given.
    private readonly int _reserveRows;
    private readonly int[] _currentK;
    private readonly int[] _currentV;

    public int NumLayers => _pool.NumLayers;
    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Upper bound: every pool page assigned to this one sequence.</summary>
    public int Capacity => _pool.MaxPages * _pool.PageSize;

    /// <summary>Every committed token can be rolled back; <see cref="Truncate"/> also returns the freed pages to the pool.</summary>
    public int MaxRollback => CurrentLength;

    /// <summary>Pages currently held by this sequence (for observability/logging).</summary>
    public int PagesHeld { get { ThrowIfDisposed(); return _blockTable.Count; } }

    /// <summary>The sequence's page table: entry <c>i</c> is the pool page that holds rows <c>[i × PageSize, (i + 1) × PageSize)</c>.
    /// It changes when the sequence grows, rolls back or resets, and a page it no longer lists may already belong to another
    /// sequence, so a reader takes it afresh for each use.</summary>
    public IReadOnlyList<int> PageTable { get { ThrowIfDisposed(); return _blockTable; } }

    /// <param name="pool">The page pool this sequence draws from.</param>
    /// <param name="expectedTokens">The most tokens the sequence will hold (prompt plus generation), when the caller knows
    /// it. The contiguous prefix is then sized ahead in steps up to this bound, so it is not regrown every page; 0 keeps
    /// it at the valid length rounded up to a page.</param>
    public PagedKvCache(PagedKvPool pool, int expectedTokens = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(expectedTokens);
        _pool = pool;
        _reserveRows = expectedTokens == 0 ? 0 : RoundUpToPage(expectedTokens);
        _scratchK = new Tensor?[pool.NumLayers];
        _scratchV = new Tensor?[pool.NumLayers];
        _written = new int[pool.NumLayers];
        _currentK = new int[pool.NumLayers];
        _currentV = new int[pool.NumLayers];
    }

    public void AppendStep(IBackend backend, int layer, Tensor newK, Tensor newV)
    {
        ThrowIfDisposed();
        _lastBackend = backend;
        int tNew = (int)newK.Shape[2];
        int headDim = _pool.HeadDimPerLayer[layer];

        // Idempotent growth: the layer loop calls AppendStep once per layer for the SAME token range (same
        // tNew, same _currentLength, since _currentLength is pinned until AdvanceLength runs after every
        // layer). Growing to "at least this many tokens" is a no-op on the 2nd..Nth layer's call, so no
        // layer-index bookkeeping is needed — this naturally converges regardless of call order.
        int target = _currentLength + tNew;
        GrowBlockTableIfNeeded(target);
        if (_physicallyWritten < target) _physicallyWritten = target;
        if (_written[layer] < target) _written[layer] = target;
        // The rows from here on are being (re)written, so neither contiguous prefix can be trusted past this position.
        if (_currentK[layer] > _currentLength) _currentK[layer] = _currentLength;
        if (_currentV[layer] > _currentLength) _currentV[layer] = _currentLength;

        _pool.Prepare(backend, layer);
        Tensor kPages = _pool.KeyPages(layer), vPages = _pool.ValuePages(layer);
        int pos = _currentLength;
        int remaining = tNew;
        int srcOffset = 0;
        while (remaining > 0)
        {
            // Rows that sit next to each other in the pool are written in one copy: a decode step's single row, and a
            // prefill's whole run of pages when the pool handed them out in order.
            int chunkLen = AdjacentRows(pos, remaining);
            int poolRow = PoolRow(pos);
            if (chunkLen == tNew)
            {
                backend.KvCacheAppend(kPages, newK, poolRow);
                backend.KvCacheAppend(vPages, newV, poolRow);
            }
            else
            {
                // The appended rows land in more than one place in the pool: slice out the rows for this run.
                Tensor kChunk = new(new TensorShape(1, _pool.NumKvHeads, chunkLen, headDim), DType.F32);
                Tensor vChunk = new(new TensorShape(1, _pool.NumKvHeads, chunkLen, headDim), DType.F32);
                try
                {
                    StridedCopies.SliceAxis(backend, kChunk, newK, 2, srcOffset, chunkLen);
                    StridedCopies.SliceAxis(backend, vChunk, newV, 2, srcOffset, chunkLen);
                    backend.KvCacheAppend(kPages, kChunk, poolRow);
                    backend.KvCacheAppend(vPages, vChunk, poolRow);
                }
                finally
                {
                    kChunk.Dispose();
                    vChunk.Dispose();
                }
            }

            pos += chunkLen;
            srcOffset += chunkLen;
            remaining -= chunkLen;
        }
        AppendToPrefix(backend, layer, isKey: true, newK, _currentLength, tNew);
        AppendToPrefix(backend, layer, isKey: false, newV, _currentLength, tNew);
    }

    /// <summary>The row of the pool's per-layer buffers that holds the sequence's row <paramref name="pos"/>.</summary>
    private int PoolRow(int pos) => _pool.FirstRow(_blockTable[pos / _pool.PageSize]) + pos % _pool.PageSize;

    /// <summary>How many of the <paramref name="wanted"/> rows from <paramref name="pos"/> on sit next to each other in the
    /// pool: the rest of that row's page, plus each following page whose pool page is the next one.</summary>
    private int AdjacentRows(int pos, int wanted)
    {
        int slot = pos / _pool.PageSize;
        int rows = _pool.PageSize - pos % _pool.PageSize;
        while (rows < wanted && slot + 1 < _blockTable.Count && _blockTable[slot + 1] == _blockTable[slot] + 1)
        {
            rows += _pool.PageSize;
            slot++;
        }
        return Math.Min(rows, wanted);
    }

    private int RoundUpToPage(int rows) => ((Math.Max(rows, 1) + _pool.PageSize - 1) / _pool.PageSize) * _pool.PageSize;

    /// <summary>Rows a new prefix buffer gets when it must hold <paramref name="len"/> valid rows. Without a token bound that
    /// is the length rounded up to a page, the count the per-call rebuild produced. With one, the buffer is sized ahead
    /// by an eighth of the length (at least 256 rows) and never past the bound, so a sequence regrows its prefix a
    /// handful of times instead of once a page, and a prefill usually allocates it once.</summary>
    private int PrefixCapacity(int len)
    {
        int needed = RoundUpToPage(len);
        if (_reserveRows == 0) return needed;
        int ahead = RoundUpToPage(len + Math.Max(256, len / 8));
        return Math.Max(needed, Math.Min(_reserveRows, ahead));
    }

    /// <summary>Adds the rows just written at <paramref name="pos"/> to the layer's contiguous prefix, straight from the
    /// appended tensor, when the prefix is current up to there. A prefix that is behind is left for the next prefix call,
    /// which catches up from the pages.</summary>
    private void AppendToPrefix(IBackend backend, int layer, bool isKey, Tensor rows, int pos, int tNew)
    {
        Tensor?[] scratchArr = isKey ? _scratchK : _scratchV;
        int[] currentRows = isKey ? _currentK : _currentV;
        if (currentRows[layer] != pos) return;
        int headDim = _pool.HeadDimPerLayer[layer];
        int len = _physicallyWritten;
        Tensor? scratch = scratchArr[layer];
        bool grow = scratch is null || scratch.Shape[2] < len;
        int capacity = grow ? PrefixCapacity(len) : (int)scratch!.Shape[2];
        long copiedRows = tNew;
        if (pos == 0 && grow)
        {
            // Nothing to keep: the new rows, padded out to the capacity, are the buffer. Built as an op's output, only the
            // padding rows are uploaded from the host, not the whole buffer.
            scratch?.Dispose();
            scratch = new Tensor(new TensorShape(1, _pool.NumKvHeads, capacity, headDim), DType.F32);
            scratchArr[layer] = scratch;
            if (capacity == tNew)
            {
                backend.Concat(scratch, [rows], 2);
            }
            else
            {
                using Tensor pad = new(new TensorShape(1, _pool.NumKvHeads, capacity - tNew, headDim), DType.F32);
                backend.Concat(scratch, [rows, pad], 2);
            }
        }
        else
        {
            if (grow)
            {
                scratch = GrowPrefix(backend, scratch, pos, capacity, headDim, out int movedRows);
                scratchArr[layer] = scratch;
                copiedRows += movedRows;
            }
            backend.KvCacheAppend(scratch!, rows, pos);
        }
        currentRows[layer] = pos + tNew;
        AttentionEvidence.Observed(AttentionEvidenceKind.PrefixGather, isKey ? "paged.keys" : "paged.values",
            _pool.NumKvHeads * copiedRows * headDim * sizeof(float), layer: layer);
    }

    /// <summary>A buffer of <paramref name="capacity"/> rows that starts with the <paramref name="keepRows"/> rows
    /// <paramref name="old"/> holds. It is a concatenation's output: the old buffer's rows move on the device and only the
    /// padding rows are uploaded. With nothing to keep, the buffer is returned unwritten and its first device use uploads
    /// it whole; that is the order where a prefix is read before anything was appended to it.</summary>
    /// <param name="movedRows">Rows the concatenation moved: all of <paramref name="old"/>, which can be more than
    /// <paramref name="keepRows"/> when it was sized ahead.</param>
    private Tensor GrowPrefix(IBackend backend, Tensor? old, int keepRows, int capacity, int headDim, out int movedRows)
    {
        Tensor grown = new(new TensorShape(1, _pool.NumKvHeads, capacity, headDim), DType.F32);
        movedRows = 0;
        if (old is not null && keepRows > 0)
        {
            movedRows = (int)old.Shape[2];
            using Tensor pad = new(new TensorShape(1, _pool.NumKvHeads, capacity - movedRows, headDim), DType.F32);
            try
            {
                backend.Concat(grown, [old, pad], 2);
            }
            catch
            {
                // The caller still holds the old buffer; only the new one is this method's to release.
                grown.Dispose();
                throw;
            }
        }
        old?.Dispose();
        return grown;
    }

    public Tensor KeyPrefix(int layer) { ThrowIfDisposed(); return Gather(layer, isKey: true); }
    public Tensor ValuePrefix(int layer) { ThrowIfDisposed(); return Gather(layer, isKey: false); }

    private Tensor Gather(int layer, bool isKey)
    {
        if (_lastBackend is null)
            throw new InvalidOperationException("KeyPrefix/ValuePrefix called before any AppendStep — nothing written yet.");
        IBackend backend = _lastBackend;
        int len = _physicallyWritten;
        int headDim = _pool.HeadDimPerLayer[layer];
        Tensor?[] scratchArr = isKey ? _scratchK : _scratchV;
        int[] currentRows = isKey ? _currentK : _currentV;
        Tensor? scratch = scratchArr[layer];
        long copiedRows = 0;
        // Grow-only capacity (mirrors FixedKvCache's oversized-buffer-plus-explicit-valid-length pattern: the attention
        // call passes the valid key count separately and never trusts this tensor's own shape for it). A larger buffer
        // starts from the old one's rows, so growth does not send the copy back to the pages. In the usual order (append,
        // then read) the append has already added this step's rows and there is nothing left to copy here.
        if (scratch is null || scratch.Shape[2] < len)
        {
            scratch = GrowPrefix(backend, scratch, currentRows[layer], PrefixCapacity(len), headDim, out int movedRows);
            copiedRows += movedRows;
            scratchArr[layer] = scratch;
        }

        // Only the rows this layer wrote since the last call come from the pages.
        int target = Math.Min(_written[layer], len);
        int pos = Math.Min(currentRows[layer], target);
        copiedRows += target - pos;
        Tensor pages = isKey ? _pool.KeyPages(layer) : _pool.ValuePages(layer);
        while (pos < target)
        {
            // One copy per run of rows that sit next to each other in the pool.
            int chunk = AdjacentRows(pos, target - pos);
            Tensor sliced = new(new TensorShape(1, _pool.NumKvHeads, chunk, headDim), DType.F32);
            try
            {
                StridedCopies.SliceAxis(backend, sliced, pages, 2, PoolRow(pos), chunk);
                backend.KvCacheAppend(scratch, sliced, pos);
            }
            finally
            {
                sliced.Dispose();
            }
            pos += chunk;
        }
        // An append never moves a watermark past the rows its layer wrote, so catching up can only raise it.
        System.Diagnostics.Debug.Assert(currentRows[layer] <= target, "A prefix watermark is ahead of the rows its layer wrote.");
        currentRows[layer] = target;
        if (copiedRows > 0)
            AttentionEvidence.Observed(AttentionEvidenceKind.PrefixGather, isKey ? "paged.keys" : "paged.values",
                _pool.NumKvHeads * copiedRows * headDim * sizeof(float), layer: layer);
        return scratch;
    }

    private void GrowBlockTableIfNeeded(int targetTokens)
    {
        int pagesNeeded = (targetTokens + _pool.PageSize - 1) / _pool.PageSize;
        while (_blockTable.Count < pagesNeeded)
        {
            _blockTable.Add(_pool.AllocatePage());
        }
    }

    public void AdvanceLength(int by)
    {
        ThrowIfDisposed();
        if (by < 0) throw new ArgumentOutOfRangeException(nameof(by));
        _currentLength += by;
    }

    /// <summary>Rolls back to <paramref name="newLength"/> (speculative-decode rejection): shrinks both the committed length and the physically-written count (unlike <see cref="FixedKvCache.Truncate"/>, this cache tracks physical-write extent separately, and it must shrink too or <see cref="KeyPrefix"/>/<see cref="ValuePrefix"/> would keep gathering the rejected tokens' stale K/V). Also frees any page now entirely beyond the new length back to the pool.</summary>
    public void Truncate(int newLength)
    {
        ThrowIfDisposed();
        if (newLength < 0 || newLength > _currentLength) throw new ArgumentOutOfRangeException(nameof(newLength));
        int pagesNeeded = (newLength + _pool.PageSize - 1) / _pool.PageSize;
        while (_blockTable.Count > pagesNeeded)
        {
            _pool.FreePage(_blockTable[^1]);
            _blockTable.RemoveAt(_blockTable.Count - 1);
        }
        _physicallyWritten = newLength;
        _currentLength = newLength;
        ForgetRowsFrom(newLength);
    }

    /// <summary>Returns every held page to the pool and resets to an empty sequence — unlike <see cref="FixedKvCache.Reset"/> (which just zeroes a counter over its still-owned fixed buffer), this actually frees VRAM back to the shared pool for reuse by other sequences.
    /// The contiguous prefix buffers are kept for the next sequence this cache holds; they are outside the pool's budget and are freed on <see cref="Dispose"/>.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        foreach (int page in _blockTable) _pool.FreePage(page);
        _blockTable.Clear();
        _currentLength = 0;
        _physicallyWritten = 0;
        ForgetRowsFrom(0);
    }

    /// <summary>Rows at and past <paramref name="length"/> no longer exist: no layer has written them and no prefix holds them.</summary>
    private void ForgetRowsFrom(int length)
    {
        for (int layer = 0; layer < _written.Length; layer++)
        {
            if (_written[layer] > length) _written[layer] = length;
            if (_currentK[layer] > length) _currentK[layer] = length;
            if (_currentV[layer] > length) _currentV[layer] = length;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PagedKvCache));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (int page in _blockTable) _pool.FreePage(page);
        _blockTable.Clear();
        foreach (Tensor? t in _scratchK) t?.Dispose();
        foreach (Tensor? t in _scratchV) t?.Dispose();
    }
}
