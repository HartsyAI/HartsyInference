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

    public PagedKvCache(PagedKvPool pool)
    {
        _pool = pool;
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

        int pos = _currentLength;
        int remaining = tNew;
        int srcOffset = 0;
        while (remaining > 0)
        {
            int pageSlot = pos / _pool.PageSize;
            int withinPage = pos % _pool.PageSize;
            int roomInPage = _pool.PageSize - withinPage;
            int chunkLen = Math.Min(remaining, roomInPage);
            int physicalPage = _blockTable[pageSlot];
            Tensor kPage = _pool.KPage(layer, physicalPage);
            Tensor vPage = _pool.VPage(layer, physicalPage);

            if (chunkLen == tNew)
            {
                // Common case (every decode step: tNew=1 always fits in one page-write, no slice needed).
                backend.KvCacheAppend(kPage, newK, withinPage);
                backend.KvCacheAppend(vPage, newV, withinPage);
            }
            else
            {
                // Multi-page-spanning append (prefill: tNew = prompt length). Slice the chunk destined for
                // THIS page out of newK/newV first.
                Tensor kChunk = new(new TensorShape(1, _pool.NumKvHeads, chunkLen, headDim), DType.F32);
                Tensor vChunk = new(new TensorShape(1, _pool.NumKvHeads, chunkLen, headDim), DType.F32);
                try
                {
                    StridedCopies.SliceAxis(backend, kChunk, newK, 2, srcOffset, chunkLen);
                    StridedCopies.SliceAxis(backend, vChunk, newV, 2, srcOffset, chunkLen);
                    backend.KvCacheAppend(kPage, kChunk, withinPage);
                    backend.KvCacheAppend(vPage, vChunk, withinPage);
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
        // Grow-only capacity, rounded up to a page (mirrors FixedKvCache's oversized-buffer-plus-explicit-valid-length
        // pattern: the attention call passes the valid key count separately and never trusts this tensor's own shape for
        // it). The row count is unchanged from when this buffer was rebuilt on every call, because kernel admission and
        // plan keys read it. A larger buffer starts from the old one's rows, so growth does not send the copy back to
        // the pages.
        if (scratch is null || scratch.Shape[2] < len)
        {
            int capacity = ((Math.Max(len, 1) + _pool.PageSize - 1) / _pool.PageSize) * _pool.PageSize;
            Tensor grown = new(new TensorShape(1, _pool.NumKvHeads, capacity, headDim), DType.F32);
            if (scratch is not null && currentRows[layer] > 0)
            {
                backend.KvCacheAppend(grown, scratch, 0);
                copiedRows += currentRows[layer];
            }
            scratch?.Dispose();
            scratch = grown;
            scratchArr[layer] = scratch;
        }

        // Only the rows this layer wrote since the last call come from the pages.
        int target = Math.Min(_written[layer], len);
        int pos = Math.Min(currentRows[layer], target);
        copiedRows += target - pos;
        while (pos < target)
        {
            int withinPage = pos % _pool.PageSize;
            int chunk = Math.Min(target - pos, _pool.PageSize - withinPage);
            int physicalPage = _blockTable[pos / _pool.PageSize];
            Tensor page = isKey ? _pool.KPage(layer, physicalPage) : _pool.VPage(layer, physicalPage);
            if (chunk == _pool.PageSize)
            {
                // A whole page: every slot is wanted, so it copies as it is.
                backend.KvCacheAppend(scratch, page, pos);
            }
            else
            {
                Tensor sliced = new(new TensorShape(1, _pool.NumKvHeads, chunk, headDim), DType.F32);
                try
                {
                    StridedCopies.SliceAxis(backend, sliced, page, 2, withinPage, chunk);
                    backend.KvCacheAppend(scratch, sliced, pos);
                }
                finally
                {
                    sliced.Dispose();
                }
            }
            pos += chunk;
        }
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

    /// <summary>Returns every held page to the pool and resets to an empty sequence — unlike <see cref="FixedKvCache.Reset"/> (which just zeroes a counter over its still-owned fixed buffer), this actually frees VRAM back to the shared pool for reuse by other sequences.</summary>
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
