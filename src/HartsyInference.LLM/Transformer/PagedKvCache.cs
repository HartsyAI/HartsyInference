using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Block/page-based KV cache for ONE sequence, backed by a shared <see cref="PagedKvPool"/>: unlike <see cref="FixedKvCache"/> (one contiguous per-layer buffer sized to a worst-case max sequence length, hard-restricted to a single sequence per instance), this allocates fixed-size pages from the pool on demand as the sequence grows and returns them on <see cref="Dispose"/>/<see cref="Reset"/>, so VRAM isn't reserved per-sequence for a worst case and a finished sequence's pages are immediately reusable.</summary>
/// <remarks>
/// <para><b>Design: "physically paged, logically contiguous."</b> <see cref="KeyPrefix"/>/<see cref="ValuePrefix"/> gather this sequence's pages into a contiguous scratch tensor each call (via <see cref="IBackend.SliceTimeRange"/> + <see cref="IBackend.KvCacheAppend"/>, page by page) so <c>FlashAttention</c>'s existing contiguous-buffer contract is unchanged — this gets the memory-management win without a new paged-attention kernel. A true paged-attention kernel that reads pages directly (no gather) is a natural follow-up once this is correctness-proven — see docs/Checklists/LLM_DECODE_PERF_GRIND.md.</para>
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

    public int NumLayers => _pool.NumLayers;
    public int CurrentLength { get { ThrowIfDisposed(); return _currentLength; } }

    /// <summary>Pages currently held by this sequence (for observability/logging).</summary>
    public int PagesHeld { get { ThrowIfDisposed(); return _blockTable.Count; } }

    public PagedKvCache(PagedKvPool pool)
    {
        _pool = pool;
        _scratchK = new Tensor?[pool.NumLayers];
        _scratchV = new Tensor?[pool.NumLayers];
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
                    backend.SliceTimeRange(kChunk, newK, srcOffset, chunkLen);
                    backend.SliceTimeRange(vChunk, newV, srcOffset, chunkLen);
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
        Tensor? scratch = scratchArr[layer];
        // Grow-only capacity (mirrors FixedKvCache's oversized-buffer-plus-explicit-valid-length pattern —
        // GenericTransformer's attention call already passes the valid key count separately and never trusts
        // this tensor's own shape for it, see its "kFull/vFull may be a fixed-capacity buffer" comment). The
        // loop below always re-copies every valid position from the pool's pages regardless of whether
        // `scratch` was just (re)allocated, so nothing needs preserving across a resize — only grow when the
        // existing buffer is too small. `len` advances by exactly 1 every decode step, so reallocating
        // whenever it merely CHANGED (the previous check) meant a full GPU tensor allocation every single
        // round for every active sequence; rounding up to a page boundary cuts that to once per PageSize
        // tokens instead, at the cost of at most one page's worth of unused VRAM per sequence per layer.
        if (scratch is null || scratch.Shape[2] < len)
        {
            scratch?.Dispose();
            int capacity = ((Math.Max(len, 1) + _pool.PageSize - 1) / _pool.PageSize) * _pool.PageSize;
            scratch = new Tensor(new TensorShape(1, _pool.NumKvHeads, capacity, headDim), DType.F32);
            scratchArr[layer] = scratch;
        }
        if (len == 0) return scratch;   // nothing written yet; caller shouldn't reach here in practice

        int fullPages = len / _pool.PageSize;
        int rem = len % _pool.PageSize;
        int dstOffset = 0;
        for (int i = 0; i < fullPages; i++)
        {
            Tensor page = isKey ? _pool.KPage(layer, _blockTable[i]) : _pool.VPage(layer, _blockTable[i]);
            // Full page — every slot is valid, so no slice needed, just copy it whole into scratch at dstOffset.
            backend.KvCacheAppend(scratch, page, dstOffset);
            dstOffset += _pool.PageSize;
        }
        if (rem > 0)
        {
            Tensor lastPage = isKey ? _pool.KPage(layer, _blockTable[fullPages]) : _pool.VPage(layer, _blockTable[fullPages]);
            // Last page is only partially occupied — slice down to the valid prefix before copying.
            Tensor sliced = new(new TensorShape(1, _pool.NumKvHeads, rem, headDim), DType.F32);
            try
            {
                backend.SliceTimeRange(sliced, lastPage, 0, rem);
                backend.KvCacheAppend(scratch, sliced, dstOffset);
            }
            finally
            {
                sliced.Dispose();
            }
        }
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
    }

    /// <summary>Returns every held page to the pool and resets to an empty sequence — unlike <see cref="FixedKvCache.Reset"/> (which just zeroes a counter over its still-owned fixed buffer), this actually frees VRAM back to the shared pool for reuse by other sequences.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        foreach (int page in _blockTable) _pool.FreePage(page);
        _blockTable.Clear();
        _currentLength = 0;
        _physicallyWritten = 0;
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
