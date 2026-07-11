using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Thrown by <see cref="PagedKvPool.AllocatePage"/> when no free page remains. Reject policy: the
/// pool never blocks or evicts on its own — a caller that hits this must refuse new admission (or evict a
/// sequence itself); queueing/eviction policy belongs at the scheduler layer (continuous batching), not
/// inside the pool.</summary>
public sealed class KvPoolExhaustedException(string message) : Exception(message);

/// <summary>Shared page pool backing one or more <see cref="PagedKvCache"/> instances. Pre-allocates a fixed
/// budget of <see cref="PageSize"/>-token pages per layer up front (bounded VRAM, same fixed-capacity
/// philosophy as <see cref="FixedKvCache"/> — see its class doc), then hands out page INDICES via a
/// free-list shared across every layer: page index P always means "the same token range" in every layer, so
/// a cache's block table is a single list of indices, not one per layer.
///
/// <para>Multiple sequences (multiple <see cref="PagedKvCache"/> instances) share one pool — that's the
/// actual point of paging: a short sequence doesn't reserve VRAM for a worst-case max length the way
/// <see cref="FixedKvCache"/> does, and a finished sequence's pages go straight back to the free-list for the
/// next admission instead of sitting reserved.</para></summary>
public sealed class PagedKvPool : IDisposable
{
    public int PageSize { get; }
    public int MaxPages { get; }
    public int NumLayers { get; }
    public int NumKvHeads { get; }
    public IReadOnlyList<int> HeadDimPerLayer { get; }

    // [layer][pageIndex] -> Tensor [1, NumKvHeads, PageSize, headDimPerLayer[layer]]
    private readonly Tensor[][] _kPages;
    private readonly Tensor[][] _vPages;
    private readonly Stack<int> _freeList;
    private int _disposed;

    /// <summary>Free pages remaining (for observability/logging — not load-bearing for correctness).</summary>
    public int FreePageCount { get { lock (_freeList) return _freeList.Count; } }

    /// <summary>Allocates a pool with every layer sharing one <paramref name="headDim"/> (every architecture
    /// except Gemma-4).</summary>
    public PagedKvPool(int numLayers, int numKvHeads, int headDim, int pageSize, int maxPages)
        : this(numLayers, numKvHeads, UniformHeadDims(numLayers, headDim), pageSize, maxPages) { }

    /// <summary>Allocates a pool with a PER-LAYER head dimension (Gemma-4: local/SWA layers are narrower than
    /// global layers) — mirrors <see cref="FixedKvCache"/>'s two constructor overloads exactly.</summary>
    public PagedKvPool(int numLayers, int numKvHeads, int[] headDimPerLayer, int pageSize, int maxPages)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDimPerLayer.Length != numLayers) throw new ArgumentException($"headDimPerLayer has {headDimPerLayer.Length} entries, expected {numLayers}.", nameof(headDimPerLayer));
        if (pageSize <= 0) throw new ArgumentOutOfRangeException(nameof(pageSize));
        if (maxPages <= 0) throw new ArgumentOutOfRangeException(nameof(maxPages));

        NumLayers = numLayers;
        NumKvHeads = numKvHeads;
        HeadDimPerLayer = headDimPerLayer;
        PageSize = pageSize;
        MaxPages = maxPages;

        _kPages = new Tensor[numLayers][];
        _vPages = new Tensor[numLayers][];
        for (int layer = 0; layer < numLayers; layer++)
        {
            int hd = headDimPerLayer[layer];
            if (hd <= 0) throw new ArgumentOutOfRangeException(nameof(headDimPerLayer), $"layer {layer} head dim must be positive.");
            TensorShape shape = new(1, numKvHeads, pageSize, hd);
            _kPages[layer] = new Tensor[maxPages];
            _vPages[layer] = new Tensor[maxPages];
            for (int p = 0; p < maxPages; p++)
            {
                _kPages[layer][p] = new Tensor(shape, DType.F32);
                _vPages[layer][p] = new Tensor(shape, DType.F32);
            }
        }

        // Push in reverse so AllocatePage hands out 0,1,2,... on a fresh pool — deterministic, easier to reason
        // about in tests; not load-bearing for correctness (any free index would do).
        _freeList = new Stack<int>(maxPages);
        for (int p = maxPages - 1; p >= 0; p--) _freeList.Push(p);
    }

    private static int[] UniformHeadDims(int numLayers, int headDim)
    {
        int[] a = new int[numLayers];
        Array.Fill(a, headDim);
        return a;
    }

    public Tensor KPage(int layer, int pageIndex) { ThrowIfDisposed(); return _kPages[layer][pageIndex]; }
    public Tensor VPage(int layer, int pageIndex) { ThrowIfDisposed(); return _vPages[layer][pageIndex]; }

    /// <summary>Pops a free page index. Throws <see cref="KvPoolExhaustedException"/> if none remain.</summary>
    public int AllocatePage()
    {
        ThrowIfDisposed();
        lock (_freeList)
        {
            if (_freeList.Count == 0)
                throw new KvPoolExhaustedException($"PagedKvPool exhausted: all {MaxPages} pages (of {PageSize} tokens each) are in use.");
            return _freeList.Pop();
        }
    }

    /// <summary>Returns a page index to the free-list for reuse by another (or the same) sequence. Does NOT
    /// clear the page's contents — the next allocator of this index will overwrite exactly as many slots as
    /// it writes, and nothing reads a page beyond what its owning cache's block table + written-length says
    /// is valid, so stale bytes in unwritten slots are never observed.</summary>
    public void FreePage(int pageIndex)
    {
        ThrowIfDisposed();
        lock (_freeList) _freeList.Push(pageIndex);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PagedKvPool));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor[] layerPages in _kPages) foreach (Tensor t in layerPages) t.Dispose();
        foreach (Tensor[] layerPages in _vPages) foreach (Tensor t in layerPages) t.Dispose();
    }
}
