using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Thrown by <see cref="PagedKvPool.AllocatePage"/> when no free page remains; reject policy — the pool never blocks or evicts on its own, a caller that hits this must refuse new admission (or evict a sequence itself), since queueing/eviction policy belongs at the scheduler layer, not inside the pool.</summary>
public sealed class KvPoolExhaustedException(string message) : Exception(message);

/// <summary>Shared page pool backing one or more <see cref="PagedKvCache"/> instances: holds a fixed budget of <see cref="PageSize"/>-token pages per layer and hands out page INDICES via a free-list shared across every layer, so a cache's block table is a single list of indices, not one per layer.</summary>
/// <remarks>
/// <para>Multiple sequences (multiple <see cref="PagedKvCache"/> instances) share one pool — that's the actual point of paging: a short sequence doesn't reserve VRAM for a worst-case max length the way <see cref="FixedKvCache"/> does, and a finished sequence's pages go straight back to the free-list for the next admission instead of sitting reserved.</para>
/// <para><b>Layout.</b> Each layer holds one key buffer and one value buffer, <c>[1, kvHeads, pages × PageSize, headDim]</c>, laid out like a contiguous cache: page <c>p</c> is rows <c>[p × PageSize, (p + 1) × PageSize)</c> of every head (<see cref="FirstRow"/>). A row of a sequence is one table lookup away from its place in the buffer, and pages that sit next to each other are written and read as one run.</para>
/// <para><b>Memory follows use.</b> A layer's buffers hold the pages handed out so far plus some headroom, never more than <see cref="MaxPages"/>, and <see cref="Prepare"/> regrows them on the device when a higher page is handed out. A pool with a large budget and a short sequence therefore does not hold the whole budget. The buffers do not shrink when pages are freed. Regrowing replaces the layer's tensors, so a caller takes <see cref="KeyPages"/> and <see cref="ValuePages"/> afresh after each <see cref="Prepare"/> and never keeps them.</para>
/// </remarks>
public sealed class PagedKvPool : IDisposable
{
    public int PageSize { get; }
    public int MaxPages { get; }
    public int NumLayers { get; }
    public int NumKvHeads { get; }
    public IReadOnlyList<int> HeadDimPerLayer { get; }

    // [layer] -> Tensor [1, NumKvHeads, _layerPages[layer] * PageSize, headDimPerLayer[layer]]; null until the layer's first write
    private readonly Tensor?[] _kPages;
    private readonly Tensor?[] _vPages;
    // Pages each layer's buffers hold now.
    private readonly int[] _layerPages;
    private readonly Stack<int> _freeList;
    // One past the highest page index ever handed out: what every layer's buffers must reach before a write.
    private int _highWaterPages;
    private int _disposed;

    /// <summary>Free pages remaining (for observability/logging — not load-bearing for correctness).</summary>
    public int FreePageCount { get { lock (_freeList) return _freeList.Count; } }

    /// <summary>Allocates a pool with every layer sharing one <paramref name="headDim"/> (every architecture except Gemma-4).</summary>
    public PagedKvPool(int numLayers, int numKvHeads, int headDim, int pageSize, int maxPages)
        : this(numLayers, numKvHeads, UniformHeadDims(numLayers, headDim), pageSize, maxPages) { }

    /// <summary>Allocates a pool with a PER-LAYER head dimension (Gemma-4: local/SWA layers are narrower than global layers) — mirrors <see cref="FixedKvCache"/>'s two constructor overloads exactly.</summary>
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

        // A row of a layer's buffers is addressed with an int, as a contiguous cache's rows are.
        if ((long)maxPages * pageSize > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(maxPages), $"{maxPages} pages of {pageSize} tokens are more rows than one buffer addresses.");
        _kPages = new Tensor?[numLayers];
        _vPages = new Tensor?[numLayers];
        _layerPages = new int[numLayers];
        for (int layer = 0; layer < numLayers; layer++)
        {
            if (headDimPerLayer[layer] <= 0) throw new ArgumentOutOfRangeException(nameof(headDimPerLayer), $"layer {layer} head dim must be positive.");
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

    /// <summary>How many pages of <paramref name="pageSize"/> tokens fit in <paramref name="bytesBudget"/> for a model with these per-layer KV head dims, at F32 for K and V. Floors at 8 pages so a tiny budget still admits one short sequence.</summary>
    public static int PageCountForBudget(int numKvHeads, int[] headDimPerLayer, int pageSize, long bytesBudget)
    {
        long bytesPerPage = 0;
        foreach (int headDim in headDimPerLayer)
            bytesPerPage += (long)numKvHeads * headDim * pageSize * sizeof(float) * 2; // ×2 for K and V
        return (int)Math.Max(8, bytesBudget / Math.Max(1, bytesPerPage));
    }

    /// <summary>The key pages of <paramref name="layer"/>, <c>[1, kvHeads, pages × PageSize, headDim]</c>: valid until the
    /// layer's next <see cref="Prepare"/>.</summary>
    public Tensor KeyPages(int layer) { ThrowIfDisposed(); return Pages(_kPages, layer); }

    /// <summary>The value pages of <paramref name="layer"/>, <c>[1, kvHeads, pages × PageSize, headDim]</c>: valid until the
    /// layer's next <see cref="Prepare"/>.</summary>
    public Tensor ValuePages(int layer) { ThrowIfDisposed(); return Pages(_vPages, layer); }

    private static Tensor Pages(Tensor?[] pages, int layer) => pages[layer]
        ?? throw new InvalidOperationException($"PagedKvPool layer {layer} has no pages yet; they are allocated at the layer's first write.");

    /// <summary>The row of a layer's buffers where page <paramref name="pageIndex"/> starts.</summary>
    public int FirstRow(int pageIndex) => pageIndex * PageSize;

    /// <summary>Makes <paramref name="layer"/>'s buffers hold every page handed out so far; called before each write to the
    /// layer. The buffers are allocated on <paramref name="backend"/>'s device without an upload (rows no sequence has
    /// written are never read, so they start uninitialized), and when they must grow, the rows they already hold are
    /// copied on the device. Growth adds an eighth of headroom so a long decode regrows rarely. The append path calls
    /// this, and the caller already serializes that path per device.</summary>
    public void Prepare(IBackend backend, int layer)
    {
        ThrowIfDisposed();
        int needed;
        lock (_freeList) needed = _highWaterPages;
        int held = _layerPages[layer];
        if (held >= needed) return;
        int pages = (int)Math.Min(MaxPages, needed + Math.Max((long)needed / 8, 4));
        TensorShape shape = new(1, NumKvHeads, (long)pages * PageSize, HeadDimPerLayer[layer]);
        _kPages[layer] = Regrown(backend, _kPages[layer], shape, held * PageSize);
        _vPages[layer] = Regrown(backend, _vPages[layer], shape, held * PageSize);
        _layerPages[layer] = pages;
    }

    private static Tensor Regrown(IBackend backend, Tensor? old, TensorShape shape, int heldRows)
    {
        Tensor grown = new(shape, DType.F32);
        try
        {
            backend.ResidentAllocateKv(grown);
            if (old is not null) backend.ScatterSeqHeadMajor(grown, old, 0, heldRows);
        }
        catch
        {
            grown.Dispose();
            throw;
        }
        old?.Dispose();
        return grown;
    }

    /// <summary>Pops a free page index. Throws <see cref="KvPoolExhaustedException"/> if none remain.</summary>
    public int AllocatePage()
    {
        ThrowIfDisposed();
        lock (_freeList)
        {
            if (_freeList.Count == 0)
                throw new KvPoolExhaustedException($"PagedKvPool exhausted: all {MaxPages} pages (of {PageSize} tokens each) are in use.");
            int page = _freeList.Pop();
            if (page >= _highWaterPages) _highWaterPages = page + 1;
            return page;
        }
    }

    /// <summary>Returns a page index to the free-list for reuse; does NOT clear the page's contents — the next allocator overwrites exactly as many slots as it writes, and nothing reads a page beyond what its owning cache's block table + written-length says is valid, so stale bytes in unwritten slots are never observed.</summary>
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
        foreach (Tensor? t in _kPages) t?.Dispose();
        foreach (Tensor? t in _vPages) t?.Dispose();
    }
}
