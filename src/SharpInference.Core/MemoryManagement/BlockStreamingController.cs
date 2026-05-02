using SharpInference.Core.Backends;

namespace SharpInference.Core.MemoryManagement;

/// <summary>
/// Drives a sliding window of resident blocks across a sequential model's forward
/// pass: prefetches blocks ahead of where compute is, awaits them just-in-time, and
/// evicts blocks that have fallen behind the retain window. Backend-agnostic — any
/// implementation of <see cref="IStreamingWeightCache"/> works.
///
/// <para>This is the SharpInference equivalent of ComfyUI's "lowvram" model-streaming
/// behaviour. The pattern is the same: at any given moment a small <em>working set</em>
/// of blocks lives on the device while the rest stay on host memory; PCIe transfers
/// of upcoming blocks overlap with compute on the SMs so the bandwidth cost is hidden.</para>
///
/// <para><b>Lifecycle expected from the caller:</b></para>
/// <code>
/// var controller = new BlockStreamingController(backend.StreamingCache!, blocks, prefetchAhead: 1);
/// controller.Prime();                        // upload [0..prefetchAhead]
/// for (int i = 0; i &lt; blocks.Count; i++)
/// {
///     controller.BeforeBlockForward(i);      // await i, prefetch i+ahead, evict i-retain-1
///     blocks[i].Forward(...);                // resident weights are guaranteed live here
/// }
/// controller.EvictAll();                     // free everything once forward is done
/// </code>
///
/// <para><b>Threading:</b> single-threaded. Call from the inference thread that owns
/// the backend's CUDA context. The cache itself uses GPU streams to overlap, but the
/// controller's bookkeeping is host-side serial.</para>
/// </summary>
public sealed class BlockStreamingController : IDisposable
{
    private enum BlockState
    {
        /// <summary>No upload has been issued for this block; weights live only on host.</summary>
        NotUploaded = 0,
        /// <summary>Upload is in flight; <see cref="_tokens"/> holds the await token.</summary>
        Uploading = 1,
        /// <summary>Upload has been awaited; weights are usable on the compute stream.</summary>
        Resident = 2,
        /// <summary>Memory has been freed via <see cref="IStreamingWeightCache.EvictAsync"/>.
        /// Re-uploading goes back through <see cref="BlockState.NotUploaded"/>.</summary>
        Evicted = 3,
    }

    private readonly IStreamingWeightCache _cache;
    private readonly IReadOnlyList<IStreamingBlock> _blocks;
    private readonly int _prefetchAhead;
    private readonly int _retainBehind;

    private readonly BlockState[] _state;
    private readonly StreamingUploadToken[] _tokens;

    private bool _disposed;

    /// <summary>Constructs a streaming controller.</summary>
    /// <param name="cache">The backend's streaming cache (typically <c>backend.StreamingCache</c>).</param>
    /// <param name="blocks">Ordered list of blocks. Iteration index passed to
    /// <see cref="BeforeBlockForward"/> indexes into this list.</param>
    /// <param name="prefetchAhead">How many blocks ahead of the current one to keep
    /// uploading. <c>0</c> = synchronous (await each block when needed, no overlap).
    /// <c>1</c> = one block in flight while compute runs (typical). Higher values
    /// improve hiding when compute is fast vs PCIe but consume more VRAM.</param>
    /// <param name="retainBehind">How many already-used blocks to keep cached for
    /// reuse. <c>0</c> = evict immediately after use (default; needed for tight VRAM
    /// budgets). Higher values help if blocks are revisited (rare in transformer
    /// inference; common in some encoder-decoder patterns).</param>
    public BlockStreamingController(
        IStreamingWeightCache cache,
        IReadOnlyList<IStreamingBlock> blocks,
        int prefetchAhead = 1,
        int retainBehind = 0)
    {
        if (cache is null) throw new ArgumentNullException(nameof(cache));
        if (blocks is null) throw new ArgumentNullException(nameof(blocks));
        if (prefetchAhead < 0) throw new ArgumentOutOfRangeException(nameof(prefetchAhead));
        if (retainBehind < 0) throw new ArgumentOutOfRangeException(nameof(retainBehind));

        _cache = cache;
        _blocks = blocks;
        _prefetchAhead = prefetchAhead;
        _retainBehind = retainBehind;
        _state = new BlockState[blocks.Count];
        _tokens = new StreamingUploadToken[blocks.Count];
    }

    /// <summary>The number of blocks under management.</summary>
    public int BlockCount => _blocks.Count;

    /// <summary>Total estimated weight bytes summed across all blocks.</summary>
    public long EstimatedTotalWeightBytes
    {
        get
        {
            long total = 0;
            for (int i = 0; i < _blocks.Count; i++) total += _blocks[i].EstimatedWeightBytes;
            return total;
        }
    }

    /// <summary>Begins async uploads for blocks <c>[0, prefetchAhead]</c> so the very
    /// first call to <see cref="BeforeBlockForward"/> doesn't have to do a cold-start
    /// synchronous load. Idempotent — safe to call multiple times.</summary>
    public void Prime()
    {
        ThrowIfDisposed();
        // Prefetch [0, prefetchAhead] inclusive. Block 0 gets uploaded; blocks 1..ahead
        // sit Uploading until BeforeBlockForward(0) gets called and bumps the front.
        for (int i = 0; i <= _prefetchAhead && i < _blocks.Count; i++)
        {
            EnsureUploading(i);
        }
    }

    /// <summary>Call IMMEDIATELY before block <paramref name="blockIdx"/>'s forward
    /// pass. Three responsibilities, in order:
    /// <list type="number">
    /// <item><description>Await block <paramref name="blockIdx"/> if its upload is in
    /// flight (cheap if already <see cref="BlockState.Resident"/>; cold-start upload
    /// + sync await if <see cref="BlockState.NotUploaded"/>).</description></item>
    /// <item><description>Kick off prefetch for block <c>blockIdx + prefetchAhead</c>
    /// if it isn't already uploading.</description></item>
    /// <item><description>Evict block <c>blockIdx - retainBehind - 1</c> if it's still
    /// resident and outside the retain window.</description></item>
    /// </list>
    /// All three are no-ops when their target index is out of <c>[0, BlockCount)</c>.</summary>
    public void BeforeBlockForward(int blockIdx)
    {
        ThrowIfDisposed();
        if (blockIdx < 0 || blockIdx >= _blocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(blockIdx),
                $"Block index {blockIdx} out of range [0, {_blocks.Count}).");
        }

        // 1. Make sure block N is resident before we hand it to the caller.
        EnsureResident(blockIdx);

        // 2. Prefetch the block one step further ahead than we already had queued.
        //    With prefetchAhead=1 this preserves a single in-flight upload; with
        //    higher values it keeps the pipeline deep.
        int prefetchTarget = blockIdx + _prefetchAhead;
        if (prefetchTarget < _blocks.Count)
        {
            EnsureUploading(prefetchTarget);
        }

        // 3. Evict blocks that have rolled out of the retain window. The slot to free
        //    is the one we'd retain "if we kept retainBehind blocks" — i.e. just past
        //    the end of the window walking backward.
        int evictTarget = blockIdx - _retainBehind - 1;
        if (evictTarget >= 0)
        {
            EvictBlock(evictTarget);
        }
    }

    /// <summary>Frees every block that's still resident. Call when the model is
    /// finished — e.g. at the end of the denoising loop, before the VAE pipeline
    /// stage starts. Safe to call multiple times.</summary>
    public void EvictAll()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _blocks.Count; i++)
        {
            EvictBlock(i);
        }
    }

    // ── State transitions ───────────────────────────────────────────────

    /// <summary>Drives block to <see cref="BlockState.Uploading"/> if it's currently
    /// <see cref="BlockState.NotUploaded"/> or <see cref="BlockState.Evicted"/>.
    /// No-op for blocks already <see cref="BlockState.Uploading"/> or
    /// <see cref="BlockState.Resident"/>.</summary>
    private void EnsureUploading(int idx)
    {
        BlockState state = _state[idx];
        if (state == BlockState.Uploading || state == BlockState.Resident)
        {
            return; // already on the way (or already there)
        }
        StreamingUploadToken token = _cache.BeginUploadAsync(_blocks[idx].EnumerateWeights());
        _tokens[idx] = token;
        // Even if the cache returned Empty (all weights were already cached from a
        // prior pipeline phase, e.g. the same tensors used elsewhere), we mark the
        // block Uploading rather than skipping ahead: the next EnsureResident call
        // will await (no-op on Empty) and transition to Resident, keeping the state
        // machine consistent.
        _state[idx] = BlockState.Uploading;
    }

    /// <summary>Drives block to <see cref="BlockState.Resident"/>. If currently
    /// <see cref="BlockState.NotUploaded"/> or <see cref="BlockState.Evicted"/>, performs
    /// a synchronous cold-start: BeginUpload + immediate Await. This is the slow path
    /// — it means our prefetch window wasn't deep enough.</summary>
    private void EnsureResident(int idx)
    {
        switch (_state[idx])
        {
            case BlockState.Resident:
                return;

            case BlockState.Uploading:
                _cache.AwaitWeights(_tokens[idx]);
                _tokens[idx] = StreamingUploadToken.Empty;
                _state[idx] = BlockState.Resident;
                return;

            case BlockState.NotUploaded:
            case BlockState.Evicted:
                // Cold path: caller asked for this block before prefetch reached it.
                // Issue the upload synchronously (BeginUpload + immediate Await) so
                // the caller's compute doesn't see uninitialised weights. Worth a
                // log line at the consumer level so prefetch tuning is observable.
                EnsureUploading(idx);
                _cache.AwaitWeights(_tokens[idx]);
                _tokens[idx] = StreamingUploadToken.Empty;
                _state[idx] = BlockState.Resident;
                return;

            default:
                throw new InvalidOperationException($"Unknown block state {_state[idx]}.");
        }
    }

    /// <summary>Drives block to <see cref="BlockState.Evicted"/>. No-op for blocks
    /// already <see cref="BlockState.Evicted"/> or <see cref="BlockState.NotUploaded"/>.
    /// If a block is mid-upload (<see cref="BlockState.Uploading"/>) we await it first
    /// before evicting — eviction implies the dptr exists, which only happens after
    /// the upload completes from the cache's perspective.</summary>
    private void EvictBlock(int idx)
    {
        switch (_state[idx])
        {
            case BlockState.NotUploaded:
            case BlockState.Evicted:
                return;

            case BlockState.Uploading:
                // The dptr was registered immediately at BeginUpload time; eviction
                // doesn't have to wait for the H2D copy to finish. The cache's
                // EvictAsync queues a free on the compute stream which is naturally
                // ordered after any prior op — including the upload itself once the
                // wait-event chain catches up.
                _cache.AwaitWeights(_tokens[idx]);
                _tokens[idx] = StreamingUploadToken.Empty;
                goto case BlockState.Resident;

            case BlockState.Resident:
                _cache.EvictAsync(_blocks[idx].EnumerateWeights());
                _state[idx] = BlockState.Evicted;
                return;

            default:
                throw new InvalidOperationException($"Unknown block state {_state[idx]}.");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(BlockStreamingController));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Best-effort: flush any in-flight uploads + free everything still cached.
        // Errors during shutdown are swallowed since there's nowhere useful to
        // surface them — the cache itself logs/throws as appropriate.
        for (int i = 0; i < _blocks.Count; i++)
        {
            try { EvictBlock(i); } catch { /* shutdown best-effort */ }
        }
    }
}
