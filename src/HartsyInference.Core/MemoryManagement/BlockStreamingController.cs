using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Drives a sliding window of resident blocks: prefetches ahead, awaits just-in-time, evicts past the retain window.</summary>
/// <remarks>HartsyInference equivalent of ComfyUI's "lowvram" model-streaming. Single-threaded; call from the inference
/// thread that owns the backend's CUDA context.</remarks>
public sealed class BlockStreamingController : IDisposable
{
    private readonly IStreamingWeightCache _cache;
    private readonly IReadOnlyList<IStreamingBlock> _blocks;
    private readonly int _prefetchAhead;
    private readonly int _retainBehind;

    private readonly BlockState[] _state;
    private readonly StreamingUploadToken[] _tokens;

    private bool _disposed;

    /// <summary>Constructs a streaming controller with the given prefetch depth and retain window.</summary>
    /// <remarks><paramref name="prefetchAhead"/> is the in-flight depth (0 = synchronous, 1 = typical overlap); <paramref name="retainBehind"/> keeps
    /// already-used blocks cached for reuse (0 = evict immediately).</remarks>
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

        // Block-swap re-uploads every block each forward, so pin the host sources: pageable H2D silently degrades
        // to a synchronous staged copy — zero overlap with compute, which made deeper prefetch windows useless
        // (LTX-2.3 measured ~7 GB/s serialized vs the pinned ~13 GB/s overlapped on this PCIe gen3 host).
        // One-time registration per source, graceful per-weight fallback on failure. HARTSY_STREAM_PIN=0 disables.
        if (Environment.GetEnvironmentVariable("HARTSY_STREAM_PIN") != "0")
        {
            cache.PinUploadSource = true;
        }
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

    /// <summary>Begins async uploads for <c>[0, prefetchAhead]</c> so <see cref="BeforeBlockForward"/> skips a cold-start load; idempotent.</summary>
    public void Prime()
    {
        ThrowIfDisposed();
        for (int i = 0; i <= _prefetchAhead && i < _blocks.Count; i++)
        {
            EnsureUploading(i);
        }
    }

    /// <summary>Call before block <paramref name="blockIdx"/>'s forward pass: awaits it, prefetches ahead, evicts past retain window.</summary>
    /// <remarks>Kicks off prefetch for <c>blockIdx + prefetchAhead</c> and evicts <c>blockIdx - retainBehind - 1</c> if outside the retain
    /// window. Indexes outside <c>[0, BlockCount)</c> are no-ops.</remarks>
    public void BeforeBlockForward(int blockIdx)
    {
        ThrowIfDisposed();
        if (blockIdx < 0 || blockIdx >= _blocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(blockIdx),
                $"Block index {blockIdx} out of range [0, {_blocks.Count}).");
        }

        EnsureResident(blockIdx);

        int prefetchTarget = blockIdx + _prefetchAhead;
        if (prefetchTarget < _blocks.Count)
        {
            EnsureUploading(prefetchTarget);
        }

        int evictTarget = blockIdx - _retainBehind - 1;
        if (evictTarget >= 0)
        {
            EvictBlock(evictTarget);
        }
    }

    /// <summary>Frees every resident block, drains in-flight async ops, and releases backend pool reservations to the driver.</summary>
    /// <remarks>Call between streaming and eager-allocation phases — CUDA's stream-ordered allocator pool is invisible to subsequent sync
    /// allocations unless explicitly trimmed. Safe to call multiple times.</remarks>
    public void EvictAll()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _blocks.Count; i++)
        {
            EvictBlock(i);
        }
        _cache.DrainAndReleasePool();
    }

    /// <summary>Drives NotUploaded/Evicted block to <see cref="BlockState.Uploading"/>.</summary>
    /// <remarks>No-op if already Uploading or Resident.</remarks>
    private void EnsureUploading(int idx)
    {
        BlockState state = _state[idx];
        if (state == BlockState.Uploading || state == BlockState.Resident)
        {
            return;
        }
        StreamingUploadToken token = _cache.BeginUploadAsync(_blocks[idx].EnumerateWeights());
        _tokens[idx] = token;
        // Even if the cache returned Empty (all weights were cached from a prior phase),
        // mark the block Uploading so the next EnsureResident call awaits (no-op on Empty)
        // and transitions to Resident — keeps the state machine consistent.
        _state[idx] = BlockState.Uploading;
    }

    /// <summary>Drives block to <see cref="BlockState.Resident"/>.</summary>
    /// <remarks>Cold-start path (NotUploaded/Evicted) issues BeginUpload + immediate Await — means prefetch wasn't deep enough.</remarks>
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
                EnsureUploading(idx);
                _cache.AwaitWeights(_tokens[idx]);
                _tokens[idx] = StreamingUploadToken.Empty;
                _state[idx] = BlockState.Resident;
                return;

            default:
                throw new InvalidOperationException($"Unknown block state {_state[idx]}.");
        }
    }

    /// <summary>Drives block to <see cref="BlockState.Evicted"/>; no-op for already-Evicted/NotUploaded.</summary>
    /// <remarks>Mid-upload blocks are awaited first — eviction implies the dptr exists, which only holds after the upload completes from
    /// the cache's perspective.</remarks>
    private void EvictBlock(int idx)
    {
        switch (_state[idx])
        {
            case BlockState.NotUploaded:
            case BlockState.Evicted:
                return;

            case BlockState.Uploading:
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
        // Best-effort flush during shutdown — log and continue rather than abort cleanup.
        for (int i = 0; i < _blocks.Count; i++)
        {
            try
            {
                EvictBlock(i);
            }
            catch (Exception ex)
            {
                Logs.Warning($"BlockStreamingController.Dispose: eviction of block {i} failed: {ex}");
            }
        }
    }

    private enum BlockState
    {
        /// <summary>No upload has been issued for this block; weights live only on host.</summary>
        NotUploaded = 0,

        /// <summary>Upload is in flight; <see cref="_tokens"/> holds the await token.</summary>
        Uploading = 1,

        /// <summary>Upload has been awaited; weights are usable on the compute stream.</summary>
        Resident = 2,

        /// <summary>Freed via <see cref="IStreamingWeightCache.EvictAsync"/>; re-upload goes back to <see cref="BlockState.NotUploaded"/>.</summary>
        Evicted = 3,
    }
}
