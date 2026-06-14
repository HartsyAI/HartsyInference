using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Drives a sliding window of resident blocks across a sequential model's forward pass: prefetches blocks ahead of compute, awaits them just-in-time, and evicts blocks that have fallen behind the retain window. HartsyInference equivalent of ComfyUI's "lowvram" model-streaming. Single-threaded; call from the inference thread that owns the backend's CUDA context.</summary>
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

        /// <summary>Memory has been freed via <see cref="IStreamingWeightCache.EvictAsync"/>. Re-uploading goes back through <see cref="BlockState.NotUploaded"/>.</summary>
        Evicted = 3,
    }

    private readonly IStreamingWeightCache _cache;
    private readonly IReadOnlyList<IStreamingBlock> _blocks;
    private readonly int _prefetchAhead;
    private readonly int _retainBehind;

    private readonly BlockState[] _state;
    private readonly StreamingUploadToken[] _tokens;

    private bool _disposed;

    /// <summary>Constructs a streaming controller. <paramref name="prefetchAhead"/> is the in-flight depth (0 = synchronous, 1 = typical overlap). <paramref name="retainBehind"/> keeps already-used blocks cached for reuse (0 = evict immediately).</summary>
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

    /// <summary>Begins async uploads for blocks <c>[0, prefetchAhead]</c> so the first <see cref="BeforeBlockForward"/> call doesn't pay a cold-start synchronous load. Idempotent.</summary>
    public void Prime()
    {
        ThrowIfDisposed();
        for (int i = 0; i <= _prefetchAhead && i < _blocks.Count; i++)
        {
            EnsureUploading(i);
        }
    }

    /// <summary>Call IMMEDIATELY before block <paramref name="blockIdx"/>'s forward pass. Awaits block <paramref name="blockIdx"/>, kicks off prefetch for <c>blockIdx + prefetchAhead</c>, and evicts <c>blockIdx - retainBehind - 1</c> if outside the retain window. Indexes outside <c>[0, BlockCount)</c> are no-ops.</summary>
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

    /// <summary>Frees every block that's still resident, then drains in-flight async ops and releases backend pool reservations to the driver. Call between streaming and eager-allocation phases — CUDA's stream-ordered allocator pool is invisible to subsequent sync allocations unless explicitly trimmed. Safe to call multiple times.</summary>
    public void EvictAll()
    {
        ThrowIfDisposed();
        for (int i = 0; i < _blocks.Count; i++)
        {
            EvictBlock(i);
        }
        _cache.DrainAndReleasePool();
    }

    /// <summary>Drives block to <see cref="BlockState.Uploading"/> if it's currently <see cref="BlockState.NotUploaded"/> or <see cref="BlockState.Evicted"/>. No-op for blocks already <see cref="BlockState.Uploading"/> or <see cref="BlockState.Resident"/>.</summary>
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

    /// <summary>Drives block to <see cref="BlockState.Resident"/>. Cold-start path (NotUploaded/Evicted) issues BeginUpload + immediate Await — slow path that means prefetch wasn't deep enough.</summary>
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

    /// <summary>Drives block to <see cref="BlockState.Evicted"/>. No-op for already-Evicted/NotUploaded. Mid-upload blocks are awaited first — eviction implies the dptr exists, which only holds after the upload completes from the cache's perspective.</summary>
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
}
