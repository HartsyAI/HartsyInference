using HartsyInference.Core.Backends;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>The left context of every causal operator in a network that is run a chunk at a time, such as the causal
/// convolutions of a video VAE decoder (the <c>feat_cache</c> and <c>feat_idx</c> pair of the reference implementations).
/// Operators are identified by the order the forward pass reaches them, so a pass calls <see cref="BeginPass"/> and then asks
/// for each operator's state in the same order every time; the states are created on the first pass.</summary>
public sealed class CausalStreamCache : IStreamingState
{
    private readonly IBackend _backend;
    private readonly int _frameAxis;
    private readonly List<FrameTailCache> _tails = [];
    private readonly List<bool> _used = [];
    private readonly List<int> _stagedUses = [];
    private int _tailIndex;
    private int _flagIndex;
    private int _disposed;

    /// <summary>Creates an empty cache for activations whose frames run along <paramref name="frameAxis"/>.</summary>
    public CausalStreamCache(IBackend backend, int frameAxis)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegative(frameAxis);
        _backend = backend;
        _frameAxis = frameAxis;
    }

    /// <summary>Passes committed since the stream started or was last reset.</summary>
    public int CommittedPasses { get; private set; }

    /// <summary>Rewinds to the first operator; call at the start of every forward pass.</summary>
    public void BeginPass()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _tailIndex = 0;
        _flagIndex = 0;
    }

    /// <summary>The input tail of the next causal operator in traversal order, holding up to <paramref name="tailFrames"/>
    /// frames.</summary>
    /// <exception cref="InvalidOperationException">The operator at this position asked for a different tail length on an
    /// earlier pass: the traversal order changed.</exception>
    public FrameTailCache NextTail(int tailFrames)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int index = _tailIndex++;
        if (index == _tails.Count)
            _tails.Add(new FrameTailCache(_backend, tailFrames, _frameAxis));
        FrameTailCache tail = _tails[index];
        if (tail.TailFrames != tailFrames)
            throw new InvalidOperationException($"Causal operator {index} kept {tail.TailFrames} frames before and asks for "
                + $"{tailFrames} now; the pass visits its operators in a different order.");
        return tail;
    }

    /// <summary>Whether the next flagged operator in traversal order runs for the first time in this stream. The use is staged,
    /// so the answer changes only once the pass commits.</summary>
    public bool NextFirstUse()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int index = _flagIndex++;
        if (index == _used.Count)
            _used.Add(false);
        if (_used[index])
            return false;
        _stagedUses.Add(index);
        return true;
    }

    /// <inheritdoc />
    public void Commit()
    {
        foreach (FrameTailCache tail in _tails)
            tail.Commit();
        foreach (int index in _stagedUses)
            _used[index] = true;
        _stagedUses.Clear();
        CommittedPasses++;
    }

    /// <inheritdoc />
    public void Discard()
    {
        foreach (FrameTailCache tail in _tails)
            tail.Discard();
        _stagedUses.Clear();
    }

    /// <inheritdoc />
    public void Reset()
    {
        foreach (FrameTailCache tail in _tails)
            tail.Dispose();
        _tails.Clear();
        _used.Clear();
        _stagedUses.Clear();
        CommittedPasses = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Reset();
    }
}
