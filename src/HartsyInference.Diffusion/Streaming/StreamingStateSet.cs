namespace HartsyInference.Diffusion.Streaming;

/// <summary>Every <see cref="IStreamingState"/> of one stream, committed, discarded and reset together. A model registers the
/// states of its layers here and the sampler drives the set, so a chunk either advances all of them or none.</summary>
public sealed class StreamingStateSet : IStreamingState
{
    private readonly List<IStreamingState> _states = [];
    private int _disposed;

    /// <summary>Chunks committed since the stream started or was last reset.</summary>
    public int CommittedChunks { get; private set; }

    /// <summary>Registers <paramref name="state"/>, which the set now owns, and returns it.</summary>
    public T Add<T>(T state)
        where T : IStreamingState
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _states.Add(state);
        return state;
    }

    /// <inheritdoc />
    public void Commit()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        foreach (IStreamingState state in _states)
            state.Commit();
        CommittedChunks++;
    }

    /// <inheritdoc />
    public void Discard()
    {
        foreach (IStreamingState state in _states)
            state.Discard();
    }

    /// <inheritdoc />
    public void Reset()
    {
        foreach (IStreamingState state in _states)
            state.Reset();
        CommittedChunks = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (IStreamingState state in _states)
            state.Dispose();
        _states.Clear();
    }
}
