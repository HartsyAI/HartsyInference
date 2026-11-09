using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>The attention context a chunk-causal layer keeps from earlier chunks: one or more parallel token streams (keys
/// and values, say), appended a chunk at a time and trimmed by a <see cref="TokenWindowPolicy"/> to a permanent sink plus a
/// sliding history. <see cref="Joined"/> is the context a chunk attends to, sink first, then the history oldest first.</summary>
/// <remarks>Tokens run along <c>tokenAxis</c> of every stream; a chunk's token count divided by its units is the tokens of
/// one unit (a frame). The window owns what it holds and pins it on the backend. Appending is staged: a chunk joins the
/// window at <see cref="Commit"/>.</remarks>
public sealed class RollingTokenWindow : IStreamingState
{
    private readonly IBackend _backend;
    private readonly TokenWindowPolicy _policy;
    private readonly int _tokenAxis;
    private readonly List<Entry> _sink = [];
    private readonly List<Entry> _history = [];
    private Entry? _staged;
    private Tensor[]? _joined;
    private bool _joinedBorrowed;
    private int _disposed;

    /// <summary>Creates an empty window of <paramref name="streams"/> parallel streams.</summary>
    public RollingTokenWindow(IBackend backend, TokenWindowPolicy policy, int streams, int tokenAxis)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(policy);
        ArgumentOutOfRangeException.ThrowIfLessThan(streams, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(tokenAxis);
        if (policy.SinkChunks < 0 || policy.MaxHistoryChunks < 0 || policy.MaxHistoryUnits < 0)
            throw new ArgumentException("A token window keeps a non-negative number of chunks and units.", nameof(policy));
        _backend = backend;
        _policy = policy;
        Streams = streams;
        _tokenAxis = tokenAxis;
    }

    /// <summary>Parallel streams per chunk.</summary>
    public int Streams { get; }

    /// <summary>Whether the committed window holds nothing.</summary>
    public bool IsEmpty => _sink.Count == 0 && _history.Count == 0;

    /// <summary>Units held by the sink.</summary>
    public int SinkUnits => Units(_sink);

    /// <summary>Units held by the sliding history.</summary>
    public int HistoryUnits => Units(_history);

    /// <summary>Tokens of the committed window along the token axis.</summary>
    public long Tokens => TokensOf(_sink) + TokensOf(_history);

    /// <summary>Stages a chunk of <paramref name="units"/> units, one tensor per stream. The window takes ownership of the
    /// tensors, also when it throws; an earlier staged chunk is released.</summary>
    public void Stage(ReadOnlySpan<Tensor> streams, int units)
    {
        Tensor[] owned = streams.ToArray();
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ArgumentOutOfRangeException.ThrowIfLessThan(units, 1);
            if (owned.Length != Streams)
                throw new ArgumentException($"The window holds {Streams} streams; {owned.Length} were given.", nameof(streams));
            long tokens = -1;
            foreach (Tensor stream in owned)
            {
                ArgumentNullException.ThrowIfNull(stream);
                if (_tokenAxis >= stream.Shape.Rank)
                    throw new ArgumentException($"{stream.Shape} has no token axis {_tokenAxis}.", nameof(streams));
                long count = stream.Shape[_tokenAxis];
                if (count == 0 || count % units != 0 || (tokens >= 0 && count != tokens))
                    throw new ArgumentException($"Every stream must hold the same whole number of tokens per unit; got {stream.Shape} "
                        + $"for {units} units.", nameof(streams));
                tokens = count;
            }
        }
        catch
        {
            foreach (Tensor? stream in owned)
                stream?.Dispose();
            throw;
        }

        Discard();
        foreach (Tensor stream in owned)
            _backend.PinActivation(stream);
        _staged = new Entry(owned, units);
    }

    /// <summary>Stream <paramref name="stream"/> of the committed window joined along the token axis, or null while the window
    /// is empty. Borrowed: valid until the next commit or reset.</summary>
    public Tensor? Joined(int stream)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (stream < 0 || stream >= Streams)
            throw new ArgumentOutOfRangeException(nameof(stream), stream, $"The window holds {Streams} streams.");
        if (IsEmpty)
            return null;
        _joined ??= BuildJoined();
        return _joined[stream];
    }

    /// <inheritdoc />
    public void Commit()
    {
        if (_staged is not Entry staged)
            return;
        _staged = null;
        ReleaseJoined();
        if (_sink.Count < _policy.SinkChunks)
        {
            _sink.Add(staged);
            return;
        }

        _history.Add(staged);
        Trim();
    }

    /// <inheritdoc />
    public void Discard()
    {
        if (_staged is not Entry staged)
            return;
        _staged = null;
        Release(staged);
    }

    /// <inheritdoc />
    public void Reset()
    {
        Discard();
        ReleaseJoined();
        foreach (Entry entry in _sink)
            Release(entry);
        foreach (Entry entry in _history)
            Release(entry);
        _sink.Clear();
        _history.Clear();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Reset();
    }

    // Drops the oldest history until both bounds hold; a chunk that only partly fits the unit bound keeps its newest units.
    private void Trim()
    {
        if (_policy.MaxHistoryChunks is int maxChunks)
        {
            while (_history.Count > maxChunks)
                DropOldest();
        }

        if (_policy.MaxHistoryUnits is not int maxUnits)
            return;
        int excess = Units(_history) - maxUnits;
        while (excess > 0 && _history.Count > 0)
        {
            Entry oldest = _history[0];
            if (oldest.Units <= excess)
            {
                excess -= oldest.Units;
                DropOldest();
                continue;
            }

            int kept = oldest.Units - excess;
            Tensor[] cut = new Tensor[Streams];
            try
            {
                for (int s = 0; s < Streams; s++)
                {
                    long perUnit = oldest.Streams[s].Shape[_tokenAxis] / oldest.Units;
                    cut[s] = StreamingTensors.Slice(_backend, oldest.Streams[s], _tokenAxis, excess * perUnit, kept * perUnit);
                    _backend.PinActivation(cut[s]);
                }
            }
            catch
            {
                foreach (Tensor? tensor in cut)
                    tensor?.Dispose();
                throw;
            }

            Release(oldest);
            _history[0] = new Entry(cut, kept);
            excess = 0;
        }
    }

    private void DropOldest()
    {
        Release(_history[0]);
        _history.RemoveAt(0);
    }

    // A window of one chunk is that chunk's own tensors; anything longer is a joined copy the window owns.
    private Tensor[] BuildJoined()
    {
        int entries = _sink.Count + _history.Count;
        if (entries == 1)
        {
            _joinedBorrowed = true;
            return _sink.Count == 1 ? _sink[0].Streams : _history[0].Streams;
        }

        _joinedBorrowed = false;
        Tensor[] joined = new Tensor[Streams];
        Tensor[] parts = new Tensor[entries];
        try
        {
            for (int s = 0; s < Streams; s++)
            {
                int at = 0;
                foreach (Entry entry in _sink)
                    parts[at++] = entry.Streams[s];
                foreach (Entry entry in _history)
                    parts[at++] = entry.Streams[s];
                joined[s] = StreamingTensors.Concat(_backend, parts, _tokenAxis);
                _backend.PinActivation(joined[s]);
            }

            return joined;
        }
        catch
        {
            foreach (Tensor? tensor in joined)
                tensor?.Dispose();
            throw;
        }
    }

    private void ReleaseJoined()
    {
        if (_joined is null)
            return;
        if (!_joinedBorrowed)
        {
            foreach (Tensor tensor in _joined)
            {
                _backend.UnpinActivation(tensor);
                tensor.Dispose();
            }
        }

        _joined = null;
    }

    private void Release(Entry entry)
    {
        foreach (Tensor tensor in entry.Streams)
        {
            _backend.UnpinActivation(tensor);
            tensor.Dispose();
        }
    }

    private static int Units(List<Entry> entries)
    {
        int units = 0;
        foreach (Entry entry in entries)
            units += entry.Units;
        return units;
    }

    private long TokensOf(List<Entry> entries)
    {
        long tokens = 0;
        foreach (Entry entry in entries)
            tokens += entry.Streams[0].Shape[_tokenAxis];
        return tokens;
    }

    private readonly record struct Entry(Tensor[] Streams, int Units);
}
