using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>The attention context a chunk-causal layer keeps from earlier chunks: one or more parallel token streams (keys
/// and values, say), appended a chunk at a time and trimmed by a <see cref="TokenWindowPolicy"/> to a permanent sink plus a
/// sliding history. <see cref="CopyParts"/> lists the context a chunk attends to, sink first, then the history oldest first;
/// the layer joins those parts with its own chunk in one concatenation, so the window never holds a second, joined copy.</summary>
/// <remarks>Tokens run along <c>tokenAxis</c> of every stream; a chunk's token count divided by its units is the tokens of
/// one unit (a frame). The window owns what it holds and pins it on the backend. Appending is staged: a chunk joins the
/// window at <see cref="Commit"/>. With a <see cref="StorageDType"/> the window keeps its streams narrower than they arrive
/// (F32 activations stored as BF16, say): each chunk is converted as it is staged, and <see cref="CopyParts"/> hands back the
/// stored parts, which the caller widens when it joins them.</remarks>
public sealed class RollingTokenWindow : IStreamingState
{
    private readonly IBackend _backend;
    private readonly TokenWindowPolicy _policy;
    private readonly int _tokenAxis;
    private readonly DType? _storage;
    private readonly List<Entry> _sink = [];
    private readonly List<Entry> _history = [];
    private Entry? _staged;
    private int _disposed;

    /// <summary>Creates an empty window of <paramref name="streams"/> parallel streams.</summary>
    /// <param name="storageDType">The dtype every staged stream is converted to and held in; null keeps streams as staged.</param>
    public RollingTokenWindow(IBackend backend, TokenWindowPolicy policy, int streams, int tokenAxis, DType? storageDType = null)
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
        _storage = storageDType;
    }

    /// <summary>The dtype the window holds its streams in, or null when it holds them as staged.</summary>
    public DType? StorageDType => _storage;

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

        try
        {
            Narrow(owned);
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

    // Converts each stream to the storage dtype in place in the array, one at a time, releasing the original as soon as its
    // copy exists so the narrower copy never coexists with more than one full-width stream.
    private void Narrow(Tensor[] streams)
    {
        if (_storage is not DType storage)
            return;
        for (int i = 0; i < streams.Length; i++)
        {
            if (streams[i].DType == storage)
                continue;
            Tensor stored = StreamingTensors.CastTo(_backend, streams[i], storage);
            streams[i].Dispose();
            streams[i] = stored;
        }
    }

    /// <summary>Committed chunks in the window, sink and history together: the parts <see cref="CopyParts"/> writes.</summary>
    public int PartCount => _sink.Count + _history.Count;

    /// <summary>The axis tokens run along in every stream.</summary>
    public int TokenAxis => _tokenAxis;

    /// <summary>Writes stream <paramref name="stream"/> of every committed chunk into <paramref name="destination"/>, sink first,
    /// then the history oldest first, and returns how many were written. The tensors are borrowed: valid until the next commit
    /// or reset.</summary>
    public int CopyParts(int stream, Span<Tensor> destination)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (stream < 0 || stream >= Streams)
            throw new ArgumentOutOfRangeException(nameof(stream), stream, $"The window holds {Streams} streams.");
        if (destination.Length < PartCount)
            throw new ArgumentException($"The window holds {PartCount} chunks; the destination has room for {destination.Length}.",
                nameof(destination));
        int at = 0;
        foreach (Entry entry in _sink)
            destination[at++] = entry.Streams[stream];
        foreach (Entry entry in _history)
            destination[at++] = entry.Streams[stream];
        return at;
    }

    /// <inheritdoc />
    public void Commit()
    {
        if (_staged is not Entry staged)
            return;
        _staged = null;
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
