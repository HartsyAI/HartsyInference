using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Generation;

/// <summary>Supplies the embedding row a graph step reads for a token, when the backend stages that row from the host (<see cref="IBackend.StagesDecodeEmbedRows"/>).</summary>
internal interface IGraphDecodeRowSource
{
    /// <summary>Pointer to the F32 embedding row for <paramref name="tokenId"/>: the values the eager lookup produces, valid until the next call.</summary>
    nint GraphDecodeEmbedRow(int tokenId);
}

/// <summary>Per-sequence CUDA-graph decode state, captured once and replayed for as long as that sequence stays the scheduler's sole active sequence; one-way — once retired (the sequence stops being solo) it is disposed and never resurrected, since resuming a captured graph after an eager interlude is unsafe.</summary>
internal sealed class GraphDecodeSession
{
    private readonly IBackend _backend;
    private object? _graph;
    private readonly ulong _devicePos;
    private readonly ulong _deviceTokenId;
    private readonly ulong _history;
    private readonly ulong _historyCount;
    private readonly ulong _rng;
    private readonly ulong _embedRow;
    private readonly int _embedWidth;
    private readonly IGraphDecodeRowSource? _rowSource;
    private int _tokenToEmbed;
    private int _disposed;

    /// <summary>Absolute position of the token this session is about to generate next; advanced by the scheduler after every replay.</summary>
    public int Pos { get; set; }

    /// <param name="embedRow">The host-staged embedding row (<see cref="IBackend.AllocDecodeEmbedRow"/>), or 0 when the step embeds on device.</param>
    /// <param name="embedWidth">Values in the staged row (the hidden size); unused when <paramref name="rowSource"/> is null.</param>
    /// <param name="firstToken">The token the first replay embeds; each later replay embeds the token the previous replay sampled.</param>
    /// <param name="rowSource">Supplies each step's row; null when the step embeds on device.</param>
    public GraphDecodeSession(IBackend backend, object graph, ulong devicePos, ulong deviceTokenId,
        ulong history, ulong historyCount, int pos, ulong rng = 0,
        ulong embedRow = 0, int embedWidth = 0, int firstToken = 0, IGraphDecodeRowSource? rowSource = null)
    {
        _backend = backend;
        _graph = graph;
        _devicePos = devicePos;
        _deviceTokenId = deviceTokenId;
        _history = history;
        _historyCount = historyCount;
        _rng = rng;
        _embedRow = embedRow;
        _embedWidth = embedWidth;
        _rowSource = rowSource;
        _tokenToEmbed = firstToken;
        Pos = pos;
    }

    /// <summary>Replays the captured graph once and returns the newly-sampled token id; does NOT advance <see cref="Pos"/> or write the next replay's device-side position — the caller does both.</summary>
    public int Replay()
    {
        // A host-staged step embeds the row written here, before the launch. The previous replay has completed (LaunchGraph
        // waits on its fence), so the row is not being read while it is written.
        if (_rowSource is not null)
            _backend.WriteDecodeEmbedRow(_embedRow, _rowSource.GraphDecodeEmbedRow(_tokenToEmbed), _embedWidth);
        _backend.LaunchGraph(_graph!);
        _tokenToEmbed = _backend.ReadDeviceTokenId(_deviceTokenId);
        return _tokenToEmbed;
    }

    /// <summary>Writes the device-side position buffer ahead of the NEXT replay.</summary>
    public void WriteNextPos() => _backend.WriteDevicePos(_devicePos, Pos + 1, Pos);

    /// <summary>Frees the captured graph and every device buffer allocated for it; idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        if (_graph is not null) { _backend.DisposeGraph(_graph); _graph = null; }
        _backend.FreeDevicePos(_devicePos);
        _backend.FreeDeviceTokenId(_deviceTokenId);
        _backend.FreeDeviceHistory(_history);
        _backend.FreeDeviceCounter(_historyCount);
        _backend.FreeDeviceRng(_rng);
        _backend.FreeDecodeEmbedRow(_embedRow);
    }
}
