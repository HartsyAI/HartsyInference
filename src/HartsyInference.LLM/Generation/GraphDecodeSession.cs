using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Generation;

/// <summary>Per-sequence CUDA-graph decode state, captured once and replayed for as long as that sequence stays the scheduler's sole active sequence; one-way — once retired (the sequence stops being solo) it is disposed and never resurrected, since resuming a captured graph after an eager interlude is unsafe.</summary>
internal sealed class GraphDecodeSession
{
    private readonly IBackend _backend;
    private object? _graph;
    private readonly ulong _devicePos;
    private readonly ulong _deviceTokenId;
    private readonly ulong _history;
    private readonly ulong _historyCount;
    private int _disposed;

    /// <summary>Absolute position of the token this session is about to generate next; advanced by the scheduler after every replay.</summary>
    public int Pos { get; set; }

    public GraphDecodeSession(IBackend backend, object graph, ulong devicePos, ulong deviceTokenId,
        ulong history, ulong historyCount, int pos)
    {
        _backend = backend;
        _graph = graph;
        _devicePos = devicePos;
        _deviceTokenId = deviceTokenId;
        _history = history;
        _historyCount = historyCount;
        Pos = pos;
    }

    /// <summary>Replays the captured graph once and returns the newly-sampled token id; does NOT advance <see cref="Pos"/> or write the next replay's device-side position — the caller does both.</summary>
    public int Replay()
    {
        _backend.LaunchGraph(_graph!);
        return _backend.ReadDeviceTokenId(_deviceTokenId);
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
    }
}
