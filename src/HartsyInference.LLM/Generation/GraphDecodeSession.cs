using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Generation;

/// <summary>Per-sequence CUDA-graph decode state, captured once (see <see cref="DynamicBatchScheduler"/>'s
/// admission-time capture) and replayed for as long as that sequence stays the scheduler's sole active
/// sequence. Mirrors exactly what <see cref="TextGenerationPipeline"/>'s <c>GenerateGraphDecode</c> tracks as
/// method-locals for one whole generation — moved to fields here because a scheduler-owned session must
/// survive across many separate decode rounds (and potentially long gaps between them while other sequences
/// run), not just one method call. One-way: once a session is retired (the sequence stops being solo), it is
/// disposed and never resurrected — see the scheduler's retirement doc for why resuming a captured graph
/// after an eager interlude is unsafe.</summary>
internal sealed class GraphDecodeSession
{
    private readonly IBackend _backend;
    private object? _graph;
    private readonly ulong _devicePos;
    private readonly ulong _deviceTokenId;
    private readonly ulong _history;
    private readonly ulong _historyCount;
    private int _disposed;

    /// <summary>Absolute position of the token this session is about to generate next — mirrors
    /// <c>GenerateGraphDecode</c>'s local <c>pos</c>. Advanced by the scheduler after every replay.</summary>
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

    /// <summary>Replays the captured graph once and returns the newly-sampled token id. Does NOT advance
    /// <see cref="Pos"/> or write the next replay's device-side position — the caller does both (mirrors
    /// <c>GenerateGraphDecode</c>'s loop body exactly, so the scheduler's per-round dispatch can replicate its
    /// token-for-token behavior).</summary>
    public int Replay()
    {
        _backend.LaunchGraph(_graph!);
        return _backend.ReadDeviceTokenId(_deviceTokenId);
    }

    /// <summary>Writes the device-side position buffer ahead of the NEXT replay — mirrors
    /// <c>GenerateGraphDecode</c>'s per-step <c>WriteDevicePos(devicePos, pos + 1, pos)</c> call.</summary>
    public void WriteNextPos() => _backend.WriteDevicePos(_devicePos, Pos + 1, Pos);

    /// <summary>Frees the captured graph and every device buffer allocated for it — mirrors
    /// <c>GenerateGraphDecode</c>'s <c>finally</c> block exactly. Idempotent.</summary>
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
