using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>How the self-attention window of a chunk-causal LTX-2 video stream is bounded.</summary>
public sealed record LtxVideo2ChunkWindow
{
    /// <summary>Most frames the window holds, the sink included (upstream <c>kv_max_frames</c>); the sliding history holds
    /// what is left after the sink. A window no larger than the sink keeps no history.</summary>
    public int MaxFrames { get; init; } = 11;

    /// <summary>Every <c>HistoryLayerStride</c>-th block keeps the sliding history, counted from <see cref="HistoryLayerOffset"/>,
    /// and the last block always does; the others attend to the sink and their own chunk only. 1 keeps it on every block
    /// (upstream's default and the trained setup); a larger stride divides the history's memory by about that much and is an
    /// approximation (upstream <c>SANA_WM_REFINER_HISTORY_LAYER_STRIDE</c>).</summary>
    public int HistoryLayerStride { get; init; } = 1;

    /// <summary>The first block that keeps the history when <see cref="HistoryLayerStride"/> is above 1.</summary>
    public int HistoryLayerOffset { get; init; }

    /// <summary>Whether block <paramref name="layer"/> of <paramref name="layers"/> keeps the sliding history.</summary>
    public bool KeepsHistory(int layer, int layers)
    {
        if (HistoryLayerStride <= 1)
            return true;
        return layer == layers - 1 || ((layer - HistoryLayerOffset) % HistoryLayerStride + HistoryLayerStride) % HistoryLayerStride == 0;
    }
}

/// <summary>The attention state one block of a chunk-causal LTX-2 video stream carries: the sink frames' key and value
/// projections, kept for the whole stream, and a sliding window of the keys and values of the frames already produced.</summary>
/// <remarks>The sink key is kept as the raw <c>to_k</c> projection, before the QK norm and the rotation, because each chunk
/// re-rotates it to a new position just ahead of the window; the norm is deterministic, so norm-then-rotate at use equals
/// upstream's cached post-norm key rotated at use. History keys are kept after the rotation, at the absolute frames they
/// were produced at. Every tensor is token-major <c>[tokens, inner]</c> in the stream's activation dtype.</remarks>
internal sealed class LtxVideo2LayerStream
{
    /// <summary>Index of the key stream in <see cref="History"/>.</summary>
    public const int KeyStream = 0;

    /// <summary>Index of the value stream in <see cref="History"/>.</summary>
    public const int ValueStream = 1;

    public LtxVideo2LayerStream(IBackend backend, StreamingStateSet sink, StreamingStateSet history, int historyFrames)
    {
        SinkKey = sink.Add(new StreamingTensorSlot(backend));
        SinkValue = sink.Add(new StreamingTensorSlot(backend));
        if (historyFrames > 0)
            History = history.Add(new RollingTokenWindow(backend, TokenWindowPolicy.RecentUnits(historyFrames), streams: 2, tokenAxis: 0));
    }

    /// <summary>The sink frames' raw key projection.</summary>
    public StreamingTensorSlot SinkKey { get; }

    /// <summary>The sink frames' value projection.</summary>
    public StreamingTensorSlot SinkValue { get; }

    /// <summary>Rotated keys and values of the newest frames, or null when this block keeps no history.</summary>
    public RollingTokenWindow? History { get; }
}

/// <summary>The state a chunk-causal LTX-2 video stream carries between chunks, for every transformer block, and the
/// stream's position. The sink is captured once, before the first chunk; every later chunk reads the committed state on
/// each denoising pass, stages its own keys and values on the pass that closes it, and <see cref="Commit"/> advances every
/// block at once.</summary>
public sealed class LtxVideo2ChunkState : IDisposable
{
    private readonly StreamingStateSet _sink = new();
    private readonly StreamingStateSet _history = new();
    private readonly LtxVideo2LayerStream[] _layers;
    private int _disposed;

    /// <summary>Creates the empty state of a stream of <paramref name="tokensPerFrame"/> tokens per latent frame.</summary>
    /// <param name="sinkFrames">Leading frames kept as the permanent anchor; they are never refined.</param>
    public LtxVideo2ChunkState(IBackend backend, LtxVideo2Config config, int tokensPerFrame, int sinkFrames, LtxVideo2ChunkWindow window)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(window);
        ArgumentOutOfRangeException.ThrowIfLessThan(tokensPerFrame, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(sinkFrames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(window.MaxFrames, sinkFrames);
        ArgumentOutOfRangeException.ThrowIfLessThan(window.HistoryLayerStride, 1);
        TokensPerFrame = tokensPerFrame;
        SinkFrames = sinkFrames;
        MaxHistoryFrames = window.MaxFrames - sinkFrames;
        _layers = new LtxVideo2LayerStream[config.NumLayers];
        bool built = false;
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                bool keeps = MaxHistoryFrames > 0 && window.KeepsHistory(i, _layers.Length);
                _layers[i] = new LtxVideo2LayerStream(backend, _sink, _history, keeps ? MaxHistoryFrames : 0);
                if (keeps)
                    LastHistoryLayer = i;
            }

            built = true;
        }
        finally
        {
            if (!built)
            {
                _sink.Dispose();
                _history.Dispose();
            }
        }
    }

    /// <summary>Tokens of one latent frame.</summary>
    public int TokensPerFrame { get; }

    /// <summary>Frames of the permanent sink.</summary>
    public int SinkFrames { get; }

    /// <summary>Most frames the sliding history holds.</summary>
    public int MaxHistoryFrames { get; }

    /// <summary>The last block that keeps the sliding history, or -1 when none does: the pass that captures a chunk's keys
    /// and values stops there.</summary>
    public int LastHistoryLayer { get; } = -1;

    /// <summary>Whether the sink has been captured.</summary>
    public bool SinkCaptured { get; private set; }

    /// <summary>Frames produced and committed after the sink.</summary>
    public int CommittedFrames { get; private set; }

    /// <summary>Frames the sliding history holds now: the frames just before the next chunk.</summary>
    public int HistoryFrames => Math.Min(CommittedFrames, MaxHistoryFrames);

    /// <summary>The absolute latent frame of the next chunk's first frame.</summary>
    public int NextFrame => SinkFrames + CommittedFrames;

    /// <summary>Whether block <paramref name="layer"/> keeps the sliding history.</summary>
    public bool KeepsHistory(int layer) => Layer(layer).History is not null;

    /// <summary>The frame the sink is rotated to for the next chunk on a block that keeps the history: just ahead of the
    /// window (upstream's <c>sink_rope_offset</c>).</summary>
    public int SinkFrameBesideHistory => NextFrame - HistoryFrames - SinkFrames;

    /// <summary>The frame the sink is rotated to for the next chunk on a block without history: just ahead of the chunk.</summary>
    public int SinkFrameWithoutHistory => NextFrame - SinkFrames;

    /// <summary>Makes the sink staged by the capture pass current.</summary>
    public void CommitSink()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (SinkCaptured)
            throw new InvalidOperationException("The stream's sink is already captured.");
        foreach (LtxVideo2LayerStream layer in _layers)
        {
            if (!layer.SinkKey.HasStaged || !layer.SinkValue.HasStaged)
                throw new InvalidOperationException("The sink capture pass did not reach every block.");
        }

        _sink.Commit();
        SinkCaptured = true;
    }

    /// <summary>Makes the keys and values staged by a chunk of <paramref name="frames"/> latent frames current and slides the
    /// window.</summary>
    public void Commit(int frames)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        _history.Commit();
        CommittedFrames = checked(CommittedFrames + frames);
    }

    /// <summary>Drops what a failed or abandoned pass staged; the committed state is unchanged.</summary>
    public void Discard()
    {
        _history.Discard();
        _sink.Discard();
    }

    /// <summary>Returns to the state of a stream that has not started.</summary>
    public void Reset()
    {
        _history.Reset();
        _sink.Reset();
        SinkCaptured = false;
        CommittedFrames = 0;
    }

    /// <summary>Bytes the committed state reaches once the window is full, with every tensor in <paramref name="dtype"/>: the
    /// sink's raw key and value plus the history's key and value, per block.</summary>
    public static long EstimateBytes(LtxVideo2Config config, int tokensPerFrame, int sinkFrames, LtxVideo2ChunkWindow window, DType dtype)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(window);
        long rowBytes = dtype.ComputeByteCount(config.InnerDim);
        long sink = 2L * sinkFrames * tokensPerFrame * rowBytes * config.NumLayers;
        int historyFrames = Math.Max(0, window.MaxFrames - sinkFrames);
        int historyLayers = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (historyFrames > 0 && window.KeepsHistory(i, config.NumLayers))
                historyLayers++;
        }

        return sink + 2L * historyFrames * tokensPerFrame * rowBytes * historyLayers;
    }

    internal LtxVideo2LayerStream Layer(int index) => _layers[index];

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _history.Dispose();
        _sink.Dispose();
    }
}
