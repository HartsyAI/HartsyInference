using System.Threading.Channels;

namespace SharpInference.Audio.Streaming;

/// <summary>Producer/consumer surface that bridges a synchronous inference loop (running
/// on a background thread) to an async <see cref="IAsyncEnumerable{AudioChunk}"/> stream.
///
/// <para>The producer (typically the pipeline's generation worker) calls
/// <see cref="Put"/> for each emitted audio chunk and <see cref="Complete"/> once
/// generation has finished. The consumer iterates <see cref="ReadAllAsync"/> from any
/// async context (Server SSE handler, CLI streaming demo, an Electron renderer that polls
/// it, etc.) and is automatically backpressured when the channel fills.</para>
///
/// <para>Mirrors dotLLM's streaming output convention so audio + LLM streams compose
/// uniformly. The bounded-channel capacity defaults to 64 chunks (~8.5 s at 7.5 Hz frame
/// rate, plenty of head-room for transient consumer stalls without blocking the
/// generation thread for long).</para></summary>
public sealed class AudioStreamer : IDisposable
{
    private readonly Channel<AudioChunk> _channel;
    private int _disposed;

    /// <summary>Creates a streamer backed by a bounded channel. <paramref name="capacity"/>
    /// controls how many chunks can be queued before <see cref="Put"/> blocks.</summary>
    public AudioStreamer(int capacity = 64)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "capacity must be >= 1");
        _channel = Channel.CreateBounded<AudioChunk>(new BoundedChannelOptions(capacity)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });
    }

    /// <summary>Enqueues a chunk. Called from the inference worker thread. Returns when
    /// the chunk has been accepted by the channel (may block briefly if the consumer is
    /// behind by <c>capacity</c> chunks). Throws if the streamer has already been
    /// completed or disposed.</summary>
    public ValueTask Put(AudioChunk chunk, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _channel.Writer.WriteAsync(chunk, cancellationToken);
    }

    /// <summary>Signals that no further chunks will be produced. Idempotent — repeat calls
    /// are no-ops. After this, <see cref="ReadAllAsync"/> will drain remaining chunks and
    /// then complete.</summary>
    public void Complete() => _channel.Writer.TryComplete();

    /// <summary>Async consumer surface. Yields chunks as they become available; completes
    /// when the producer calls <see cref="Complete"/>.</summary>
    public IAsyncEnumerable<AudioChunk> ReadAllAsync(CancellationToken cancellationToken = default)
        => _channel.Reader.ReadAllAsync(cancellationToken);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _channel.Writer.TryComplete();
    }
}
