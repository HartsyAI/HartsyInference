namespace SharpInference.Audio.Streaming;

/// <summary>Lock-protected circular PCM buffer. Producers push freshly-captured samples
/// (microphone callback, decoded file chunk) via <see cref="Write"/>; consumers
/// <see cref="Peek"/> a sliding window without advancing, then <see cref="Discard"/> the
/// hop step between successive STFT frames.
///
/// <para>Used as the input side of streaming STT (Whisper / Parakeet / Moonshine
/// streaming) and any voice-conversion path with live microphone input. For streaming
/// TTS the audio flows the OTHER way and uses <see cref="AudioStreamer"/> instead.</para>
///
/// <para>The buffer drops the oldest samples when <see cref="Write"/> would overflow.
/// That's the right default for STT — falling behind real-time means we're losing
/// audio, and silently retaining stale audio would extend the latency unboundedly.
/// Callers that need a stricter policy can check <see cref="FreeSpace"/> before writing
/// and apply their own backpressure.</para>
///
/// <para>All positions are absolute sample counts since construction (or last
/// <see cref="Reset"/>) — that's what STT pipelines use to time-stamp transcripts. The
/// ring index itself is not exposed.</para></summary>
public sealed class AudioRingBuffer
{
    private readonly float[] _buffer;
    private readonly int _capacity;
    private readonly object _lock = new();
    private int _writeIdx;
    private int _available;
    private long _readPosition;
    private long _writePosition;
    private long _samplesDropped;

    /// <summary>Creates a buffer that can hold <paramref name="capacity"/> samples
    /// before old data starts getting overwritten.</summary>
    public AudioRingBuffer(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "capacity must be > 0");
        _capacity = capacity;
        _buffer = new float[capacity];
    }

    public int Capacity => _capacity;

    /// <summary>Samples currently available to read.</summary>
    public int Available { get { lock (_lock) return _available; } }

    /// <summary>Free space remaining before <see cref="Write"/> starts dropping the
    /// oldest samples.</summary>
    public int FreeSpace { get { lock (_lock) return _capacity - _available; } }

    /// <summary>Absolute sample index the next <see cref="Read"/> / <see cref="Peek"/>
    /// (with offset 0) will return. Advances on <see cref="Read"/>, <see cref="Discard"/>,
    /// and when <see cref="Write"/> overflows.</summary>
    public long ReadPosition { get { lock (_lock) return _readPosition; } }

    /// <summary>Absolute count of samples ever written, including dropped ones.</summary>
    public long WritePosition { get { lock (_lock) return _writePosition; } }

    /// <summary>Total samples lost to overflow. A non-zero value means the consumer
    /// is falling behind real-time — STT pipelines should surface this to the user.</summary>
    public long SamplesDropped { get { lock (_lock) return _samplesDropped; } }

    /// <summary>Appends <paramref name="samples"/> to the buffer. If this would exceed
    /// <see cref="Capacity"/>, the oldest queued samples are evicted to make room — the
    /// new samples always fit. Returns the number of samples evicted, which equals the
    /// increment of <see cref="SamplesDropped"/>.</summary>
    public int Write(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty) return 0;
        if (samples.Length > _capacity)
        {
            // Caller is writing more than fits — keep only the trailing capacity worth.
            samples = samples[^_capacity..];
        }
        lock (_lock)
        {
            int incoming = samples.Length;
            int overflow = Math.Max(0, _available + incoming - _capacity);
            if (overflow > 0)
            {
                _readPosition += overflow;
                _samplesDropped += overflow;
                _available -= overflow;
            }

            // Copy in two phases if we'd wrap around the end of the underlying array.
            int firstSpan = Math.Min(incoming, _capacity - _writeIdx);
            samples[..firstSpan].CopyTo(_buffer.AsSpan(_writeIdx, firstSpan));
            int remaining = incoming - firstSpan;
            if (remaining > 0)
            {
                samples.Slice(firstSpan, remaining).CopyTo(_buffer.AsSpan(0, remaining));
            }

            _writeIdx = (_writeIdx + incoming) % _capacity;
            _available += incoming;
            _writePosition += incoming;
            return overflow;
        }
    }

    /// <summary>Copies up to <paramref name="output"/>.Length samples into the output
    /// span without advancing the read position. <paramref name="offset"/> is measured
    /// from the oldest available sample (i.e. the next sample <see cref="Read"/> would
    /// return). Returns the number actually copied — fewer than requested when the
    /// requested range extends past <see cref="Available"/>.</summary>
    public int Peek(Span<float> output, int offset = 0)
    {
        if (output.IsEmpty) return 0;
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset), offset, "offset must be >= 0");
        lock (_lock)
        {
            int startInRing = offset >= _available ? -1 : offset;
            if (startInRing < 0) return 0;
            int copyLen = Math.Min(output.Length, _available - startInRing);
            int readIdx = (_writeIdx - _available + startInRing + _capacity) % _capacity;
            int firstSpan = Math.Min(copyLen, _capacity - readIdx);
            _buffer.AsSpan(readIdx, firstSpan).CopyTo(output);
            int remaining = copyLen - firstSpan;
            if (remaining > 0)
            {
                _buffer.AsSpan(0, remaining).CopyTo(output[firstSpan..]);
            }
            return copyLen;
        }
    }

    /// <summary>Copies and consumes up to <paramref name="output"/>.Length samples.
    /// Returns the number actually moved.</summary>
    public int Read(Span<float> output)
    {
        int copied = Peek(output);
        if (copied > 0) Discard(copied);
        return copied;
    }

    /// <summary>Advances the read position by <paramref name="sampleCount"/> samples
    /// without copying. Caps at <see cref="Available"/>.</summary>
    public void Discard(int sampleCount)
    {
        if (sampleCount <= 0) return;
        lock (_lock)
        {
            int drop = Math.Min(sampleCount, _available);
            _available -= drop;
            _readPosition += drop;
        }
    }

    /// <summary>Empties the buffer and clears all counters. The underlying storage is
    /// retained.</summary>
    public void Reset()
    {
        lock (_lock)
        {
            _writeIdx = 0;
            _available = 0;
            _readPosition = 0;
            _writePosition = 0;
            _samplesDropped = 0;
        }
    }
}
