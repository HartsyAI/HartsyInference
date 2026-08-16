using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>What a session is currently doing. Post-trigger capture is tracked here so a detection can hand
/// the utterance to transcription without the worker holding any lock while that runs.</summary>
public enum WakeSessionState
{
    /// <summary>Connected but no <c>hello</c> received yet.</summary>
    Handshake,
    /// <summary>Streaming audio and scoring wake words.</summary>
    Listening,
    /// <summary>A word fired; buffering the command utterance that follows it.</summary>
    Capturing,
}

/// <summary>One connected satellite. Owns that device's audio buffer and detection pipeline, and is keyed by
/// device id rather than by connection so a reconnect resumes the same configuration.
///
/// <para>The socket loop writes audio in and the wake worker reads it out — a single-producer/single-consumer
/// handoff through <see cref="AudioRingBuffer"/>, which drops oldest on overflow. That is the correct failure
/// mode for an always-on listener: if inference ever falls behind, losing the oldest audio degrades detection
/// briefly, whereas blocking the socket would stall the device and cascade into its reconnect logic.</para></summary>
public sealed class WakeSession : IDisposable
{
    /// <summary>Audio buffered between the socket and the worker (2 s). Sized to absorb scheduling jitter and
    /// short inference stalls without becoming a latency reservoir.</summary>
    public const int RingCapacitySamples = 32_000;

    /// <summary>Rolling raw audio retained for post-trigger use — transcription and speaker identification both
    /// need the audio from before the word fired, not just after.</summary>
    public const int CaptureCapacitySamples = 16_000 * 15;

    private readonly AudioRingBuffer _ring = new(RingCapacitySamples);
    private readonly float[] _capture = new float[CaptureCapacitySamples];
    private readonly object _captureLock = new();
    private long _captureWritten;
    private long _expectedSequence = -1;
    private int _disposed;

    /// <summary>Stable identity from the client's <c>hello</c>; survives reconnects.</summary>
    public string DeviceId { get; }

    /// <summary>The device's detection pipeline. Touched only by the wake worker thread.</summary>
    public WakeDetectionPipeline Pipeline { get; }

    /// <summary>Frame codec for this connection, replaced when the device reconnects.</summary>
    public WakeFrameCodec? Codec { get; set; }

    public WakeSessionState State { get; set; } = WakeSessionState.Handshake;

    /// <summary>When the current connection last produced any traffic; drives the liveness timeout.</summary>
    public DateTimeOffset LastActivityUtc { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>Samples dropped because the worker fell behind. Non-zero here means the machine is oversubscribed.</summary>
    public long SamplesDropped => _ring.SamplesDropped;

    public WakeSession(string deviceId, WakeDetectionPipeline pipeline)
    {
        DeviceId = deviceId;
        Pipeline = pipeline;
    }

    /// <summary>Queues audio from the socket and mirrors it into the capture buffer. <paramref name="sequence"/>
    /// is the client's frame counter; a gap means audio was lost in flight, so the models are reset rather than
    /// fed a splice across the discontinuity.</summary>
    public void Enqueue(ReadOnlySpan<float> samples, long sequence)
    {
        if (_expectedSequence >= 0 && sequence != _expectedSequence)
        {
            Logs.Warning($"[Audio][Wake] Device '{DeviceId}' sequence gap: expected {_expectedSequence}, got {sequence}. Resetting detection state.");
            RequestReset = true;
        }
        _expectedSequence = sequence + 1;

        _ring.Write(samples);
        lock (_captureLock)
        {
            foreach (float sample in samples)
            {
                _capture[(int)(_captureWritten % CaptureCapacitySamples)] = sample;
                _captureWritten++;
            }
        }
    }

    /// <summary>Set when the stream became discontinuous; the worker clears it after resetting the pipeline.</summary>
    public bool RequestReset { get; set; }

    /// <summary>Drains queued audio for the worker. Returns the sample count written to <paramref name="destination"/>.</summary>
    public int Drain(Span<float> destination) => _ring.Read(destination);

    /// <summary>Marks a fresh connection: the pipeline's audio context is meaningless across a disconnect, so it
    /// is cleared rather than resumed.</summary>
    public void OnReconnected(WakeFrameCodec codec)
    {
        Codec = codec;
        State = WakeSessionState.Listening;
        LastActivityUtc = DateTimeOffset.UtcNow;
        _expectedSequence = -1;
        RequestReset = true;
        _ring.Reset();
        lock (_captureLock) _captureWritten = 0;
    }

    /// <summary>Copies the most recent <paramref name="seconds"/> of audio, oldest first — the utterance around a
    /// detection, for transcription or speaker identification.</summary>
    public float[] SnapshotRecent(double seconds)
    {
        int want = Math.Min((int)(seconds * 16_000), CaptureCapacitySamples);
        lock (_captureLock)
        {
            int have = (int)Math.Min(_captureWritten, want);
            float[] result = new float[have];
            long first = _captureWritten - have;
            for (int i = 0; i < have; i++)
                result[i] = _capture[(int)((first + i) % CaptureCapacitySamples)];
            return result;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Pipeline.Dispose();
    }
}
