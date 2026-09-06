using System.Collections.Concurrent;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>What a session is currently doing. Post-trigger capture is tracked here so a detection can hand the utterance to transcription without the worker holding any lock while that runs.</summary>
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
public sealed class WakeSession(string deviceId, WakeDetectionPipeline pipeline, RnnoiseStream? denoiser = null, SileroVadStream? vad = null) : IDisposable
{
    /// <summary>Audio buffered between the socket and the worker (2 s). Sized to absorb scheduling jitter and short inference stalls without becoming a latency reservoir.</summary>
    public const int RingCapacitySamples = 32_000;

    /// <summary>Rolling raw audio retained for post-trigger use — transcription and speaker identification both need the audio from before the word fired, not just after.</summary>
    public const int CaptureCapacitySamples = 16_000 * 15;

    private readonly AudioRingBuffer _ring = new(RingCapacitySamples);
    private readonly float[] _capture = new float[CaptureCapacitySamples];
    private readonly object _captureLock = new();
    private long _captureWritten;
    private long _expectedSequence = -1;
    private int _disposed;

    /// <summary>Stable identity from the client's <c>hello</c>; survives reconnects.</summary>
    public string DeviceId { get; } = deviceId;

    /// <summary>The device's detection pipeline. Touched only by the wake worker thread.</summary>
    public WakeDetectionPipeline Pipeline { get; } = pipeline;

    /// <summary>This device's noise suppressor, or null when suppression is off. Carries per-stream state (GRU
    /// hidden vectors, resampler and overlap-add tails) over shared weights, so it is per-session and, like
    /// <see cref="Pipeline"/>, touched only by the wake worker thread.</summary>
    public RnnoiseStream? Denoiser { get; } = denoiser;

    /// <summary>This device's end-of-speech detector, or null when no VAD weights are installed. Carries LSTM
    /// state for one audio stream, so it is per-session and, like <see cref="Pipeline"/>, touched only by the
    /// wake worker thread.</summary>
    public SileroVadStream? Vad { get; } = vad;

    /// <summary>Ticks (<see cref="Environment.TickCount64"/>) when the VAD last scored a chunk as speech.
    ///
    /// <para>Written by the worker thread and read by whichever thread-pool thread is handling a detection, so
    /// it is deliberately a plain 64-bit field read through <see cref="Interlocked"/> rather than an event: the
    /// reader wants "how long has it been quiet", which is a question about the present, not a notification
    /// about the past. A missed update just costs one more 50 ms poll.</para></summary>
    public long LastSpeechTicks
    {
        get => Interlocked.Read(ref _lastSpeechTicks);
        set => Interlocked.Exchange(ref _lastSpeechTicks, value);
    }

    private long _lastSpeechTicks;

    /// <summary>Partial 512-sample window held back for the VAD, which only accepts exact windows.</summary>
    private readonly float[] _vadWindow = new float[SileroVad.WindowSamples];
    private int _vadFilled;

    /// <summary>Feeds audio to this device's VAD and notes when it hears speech. Called only by the wake worker.</summary>
    /// <param name="samples">Int16-scaled audio, as the wake path carries it.</param>
    /// <remarks>Silero takes exactly 512 samples at ±1 scale, while the worker drains a variable count at int16
    /// scale — hence the accumulator and the divide. Whole windows only: a short final window padded with zeros
    /// would read as silence and end an utterance early.</remarks>
    public void PushVad(IBackend backend, ReadOnlySpan<float> samples)
    {
        if (Vad is null) return;
        int offset = 0;
        while (offset < samples.Length)
        {
            int take = Math.Min(SileroVad.WindowSamples - _vadFilled, samples.Length - offset);
            for (int i = 0; i < take; i++)
            {
                _vadWindow[_vadFilled + i] = samples[offset + i] / 32768f;
            }
            _vadFilled += take;
            offset += take;
            if (_vadFilled < SileroVad.WindowSamples) return;
            _vadFilled = 0;
            Vad.Push(backend, _vadWindow, out _);
            if (Vad.InSpeech)
            {
                LastSpeechTicks = Environment.TickCount64;
            }
        }
    }

    /// <summary>Frame codec for this connection, replaced when the device reconnects.</summary>
    public WakeFrameCodec? Codec { get; set; }

    public WakeSessionState State { get; set; } = WakeSessionState.Handshake;

    /// <summary>When the current connection last produced any traffic; drives the liveness timeout.</summary>
    public DateTimeOffset LastActivityUtc { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>Samples dropped because the worker fell behind. Non-zero here means the machine is oversubscribed.</summary>
    public long SamplesDropped => _ring.SamplesDropped;

    /// <summary>Queues audio from the socket and mirrors it into the capture buffer. <paramref name="sequence"/> is the client's frame counter; a gap means audio was lost in flight, so the models are reset rather than fed a splice across the discontinuity.</summary>
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

    /// <summary>Word changes waiting to be applied. A pipeline is single-threaded by contract and is owned by the worker, so configuration changes are queued from whatever thread requests them and applied by the worker between drains rather than mutating a pipeline mid-push.</summary>
    public ConcurrentQueue<(string Name, WakeHead? Head, WakeWordSettings? Settings)> PendingWords { get; } = new();

    /// <summary>Clears VAD state after a stream discontinuity. Called only by the wake worker.</summary>
    public void ResetVad()
    {
        Vad?.Reset();
        _vadFilled = 0;
    }

    /// <summary>Applies queued word changes. Called only by the wake worker.</summary>
    public void ApplyPendingWords()
    {
        while (PendingWords.TryDequeue(out (string Name, WakeHead? Head, WakeWordSettings? Settings) change))
        {
            Pipeline.RemoveWord(change.Name);
            if (change.Head is not null && change.Settings is not null)
                Pipeline.AddWord(change.Head, change.Settings);
        }
    }

    /// <summary>Drains queued audio for the worker. Returns the sample count written to <paramref name="destination"/>.</summary>
    public int Drain(Span<float> destination) => _ring.Read(destination);

    /// <summary>Marks a fresh connection: the pipeline's audio context is meaningless across a disconnect, so it is cleared rather than resumed.</summary>
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

    /// <summary>Copies the most recent <paramref name="seconds"/> of audio, oldest first — the utterance around a detection, for transcription or speaker identification.</summary>
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
        Denoiser?.Dispose();
        // The stream does not own its model by contract, and this session is the only thing holding either.
        Vad?.Model.Dispose();
    }
}
