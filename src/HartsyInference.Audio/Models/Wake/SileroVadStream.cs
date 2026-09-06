using HartsyInference.Core.Backends;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>Turns <see cref="SileroVad"/>'s per-chunk probabilities into speech segments, mirroring
/// silero-vad's <c>VADIterator</c> hysteresis: speech opens at <c>enterThreshold</c> and only closes below
/// <c>exitThreshold</c>, so a probability drifting through the band between them neither opens nor closes a
/// segment.
///
/// <para>Two debounces sit on top of that. A dip back above <c>enterThreshold</c> before
/// <c>minSilenceMs</c> has elapsed cancels the pending close, and a segment shorter than
/// <c>minSpeechMs</c> is dropped instead of emitted. Reported boundaries are padded outward by
/// <c>speechPadMs</c> so a downstream recognizer keeps the onset consonant and the trailing vowel.</para>
///
/// <para>The stream does not own its <see cref="SileroVad"/>; the model carries this stream's LSTM state, so
/// give every concurrent session its own model instance rather than sharing one.</para></summary>
public sealed class SileroVadStream
{
    private const int SampleRate = 16000;

    private readonly SileroVad _vad;
    private readonly float _enterThreshold;
    private readonly float _exitThreshold;
    private readonly int _minSpeechSamples;
    private readonly int _minSilenceSamples;
    private readonly int _speechPadSamples;

    private long _consumed;
    private long _speechStart;
    private long _speechStartRaw;
    private long _silenceStart = -1;
    private bool _triggered;

    /// <summary>Defaults are silero-vad's own: 0.5 / 0.35 with 250 ms minimum speech, 100 ms minimum silence
    /// and 30 ms of boundary padding.</summary>
    public SileroVadStream(SileroVad vad, float enterThreshold = 0.5f, float exitThreshold = 0.35f,
        int minSpeechMs = 250, int minSilenceMs = 100, int speechPadMs = 30)
    {
        ArgumentNullException.ThrowIfNull(vad);
        if (exitThreshold > enterThreshold)
            throw new ArgumentException($"Exit threshold {exitThreshold} must not exceed enter threshold {enterThreshold}.", nameof(exitThreshold));
        if (minSpeechMs < 0 || minSilenceMs < 0 || speechPadMs < 0)
            throw new ArgumentException("Durations must be non-negative.", nameof(minSpeechMs));
        _vad = vad;
        _enterThreshold = enterThreshold;
        _exitThreshold = exitThreshold;
        _minSpeechSamples = minSpeechMs * SampleRate / 1000;
        _minSilenceSamples = minSilenceMs * SampleRate / 1000;
        _speechPadSamples = speechPadMs * SampleRate / 1000;
    }

    /// <summary>The model this stream drives. Exposed because the stream does not own it — whoever constructed
    /// the pair has to dispose it, and when the stream is the only handle that has been kept, this is it.</summary>
    public SileroVad Model => _vad;

    /// <summary>Probability of the most recently pushed chunk.</summary>
    public float LastProbability { get; private set; }

    /// <summary>True while a speech segment is open and has not yet been closed by silence.</summary>
    public bool InSpeech => _triggered;

    /// <summary>Padded start of the open segment, or -1 when not in speech.</summary>
    public long SpeechStartSample => _triggered ? _speechStart : -1;

    /// <summary>Total samples pushed since the last <see cref="Reset"/>.</summary>
    public long ConsumedSamples => _consumed;

    /// <summary>Scores one 512-sample chunk of ±1-normalized audio; returns true on the chunk that closes a
    /// segment long enough to keep, filling <paramref name="segment"/> with its padded sample bounds.</summary>
    public bool Push(IBackend backend, ReadOnlySpan<float> chunk, out SileroVadSegment segment)
    {
        segment = default;
        float probability = _vad.Process(backend, chunk);
        LastProbability = probability;
        _consumed += SileroVad.WindowSamples;
        long chunkStart = _consumed - SileroVad.WindowSamples;

        if (probability >= _enterThreshold)
        {
            _silenceStart = -1;
            if (!_triggered)
            {
                _triggered = true;
                _speechStartRaw = chunkStart;
                _speechStart = Math.Max(0, chunkStart - _speechPadSamples);
            }
            return false;
        }
        if (!_triggered || probability >= _exitThreshold) return false;

        if (_silenceStart < 0) _silenceStart = chunkStart;
        if (chunkStart - _silenceStart < _minSilenceSamples) return false;

        long speechEnd = _silenceStart;
        _triggered = false;
        _silenceStart = -1;
        if (speechEnd - _speechStartRaw < _minSpeechSamples) return false;
        segment = new SileroVadSegment(_speechStart, speechEnd + _speechPadSamples);
        return true;
    }

    /// <summary>Closes an open segment at the end of the stream, so trailing speech is not lost when the
    /// audio stops before silence has been confirmed.</summary>
    public bool Flush(out SileroVadSegment segment)
    {
        segment = default;
        if (!_triggered) return false;
        _triggered = false;
        long speechEnd = _silenceStart < 0 ? _consumed : _silenceStart;
        _silenceStart = -1;
        if (speechEnd - _speechStartRaw < _minSpeechSamples) return false;
        segment = new SileroVadSegment(_speechStart, Math.Min(_consumed, speechEnd + _speechPadSamples));
        return true;
    }

    /// <summary>Drops the open segment, the sample counter and the model's LSTM state.</summary>
    public void Reset()
    {
        _vad.Reset();
        _consumed = 0;
        _speechStart = 0;
        _speechStartRaw = 0;
        _silenceStart = -1;
        _triggered = false;
        LastProbability = 0f;
    }
}

/// <summary>Speech bounds in samples from the start of the stream, already widened by the padding.</summary>
public readonly record struct SileroVadSegment(long StartSample, long EndSample)
{
    /// <summary>Length in samples.</summary>
    public long LengthSamples => EndSample - StartSample;
}
