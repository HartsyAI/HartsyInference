using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice.Audio;

/// <summary>The audio thread's per-frame work: optional RNNoise, Silero endpointing, the capture of the caller's
/// speech, and the barge-in decision. Single-threaded and allocation-free after construction.</summary>
/// <remarks>Scale contract: frames arrive at 16 kHz, ±1. RNNoise runs at int16 scale (its silence floor and log
/// offsets are absolute, so ±1 audio would sit under the floor and pass through untouched), so a frame is scaled by
/// 32768 on the way in and back on the way out; Silero and the recognizer take ±1.
/// <para>Every position is on one clock: samples pushed through the VAD. The capture ring is indexed by it, so the
/// segment <see cref="SileroVadStream"/> reports is copied straight out; the hangover, hold-off and barge-in run are
/// counted in it too, so the decisions are the same whether audio arrives in real time or in a burst.</para>
/// <para>An endpoint (the segment the VAD closes after <see cref="VoiceAgentOptions.EndOfTurnSilenceMs"/> of silence,
/// or the cut at <see cref="VoiceAgentOptions.MaxUtteranceMs"/>) is answered unless its speech both began and ended
/// while the agent's reply was audible and never grew into a barge-in: that is backchannel or the reply's own echo,
/// and answering it would talk over the reply with a reply to itself. Both ends are judged when the speech happens,
/// not when the endpoint is decided a hangover later, and the reply counts as audible for
/// <see cref="VoiceAgentOptions.BargeInHoldoffMs"/> after it leaves, because the echo of its last words arrives late.
/// Speech that began before the reply, or outlasted it, is answered.</para></remarks>
internal sealed class VoiceAudioFrontend : IDisposable
{
    /// <summary>Inbound rate; Silero and RNNoise's 16 kHz wrapper both run at it.</summary>
    public const int SampleRate = 16_000;

    /// <summary>Samples per processed frame (20 ms).</summary>
    public const int FrameSamples = SampleRate / 50;

    /// <summary>Shortest speech kept as an utterance.</summary>
    public const int MinSpeechMs = 250;

    /// <summary>Padding the VAD adds on each side of a segment.</summary>
    public const int SpeechPadMs = 30;

    private const int SamplesPerMs = SampleRate / 1000;
    private const float Int16Scale = 32768f;
    private const float InverseInt16Scale = 1f / Int16Scale;
    private const int CaptureSlackMs = 1_000;

    private readonly IBackend _cpu;
    private readonly IVadModel _vadModel;
    private readonly SileroVadStream _vad;
    private readonly RnnoiseStream? _denoiser;
    private readonly VoiceTurnSignals _signals;
    private readonly float[] _window;
    private readonly float[] _scaled;
    private readonly float[] _denoised;
    private readonly float[] _capture;
    private readonly LatencyHistogram _frameTimes = new();
    private readonly bool _bargeInEnabled;
    private readonly float _bargeInProbability;
    private readonly long _bargeInMinSamples;
    private readonly long _bargeInHoldoffSamples;
    private readonly long _maxUtteranceSamples;
    private readonly long _speechPadSamples = SpeechPadMs * SamplesPerMs;
    private int _windowFill;
    private bool _wasInSpeech;
    private long _replyAudibleUntil = -1;
    private bool _speechStartedInReply;
    private bool _speechEndedInReply;
    private int _observedTurn;
    private long _speakingSince;
    private long _bargeInRun;
    private int _bargeInTurn;
    private long _bargeInClock = -1;
    private SileroVadSegment _utterance;
    private long _hangover;
    private int _disposed;

    /// <summary>Takes ownership of <paramref name="vad"/> (disposed with this when disposable) and
    /// <paramref name="denoiser"/>; <paramref name="cpu"/> is borrowed.</summary>
    public VoiceAudioFrontend(IBackend cpu, IVadModel vad, RnnoiseStream? denoiser, VoiceTurnSignals signals, VoiceAgentOptions options)
    {
        ArgumentNullException.ThrowIfNull(cpu);
        ArgumentNullException.ThrowIfNull(vad);
        ArgumentNullException.ThrowIfNull(signals);
        ArgumentNullException.ThrowIfNull(options);
        if (denoiser is not null && denoiser.SampleRate != SampleRate)
        {
            throw new ArgumentException($"The denoiser runs at {denoiser.SampleRate} Hz; the voice front-end needs {SampleRate} Hz.", nameof(denoiser));
        }
        _cpu = cpu;
        _vadModel = vad;
        _vad = new SileroVadStream(vad, minSpeechMs: MinSpeechMs, minSilenceMs: options.EndOfTurnSilenceMs, speechPadMs: SpeechPadMs);
        _denoiser = denoiser;
        _signals = signals;
        _window = new float[vad.WindowSamples];
        _scaled = denoiser is null ? [] : new float[FrameSamples];
        _denoised = denoiser is null ? [] : new float[FrameSamples + denoiser.FrameSize];
        _capture = new float[(options.MaxUtteranceMs + options.EndOfTurnSilenceMs + CaptureSlackMs) * SamplesPerMs];
        _bargeInEnabled = options.BargeInEnabled;
        _bargeInProbability = options.BargeInProbability;
        _bargeInMinSamples = (long)options.BargeInMinMs * SamplesPerMs;
        _bargeInHoldoffSamples = (long)options.BargeInHoldoffMs * SamplesPerMs;
        _maxUtteranceSamples = (long)options.MaxUtteranceMs * SamplesPerMs;
    }

    /// <summary>Samples pushed through the VAD since the last <see cref="Reset"/>: the front-end's clock.</summary>
    public long ClockSamples => _vad.ConsumedSamples;

    /// <summary>Whether the VAD has a speech segment open.</summary>
    public bool InSpeech => _vad.InSpeech;

    /// <summary>Speech probability of the last VAD chunk.</summary>
    public float LastProbability => _vad.LastProbability;

    /// <summary>Whether frames go through RNNoise.</summary>
    public bool Denoising => _denoiser is not null;

    /// <summary>The denoiser's algorithmic lag in samples at <see cref="SampleRate"/> (0 when <see cref="Denoising"/> is
    /// false): output sample <c>i</c> is input sample <c>i</c> minus this many samples, so every decision counted on
    /// the front-end's sample clock — the endpoint hangover, the barge-in run — is this much later in wall-clock terms
    /// than the sample count alone suggests. Nothing upstream of <see cref="ProcessFrame"/> can see it; a caller that
    /// wants a wall-clock-honest duration adds it back.</summary>
    public int DenoiserLatencySamples => _denoiser?.LatencySamples ?? 0;

    /// <summary>The turn the last <see cref="VoiceFrameEvents.BargeIn"/> interrupted.</summary>
    public int BargeInTurn => _bargeInTurn;

    /// <summary>Length of the utterance the last endpoint closed.</summary>
    public int UtteranceSamples => (int)_utterance.LengthSamples;

    /// <summary>Samples from the end of that utterance's speech to the endpoint decision; 0 for a cut mid-speech at the
    /// maximum length.</summary>
    public long HangoverSamples => _hangover;

    /// <summary>First sample of that utterance on the front-end clock, padding included.</summary>
    public long UtteranceStartSample => _utterance.StartSample;

    /// <summary>Scores one 20 ms frame of 16 kHz ±1 audio and reports what it decided.</summary>
    public VoiceFrameEvents ProcessFrame(ReadOnlySpan<float> frame)
    {
        if (frame.Length != FrameSamples)
        {
            throw new ArgumentException($"A frame is {FrameSamples} samples, got {frame.Length}.", nameof(frame));
        }
        long started = MonotonicClock.NowNs();
        ReadOnlySpan<float> clean = frame;
        if (_denoiser is not null)
        {
            for (int i = 0; i < frame.Length; i++)
            {
                _scaled[i] = frame[i] * Int16Scale;
            }
            int produced = _denoiser.Process(_cpu, _scaled, _denoised);
            Span<float> denoised = _denoised.AsSpan(0, produced);
            for (int i = 0; i < denoised.Length; i++)
            {
                denoised[i] *= InverseInt16Scale;
            }
            clean = denoised;
        }
        VoiceFrameEvents events = VoiceFrameEvents.None;
        while (!clean.IsEmpty)
        {
            int take = Math.Min(_window.Length - _windowFill, clean.Length);
            clean[..take].CopyTo(_window.AsSpan(_windowFill));
            _windowFill += take;
            clean = clean[take..];
            if (_windowFill == _window.Length)
            {
                _windowFill = 0;
                events |= PushWindow();
            }
        }
        _frameTimes.Record(MonotonicClock.NowNs() - started);
        return events;
    }

    /// <summary>Copies the utterance the last endpoint closed; <paramref name="destination"/> holds exactly
    /// <see cref="UtteranceSamples"/>.</summary>
    public void CopyUtterance(Span<float> destination)
    {
        long start = _utterance.StartSample;
        long length = _utterance.LengthSamples;
        if (destination.Length != length)
        {
            throw new ArgumentException($"The utterance is {length} samples, the destination {destination.Length}.", nameof(destination));
        }
        if (start < _vad.ConsumedSamples - _capture.Length || _utterance.EndSample > _vad.ConsumedSamples)
        {
            throw new InvalidOperationException($"Utterance [{start}, {_utterance.EndSample}) is outside the capture window ending at {_vad.ConsumedSamples}.");
        }
        int index = (int)(start % _capture.Length);
        int first = (int)Math.Min(length, _capture.Length - index);
        _capture.AsSpan(index, first).CopyTo(destination);
        _capture.AsSpan(0, (int)length - first).CopyTo(destination[first..]);
    }

    /// <summary>The per-frame timings since the last call, then starts a new window.</summary>
    public LatencyHistogram.Summary TakeFrameTimes()
    {
        LatencyHistogram.Summary summary = _frameTimes.Snapshot();
        _frameTimes.Reset();
        return summary;
    }

    /// <summary>Clears every stage and the clock, for a discontinuity or the end of warm-up.</summary>
    public void Reset()
    {
        _vad.Reset();
        _denoiser?.Reset();
        _windowFill = 0;
        _wasInSpeech = false;
        _replyAudibleUntil = -1;
        _speechStartedInReply = false;
        _speechEndedInReply = false;
        _observedTurn = 0;
        _bargeInRun = 0;
        _bargeInTurn = 0;
        _bargeInClock = -1;
        _utterance = default;
        _hangover = 0;
        _frameTimes.Reset();
    }

    private VoiceFrameEvents PushWindow()
    {
        long windowStart = _vad.ConsumedSamples;
        WriteCapture(windowStart);
        bool closed = _vad.Push(_cpu, _window, out SileroVadSegment segment);
        long clock = _vad.ConsumedSamples;
        if (_signals.SpeakingTurn != 0)
        {
            _replyAudibleUntil = clock + _bargeInHoldoffSamples;
        }
        bool inReply = windowStart < _replyAudibleUntil;
        VoiceFrameEvents events = VoiceFrameEvents.None;
        if (_vad.InSpeech && !_wasInSpeech)
        {
            _speechStartedInReply = inReply;
            events |= VoiceFrameEvents.SpeechStarted;
        }
        if (_vad.LastChunkWasSpeech)
        {
            _speechEndedInReply = inReply;
        }
        events |= CheckBargeIn(windowStart);
        if (closed || (_vad.InSpeech && clock - _vad.SpeechStartSample >= _maxUtteranceSamples && _vad.Flush(out segment)))
        {
            events |= Close(segment, clock);
        }
        _wasInSpeech = _vad.InSpeech;
        return events;
    }

    private VoiceFrameEvents CheckBargeIn(long windowStart)
    {
        int turn = _signals.SpeakingTurn;
        if (turn == 0 || !_bargeInEnabled)
        {
            _observedTurn = 0;
            _bargeInRun = 0;
            return VoiceFrameEvents.None;
        }
        if (turn != _observedTurn)
        {
            _observedTurn = turn;
            _speakingSince = windowStart;
            _bargeInRun = 0;
        }
        if (turn == _bargeInTurn || windowStart - _speakingSince < _bargeInHoldoffSamples)
        {
            return VoiceFrameEvents.None;
        }
        if (_vad.LastProbability < _bargeInProbability)
        {
            _bargeInRun = 0;
            return VoiceFrameEvents.None;
        }
        _bargeInRun += _window.Length;
        if (_bargeInRun < _bargeInMinSamples)
        {
            return VoiceFrameEvents.None;
        }
        _bargeInTurn = turn;
        _bargeInClock = windowStart;
        _bargeInRun = 0;
        return VoiceFrameEvents.BargeIn;
    }

    private VoiceFrameEvents Close(SileroVadSegment segment, long clock)
    {
        _utterance = segment;
        // A segment ending at the clock was cut mid-speech; any other ends a padded stretch after its last speech.
        _hangover = segment.EndSample >= clock ? 0 : clock - (segment.EndSample - _speechPadSamples);
        bool bargedIn = _bargeInClock >= segment.StartSample;
        bool withinReply = _speechStartedInReply && _speechEndedInReply;
        return withinReply && !bargedIn ? VoiceFrameEvents.UtteranceDiscarded : VoiceFrameEvents.Endpoint;
    }

    private void WriteCapture(long position)
    {
        int index = (int)(position % _capture.Length);
        int first = Math.Min(_window.Length, _capture.Length - index);
        _window.AsSpan(0, first).CopyTo(_capture.AsSpan(index));
        _window.AsSpan(first).CopyTo(_capture);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _denoiser?.Dispose();
        (_vadModel as IDisposable)?.Dispose();
    }
}
