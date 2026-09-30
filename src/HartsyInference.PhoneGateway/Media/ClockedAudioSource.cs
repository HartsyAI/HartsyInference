using HartsyInference.Audio.Io;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>The RTP sender's clock: one dedicated thread per call that emits one G.711 frame every 20 ms on absolute
/// <see cref="MonotonicClock"/> deadlines, replacing sipsorcery's timer-paced <c>AudioExtrasSource</c>.</summary>
/// <remarks>The tick thread drains a <see cref="SpscRing{T}"/> the outbound path fills, encodes the frame in place and
/// raises <see cref="OnAudioSourceEncodedSample"/>, which <c>VoIPMediaSession</c> has bound to <c>SendAudio</c>; the
/// packet is on the wire before the handler returns. After construction the thread allocates nothing: the PCM and
/// code buffers are reused every tick, the ring is fixed, and the histogram records without allocating. A frame is
/// sent on every tick even when the ring is empty (comfort silence keeps NAT bindings and the provider's media latch
/// alive). Lateness of a whole period is absorbed by up to <see cref="ClockedAudioSourceOptions.MaxCatchUpFrames"/>
/// back-to-back frames; more than that resyncs the clock and is counted. <see cref="Flush"/> raises an epoch flag
/// that the next tick applies with <see cref="SpscRing{T}.DiscardAll"/>, so a barge-in empties the queue within one
/// period. The thread asks for <c>SCHED_FIFO</c> once at start and otherwise falls back to a short spin before each
/// deadline; the refusal reason is logged once and kept in <see cref="FifoReason"/>. Only PCMU and PCMA at 8 kHz
/// are offered.</remarks>
public sealed class ClockedAudioSource : IAudioSource, IDisposable
{
    /// <summary>The only sample rate this source produces.</summary>
    public const int SampleRate = 8000;

    /// <summary>Samples in one 20 ms frame.</summary>
    public const int FrameSamples = 160;

    /// <summary>The tick period.</summary>
    public const long PeriodNs = 20_000_000L;

    /// <summary>Ring capacity in samples: about 32 s at 8 kHz, a power of two.</summary>
    public const int RingCapacity = 1 << 18;

    private const int ThreadStackBytes = 256 * 1024;
    private const long WarmUpPeriodNs = 250_000L;
    private const int SpinIterations = 8;

    private readonly ClockedAudioSourceOptions _options;
    private readonly SpscRing<short> _ring = new(RingCapacity);
    private readonly short[] _pcm = new short[FrameSamples];
    private readonly byte[] _encoded = new byte[FrameSamples];
    private readonly LatencyHistogram _lateness;
    private readonly ManualResetEventSlim _started = new(false);
    private readonly object _lifecycleLock = new();
    private Func<AudioFormat, bool>? _formatFilter;
    private Thread? _thread;
    private volatile bool _running;
    private volatile bool _paused;
    private volatile bool _closed;
    private volatile bool _fifoActive;
    private volatile bool _faulted;
    private string _fifoReason = "";
    private int _law;
    private int _flushPending;
    private long _flushedSamples;
    private long _ticks;
    private long _framesSent;
    private long _silenceFrames;
    private long _catchUpFrames;
    private long _resyncs;
    private long _startNs;
    private long _lastTickNs;
    private long _allocatedBaseline;
    private long _allocatedNow;

    public ClockedAudioSource(ClockedAudioSourceOptions options, LatencyHistogram? lateness = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.FifoPriority < 0 || options.FifoPriority > RealtimeScheduling.MaxFifoPriority)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.FifoPriority, "FifoPriority must be 0..99.");
        }
        if (options.SpinTailNs < 0 || options.SpinTailNs >= PeriodNs)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.SpinTailNs, "SpinTailNs must be below one period.");
        }
        if (options.MaxCatchUpFrames < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxCatchUpFrames, "MaxCatchUpFrames cannot be negative.");
        }
        _options = options;
        _lateness = lateness ?? new LatencyHistogram();
        _law = (int)(options.Codec == AudioCodecPreference.Pcma ? G711Law.ALaw : G711Law.MuLaw);
    }

    /// <summary>Raised on the tick thread with one encoded 20 ms frame; <c>VoIPMediaSession</c> binds it to <c>SendAudio</c>.</summary>
    public event EncodedSampleDelegate? OnAudioSourceEncodedSample;

    /// <summary>Never raised: the session consumes <see cref="OnAudioSourceEncodedSample"/>.</summary>
    public event Action<EncodedAudioFrame>? OnAudioSourceEncodedFrameReady
    {
        add { }
        remove { }
    }

    /// <summary>Never raised: this source has no raw-sample consumers.</summary>
    public event RawAudioSampleDelegate? OnAudioSourceRawSample
    {
        add { }
        remove { }
    }

    /// <summary>Never raised: faults are exposed through <see cref="Faulted"/>.</summary>
    public event SourceErrorDelegate? OnAudioSourceError
    {
        add { }
        remove { }
    }

    /// <summary>Per-tick lateness (wake-up time minus deadline).</summary>
    public LatencyHistogram Lateness => _lateness;

    /// <summary>The law frames are encoded with; set by <see cref="SetAudioSourceFormat"/>.</summary>
    public G711Law Law => (G711Law)Volatile.Read(ref _law);

    /// <summary>True while the tick thread is running.</summary>
    public bool IsRunning => _running && !_faulted;

    /// <summary>True when the tick thread died on an exception; the call should be torn down.</summary>
    public bool Faulted => _faulted;

    /// <summary>True when the tick thread runs under <c>SCHED_FIFO</c>.</summary>
    public bool FifoActive => _fifoActive;

    /// <summary>Why FIFO was refused, or empty.</summary>
    public string FifoReason => Volatile.Read(ref _fifoReason);

    /// <summary>Samples queued for sending.</summary>
    public int Available => _ring.Available;

    /// <summary>Samples <see cref="WriteOutbound"/> can take before dropping.</summary>
    public int FreeSpace => _ring.FreeSpace;

    /// <summary>Samples refused because the ring was full.</summary>
    public long DroppedSamples => _ring.DroppedSamples;

    /// <summary>Samples thrown away by flushes.</summary>
    public long FlushedSamples => Volatile.Read(ref _flushedSamples);

    /// <summary>Ticks executed, including catch-up ticks.</summary>
    public long Ticks => Volatile.Read(ref _ticks);

    /// <summary>Frames handed to the session (ticks while not paused).</summary>
    public long FramesSent => Volatile.Read(ref _framesSent);

    /// <summary>Frames that were padded with silence because the ring ran dry.</summary>
    public long SilenceFrames => Volatile.Read(ref _silenceFrames);

    /// <summary>Extra frames emitted to absorb late wake-ups.</summary>
    public long CatchUpFrames => Volatile.Read(ref _catchUpFrames);

    /// <summary>Times the clock was re-based because the thread fell further behind than catch-up allows.</summary>
    public long Resyncs => Volatile.Read(ref _resyncs);

    /// <summary>Nanoseconds between the thread's first deadline base and its latest tick.</summary>
    public long ElapsedNs => Volatile.Read(ref _lastTickNs) - Volatile.Read(ref _startNs);

    /// <summary>Managed bytes the tick thread allocated since its baseline (taken after the FIFO attempt); zero by design.</summary>
    public long TickThreadAllocatedBytes => Volatile.Read(ref _allocatedNow) - Volatile.Read(ref _allocatedBaseline);

    /// <summary>Producer side: queues 8 kHz PCM for the tick thread. One producer thread at a time.</summary>
    /// <returns>Samples accepted; the rest were dropped and counted.</returns>
    public int WriteOutbound(ReadOnlySpan<short> pcm) => _ring.Write(pcm);

    /// <summary>Asks the tick thread to discard everything queued on its next tick.</summary>
    /// <returns>The samples queued at the moment of the request, an upper bound on what the tick will discard.</returns>
    public int Flush()
    {
        int queued = _ring.Available;
        Volatile.Write(ref _flushPending, 1);
        return queued;
    }

    /// <summary>Starts the tick thread; returns once its warm-up ticks are done and the paced loop is running.</summary>
    public void Start()
    {
        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_closed, this);
            if (_thread is not null)
            {
                throw new InvalidOperationException("ClockedAudioSource is already started.");
            }
            _running = true;
            _faulted = false;
            _started.Reset();
            _thread = new Thread(ThreadMain, ThreadStackBytes) { Name = _options.ThreadName, IsBackground = false };
            _thread.Start();
        }
        _started.Wait();
    }

    /// <summary>Stops the tick thread and waits for it. Safe to call more than once.</summary>
    public void Stop()
    {
        Thread? thread;
        lock (_lifecycleLock)
        {
            _running = false;
            thread = _thread;
            _thread = null;
        }
        if (thread is not null && thread != Thread.CurrentThread)
        {
            thread.Join();
        }
    }

    public Task StartAudio()
    {
        Start();
        return Task.CompletedTask;
    }

    public Task PauseAudio()
    {
        _paused = true;
        return Task.CompletedTask;
    }

    public Task ResumeAudio()
    {
        _paused = false;
        return Task.CompletedTask;
    }

    public Task CloseAudio()
    {
        Stop();
        return Task.CompletedTask;
    }

    public bool IsAudioSourcePaused() => _paused;

    public bool HasEncodedAudioSubscribers() => OnAudioSourceEncodedSample is not null;

    public List<AudioFormat> GetAudioSourceFormats()
    {
        List<AudioFormat> formats = G711Formats.Offer(_options.Codec);
        Func<AudioFormat, bool>? filter = _formatFilter;
        return filter is null ? formats : formats.Where(filter).ToList();
    }

    public void RestrictFormats(Func<AudioFormat, bool> filter) => _formatFilter = filter;

    public void SetAudioSourceFormat(AudioFormat audioFormat)
    {
        if (!G711Formats.TryGetLaw(audioFormat, out G711Law law))
        {
            Logs.Error($"[PhoneGateway] Negotiated audio format {audioFormat.Codec} is not G.711; keeping {Law}.");
            return;
        }
        Volatile.Write(ref _law, (int)law);
    }

    /// <summary>Raw samples from elsewhere are not accepted; the outbound path writes the ring directly.</summary>
    public void ExternalAudioSourceRawSample(AudioSamplingRatesEnum samplingRate, uint durationMilliseconds, short[] sample)
    {
    }

    private void ThreadMain()
    {
        try
        {
            if (_options.FifoPriority > 0)
            {
                bool fifo = RealtimeScheduling.TryEnterFifo(_options.FifoPriority, out string reason);
                _fifoActive = fifo;
                Volatile.Write(ref _fifoReason, reason);
                if (fifo)
                {
                    Logs.Info($"[PhoneGateway] RTP tick thread running under SCHED_FIFO {_options.FifoPriority}.");
                }
                else
                {
                    Logs.Warning($"[PhoneGateway] RTP tick thread stays on the default scheduler with a {_options.SpinTailNs / 1000} µs spin tail: {reason}");
                }
            }
            if (_options.TickCpu >= 0 && !RealtimeScheduling.TryPinToCpu(_options.TickCpu, out string pinReason))
            {
                Logs.Warning($"[PhoneGateway] RTP tick thread could not be pinned to CPU {_options.TickCpu}: {pinReason}");
            }
            // Warm-up on this very thread, at a period long enough that clock_nanosleep is really entered: the first
            // call of a P/Invoke pays a one-time binding cost, and the tick path must be compiled before the first
            // real frame. Nothing is raised.
            RunLoop(Math.Max(WarmUpPeriodNs, 2 * _options.SpinTailNs), _options.WarmUpTicks, raise: false);
            _ring.DiscardAll();
            _lateness.Reset();
            Volatile.Write(ref _ticks, 0);
            Volatile.Write(ref _silenceFrames, 0);
            Volatile.Write(ref _catchUpFrames, 0);
            Volatile.Write(ref _resyncs, 0);
            Volatile.Write(ref _flushedSamples, 0);
            _started.Set();
            // Baseline after the event set: waking the starter is the last thing on this thread that may allocate.
            long baseline = GC.GetAllocatedBytesForCurrentThread();
            Volatile.Write(ref _allocatedBaseline, baseline);
            Volatile.Write(ref _allocatedNow, baseline);
            RunLoop(PeriodNs, -1, raise: true);
        }
        catch (Exception ex)
        {
            _faulted = true;
            _started.Set();
            Logs.Error("[PhoneGateway] RTP tick thread faulted; the call must be torn down", ex);
        }
    }

    /// <summary>The paced loop. Deadlines are <c>t0 + n·period</c>, never "now + period", so lateness does not accumulate.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunLoop(long periodNs, long maxTicks, bool raise)
    {
        long spinTail = _options.SpinTailNs;
        int maxCatchUp = _options.MaxCatchUpFrames;
        long t0 = MonotonicClock.NowNs();
        Volatile.Write(ref _startNs, t0);
        Volatile.Write(ref _lastTickNs, t0);
        long n = 1;
        long done = 0;
        while (_running && (maxTicks < 0 || done < maxTicks))
        {
            long deadline = t0 + n * periodNs;
            if (_fifoActive)
            {
                MonotonicClock.SleepUntil(deadline);
            }
            else
            {
                MonotonicClock.SleepUntil(deadline - spinTail);
                while (MonotonicClock.NowNs() < deadline)
                {
                    Thread.SpinWait(SpinIterations);
                }
            }
            if (!_running)
            {
                break;
            }
            long now = MonotonicClock.NowNs();
            long late = now - deadline;
            _lateness.Record(late);
            Tick(raise);
            done++;
            n++;
            if (late >= periodNs)
            {
                long behind = late / periodNs;
                if (behind <= maxCatchUp)
                {
                    for (long i = 0; i < behind; i++)
                    {
                        Tick(raise);
                    }
                    done += behind;
                    n += behind;
                    Volatile.Write(ref _catchUpFrames, _catchUpFrames + behind);
                }
                else
                {
                    t0 = now;
                    n = 1;
                    Volatile.Write(ref _resyncs, _resyncs + 1);
                }
            }
            Volatile.Write(ref _lastTickNs, now);
            Volatile.Write(ref _allocatedNow, GC.GetAllocatedBytesForCurrentThread());
        }
    }

    /// <summary>One frame: apply a pending flush, drain up to 160 samples, pad with silence, encode, hand off.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Tick(bool raise)
    {
        if (Volatile.Read(ref _flushPending) != 0)
        {
            Volatile.Write(ref _flushPending, 0);
            int discarded = _ring.DiscardAll();
            Volatile.Write(ref _flushedSamples, _flushedSamples + discarded);
        }
        int got = _ring.Read(_pcm);
        if (got < FrameSamples)
        {
            _pcm.AsSpan(got).Clear();
            _silenceFrames++;
        }
        G711.Encode(_pcm, _encoded, (G711Law)Volatile.Read(ref _law));
        Volatile.Write(ref _ticks, _ticks + 1);
        if (raise && !_paused)
        {
            OnAudioSourceEncodedSample?.Invoke(FrameSamples, _encoded);
            Volatile.Write(ref _framesSent, _framesSent + 1);
        }
    }

    public void Dispose()
    {
        _closed = true;
        Stop();
        _started.Dispose();
    }
}
