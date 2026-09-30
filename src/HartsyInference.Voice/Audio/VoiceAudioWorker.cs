using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice.Audio;

/// <summary>T1, the voice-audio thread: drains the inbound ring in 20 ms frames through the
/// <see cref="VoiceAudioFrontend"/> and acts on what it decides.</summary>
/// <remarks>A dedicated thread, not a pool task, inside <see cref="CpuParallel.EnterInline"/> for its whole life so
/// no kernel it runs hands work to the pool and waits for it back. It sleeps on a doorbell the producer rings after
/// every write (reset, drain, then wait, so a write between the drain and the wait is never missed) and never polls.
/// The inbound ring is a lock-free <see cref="SpscRing{T}"/>; the only synchronization the per-frame path touches is
/// that ring and the doorbell. Endpoints and barge-ins (once per utterance) allocate the utterance copy and hand off
/// through <see cref="IVoiceAudioSink"/>; nothing else on this thread allocates after warm-up.
/// <para>The ring drops the newest samples when full, so the thread itself enforces the drop-oldest policy: when
/// it falls more than the backlog limit behind, it discards the oldest excess before processing, counted in
/// <see cref="DroppedSamples"/>, so a stalled call catches up to the caller instead of answering the past.</para></remarks>
internal sealed class VoiceAudioWorker : IDisposable
{
    /// <summary>Frames of low-level noise pushed through the front-end before listening, so the JIT and every lazy
    /// buffer are done before the first real frame.</summary>
    public const int WarmUpFrames = 50;

    private readonly SpscRing<float> _inbound;
    private readonly VoiceAudioFrontend _frontend;
    private readonly VoiceTurnSignals _signals;
    private readonly IVoiceAudioSink _sink;
    private readonly ManualResetEventSlim _doorbell = new(false);
    private readonly float[] _frame = new float[VoiceAudioFrontend.FrameSamples];
    private readonly long _backlogLimit;
    private readonly TaskCompletionSource _ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Thread? _thread;
    private int _frameFill;
    private long _droppedOldest;
    private long _frames;
    private long _publishedFrames;
    private long _allocatedBytes;
    private volatile bool _stopping;
    private int _disposed;

    /// <summary>Takes ownership of <paramref name="frontend"/>. <paramref name="inboundCapacity"/> is a power of two.</summary>
    public VoiceAudioWorker(VoiceAudioFrontend frontend, VoiceTurnSignals signals, IVoiceAudioSink sink, int inboundCapacity, long backlogLimitSamples)
    {
        ArgumentNullException.ThrowIfNull(frontend);
        ArgumentNullException.ThrowIfNull(signals);
        ArgumentNullException.ThrowIfNull(sink);
        ArgumentOutOfRangeException.ThrowIfLessThan(backlogLimitSamples, VoiceAudioFrontend.FrameSamples);
        _inbound = new SpscRing<float>(inboundCapacity);
        _frontend = frontend;
        _signals = signals;
        _sink = sink;
        _backlogLimit = Math.Min(backlogLimitSamples, inboundCapacity);
    }

    /// <summary>Completes once warm-up is done and frames are being processed; faults if the thread failed first.</summary>
    public Task Ready => _ready.Task;

    /// <summary>The front-end this thread drives.</summary>
    public VoiceAudioFrontend Frontend => _frontend;

    /// <summary>Inbound samples lost: refused by a full ring plus discarded as backlog.</summary>
    public long DroppedSamples => _inbound.DroppedSamples + Volatile.Read(ref _droppedOldest);

    /// <summary>Managed bytes the audio thread had allocated after its last drain; a diagnostic for the zero-allocation
    /// contract.</summary>
    public long AllocatedBytes => Volatile.Read(ref _allocatedBytes);

    /// <summary>Frames the thread had processed after its last drain (published after <see cref="AllocatedBytes"/>).</summary>
    public long ProcessedFrames => Volatile.Read(ref _publishedFrames);

    /// <summary>Producer side: queues caller audio (16 kHz, ±1) and wakes the thread. One producer thread; never blocks.</summary>
    public void Push(ReadOnlySpan<float> samples)
    {
        _inbound.Write(samples);
        _doorbell.Set();
    }

    /// <summary>Starts the thread.</summary>
    public void Start()
    {
        if (_thread is not null)
        {
            throw new InvalidOperationException("The voice audio thread is already running.");
        }
        _thread = new Thread(Run) { Name = "voice-audio", IsBackground = true };
        _thread.Start();
    }

    /// <summary>Stops the thread and waits for it; safe to call more than once and from the thread itself.</summary>
    public void Stop()
    {
        _stopping = true;
        _doorbell.Set();
        Thread? thread = _thread;
        if (thread is not null && thread != Thread.CurrentThread)
        {
            thread.Join();
        }
    }

    /// <summary>Processes every whole frame queued so far on the calling thread; the thread loop's body.</summary>
    internal void ProcessAvailable()
    {
        TrimBacklog();
        while (true)
        {
            _frameFill += _inbound.Read(_frame.AsSpan(_frameFill));
            if (_frameFill < _frame.Length)
            {
                return;
            }
            _frameFill = 0;
            _frames++;
            VoiceFrameEvents events = _frontend.ProcessFrame(_frame);
            if (events != VoiceFrameEvents.None)
            {
                Dispatch(events);
            }
        }
    }

    /// <summary>Warms the per-frame path on the calling thread, then clears every stage it touched.</summary>
    internal void WarmUp()
    {
        uint seed = 0x9E3779B9;
        for (int f = 0; f < WarmUpFrames; f++)
        {
            for (int i = 0; i < _frame.Length; i++)
            {
                seed = seed * 1664525 + 1013904223;
                _frame[i] = ((seed >> 9) * (1f / (1 << 23)) - 0.5f) * 1e-3f;
            }
            _frontend.ProcessFrame(_frame);
        }
        _frontend.Reset();
        // The doorbell allocates its wait object on the first blocking wait; take that hit here.
        _doorbell.Wait(1);
    }

    private void Run()
    {
        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        try
        {
            WarmUp();
            _ready.TrySetResult();
            while (!_stopping)
            {
                _doorbell.Reset();
                ProcessAvailable();
                Volatile.Write(ref _allocatedBytes, GC.GetAllocatedBytesForCurrentThread());
                Volatile.Write(ref _publishedFrames, _frames);
                if (_stopping)
                {
                    break;
                }
                _doorbell.Wait();
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[Voice] The audio thread failed; the session cannot hear the caller any more.", ex);
            _ready.TrySetException(ex);
            _sink.OnAudioFault(ex);
        }
    }

    private void Dispatch(VoiceFrameEvents events)
    {
        long now = MonotonicClock.NowNs();
        if ((events & VoiceFrameEvents.BargeIn) != 0 && _signals.TryBargeIn(_frontend.BargeInTurn, now))
        {
            _sink.OnBargeIn(_frontend.BargeInTurn, now);
        }
        if ((events & VoiceFrameEvents.Endpoint) != 0)
        {
            float[] utterance = new float[_frontend.UtteranceSamples];
            _frontend.CopyUtterance(utterance);
            _sink.OnUtterance(VoiceTurnInput.Utterance(utterance, _frontend.HangoverSamples, _frontend.TakeFrameTimes(), now));
        }
        if ((events & VoiceFrameEvents.UtteranceDiscarded) != 0)
        {
            _sink.OnUtteranceDiscarded(_frontend.UtteranceSamples);
        }
    }

    private void TrimBacklog()
    {
        long excess = _inbound.Available - _backlogLimit;
        if (excess <= 0)
        {
            return;
        }
        long dropped = 0;
        while (dropped < excess)
        {
            int read = _inbound.Read(_frame.AsSpan(0, (int)Math.Min(_frame.Length, excess - dropped)));
            if (read == 0)
            {
                break;
            }
            dropped += read;
        }
        // The partial frame and the model state belong to audio that is gone; resume clean on the fresh audio.
        _frameFill = 0;
        _frontend.Reset();
        Volatile.Write(ref _droppedOldest, _droppedOldest + dropped);
        Logs.Warning($"[Voice] The audio thread fell {excess / (VoiceAudioFrontend.SampleRate / 1000)} ms behind; dropped the oldest audio to catch up.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        Stop();
        _frontend.Dispose();
        // The doorbell stays undisposed: a producer racing the end of the call may still ring it, and it holds no
        // handle unless its WaitHandle is taken, which nothing here does.
    }
}
