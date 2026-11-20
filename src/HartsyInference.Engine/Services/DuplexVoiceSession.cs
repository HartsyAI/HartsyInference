using System.Threading.Channels;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>How a live conversation is paced and buffered.</summary>
public sealed record DuplexVoiceOptions
{
    /// <summary>Frames of lead a listener holds before it starts playing: a model frame that takes up to this many frame lengths longer than
    /// its slot still arrives before it is due. Each frame of lead is one more frame of delay (133 ms for Haru). Zero to 8.</summary>
    public int LeadFrames { get; init; } = 2;

    /// <summary>Sample rate of the audio pushed with <see cref="DuplexVoiceSession.PushUserAudio"/>; the session resamples it to the model's. Must
    /// divide by 50 when it differs from the model's rate (8000, 16000, 44100, 48000).</summary>
    public int InputSampleRate { get; init; }
}

/// <summary>One frame of the model's reply, with the time it took.</summary>
/// <param name="Index">Frame number, from 0.</param>
/// <param name="Audio">The model's samples for this frame, mono, at the session's rate.</param>
/// <param name="Text">The text the model spoke in this frame, usually empty.</param>
/// <param name="ComputeSeconds">Compute time of the frame.</param>
/// <param name="Late">The frame finished after the moment a listener holding <see cref="DuplexVoiceOptions.LeadFrames"/> frames would have needed it.</param>
public readonly record struct DuplexVoiceFrame(int Index, float[] Audio, string Text, double ComputeSeconds, bool Late);

/// <summary>What the pacer saw. <c>LateFrames</c> are frames that finished after a listener with the configured lead would have needed them; <c>Rebases</c> are
/// stalls of more than the lead plus two frames, after which the clock restarted; <c>DroppedUserSamples</c> is audio refused by a full buffer plus audio the pacer
/// trimmed; <c>DroppedReplyFrames</c> is frames the output queue dropped because the reader fell behind.</summary>
public sealed record DuplexVoiceStats(long Frames, long LateFrames, long Rebases, long SilenceFrames, long DroppedUserSamples, long DroppedReplyFrames,
    double MedianComputeMs, double P99ComputeMs, double MaxComputeMs, int LeadFrames, double FrameMs);

/// <summary>A live full-duplex conversation held to real time: the user's audio goes in as it arrives, the model's frames come out one per
/// frame length. A dedicated thread ticks every frame length (<see cref="MonotonicClock"/>), takes whatever the user has said since the last tick
/// (silence when nothing has arrived), runs one <see cref="IDuplexSpeechSession.Step"/> and queues the reply. A frame that takes too long
/// does not make the next ones burst: past the lead it rebases to a fresh tick and the user audio that piled up is dropped, so the delay
/// stays bounded. No echo cancellation: the user must not hear the model into their microphone.</summary>
public sealed class DuplexVoiceSession : IDisposable
{
    private readonly IDuplexSpeechSession _session;
    private readonly DuplexVoiceOptions _options;
    private readonly SpscRing<float> _input;
    private readonly Channel<DuplexVoiceFrame> _output;
    private readonly long _tickNs;
    private readonly int _frameSamples;
    private readonly Thread _thread;
    private readonly CancellationTokenSource _stop = new();
    private readonly StreamingResampler? _resampler;
    private readonly float[]? _pending;
    private int _pendingCount;
    private readonly float[] _resampled;
    private long _frames, _late, _rebases, _silence, _droppedReply, _trimmedUser;
    private double _maxComputeSeconds;
    private readonly List<double> _computeSeconds = [];
    private int _started, _disposed, _sessionDisposed;

    /// <summary>Wraps a session that has not been started: <see cref="Start"/> runs its prompt.</summary>
    public DuplexVoiceSession(IDuplexSpeechSession session, DuplexVoiceOptions? options = null)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
        _options = options ?? new DuplexVoiceOptions();
        if (_options.LeadFrames is < 0 or > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The lead is 0 to 8 frames.");
        }
        _frameSamples = session.FrameSamples;
        _tickNs = (long)(1_000_000_000.0 * _frameSamples / session.SampleRate);
        _input = new SpscRing<float>(NextPowerOfTwo(_frameSamples * (_options.LeadFrames + 4)));
        _output = Channel.CreateBounded(new BoundedChannelOptions(Math.Max(16, _options.LeadFrames * 4))
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = true,
        }, (DuplexVoiceFrame _) => Interlocked.Increment(ref _droppedReply));
        int inRate = _options.InputSampleRate;
        if (inRate < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "The input sample rate cannot be negative.");
        }
        if (inRate > 0 && inRate != session.SampleRate)
        {
            if (inRate % 50 != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(options), "An input rate other than the model's must divide by 50.");
            }
            int chunk = inRate / 50;
            _resampler = new StreamingResampler(inRate, session.SampleRate, chunk);
            _pending = new float[chunk];
            _resampled = new float[_resampler.OutputFrameSize];
        }
        else
        {
            _resampled = [];
        }
        _thread = new Thread(Run) { IsBackground = true, Name = "duplex-voice", Priority = ThreadPriority.AboveNormal };
    }

    /// <summary>Sample rate of the model's reply.</summary>
    public int SampleRate => _session.SampleRate;

    /// <summary>Samples in each reply frame.</summary>
    public int FrameSamples => _frameSamples;

    /// <summary>Length of one frame in milliseconds.</summary>
    public double FrameMs => 1000.0 * _frameSamples / _session.SampleRate;

    /// <summary>The model's frames in order. The channel completes when the conversation ends or the session is disposed.</summary>
    public ChannelReader<DuplexVoiceFrame> Frames => _output.Reader;

    /// <summary>Runs the card and the voice reference and starts the clock. The model's frame 0 is the first one queued.</summary>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.Exchange(ref _started, 1) != 0)
        {
            throw new InvalidOperationException("The conversation has already started.");
        }
        DuplexFrame first = _session.Start();
        Publish(new DuplexVoiceFrame(0, first.Audio, first.Text, first.ComputeSeconds, Late: false));
        _thread.Start();
    }

    /// <summary>Adds the user's microphone audio (mono, at <see cref="DuplexVoiceOptions.InputSampleRate"/>), any amount at a time, from one thread. What the
    /// buffer cannot hold is refused and counted; the pacer also trims, oldest first, what piles up beyond the lead (see <see cref="DuplexVoiceStats.DroppedUserSamples"/>).</summary>
    public void PushUserAudio(ReadOnlySpan<float> samples)
    {
        if (_resampler is null)
        {
            _input.Write(samples);
            return;
        }
        while (!samples.IsEmpty)
        {
            int take = Math.Min(samples.Length, _pending!.Length - _pendingCount);
            samples[..take].CopyTo(_pending.AsSpan(_pendingCount));
            _pendingCount += take;
            samples = samples[take..];
            if (_pendingCount == _pending.Length)
            {
                _resampler.Process(_pending, _resampled);
                _input.Write(_resampled);
                _pendingCount = 0;
            }
        }
    }

    /// <summary>The pacer's counters and the compute time of the frames so far.</summary>
    public DuplexVoiceStats Stats()
    {
        double[] sorted;
        lock (_computeSeconds)
        {
            sorted = [.. _computeSeconds];
        }
        Array.Sort(sorted);
        double At(double fraction) => sorted.Length == 0 ? 0 : sorted[(int)((sorted.Length - 1) * fraction)] * 1000;
        return new DuplexVoiceStats(Volatile.Read(ref _frames), Volatile.Read(ref _late), Volatile.Read(ref _rebases), Volatile.Read(ref _silence),
            _input.DroppedSamples + Volatile.Read(ref _trimmedUser), Volatile.Read(ref _droppedReply), At(0.5), At(0.99), Volatile.Read(ref _maxComputeSeconds) * 1000, _options.LeadFrames, FrameMs);
    }

    private void Run()
    {
        CancellationToken stop = _stop.Token;
        float[] user = new float[_frameSamples];
        float[] scratch = new float[_frameSamples];
        long origin = MonotonicClock.NowNs();
        long next = origin + _tickNs;
        int lead = _options.LeadFrames;
        try
        {
            while (!stop.IsCancellationRequested)
            {
                MonotonicClock.SleepUntil(next);
                if (stop.IsCancellationRequested)
                {
                    break;
                }
                // A frame is complete when its tick arrives; a short read means the user said nothing for the rest of it.
                int got = _input.Read(user);
                if (got > 0 && got < user.Length)
                {
                    // Audio that is on its way: a microphone delivers in chunks that do not line up with this clock, so give the rest of the frame a
                    // quarter of a frame to arrive rather than put a gap of zeros into the middle of speech.
                    long grace = MonotonicClock.NowNs() + _tickNs / 4;
                    while (got < user.Length && MonotonicClock.NowNs() < grace && !stop.IsCancellationRequested)
                    {
                        Thread.Sleep(1);
                        got += _input.Read(user.AsSpan(got));
                    }
                }
                if (got < user.Length)
                {
                    Array.Clear(user, got, user.Length - got);
                    if (got == 0)
                    {
                        Interlocked.Increment(ref _silence);
                    }
                }
                // Keep the delay bounded: of what piled up beyond the lead while a frame was late, the oldest is not replayed. The newest lead's
                // worth stays, so speech that arrived on time is not thrown away with it.
                int excess = _input.Available - _frameSamples * (lead + 1);
                while (excess > 0)
                {
                    int dropped = _input.Read(scratch.AsSpan(0, Math.Min(excess, scratch.Length)));
                    if (dropped == 0)
                    {
                        break;
                    }
                    Interlocked.Add(ref _trimmedUser, dropped);
                    excess -= dropped;
                }

                long started = MonotonicClock.NowNs();
                DuplexFrame frame = _session.Step(user);
                long finished = MonotonicClock.NowNs();
                int index = (int)Volatile.Read(ref _frames) + 1;
                // Frame k is scheduled at origin + k ticks (frame 1 at the first tick). A listener that waited `lead` frames plays it at
                // origin + (lead + 1 + k) ticks, so a frame that finishes later than that is late for that listener.
                long due = origin + (lead + 1L + index) * _tickNs;
                bool late = finished > due;
                if (late)
                {
                    Interlocked.Increment(ref _late);
                }
                double seconds = (finished - started) / 1e9;
                Record(seconds);
                Publish(new DuplexVoiceFrame(index, frame.Audio, frame.Text, seconds, late));
                Interlocked.Increment(ref _frames);

                next += _tickNs;
                if (finished - next > (lead + 2L) * _tickNs)
                {
                    // Behind by more than the lead plus two frames (a late frame is only late; this is a stall): take a fresh tick instead of
                    // running frames back to back to catch up.
                    next = finished + _tickNs;
                    Interlocked.Increment(ref _rebases);
                    origin = next - (index + 1L) * _tickNs;
                }
                if (frame.Done)
                {
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Duplex] The conversation stopped: {ex.Message}");
            _output.Writer.TryComplete(ex);
            FinishAfterDispose();
            return;
        }
        _output.Writer.TryComplete();
        FinishAfterDispose();
    }

    /// <summary>When <see cref="Dispose"/> gave up waiting for a step that was still running, the pacer releases the session as it leaves.</summary>
    private void FinishAfterDispose()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            DisposeSession();
        }
    }

    private void DisposeSession()
    {
        if (Interlocked.Exchange(ref _sessionDisposed, 1) == 0)
        {
            _session.Dispose();
        }
    }

    private void Record(double seconds)
    {
        lock (_computeSeconds)
        {
            if (_computeSeconds.Count < 65536)
            {
                _computeSeconds.Add(seconds);
            }
        }
        if (seconds > _maxComputeSeconds)
        {
            Volatile.Write(ref _maxComputeSeconds, seconds);
        }
    }

    /// <summary>Queues a frame. The channel drops its oldest frame when a reader has stalled (counted by the drop callback, not here).</summary>
    private void Publish(DuplexVoiceFrame frame) => _output.Writer.TryWrite(frame);

    private static int NextPowerOfTwo(int value)
    {
        int p = 1;
        while (p < value)
        {
            p <<= 1;
        }
        return p;
    }

    /// <summary>Stops the clock and releases the conversation.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _stop.Cancel();
        bool stopped = true;
        if (Volatile.Read(ref _started) != 0 && Thread.CurrentThread != _thread)
        {
            stopped = _thread.Join(TimeSpan.FromSeconds(5));
        }
        _output.Writer.TryComplete();
        if (stopped)
        {
            DisposeSession();
            _stop.Dispose();
        }
        else
        {
            Logs.Warning("[Duplex] A step was still running after 5 s; the pacer releases the conversation when it finishes.");
        }
    }
}
