using HartsyInference.Core.Runtime;
using HartsyInference.Tools;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Calls;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>A scripted call session: records the caller audio, keys and spoken lines the host hands it, plays queued
/// replies one turn at a time through the tagged read the real session has, and raises whatever events a test asks for.
/// With <see cref="EndlessTurn"/> set every read returns a full buffer, for the sender's cadence and allocation test.</summary>
internal sealed class FakeCallSession(uint callId, ToolRegistry tools) : IVoiceCallSession
{
    public const float EndlessLevel = 0.25f;

    private readonly object _lock = new();
    private readonly Queue<(int Turn, float[] Samples)> _replies = new();
    private readonly List<float> _inbound = [];
    private readonly List<char> _dtmf = [];
    private readonly List<string> _spoken = [];
    private readonly TaskCompletionSource _started = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private float[]? _current;
    private int _currentTurn;
    private int _offset;
    private int _endCalls;
    private int _disposed;
    private int _endlessTurn;

    public event Action<VoiceAgentEvent>? EventRaised;

    public uint CallId { get; } = callId;

    /// <summary>The tools the host bound to this call.</summary>
    public ToolRegistry Tools { get; } = tools;

    public int OutboundSampleRate => 16_000;

    /// <summary><see cref="Queued"/>, or <see cref="EndlessQueuedSamples"/> while <see cref="EndlessTurn"/> is set.</summary>
    public int OutboundQueuedSamples => EndlessTurn > 0 ? EndlessQueuedSamples : Queued;

    /// <summary>What <see cref="OutboundQueuedSamples"/> reports for an endless turn.</summary>
    public int EndlessQueuedSamples { get; set; }

    /// <summary>Thrown by <see cref="StartAsync"/>.</summary>
    public Exception? StartFailure { get; init; }

    /// <summary>When set, <see cref="StartAsync"/> waits for it, so a test can act while the session is still starting.</summary>
    public Task? StartGate { get; init; }

    /// <summary>When set, <see cref="EndAsync"/> completes only with it, so a test can act while the session is still ending.</summary>
    public Task? EndGate { get; init; }

    /// <summary>Thrown by the next <see cref="ReadOutbound"/>.</summary>
    public Exception? ReadFailure { get; set; }

    /// <summary>When positive, every read fills the whole buffer with <see cref="EndlessLevel"/>, tagged with this turn.</summary>
    public int EndlessTurn
    {
        get => Volatile.Read(ref _endlessTurn);
        set => Volatile.Write(ref _endlessTurn, value);
    }

    /// <summary>Completes when the host started the session.</summary>
    public Task Started => _started.Task;

    public bool EndCalled => Volatile.Read(ref _endCalls) > 0;

    public bool Disposed => Volatile.Read(ref _disposed) != 0;

    public float[] InboundSamples
    {
        get
        {
            lock (_lock)
            {
                return [.. _inbound];
            }
        }
    }

    public char[] Dtmf
    {
        get
        {
            lock (_lock)
            {
                return [.. _dtmf];
            }
        }
    }

    public string[] Spoken
    {
        get
        {
            lock (_lock)
            {
                return [.. _spoken];
            }
        }
    }

    /// <summary>Samples queued and not yet read.</summary>
    public int Queued
    {
        get
        {
            lock (_lock)
            {
                return (_current is null ? 0 : _current.Length - _offset) + _replies.Sum(r => r.Samples.Length);
            }
        }
    }

    /// <summary>Queues <paramref name="samples"/> of reply audio for <paramref name="turnId"/> at <paramref name="level"/>.</summary>
    public void QueueReply(int turnId, int samples, float level = 0.5f)
    {
        float[] audio = new float[samples];
        Array.Fill(audio, level);
        lock (_lock)
        {
            _replies.Enqueue((turnId, audio));
        }
    }

    /// <summary>Raises <paramref name="item"/> on the calling thread, the way the real session's pump would.</summary>
    public void Raise(VoiceAgentEvent item) => EventRaised?.Invoke(item);

    public void Raise(VoiceAgentEventKind kind, int turnId = 0, VoiceAgentState state = VoiceAgentState.Created, string? text = null,
        VoiceTurnMetrics? metrics = null) =>
        Raise(new VoiceAgentEvent { Kind = kind, TurnId = turnId, State = state, Text = text, Metrics = metrics, TimestampNs = MonotonicClock.NowNs() });

    public async Task StartAsync(CancellationToken cancel)
    {
        if (StartGate is not null)
        {
            await StartGate.WaitAsync(cancel);
        }
        if (StartFailure is not null)
        {
            throw StartFailure;
        }
        _started.TrySetResult();
    }

    public void PushInbound(ReadOnlySpan<float> samples)
    {
        lock (_lock)
        {
            foreach (float sample in samples)
            {
                _inbound.Add(sample);
            }
        }
    }

    public int ReadOutbound(Span<float> destination, out int turnId)
    {
        if (ReadFailure is { } failure)
        {
            ReadFailure = null;
            throw failure;
        }
        int endless = Volatile.Read(ref _endlessTurn);
        if (endless > 0)
        {
            destination.Fill(EndlessLevel);
            turnId = endless;
            return destination.Length;
        }
        lock (_lock)
        {
            if (_current is null || _offset == _current.Length)
            {
                if (!_replies.TryDequeue(out (int Turn, float[] Samples) next))
                {
                    _current = null;
                    destination.Clear();
                    turnId = 0;
                    return 0;
                }
                _current = next.Samples;
                _currentTurn = next.Turn;
                _offset = 0;
            }
            int count = Math.Min(destination.Length, _current.Length - _offset);
            _current.AsSpan(_offset, count).CopyTo(destination);
            destination[count..].Clear();
            _offset += count;
            turnId = _currentTurn;
            return count;
        }
    }

    public Task SpeakAsync(string text, CancellationToken cancel)
    {
        lock (_lock)
        {
            _spoken.Add(text);
        }
        return Task.CompletedTask;
    }

    public void PushDtmf(char digit)
    {
        lock (_lock)
        {
            _dtmf.Add(digit);
        }
    }

    /// <summary>Ends the session the way a failing audio thread does: <c>StateChanged(Ended)</c> with nobody asking.</summary>
    public void EndOnItsOwn() => Raise(VoiceAgentEventKind.StateChanged, state: VoiceAgentState.Ended);

    public Task EndAsync()
    {
        if (Interlocked.Increment(ref _endCalls) == 1)
        {
            Raise(VoiceAgentEventKind.StateChanged, state: VoiceAgentState.Ended);
        }
        return EndGate ?? Task.CompletedTask;
    }

    public ValueTask DisposeAsync()
    {
        Interlocked.Exchange(ref _disposed, 1);
        return ValueTask.CompletedTask;
    }
}
