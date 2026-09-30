using System.Threading.Channels;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Cpu;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice;

/// <summary>One phone call's voice agent: caller audio in, spoken replies out, with endpointing, barge-in, tool calls
/// and per-turn latency metrics.</summary>
/// <remarks>Three workers, and the order in which they may lock:
/// <list type="bullet">
/// <item>T1, the audio thread (<see cref="VoiceAudioWorker"/>): denoises and scores caller audio in 20 ms frames,
/// decides endpoints and barge-ins. Lock-free on its per-frame path and allocation-free after warm-up.</item>
/// <item>T2, the model set's GPU thread: recognition and synthesis, one job at a time, one device-gate hold per job.</item>
/// <item>T3, the turn loop (async): transcript → conversation → <see cref="ToolLoop"/> on the given
/// <see cref="ITextService"/> (thinking off) → sentences → synthesis on T2 → resampler → the outbound queue.</item>
/// </list>
/// The session's state lock is never held across an await; the rings and queues it may touch while holding it are
/// leaves, and T1 never takes it. Events are raised on a pool thread, never on T1 or T2, in order.
/// <para>Threading contract for the host: <see cref="PushInbound"/> from one producer thread and
/// <see cref="ReadOutbound"/> from one consumer thread; both never block. Everything else is thread-safe.</para></remarks>
public sealed partial class VoiceAgentSession : IAsyncDisposable
{
    private const int RingSeconds = 30;
    private const int MaxSynthesisInFlight = 2;
    private const string OverlapDiscardReason = "speech within the reply that did not barge in";

    private readonly VoiceModelSet _models;
    private readonly ITextService _text;
    private readonly ToolRegistry _tools;
    private readonly VoiceAgentOptions _options;
    private readonly ModelSpec _llm;
    private readonly VoiceTurnSignals _signals = new();
    private readonly CpuBackend _cpu = new();
    private readonly VoiceOutbound _outbound;
    private readonly VoiceAudioWorker _audio;
    private readonly VoiceConversation _conversation;
    private readonly Channel<VoiceTurnInput> _inputs = Channel.CreateUnbounded<VoiceTurnInput>(new UnboundedChannelOptions { SingleReader = true });
    private readonly Channel<VoiceAgentEvent> _events = Channel.CreateUnbounded<VoiceAgentEvent>(new UnboundedChannelOptions { SingleReader = true });
    private readonly CancellationTokenSource _ending = new();
    private readonly object _stateLock = new();
    private readonly List<VoiceTranscriptEntry> _transcript = [];
    private readonly Task _eventPump;
    private VoiceAgentState _state = VoiceAgentState.Created;
    private Task _turnLoop = Task.CompletedTask;
    private Task? _end;
    private int _endRequested;
    private int _turnCounter;
    private int _discardedUtterances;
    private long _reportedDrops;

    /// <summary>Creates a session on <paramref name="models"/>, answering with <paramref name="text"/> and offering the
    /// tools in <paramref name="tools"/>. The model and device fields of <paramref name="options"/> must match the
    /// model set's. Call <see cref="StartAsync"/> before pushing audio.</summary>
    public VoiceAgentSession(VoiceModelSet models, ITextService text, ToolRegistry tools, VoiceAgentOptions options)
        : this(models, text, tools, options, inboundCapacity: 0, outboundCapacity: 0)
    {
    }

    /// <summary>Test seam: ring capacities (powers of two; 0 selects 30 s).</summary>
    internal VoiceAgentSession(VoiceModelSet models, ITextService text, ToolRegistry tools, VoiceAgentOptions options,
        int inboundCapacity, int outboundCapacity)
    {
        ArgumentNullException.ThrowIfNull(models);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        RequireSameModels(models.Options, options);
        VoiceTurnOutput.Validate(models.SynthesisSampleRate, options.OutboundSampleRate);
        _models = models;
        _text = text;
        _tools = tools;
        _options = options;
        _llm = VoiceModelSet.ResolveLlm(options);
        _conversation = new VoiceConversation(options.SystemPrompt);
        long backlogLimit = (long)VoiceAudioFrontend.SampleRate * RingSeconds;
        inboundCapacity = inboundCapacity > 0 ? inboundCapacity : (int)BitOperations.RoundUpToPowerOf2((uint)backlogLimit);
        outboundCapacity = outboundCapacity > 0 ? outboundCapacity : (int)BitOperations.RoundUpToPowerOf2((uint)(options.OutboundSampleRate * RingSeconds));
        _outbound = new VoiceOutbound(outboundCapacity, _signals);
        IVadModel vad = models.CreateVadModel();
        RnnoiseStream? denoiser = null;
        try
        {
            denoiser = models.CreateDenoiser();
            VoiceAudioFrontend frontend = new(_cpu, vad, denoiser, _signals, options);
            _audio = new VoiceAudioWorker(frontend, _signals, new AudioSink(this), inboundCapacity, Math.Min(backlogLimit, inboundCapacity));
        }
        catch
        {
            (vad as IDisposable)?.Dispose();
            denoiser?.Dispose();
            _cpu.Dispose();
            throw;
        }
        // A single long-lived loop, not a fan-out: it delivers events in order on a pool thread.
        _eventPump = PumpEventsAsync();
    }

    /// <summary>Raised for every <see cref="VoiceAgentEvent"/>, in order, on a thread-pool thread. A throwing handler is
    /// logged and does not stop later events.</summary>
    public event Action<VoiceAgentEvent>? EventRaised;

    /// <summary>What the session is doing now.</summary>
    public VoiceAgentState State
    {
        get
        {
            lock (_stateLock)
            {
                return _state;
            }
        }
    }

    /// <summary>The call so far, oldest first.</summary>
    public IReadOnlyList<VoiceTranscriptEntry> Transcript
    {
        get
        {
            lock (_stateLock)
            {
                return [.. _transcript];
            }
        }
    }

    /// <summary>Rate of the audio <see cref="ReadOutbound"/> returns.</summary>
    public int OutboundSampleRate => _options.OutboundSampleRate;

    /// <summary>Caller samples lost because the audio thread fell behind.</summary>
    public long InboundDroppedSamples => _audio.DroppedSamples;

    /// <summary>Caller utterances not answered (<see cref="VoiceAgentEventKind.UtteranceDiscarded"/>).</summary>
    public int DiscardedUtterances => Volatile.Read(ref _discardedUtterances);

    /// <summary>Whether the per-frame path runs RNNoise.</summary>
    public bool Denoising => _audio.Frontend.Denoising;

    /// <summary>Starts the audio thread, warms its per-frame path and begins listening. A start that fails or is
    /// cancelled ends the session.</summary>
    public async Task StartAsync(CancellationToken cancel = default)
    {
        lock (_stateLock)
        {
            if (_state != VoiceAgentState.Created)
            {
                throw new InvalidOperationException($"The session was already started (state {_state}).");
            }
            SetStateLocked(VoiceAgentState.Warming, 0);
        }
        try
        {
            // Checked first: WaitAsync returns a finished warm-up without looking at the token.
            cancel.ThrowIfCancellationRequested();
            _audio.Start();
            await _audio.Ready.WaitAsync(cancel).ConfigureAwait(false);
        }
        catch
        {
            // Half-started is not a state a caller can use or retry: stop the audio thread and release the models,
            // keeping the start's own exception if that cleanup fails too.
            try
            {
                await EndAsync().ConfigureAwait(false);
            }
            catch (Exception endError)
            {
                Logs.Error("[Voice] Ending a session whose start failed also failed.", endError);
            }
            throw;
        }
        if (Volatile.Read(ref _endRequested) != 0)
        {
            // Ended while it was warming; there is nothing left to start.
            return;
        }
        // A single long-lived loop: turns run one after another, never in parallel.
        _turnLoop = RunTurnsAsync();
        SetState(VoiceAgentState.Listening, 0);
    }

    /// <summary>Queues caller audio: 16 kHz mono, ±1. Never blocks; when the audio thread is 30 s behind the oldest
    /// audio is dropped and counted in <see cref="InboundDroppedSamples"/>. Call from one thread.</summary>
    public void PushInbound(ReadOnlySpan<float> samples)
    {
        if (Volatile.Read(ref _endRequested) != 0)
        {
            return;
        }
        _audio.Push(samples);
    }

    /// <summary>Fills <paramref name="destination"/> with reply audio at <see cref="OutboundSampleRate"/>, ±1, and
    /// zero-fills what the queue could not supply. Returns how many samples were reply audio. Never blocks; call from one
    /// thread at the playback cadence. A barge-in drops the queued reply on the next call.</summary>
    public int ReadOutbound(Span<float> destination) => _outbound.Read(destination);

    /// <summary>Says <paramref name="text"/> without asking the model (a greeting, a hold message). It is queued behind
    /// any turn in progress, becomes part of the conversation, and can be barged in on. Completes when it has played
    /// or was interrupted; cancelled if the session ends first.</summary>
    public Task SpeakAsync(string text, CancellationToken cancel = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        TaskCompletionSource completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        if (!_inputs.Writer.TryWrite(VoiceTurnInput.Speak(text, completion, MonotonicClock.NowNs())))
        {
            throw new InvalidOperationException("The session has ended.");
        }
        return cancel.CanBeCanceled ? completion.Task.WaitAsync(cancel) : completion.Task;
    }

    /// <summary>A key the caller pressed (0-9, *, #, A-D): answered as the user message <c>[DTMF n]</c>, queued behind
    /// any turn in progress.</summary>
    public void PushDtmf(char digit)
    {
        char key = char.ToUpperInvariant(digit);
        if (key is not ((>= '0' and <= '9') or '*' or '#' or (>= 'A' and <= 'D')))
        {
            throw new ArgumentOutOfRangeException(nameof(digit), digit, "A DTMF key is 0-9, *, #, or A-D.");
        }
        if (!_inputs.Writer.TryWrite(VoiceTurnInput.Dtmf(key, MonotonicClock.NowNs())))
        {
            throw new InvalidOperationException("The session has ended.");
        }
    }

    /// <summary>Ends the call: cancels the turn in progress and any queued prompt, stops the audio thread, raises the
    /// last events and releases the session's models. Idempotent. A tool handler must not await it: the turn it runs in
    /// is part of what this waits for.</summary>
    public Task EndAsync()
    {
        TaskCompletionSource ended = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task? prior = Interlocked.CompareExchange(ref _end, ended.Task, null);
        if (prior is not null)
        {
            return prior;
        }
        _ = CompleteEndAsync(ended);
        return ended.Task;
    }

    /// <summary>Same as <see cref="EndAsync"/>.</summary>
    public ValueTask DisposeAsync() => new(EndAsync());

    private async Task CompleteEndAsync(TaskCompletionSource ended)
    {
        try
        {
            await EndCoreAsync().ConfigureAwait(false);
            ended.TrySetResult();
        }
        catch (Exception ex)
        {
            Logs.Error("[Voice] Ending the session failed.", ex);
            ended.TrySetException(ex);
        }
    }

    private async Task EndCoreAsync()
    {
        Volatile.Write(ref _endRequested, 1);
        _inputs.Writer.TryComplete();
        await _ending.CancelAsync().ConfigureAwait(false);
        _audio.Stop();
        await _turnLoop.ConfigureAwait(false);
        while (_inputs.Reader.TryRead(out VoiceTurnInput? pending))
        {
            pending.Completion?.TrySetCanceled();
        }
        _audio.Dispose();
        _cpu.Dispose();
        lock (_stateLock)
        {
            SetStateLocked(VoiceAgentState.Ended, 0);
        }
        _events.Writer.TryComplete();
        await _eventPump.ConfigureAwait(false);
    }

    private static void RequireSameModels(VoiceAgentOptions loaded, VoiceAgentOptions requested)
    {
        if (loaded.SttModel != requested.SttModel || loaded.TtsModel != requested.TtsModel || loaded.AudioDevice != requested.AudioDevice
            || loaded.Denoise != requested.Denoise || loaded.CpuThreadCap != requested.CpuThreadCap)
        {
            throw new ArgumentException(
                $"The session's models ({requested.SttModel}, {requested.TtsModel} on {requested.AudioDevice}, denoise {requested.Denoise}, "
                + $"cpu cap {requested.CpuThreadCap}) differ from the model set's ({loaded.SttModel}, {loaded.TtsModel} on {loaded.AudioDevice}, "
                + $"denoise {loaded.Denoise}, cpu cap {loaded.CpuThreadCap}).",
                nameof(requested));
        }
    }

    private void SetState(VoiceAgentState state, int turnId)
    {
        lock (_stateLock)
        {
            SetStateLocked(state, turnId);
        }
    }

    // The event is queued under the lock so state events keep the order of the transitions; the channel is a leaf.
    private void SetStateLocked(VoiceAgentState state, int turnId)
    {
        if (_state == state || _state == VoiceAgentState.Ended)
        {
            return;
        }
        _state = state;
        Emit(new VoiceAgentEvent { Kind = VoiceAgentEventKind.StateChanged, State = state, TurnId = turnId, TimestampNs = MonotonicClock.NowNs() });
    }

    private void Emit(VoiceAgentEvent item) => _events.Writer.TryWrite(item);

    private void Emit(VoiceAgentEventKind kind, int turnId, string? text = null, NativeToolCall? call = null) =>
        Emit(new VoiceAgentEvent { Kind = kind, TurnId = turnId, Text = text, ToolCall = call, TimestampNs = MonotonicClock.NowNs() });

    private void AddTranscript(int turnId, TextRole role, string text, bool interrupted)
    {
        lock (_stateLock)
        {
            _transcript.Add(new VoiceTranscriptEntry(turnId, role, text, interrupted));
        }
    }

    // No logging here: the audio thread calls this too.
    private void CountDiscarded(int turnId, string reason)
    {
        Interlocked.Increment(ref _discardedUtterances);
        Emit(VoiceAgentEventKind.UtteranceDiscarded, turnId, reason);
    }

    private async Task PumpEventsAsync()
    {
        await foreach (VoiceAgentEvent item in _events.Reader.ReadAllAsync().ConfigureAwait(false))
        {
            Action<VoiceAgentEvent>? handler = EventRaised;
            if (handler is null)
            {
                continue;
            }
            try
            {
                handler(item);
            }
            catch (Exception ex)
            {
                Logs.Error($"[Voice] A {item.Kind} event handler threw.", ex);
            }
        }
    }

    /// <summary>The audio thread's hand-offs: each only queues work for other threads.</summary>
    private sealed class AudioSink(VoiceAgentSession session) : IVoiceAudioSink
    {
        public void OnUtterance(VoiceTurnInput utterance)
        {
            if (!session._inputs.Writer.TryWrite(utterance))
            {
                Logs.Debug("[Voice] An utterance arrived after the session ended; dropped.");
            }
        }

        public void OnUtteranceDiscarded(int samples) => session.CountDiscarded(0, OverlapDiscardReason);

        public void OnBargeIn(int turnId, long detectNs) =>
            session.Emit(new VoiceAgentEvent { Kind = VoiceAgentEventKind.BargeIn, TurnId = turnId, TimestampNs = detectNs });

        public void OnAudioFault(Exception error)
        {
            session.Emit(new VoiceAgentEvent { Kind = VoiceAgentEventKind.Error, Text = error.Message, Error = error, TimestampNs = MonotonicClock.NowNs() });
            // The call cannot continue without its audio thread; end it from the pool, not from the failing thread.
            ThreadPool.UnsafeQueueUserWorkItem(static s => _ = s.EndAsync(), session, preferLocal: false);
        }
    }
}
