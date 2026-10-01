using System.Collections.Concurrent;
using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneLink;
using HartsyInference.Tools;
using HartsyInference.Voice;
using HartsyInference.VoiceHost.Link;
using HartsyInference.VoiceHost.Runtime;
using HartsyInference.VoiceHost.Tools;

namespace HartsyInference.VoiceHost.Calls;

/// <summary>One call on a gateway connection: its session, caller audio converted into it, the outbound state the sender
/// thread keeps, session events turned into link frames, and the telephony tools bound to it.</summary>
/// <remarks>Threads: the link reader pushes caller audio and DTMF and completes tool results; the sender thread alone reads
/// reply audio and owns the sender-only fields; session events arrive on a pool thread, in order; <see cref="EndAsync"/>
/// runs once from whichever of them ends the call. Any failure of the session, from its start to its last frame, ends
/// this call with <c>CallEnd(Failed)</c> and goes no further: the connection and the other calls carry on.
/// <para>Barge-in: the session's <c>BargeIn(T)</c> becomes <c>Flush(T)</c> on the sender, which from then on drops any
/// audio tagged T or lower that the session still hands it, so nothing of the cancelled reply follows the flush on the
/// wire. A turn's <c>OutboundEnd</c> goes out once its last audio has, because the gateway's resampler holds the last
/// frame until it.</para>
/// <para>Hang-up: the gateway sends its BYE the moment it runs <c>hangup</c>, so the host asks only once the goodbye
/// has played. The request belongs to one turn: the handler marks that the tool ran, and the session's
/// <c>ToolResult</c> event for it names the turn (the events arrive in order, after the handler and before that turn's
/// <c>TurnCompleted</c>); a <c>ToolResult</c> without the handler having run (the tool disabled) arms nothing, and a
/// later turn finishing first disarms it rather than hang up on a caller who is still talking. When the turn that
/// called the tool ends, the host waits for that turn's audio to drain to the gateway
/// (<see cref="DrainedTurn"/>: its <c>OutboundEnd</c> written, or the turn flushed, or it had no audio), then
/// <see cref="PhoneLinkServerOptions.HangupMarginMs"/> for what is still queued downstream; the whole wait is capped
/// at the audio the session still held plus <see cref="PhoneLinkServerOptions.HangupSlackMs"/>, and never exceeds
/// <see cref="PhoneLinkServerOptions.HangupMaxWaitMs"/>, so a goodbye that never drains cannot keep the call open. A
/// barge-in does not cancel it: the model chose to end the call, the flush empties the goodbye, and the BYE goes out at
/// once.</para></remarks>
internal sealed class VoiceCall
{
    private const float FromPcm16 = 1f / 32768f;
    private const int HangupPollMs = 10;

    private readonly LinkConnection _connection;
    private readonly IVoiceCallSessionFactory _factory;
    private readonly PhoneLinkServerOptions _options;
    private readonly CallStartMessage _start;
    private readonly short[] _inboundPcm = new short[LinkProtocol.InboundFrameSamples];
    private readonly float[] _inboundFloat = new float[LinkProtocol.InboundFrameSamples];
    private readonly ConcurrentDictionary<uint, TaskCompletionSource<ToolResultMessage>> _pendingTools = new();
    private readonly object _lifecycleLock = new();
    private readonly long _startedNs = MonotonicClock.NowNs();
    private IVoiceCallSession? _session;
    private Task? _end;
    private bool _ending;
    private bool _counted;
    private volatile bool _audioReady;
    private int _completedTurn;
    private int _flushedTurn;
    private int _drainedTurn;
    private int _hangupRequested;
    private int _hangupTurn;
    private long _inboundFrames;
    private long _inboundConcealed;
    private long _inboundBeforeStart;
    private long _outboundSamples;
    private long _staleSamples;
    private long _flushes;

    public VoiceCall(LinkConnection connection, uint callId, CallStartMessage start, IVoiceCallSessionFactory factory, PhoneLinkServerOptions options)
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentNullException.ThrowIfNull(start);
        ArgumentNullException.ThrowIfNull(factory);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfZero(callId);
        _connection = connection;
        CallId = callId;
        _start = start;
        _factory = factory;
        _options = options;
    }

    public uint CallId { get; }

    /// <summary>True when the gateway re-attached this call after the link dropped.</summary>
    public bool Resumed => _start.Resume;

    /// <summary>True once the session has started and until the call begins ending: caller audio is pushed and reply
    /// audio read only then.</summary>
    public bool AudioReady => _audioReady;

    /// <summary>True once <see cref="EndAsync"/> has been called.</summary>
    public bool IsEnding => Volatile.Read(ref _end) is not null;

    /// <summary>Completes when the call has fully ended.</summary>
    public Task Ended => Volatile.Read(ref _end) ?? Task.CompletedTask;

    /// <summary>The latest turn the session reported finished.</summary>
    public int CompletedTurn => Volatile.Read(ref _completedTurn);

    /// <summary>Samples of reply audio sent to the gateway.</summary>
    public long OutboundSamples => Volatile.Read(ref _outboundSamples);

    /// <summary>Samples the sender dropped because a flush had superseded their turn.</summary>
    public long StaleSamples => Volatile.Read(ref _staleSamples);

    /// <summary>Caller frames pushed into the session.</summary>
    public long InboundFrames => Volatile.Read(ref _inboundFrames);

    /// <summary>Caller frames that arrived before the session had started, and were dropped.</summary>
    public long InboundBeforeStart => Volatile.Read(ref _inboundBeforeStart);

    /// <summary>The session, once created; for tests.</summary>
    internal IVoiceCallSession? Session => Volatile.Read(ref _session);

    /// <summary>The highest turn flushed on the wire. Written by the sender only.</summary>
    internal int FlushedTurn => Volatile.Read(ref _flushedTurn);

    /// <summary>The latest turn whose reply audio is all on the link: that turn and every earlier one had its
    /// <c>OutboundEnd</c> written, was flushed, or had no audio. Written by the sender only.</summary>
    public int DrainedTurn => Volatile.Read(ref _drainedTurn);

    /// <summary>Sender-only: the turn whose audio went out and whose <c>OutboundEnd</c> has not; 0 for none.</summary>
    internal int OpenTurn { get; set; }

    /// <summary>Sender-only: reply audio went out on the previous tick, so no prebuffer is due.</summary>
    internal bool Bursting { get; set; }

    /// <summary>Sender-only: <c>CallEnd</c> went out; nothing more is sent for this call.</summary>
    internal bool EndSent { get; set; }

    /// <summary>Creates and starts the session off the link reader thread, then says the greeting or, on a resumed call,
    /// the apology. A failure ends the call with <c>CallEnd(Failed)</c>.</summary>
    public async Task StartAsync()
    {
        try
        {
            await StartCoreAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // The expected failures are handled inside; this keeps an unexpected one from leaving the call open unobserved.
            Logs.Error($"[VoiceHost] Call {CallId}: starting it failed unexpectedly", ex);
            await EndAsync(LinkCallEndReason.Failed, "the call failed to start").ConfigureAwait(false);
        }
    }

    private async Task StartCoreAsync()
    {
        await Task.Yield();
        lock (_lifecycleLock)
        {
            if (_ending)
            {
                return;
            }
            HostRuntimeTuning.CallStarted();
            _counted = true;
        }
        IVoiceCallSession session;
        try
        {
            ToolRegistry tools = VoiceHostTools.Build(_options.Tools, RequestToolAsync, RequestHangup, _options.Clock);
            session = _factory.Create(CallId, tools);
            session.EventRaised += OnSessionEvent;
        }
        catch (Exception ex)
        {
            Logs.Error($"[VoiceHost] Call {CallId}: its voice session could not be created", ex);
            await EndAsync(LinkCallEndReason.Failed, "the voice session could not be created").ConfigureAwait(false);
            return;
        }
        lock (_lifecycleLock)
        {
            if (!_ending)
            {
                _session = session;
            }
        }
        if (Volatile.Read(ref _session) != session)
        {
            // Ended while the session was being built; nothing else will release it.
            await ReleaseAsync(session).ConfigureAwait(false);
            return;
        }
        try
        {
            await session.StartAsync(CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[VoiceHost] Call {CallId}: its voice session failed to start", ex);
            await EndAsync(LinkCallEndReason.Failed, "the voice session failed to start").ConfigureAwait(false);
            return;
        }
        lock (_lifecycleLock)
        {
            _audioReady = !_ending;
        }
        Logs.Info($"[VoiceHost] Call {CallId} ({_start.Direction}{(Resumed ? ", resumed" : "")}) is up.");
        string? line = Resumed ? _options.ResumeApology : _options.Greeting;
        if (_audioReady && !string.IsNullOrWhiteSpace(line))
        {
            _ = SayAsync(session, line);
        }
    }

    /// <summary>Reader thread: one 20 ms caller frame into the session, PCM16 → ±1.</summary>
    public void PushInbound(in LinkFrame frame)
    {
        int count = frame.ReadPcm(_inboundPcm);
        IVoiceCallSession? session = Volatile.Read(ref _session);
        if (!_audioReady || session is null)
        {
            Volatile.Write(ref _inboundBeforeStart, _inboundBeforeStart + 1);
            return;
        }
        for (int i = 0; i < count; i++)
        {
            _inboundFloat[i] = _inboundPcm[i] * FromPcm16;
        }
        try
        {
            session.PushInbound(_inboundFloat.AsSpan(0, count));
        }
        catch (Exception ex)
        {
            Fault("taking caller audio", ex);
            return;
        }
        Volatile.Write(ref _inboundFrames, _inboundFrames + 1);
        if (frame.Concealed)
        {
            Volatile.Write(ref _inboundConcealed, _inboundConcealed + 1);
        }
    }

    /// <summary>Reader thread: a key the caller pressed.</summary>
    public void PushDtmf(LinkDtmf dtmf)
    {
        IVoiceCallSession? session = Volatile.Read(ref _session);
        if (!_audioReady || session is null)
        {
            return;
        }
        try
        {
            session.PushDtmf(dtmf.Digit);
        }
        catch (InvalidOperationException)
        {
            // The session ended between the check and the push; the call is ending with it.
        }
    }

    /// <summary>Reader thread: the gateway answered one of this call's tool requests.</summary>
    public void CompleteTool(uint requestId, ToolResultMessage result)
    {
        if (_pendingTools.TryRemove(requestId, out TaskCompletionSource<ToolResultMessage>? pending))
        {
            pending.TrySetResult(result);
            return;
        }
        Logs.Debug($"[VoiceHost] Call {CallId}: a ToolResult for request {requestId} arrived after it was given up.");
    }

    /// <summary>Reader thread: the gateway applied a flush.</summary>
    public void OnFlushAck(LinkFlushAck ack) =>
        Logs.Debug($"[VoiceHost] Call {CallId}: the gateway flushed turn {ack.TurnId}, dropping {ack.MsDiscarded} ms.");

    /// <summary>Sender thread: reply audio of one turn, tagged with it.</summary>
    public int ReadOutbound(Span<float> destination, out int turnId)
    {
        IVoiceCallSession? session = Volatile.Read(ref _session);
        if (session is null)
        {
            turnId = 0;
            return 0;
        }
        return session.ReadOutbound(destination, out turnId);
    }

    /// <summary>Sender thread: <c>Flush(turn)</c> is on the wire; audio of that turn or older is dropped from here on, so
    /// none of it is left to play.</summary>
    internal void MarkFlushed(int turn)
    {
        if (turn > _flushedTurn)
        {
            Volatile.Write(ref _flushedTurn, turn);
        }
        if (OpenTurn <= _flushedTurn)
        {
            OpenTurn = 0;
        }
        MarkDrained(turn);
        Volatile.Write(ref _flushes, _flushes + 1);
    }

    /// <summary>Sender thread: every turn up to <paramref name="turn"/> has all its reply audio on the link.</summary>
    internal void MarkDrained(int turn)
    {
        if (turn > _drainedTurn)
        {
            Volatile.Write(ref _drainedTurn, turn);
        }
    }

    /// <summary>Sender thread: bookkeeping for audio sent or dropped as stale.</summary>
    internal void CountOutbound(int sent, int stale)
    {
        if (sent > 0)
        {
            Volatile.Write(ref _outboundSamples, _outboundSamples + sent);
        }
        if (stale > 0)
        {
            Volatile.Write(ref _staleSamples, _staleSamples + stale);
        }
    }

    /// <summary>The session failed on a link thread: log it once and end the call from the pool.</summary>
    internal void Fault(string doing, Exception error)
    {
        if (IsEnding)
        {
            return;
        }
        Logs.Error($"[VoiceHost] Call {CallId}: the voice session failed {doing}; ending the call", error);
        ThreadPool.UnsafeQueueUserWorkItem(static call => _ = call.EndAsync(LinkCallEndReason.Failed, "the voice session failed"), this, preferLocal: false);
    }

    /// <summary>Ends the call once: stops its audio, tells the gateway when <paramref name="tellGateway"/> is set (not when
    /// the gateway ended it or the link is gone), gives up its pending tool requests and ends the session.</summary>
    public Task EndAsync(LinkCallEndReason? tellGateway, string why)
    {
        TaskCompletionSource ended = new(TaskCreationOptions.RunContinuationsAsynchronously);
        Task? prior = Interlocked.CompareExchange(ref _end, ended.Task, null);
        if (prior is not null)
        {
            return prior;
        }
        _ = CompleteEndAsync(ended, tellGateway, why);
        return ended.Task;
    }

    private async Task CompleteEndAsync(TaskCompletionSource ended, LinkCallEndReason? tellGateway, string why)
    {
        IVoiceCallSession? session;
        lock (_lifecycleLock)
        {
            _ending = true;
            _audioReady = false;
            session = _session;
        }
        _connection.RemoveCall(this);
        if (tellGateway is { } reason)
        {
            _connection.Enqueue(new HostControlItem(LinkMessageType.CallEnd, CallId, (uint)reason, 0, null, this));
        }
        foreach (uint requestId in _pendingTools.Keys)
        {
            if (_pendingTools.TryRemove(requestId, out TaskCompletionSource<ToolResultMessage>? pending))
            {
                pending.TrySetResult(new ToolResultMessage { Status = LinkToolStatus.Failed, Message = "The call has ended." });
            }
        }
        if (session is not null)
        {
            await ReleaseAsync(session).ConfigureAwait(false);
        }
        if (_counted)
        {
            HostRuntimeTuning.CallEnded();
        }
        double seconds = (MonotonicClock.NowNs() - _startedNs) / 1e9;
        Logs.Info(
            $"[VoiceHost] Call {CallId} ended ({why}) after {seconds.ToString("0.0", CultureInfo.InvariantCulture)} s: inbound frames={InboundFrames} "
            + $"concealed={Volatile.Read(ref _inboundConcealed)} beforeStart={InboundBeforeStart}; outbound samples={OutboundSamples} "
            + $"stale={StaleSamples} flushes={Volatile.Read(ref _flushes)}.");
        ended.TrySetResult();
    }

    /// <summary>Ends and disposes a session, each step bounded, so a wedged session cannot hold the host.</summary>
    private async Task ReleaseAsync(IVoiceCallSession session)
    {
        TimeSpan limit = TimeSpan.FromMilliseconds(_options.CallEndTimeoutMs);
        try
        {
            await session.EndAsync().WaitAsync(limit).ConfigureAwait(false);
            await session.DisposeAsync().AsTask().WaitAsync(limit).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            Logs.Error($"[VoiceHost] Call {CallId}: its voice session did not stop within {_options.CallEndTimeoutMs} ms; abandoning it.");
        }
        catch (Exception ex)
        {
            Logs.Error($"[VoiceHost] Call {CallId}: ending its voice session failed", ex);
        }
        finally
        {
            session.EventRaised -= OnSessionEvent;
        }
    }

    private async Task SayAsync(IVoiceCallSession session, string line)
    {
        try
        {
            await session.SpeakAsync(line, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is OperationCanceledException or InvalidOperationException && IsEnding)
        {
            Logs.Debug($"[VoiceHost] Call {CallId}: the opening line was cut short by the end of the call.");
        }
        catch (Exception ex)
        {
            Logs.Warning($"[VoiceHost] Call {CallId}: the opening line did not play: {ex.Message}");
        }
    }

    /// <summary>Session events, on the session's pool thread and in order, as link frames and call decisions.</summary>
    private void OnSessionEvent(VoiceAgentEvent item)
    {
        uint? turn = item.TurnId > 0 ? (uint)item.TurnId : null;
        switch (item.Kind)
        {
            case VoiceAgentEventKind.StateChanged when item.State == VoiceAgentState.Ended:
                if (!IsEnding)
                {
                    _ = EndAsync(LinkCallEndReason.Failed, "the voice session ended on its own");
                }
                return;
            case VoiceAgentEventKind.StateChanged:
                SendEvent(new LinkEventMessage { Kind = LinkEventKind.State, State = item.State.ToString(), TurnId = turn });
                return;
            case VoiceAgentEventKind.UserTranscript:
                SendEvent(new LinkEventMessage { Kind = LinkEventKind.TranscriptFinal, Text = item.Text, TurnId = turn });
                return;
            case VoiceAgentEventKind.BargeIn when turn is { } flushed:
                _connection.Enqueue(new HostControlItem(LinkMessageType.Flush, CallId, flushed, 0, null, this));
                return;
            case VoiceAgentEventKind.TurnCompleted:
                OnTurnCompleted(item);
                return;
            case VoiceAgentEventKind.ToolResult when item.ToolCall?.Name == VoiceHostTools.Hangup:
                OnHangupResult(item.TurnId);
                return;
            case VoiceAgentEventKind.Error:
                Logs.Warning($"[VoiceHost] Call {CallId}, turn {item.TurnId}: {item.Text}");
                return;
            default:
                Logs.Debug($"[VoiceHost] Call {CallId}: {item.Kind} (turn {item.TurnId}).");
                return;
        }
    }

    private void OnTurnCompleted(VoiceAgentEvent item)
    {
        Volatile.Write(ref _completedTurn, Math.Max(Volatile.Read(ref _completedTurn), item.TurnId));
        if (item.Metrics is { } metrics && metrics.TotalMs is double total)
        {
            SendEvent(new LinkEventMessage
            {
                Kind = LinkEventKind.TurnLatency,
                TurnId = (uint)item.TurnId,
                Latency = new TurnLatency
                {
                    SttMs = Ms(metrics.SttMs),
                    LlmFirstTokenMs = Ms(metrics.LlmTtftMs),
                    LlmFirstSentenceMs = Ms(metrics.LlmFirstSentenceMs),
                    TtsFirstChunkMs = Ms(metrics.TtsFirstChunkMs),
                    TransportMs = Ms(metrics.TransportMs),
                    TotalMs = (int)Math.Round(total),
                },
            });
        }
        int armed = Volatile.Read(ref _hangupTurn);
        if (armed == 0 || item.TurnId < armed)
        {
            return;
        }
        Volatile.Write(ref _hangupTurn, 0);
        if (item.TurnId > armed)
        {
            Logs.Warning($"[VoiceHost] Call {CallId}: turn {armed} asked to hang up but never finished, and turn {item.TurnId} did; staying on the line.");
            return;
        }
        // Interrupted or not: the model chose to end the call, and a barge-in only empties the goodbye.
        _ = HangUpAfterGoodbyeAsync(item.TurnId);
    }

    /// <summary>The session recorded a <c>hangup</c> result for <paramref name="turn"/>: arm the hang-up for that turn, if
    /// the handler really ran (a disabled tool's error result also carries the name).</summary>
    private void OnHangupResult(int turn)
    {
        if (Interlocked.Exchange(ref _hangupRequested, 0) != 0 && turn > 0)
        {
            Volatile.Write(ref _hangupTurn, turn);
        }
    }

    private void SendEvent(LinkEventMessage message) =>
        _connection.Enqueue(new HostControlItem(LinkMessageType.Event, CallId, 0, 0, message, this));

    /// <summary>The <c>hangup</c> tool's request, sent once the goodbye of <paramref name="turn"/>, the turn that called
    /// it, has played: its audio drained to the gateway plus the downstream margin, or the cap (see the class remarks).
    /// The gateway sends its BYE and answers, then the host ends the call on its side too, whatever the answer. Nobody
    /// awaits it, so anything unexpected ends the call with <c>CallEnd(Failed)</c> instead of leaving it open.</summary>
    private async Task HangUpAfterGoodbyeAsync(int turn)
    {
        try
        {
            await HangUpAfterGoodbyeCoreAsync(turn).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[VoiceHost] Call {CallId}: hanging up after turn {turn} failed; ending the call", ex);
            await EndAsync(LinkCallEndReason.Failed, "the hang-up failed").ConfigureAwait(false);
        }
    }

    private async Task HangUpAfterGoodbyeCoreAsync(int turn)
    {
        long startNs = MonotonicClock.NowNs();
        IVoiceCallSession? session = Session;
        long queuedMs = session is null ? 0 : (long)session.OutboundQueuedSamples * 1000 / Math.Max(1, session.OutboundSampleRate);
        long capMs = Math.Min(_options.HangupMaxWaitMs, queuedMs + _options.HangupSlackMs);
        long deadlineNs = startNs + capMs * 1_000_000L;
        bool drained = await PollAsync(() => DrainedTurn >= turn, deadlineNs).ConfigureAwait(false);
        if (drained && FlushedTurn < turn)
        {
            // The goodbye went out whole: let the prebuffer lead, the gateway's held frame and the far end's jitter
            // buffer play it before the BYE.
            await PollAsync(static () => false, Math.Min(MonotonicClock.NowNs() + _options.HangupMarginMs * 1_000_000L, deadlineNs)).ConfigureAwait(false);
        }
        if (IsEnding)
        {
            return;
        }
        long waitedMs = (MonotonicClock.NowNs() - startNs) / 1_000_000L;
        if (drained)
        {
            Logs.Info($"[VoiceHost] Call {CallId}: the goodbye of turn {turn} has played; hanging up ({waitedMs} ms after the turn ended).");
        }
        else
        {
            Logs.Warning($"[VoiceHost] Call {CallId}: the goodbye of turn {turn} was still going out after {capMs} ms; hanging up anyway.");
        }
        ToolResultMessage result = await RequestToolAsync(new ToolRequestMessage { Name = VoiceHostTools.Hangup }, CancellationToken.None).ConfigureAwait(false);
        if (result.Status != LinkToolStatus.Ok)
        {
            Logs.Warning($"[VoiceHost] Call {CallId}: the gateway answered hangup with {result.Status} ({result.Message}); ending the call from the host.");
        }
        await EndAsync(LinkCallEndReason.Completed, "the agent hung up").ConfigureAwait(false);
    }

    /// <summary>The <c>hangup</c> handler ran; the session's <c>ToolResult</c> event that follows names the turn.</summary>
    private void RequestHangup() => Volatile.Write(ref _hangupRequested, 1);

    /// <summary>Waits until <paramref name="condition"/> holds (true), or until <paramref name="deadlineNs"/> or the end of
    /// the call (false). It polls, so the sender, which publishes what it waits on, never signals and never allocates.</summary>
    private async Task<bool> PollAsync(Func<bool> condition, long deadlineNs)
    {
        while (!condition())
        {
            long leftMs = (deadlineNs - MonotonicClock.NowNs()) / 1_000_000L;
            if (leftMs <= 0 || IsEnding)
            {
                return false;
            }
            await Task.Delay((int)Math.Min(leftMs, HangupPollMs)).ConfigureAwait(false);
        }
        return true;
    }

    /// <summary>Sends a telephony request for this call and waits for the gateway's answer, up to the tool timeout; a
    /// timeout or the end of the call comes back as <c>Failed</c> so the model hears why.</summary>
    private async Task<ToolResultMessage> RequestToolAsync(ToolRequestMessage request, CancellationToken cancel)
    {
        if (IsEnding)
        {
            return new ToolResultMessage { Status = LinkToolStatus.Failed, Message = "The call has ended." };
        }
        uint requestId = _connection.NextRequestId();
        TaskCompletionSource<ToolResultMessage> pending = new(TaskCreationOptions.RunContinuationsAsynchronously);
        _pendingTools[requestId] = pending;
        try
        {
            _connection.Enqueue(new HostControlItem(LinkMessageType.ToolRequest, CallId, requestId, 0, request, this));
            return await pending.Task.WaitAsync(TimeSpan.FromMilliseconds(_options.ToolTimeoutMs), cancel).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            Logs.Warning($"[VoiceHost] Call {CallId}: the gateway did not answer {request.Name} within {_options.ToolTimeoutMs} ms.");
            return new ToolResultMessage { Status = LinkToolStatus.Failed, Message = $"The phone system did not answer within {_options.ToolTimeoutMs} ms." };
        }
        finally
        {
            _pendingTools.TryRemove(requestId, out _);
        }
    }

    private static int? Ms(double? value) => value is double ms ? (int)Math.Round(ms) : null;
}
