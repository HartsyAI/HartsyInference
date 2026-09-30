using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Media;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Transport;
using HartsyInference.PhoneLink;
using SIPSorcery.SIP;
using SIPSorcery.SIP.App;
using SIPSorcery.Sys;
using SIPSorceryMedia.Abstractions;

namespace HartsyInference.PhoneGateway.Sip;

/// <summary>The one-call state machine: answers or places a call, builds its media (tick source, jitter buffer, pump,
/// outbound path, prompts), announces it to the voice host, relays DTMF and telephony tools, and tears everything
/// down exactly once whichever side ends the call.</summary>
/// <remarks>Inbound screening runs before anything is allocated: a second INVITE while a call is up gets <c>486 Busy
/// Here</c>, a caller the policy refuses gets <c>603 Decline</c>, and an INVITE while the host link is down gets
/// <c>503</c>. Tool requests arrive on the link reader thread and run on the pool so that thread never waits on SIP.
/// <c>OnCallHungup</c> fires for a remote BYE and for our own <c>Hangup()</c>, so the end reason is decided before
/// hanging up and teardown is guarded by the state under <c>_stateLock</c>, which is never held across an await or
/// a sipsorcery call.</remarks>
public sealed class CallController : IDisposable
{
    private readonly SipAccount _account;
    private readonly EngineLink _link;
    private readonly CallControllerOptions _options;
    private readonly PromptPlayer _prompts;
    private readonly GatewayMetrics _metrics;
    private readonly LinkOutageGuard _guard;
    private readonly object _stateLock = new();
    private SIPUserAgent? _agent;
    private CallState _state;
    private ActiveCall? _current;
    private uint _nextCallId;
    private LinkCallEndReason? _pendingEndReason;
    private bool _pendingTellHost = true;
    private string? _ringingSipCallId;
    private string _lastCallFailure = "";

    public CallController(SipAccount account, EngineLink link, CallControllerOptions options, PromptPlayer prompts, GatewayMetrics metrics)
    {
        ArgumentNullException.ThrowIfNull(account);
        ArgumentNullException.ThrowIfNull(link);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(prompts);
        ArgumentNullException.ThrowIfNull(metrics);
        if (options.RtpPortStart < 1024 || options.RtpPortEnd > 65534 || options.RtpPortEnd <= options.RtpPortStart)
        {
            throw new ArgumentException($"RTP port range {options.RtpPortStart}-{options.RtpPortEnd} is not valid.", nameof(options));
        }
        if (options.GreetingPromptFile is not null && !File.Exists(options.GreetingPromptFile))
        {
            throw new FileNotFoundException("sip.greetingPromptFile does not exist.", options.GreetingPromptFile);
        }
        _account = account;
        _link = link;
        _options = options;
        _prompts = prompts;
        _metrics = metrics;
        _guard = new LinkOutageGuard(options.Outage)
        {
            OutageStarted = metrics.Outage,
            PlayPrompt = PlayGuardPrompt,
            ResumeCall = ResumeCall,
            HangUp = () =>
            {
                metrics.OutageHangup();
                HangUp(LinkCallEndReason.Failed);
            },
        };
    }

    public CallState State
    {
        get
        {
            lock (_stateLock)
            {
                return _state;
            }
        }
    }

    /// <summary>The live call, or null.</summary>
    public CallSummary? Current
    {
        get
        {
            lock (_stateLock)
            {
                return _current is null ? null : new CallSummary(_current.CallId, _current.SipCallId, _current.Direction, _current.Remote, _current.StartedUtc, _state);
            }
        }
    }

    /// <summary>Why the last outbound call attempt failed, for the admin endpoint.</summary>
    public string LastCallFailure => Volatile.Read(ref _lastCallFailure);

    public void Start()
    {
        lock (_stateLock)
        {
            if (_agent is not null)
            {
                throw new InvalidOperationException("CallController is already started.");
            }
            SIPUserAgent agent = new(_account.Transport, null);
            agent.OnIncomingCall += OnIncomingCall;
            agent.OnCallHungup += OnCallHungup;
            agent.OnDtmfTone += OnDtmfTone;
            agent.ClientCallFailed += (_, message, _) => Volatile.Write(ref _lastCallFailure, message ?? "");
            _agent = agent;
        }
        // sipsorcery's user agent ignores a new INVITE while it has a call, so the busy answer is given at the transport.
        _account.Transport.SIPTransportRequestReceived += OnTransportRequest;
        _link.Connected = OnLinkConnected;
        _link.Disconnected = _ => _guard.LinkDisconnected();
        _link.OutboundAudio = OnOutboundAudio;
        _link.OutboundEnd = OnOutboundEnd;
        _link.Flush = OnFlush;
        _link.CallEnd = OnHostCallEnd;
        _link.ToolRequest = OnToolRequest;
        _link.Event = (_, e) => Logs.Info($"[PhoneGateway] host event {e.Kind}{(e.State is null ? "" : " " + e.State)}{(e.Text is null ? "" : ": " + e.Text)}");
        _link.Error = (callId, e) => Logs.Error($"[PhoneGateway] host error (call {callId}): {e.Text}");
        _metrics.MediaProbe = SnapshotMedia;
    }

    /// <summary>Places a call. <paramref name="destination"/> is a SIP URI, or a number dialled through the registrar.</summary>
    public async Task<CallPlacementResult> PlaceCallAsync(string destination)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(destination);
        SIPUserAgent agent = _agent ?? throw new InvalidOperationException("CallController is not started.");
        string uri = destination.Contains(':') ? destination : $"sip:{destination}@{_account.Registrar}";
        if (!destination.Contains(':') && _account.Registrar.Length == 0)
        {
            return new CallPlacementResult(false, "a bare number needs a registrar; give a full sip: URI");
        }
        lock (_stateLock)
        {
            if (_state != CallState.Idle)
            {
                return new CallPlacementResult(false, "busy");
            }
            if (!_link.IsConnected)
            {
                return new CallPlacementResult(false, "voice host unavailable");
            }
            _state = CallState.Ringing;
        }
        ActiveCall call = CreateCall(LinkCallDirection.Outbound, _account.Username, destination, "");
        bool answered;
        try
        {
            Volatile.Write(ref _lastCallFailure, "");
            answered = await agent.Call(uri, _account.Username, _account.Password, call.Session, _options.RingTimeoutSeconds).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Outbound call to {uri} failed", ex);
            Volatile.Write(ref _lastCallFailure, ex.Message);
            answered = false;
        }
        if (!answered)
        {
            AbandonRinging(call);
            _metrics.Failed();
            return new CallPlacementResult(false, LastCallFailure.Length == 0 ? "not answered" : LastCallFailure);
        }
        call.SipCallId = agent.Dialogue?.CallId ?? "";
        Activate(call);
        return CallPlacementResult.Ok;
    }

    /// <summary>Ends the live call from our side with <paramref name="reason"/> and tells the host.</summary>
    public void HangUp(LinkCallEndReason reason) => HangUp(reason, tellHost: true);

    private void HangUp(LinkCallEndReason reason, bool tellHost)
    {
        SIPUserAgent? agent;
        lock (_stateLock)
        {
            if (_state is CallState.Idle or CallState.Ending)
            {
                return;
            }
            _pendingEndReason = reason;
            _pendingTellHost = tellHost;
            agent = _agent;
        }
        try
        {
            if (agent is not null && agent.IsCalling)
            {
                agent.Cancel();
            }
            else
            {
                agent?.Hangup();
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] SIP hangup failed; tearing the call down anyway", ex);
        }
        EndCall(reason, tellHost);
    }

    /// <summary>Answers a new INVITE with 486 while a call is up or being set up. Re-INVITEs (with a To tag) and the
    /// INVITE currently being answered are left to the user agent.</summary>
    private Task OnTransportRequest(SIPEndPoint localEndPoint, SIPEndPoint remoteEndPoint, SIPRequest request)
    {
        if (request.Method != SIPMethodsEnum.INVITE || !string.IsNullOrEmpty(request.Header.To?.ToTag))
        {
            return Task.CompletedTask;
        }
        string? callId = request.Header.CallId;
        lock (_stateLock)
        {
            if (_state == CallState.Idle || callId == _ringingSipCallId || (_current is not null && callId == _current.SipCallId))
            {
                return Task.CompletedTask;
            }
        }
        Logs.Info($"[PhoneGateway] INVITE from {request.Header.From?.FromURI?.User} while busy: 486 Busy Here.");
        _metrics.RejectedBusy();
        SIPResponse busy = SIPResponse.GetResponse(request, SIPResponseStatusCodesEnum.BusyHere, null);
        return _account.Transport.SendResponseAsync(busy);
    }

    private void OnIncomingCall(SIPUserAgent agent, SIPRequest request)
    {
        string? caller = request.Header.From?.FromURI?.User;
        string? called = request.URI?.User;
        lock (_stateLock)
        {
            if (_state != CallState.Idle)
            {
                // Normally answered by OnTransportRequest before the user agent sees it.
                _metrics.RejectedBusy();
                Reject(agent, request, SIPResponseStatusCodesEnum.BusyHere, "Busy Here");
                return;
            }
            if (!_link.IsConnected)
            {
                Logs.Warning($"[PhoneGateway] INVITE from {caller} while the voice host is down: 503.");
                _metrics.RejectedHostDown();
                Reject(agent, request, SIPResponseStatusCodesEnum.ServiceUnavailable, "Voice host unavailable");
                return;
            }
            if (!IsAllowed(caller))
            {
                Logs.Info($"[PhoneGateway] INVITE from {caller} declined by policy {_options.InboundPolicy}: 603.");
                _metrics.Declined();
                Reject(agent, request, SIPResponseStatusCodesEnum.Decline, "Decline");
                return;
            }
            _state = CallState.Ringing;
            _ringingSipCallId = request.Header.CallId;
        }
        _ = AnswerAsync(agent, request, caller, called);
    }

    private async Task AnswerAsync(SIPUserAgent agent, SIPRequest request, string? caller, string? called)
    {
        ActiveCall call = CreateCall(LinkCallDirection.Inbound, caller, called, request.Header.CallId ?? "");
        bool answered;
        try
        {
            SIPServerUserAgent uas = agent.AcceptCall(request);
            answered = await agent.Answer(uas, call.Session, _account.PublicAddress.LastResolved).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Answering the call from {caller} failed", ex);
            answered = false;
        }
        if (!answered)
        {
            AbandonRinging(call);
            _metrics.Failed();
            return;
        }
        Activate(call);
    }

    private static void Reject(SIPUserAgent agent, SIPRequest request, SIPResponseStatusCodesEnum status, string reason)
    {
        try
        {
            SIPServerUserAgent uas = agent.AcceptCall(request);
            uas.Reject(status, reason);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Could not send {status} to an INVITE", ex);
        }
    }

    private bool IsAllowed(string? caller) => _options.InboundPolicy switch
    {
        InboundPolicy.AllowAll => true,
        InboundPolicy.Allowlist => caller is not null && _options.Allowlist.Contains(caller, StringComparer.Ordinal),
        _ => false,
    };

    private ActiveCall CreateCall(LinkCallDirection direction, string? callerId, string? called, string sipCallId)
    {
        uint callId = Interlocked.Increment(ref _nextCallId);
        CallRecorder? recorder = _options.Recording.Enabled ? CallRecorder.TryOpen(_options.Recording.Directory, sipCallId.Length == 0 ? callId.ToString() : sipCallId) : null;
        ClockedAudioSource source = new(_options.Tick with { Codec = _options.Codec });
        InboundAudioSink sink = new(_options.Codec);
        PhoneMediaSession session = new(
            new MediaEndPoints { AudioSource = source, AudioSink = sink },
            _options.BindAddress,
            new PortRange(_options.RtpPortStart, _options.RtpPortEnd, shuffle: true),
            _account.PublicAddress.LastResolved);
        RtpJitterBuffer jitter = new();
        InboundAudioPath inbound = new(jitter, _link, callId, recorder);
        OutboundAudioPath outbound = new(source, recorder);
        session.OnRtpPacketReceived += inbound.HandleRtpPacket;
        return new ActiveCall(callId, direction, callerId, called, sipCallId, source, session, jitter, inbound, outbound, recorder);
    }

    /// <summary>Media is up: start the pump, announce the call, play the greeting.</summary>
    private void Activate(ActiveCall call)
    {
        lock (_stateLock)
        {
            _current = call;
            _state = CallState.Active;
            _pendingEndReason = null;
            _pendingTellHost = true;
            _ringingSipCallId = null;
        }
        call.Inbound.Start();
        if (_link.IsConnected)
        {
            call.Outbound.Configure(_link.OutboundRate);
        }
        _link.SendCallStart(call.CallId, call.ToCallStart(resume: false));
        _guard.CallStarted();
        _metrics.CallStarted(call.Direction == LinkCallDirection.Inbound);
        Logs.Info($"[PhoneGateway] Call {call.CallId} {call.Direction} {call.Remote} active (SIP {call.SipCallId}); fifo={call.Source.FifoActive}.");
        if (_options.GreetingPromptFile is not null)
        {
            try
            {
                _prompts.PlayFile(call.Outbound, _options.GreetingPromptFile);
            }
            catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                Logs.Error("[PhoneGateway] Greeting prompt could not be played", ex);
            }
        }
    }

    /// <summary>A call that never became active: release its media and go idle.</summary>
    private void AbandonRinging(ActiveCall call)
    {
        lock (_stateLock)
        {
            _state = CallState.Idle;
            _current = null;
            _ringingSipCallId = null;
        }
        call.Session.OnRtpPacketReceived -= call.Inbound.HandleRtpPacket;
        call.Session.Close("not answered");
        call.Dispose();
    }

    private void OnCallHungup(SIPDialogue dialogue)
    {
        LinkCallEndReason reason;
        bool tellHost;
        lock (_stateLock)
        {
            reason = _pendingEndReason ?? LinkCallEndReason.RemoteHangup;
            tellHost = _pendingTellHost;
        }
        EndCall(reason, tellHost);
    }

    /// <summary>Tears the live call down exactly once and reports it.</summary>
    private void EndCall(LinkCallEndReason reason, bool tellHost)
    {
        ActiveCall? call;
        lock (_stateLock)
        {
            if (_state is CallState.Idle or CallState.Ending || _current is null)
            {
                return;
            }
            call = _current;
            _state = CallState.Ending;
        }
        _guard.CallEnded();
        if (tellHost)
        {
            _link.SendCallEnd(call.CallId, reason);
        }
        call.Session.OnRtpPacketReceived -= call.Inbound.HandleRtpPacket;
        call.Inbound.Stop();
        try
        {
            call.Session.Close("call ended");
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] Closing the media session failed", ex);
        }
        call.Source.Stop();
        long seconds = (MonotonicClock.NowNs() - call.StartedNs) / 1_000_000_000L;
        LogCallEnd(call, reason, seconds);
        call.Dispose();
        _metrics.CallEnded(seconds);
        lock (_stateLock)
        {
            _current = null;
            _state = CallState.Idle;
            _pendingEndReason = null;
            _pendingTellHost = true;
        }
    }

    private static void LogCallEnd(ActiveCall call, LinkCallEndReason reason, long seconds)
    {
        LatencyHistogram.Summary late = call.Source.Lateness.Snapshot();
        RtpJitterBuffer j = call.Jitter;
        Logs.Info(
            $"[PhoneGateway] Call {call.CallId} ended {reason} after {seconds} s: rtp out frames={call.Source.FramesSent} silence={call.Source.SilenceFrames} " +
            $"catchUp={call.Source.CatchUpFrames} resyncs={call.Source.Resyncs} fifo={call.Source.FifoActive} " +
            $"lateness p50={late.P50Us}us p99={late.P99Us}us max={late.MaxUs}us; rtp in received={j.Received} late={j.Late} lost={j.Lost} " +
            $"dup={j.Duplicate} reorder={j.Reordered} resets={j.Resets}; pump frames={call.Inbound.FramesPumped} concealed={call.Inbound.FramesConcealed} " +
            $"droppedByLink={call.Inbound.DroppedByLink}; outbound frames={call.Outbound.FramesQueued} ringDropped={call.Outbound.DroppedSamples}.");
    }

    private void OnDtmfTone(byte tone, int durationRtpUnits)
    {
        ActiveCall? call = CurrentActive();
        if (call is null)
        {
            return;
        }
        char digit = DtmfDigit(tone);
        if (!LinkDtmf.IsDigit(digit))
        {
            return;
        }
        _metrics.Dtmf();
        // sipsorcery reports the RTP event duration in 8 kHz timestamp units.
        _link.SendDtmf(call.CallId, new LinkDtmf(digit, (ushort)Math.Clamp(durationRtpUnits / 8, 0, ushort.MaxValue)));
    }

    private void OnLinkConnected(LinkHelloAck ack)
    {
        ActiveCall? call = CurrentActive();
        call?.Outbound.Configure(ack.OutboundRate);
        _guard.LinkConnected();
    }

    private void ResumeCall()
    {
        ActiveCall? call = CurrentActive();
        if (call is null)
        {
            return;
        }
        call.Outbound.Configure(_link.OutboundRate);
        _link.SendCallStart(call.CallId, call.ToCallStart(resume: true));
    }

    private int PlayGuardPrompt(PromptKind kind)
    {
        ActiveCall? call = CurrentActive();
        return call is null ? 0 : _prompts.Play(call.Outbound, kind);
    }

    private void OnOutboundAudio(uint callId, uint turnId, ReadOnlySpan<short> pcm)
    {
        ActiveCall? call = CurrentActive();
        if (call is not null && call.CallId == callId)
        {
            call.Outbound.Write(pcm);
        }
    }

    private void OnOutboundEnd(uint callId, uint turnId)
    {
        ActiveCall? call = CurrentActive();
        if (call is not null && call.CallId == callId)
        {
            call.Outbound.EndTurn();
        }
    }

    private uint OnFlush(uint callId, uint turnId)
    {
        ActiveCall? call = CurrentActive();
        return call is not null && call.CallId == callId ? call.Outbound.Flush() : 0;
    }

    private void OnHostCallEnd(uint callId, LinkCallEndReason reason)
    {
        ActiveCall? call = CurrentActive();
        if (call is null || call.CallId != callId)
        {
            return;
        }
        Logs.Info($"[PhoneGateway] Host ended call {callId} ({reason}).");
        _ = Task.Run(() => HangUp(LinkCallEndReason.Completed, tellHost: false));
    }

    private void OnToolRequest(uint callId, uint requestId, ToolRequestMessage request)
    {
        ActiveCall? call = CurrentActive();
        if (call is null || call.CallId != callId)
        {
            Reply(callId, requestId, LinkToolStatus.Failed, "no such call");
            return;
        }
        _ = Task.Run(() => RunToolAsync(call, requestId, request));
    }

    private async Task RunToolAsync(ActiveCall call, uint requestId, ToolRequestMessage request)
    {
        SIPUserAgent? agent = _agent;
        try
        {
            switch (request.Name)
            {
                case "hangup":
                    Reply(call.CallId, requestId, LinkToolStatus.Ok, null);
                    HangUp(LinkCallEndReason.Completed, tellHost: false);
                    return;
                case "send_dtmf":
                {
                    string digits = Argument(request, "digits") ?? throw new ArgumentException("send_dtmf needs 'digits'.");
                    int gapMs = ArgumentInt(request, "gapMs") ?? _options.DtmfGapMs;
                    if (digits.Length == 0 || digits.Length > 32 || !digits.All(LinkDtmf.IsDigit))
                    {
                        throw new ArgumentException("send_dtmf 'digits' must be 1..32 DTMF keys (0-9 * # A-D).");
                    }
                    if (agent is null)
                    {
                        throw new InvalidOperationException("no SIP agent");
                    }
                    foreach (char digit in digits)
                    {
                        await agent.SendDtmf(DtmfCode(digit)).ConfigureAwait(false);
                        await Task.Delay(Math.Clamp(gapMs, 0, 2000)).ConfigureAwait(false);
                    }
                    Reply(call.CallId, requestId, LinkToolStatus.Ok, null);
                    return;
                }
                case "transfer":
                {
                    string target = Argument(request, "target") ?? throw new ArgumentException("transfer needs 'target'.");
                    if (agent is null)
                    {
                        throw new InvalidOperationException("no SIP agent");
                    }
                    SIPURI uri = SIPURI.ParseSIPURIRelaxed(target);
                    bool ok = await agent.BlindTransfer(uri, TimeSpan.FromSeconds(_options.TransferTimeoutSeconds), CancellationToken.None).ConfigureAwait(false);
                    Reply(call.CallId, requestId, ok ? LinkToolStatus.Ok : LinkToolStatus.Failed, ok ? null : "transfer was not accepted");
                    return;
                }
                case "hold":
                    agent?.PutOnHold();
                    Reply(call.CallId, requestId, LinkToolStatus.Ok, null);
                    return;
                case "unhold":
                    agent?.TakeOffHold();
                    Reply(call.CallId, requestId, LinkToolStatus.Ok, null);
                    return;
                case "play_prompt":
                {
                    string? file = Argument(request, "file");
                    string? name = Argument(request, "name");
                    if (file is not null)
                    {
                        _prompts.PlayFile(call.Outbound, file);
                    }
                    else if (name is not null && Enum.TryParse(name.Replace("-", "").Replace("_", ""), ignoreCase: true, out PromptKind kind))
                    {
                        _prompts.Play(call.Outbound, kind);
                    }
                    else if (Argument(request, "text") is not null)
                    {
                        Reply(call.CallId, requestId, LinkToolStatus.Unsupported, "the gateway has no TTS; synthesize on the host and send audio");
                        return;
                    }
                    else
                    {
                        throw new ArgumentException("play_prompt needs 'file' or 'name' (one-moment, goodbye).");
                    }
                    Reply(call.CallId, requestId, LinkToolStatus.Ok, null);
                    return;
                }
                default:
                    Reply(call.CallId, requestId, LinkToolStatus.Unsupported, $"unknown tool '{request.Name}'");
                    return;
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Tool {request.Name} failed", ex);
            Reply(call.CallId, requestId, LinkToolStatus.Failed, ex.Message);
        }
    }

    private void Reply(uint callId, uint requestId, LinkToolStatus status, string? message)
    {
        _metrics.Tool(status != LinkToolStatus.Ok);
        try
        {
            _link.SendToolResult(callId, requestId, new ToolResultMessage { Status = status, Message = message });
        }
        catch (TimeoutException ex)
        {
            Logs.Error("[PhoneGateway] Could not send a tool result; the link is not draining", ex);
        }
    }

    private static string? Argument(ToolRequestMessage request, string name) =>
        request.Arguments is JsonElement args && args.ValueKind == JsonValueKind.Object && args.TryGetProperty(name, out JsonElement value)
            ? value.ValueKind == JsonValueKind.String ? value.GetString() : value.ToString()
            : null;

    private static int? ArgumentInt(ToolRequestMessage request, string name) =>
        request.Arguments is JsonElement args && args.ValueKind == JsonValueKind.Object && args.TryGetProperty(name, out JsonElement value)
            && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int number)
            ? number
            : null;

    /// <summary>RFC 4733 event id to key: 0-9, 10 = *, 11 = #, 12-15 = A-D.</summary>
    private static char DtmfDigit(byte tone) => tone switch
    {
        <= 9 => (char)('0' + tone),
        10 => '*',
        11 => '#',
        <= 15 => (char)('A' + tone - 12),
        _ => '\0',
    };

    private static byte DtmfCode(char digit) => digit switch
    {
        >= '0' and <= '9' => (byte)(digit - '0'),
        '*' => 10,
        '#' => 11,
        _ => (byte)(12 + digit - 'A'),
    };

    private ActiveCall? CurrentActive()
    {
        lock (_stateLock)
        {
            return _state == CallState.Active ? _current : null;
        }
    }

    private MediaSnapshot? SnapshotMedia()
    {
        ActiveCall? call = CurrentActive();
        if (call is null)
        {
            return null;
        }
        long[] buckets = new long[LatencyHistogram.BucketCount];
        call.Source.Lateness.CopyCounts(buckets);
        RtpJitterBuffer j = call.Jitter;
        return new MediaSnapshot(
            call.Source.FramesSent, call.Source.SilenceFrames, call.Source.CatchUpFrames, call.Source.Resyncs, call.Source.FifoActive,
            call.Source.Lateness.Snapshot(), buckets,
            j.Received, j.Late, j.Lost, j.Duplicate, j.Reordered, j.Resets, j.DepthMs,
            call.Inbound.FramesPumped, call.Inbound.FramesConcealed, call.Inbound.DroppedByLink, call.Outbound.DroppedSamples);
    }

    public void Dispose()
    {
        HangUp(LinkCallEndReason.LocalHangup);
        _account.Transport.SIPTransportRequestReceived -= OnTransportRequest;
        _guard.Dispose();
        SIPUserAgent? agent;
        lock (_stateLock)
        {
            agent = _agent;
            _agent = null;
        }
        agent?.Dispose();
    }

    /// <summary>Everything one call owns; disposed exactly once by <see cref="EndCall"/> or <see cref="AbandonRinging"/>.</summary>
    private sealed class ActiveCall(
        uint callId, LinkCallDirection direction, string? callerId, string? called, string sipCallId,
        ClockedAudioSource source, PhoneMediaSession session, RtpJitterBuffer jitter,
        InboundAudioPath inbound, OutboundAudioPath outbound, CallRecorder? recorder) : IDisposable
    {
        public uint CallId => callId;
        public LinkCallDirection Direction => direction;
        public string? CallerId => callerId;
        public string? Called => called;
        public string SipCallId { get; set; } = sipCallId;
        public string Remote => direction == LinkCallDirection.Inbound ? callerId ?? "anonymous" : called ?? "";
        public ClockedAudioSource Source => source;
        public PhoneMediaSession Session => session;
        public RtpJitterBuffer Jitter => jitter;
        public InboundAudioPath Inbound => inbound;
        public OutboundAudioPath Outbound => outbound;
        public long StartedNs { get; } = MonotonicClock.NowNs();
        public DateTime StartedUtc { get; } = DateTime.UtcNow;

        public CallStartMessage ToCallStart(bool resume) => new()
        {
            Direction = direction,
            CallerId = callerId,
            Called = called,
            SipCallId = SipCallId,
            Resume = resume,
        };

        public void Dispose()
        {
            inbound.Dispose();
            source.Dispose();
            recorder?.Dispose();
            session.Dispose();
        }
    }
}
