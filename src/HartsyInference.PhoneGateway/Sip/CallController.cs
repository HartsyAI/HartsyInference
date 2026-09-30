using System.Net.Sockets;
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
/// <remarks>Every new INVITE is screened on the transport before the user agent sees it: <c>486 Busy Here</c> while a
/// call is up, <c>503</c> while the host link is down, <c>603 Decline</c> by policy. Each refusal goes out on its own
/// server transaction, which answers the caller's retransmissions and absorbs the ACK, and is recorded in an
/// <see cref="InviteRejectionLedger"/> so it is counted once and a copy is never offered as a call. A call whose tick or
/// pump thread faults is ended at once: BYE, <c>CallEnd(Failed)</c> to the host, <c>calls_media_fault_total</c>; if
/// the fault lands before the call was announced, the call is never announced and is ended with a BYE. Tool requests
/// arrive on the link reader thread and run on the pool so that thread never waits on SIP. <c>OnCallHungup</c> fires
/// for a remote BYE and for our own <c>Hangup()</c>, so the end reason is decided before hanging up and teardown is
/// guarded by the state under <c>_stateLock</c>, which is never held across an await or a sipsorcery call.</remarks>
public sealed class CallController : IDisposable
{
    // Four 200 OK retransmissions at T1 = 500 ms doubling; a caller that has not ACKed by then is not getting them.
    private const int AckWaitMs = 8000;
    private const int AckPollMs = 20;

    private readonly SipAccount _account;
    private readonly EngineLink _link;
    private readonly CallControllerOptions _options;
    private readonly PromptPlayer _prompts;
    private readonly GatewayMetrics _metrics;
    private readonly LinkOutageGuard _guard;
    private readonly InviteRejectionLedger _rejections = new();
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
        _guard = new LinkOutageGuard(options.Outage, () => link.IsConnected)
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

    /// <summary>Test seam: called with each call's tick source and inbound pump right after they are created, before
    /// either starts.</summary>
    internal Action<ClockedAudioSource, InboundAudioPath>? MediaCreated { get; set; }

    public void Start()
    {
        lock (_stateLock)
        {
            if (_agent is not null)
            {
                throw new InvalidOperationException("CallController is already started.");
            }
            // The screen must see an INVITE before the user agent does: sipsorcery offers every copy of an INVITE to
            // OnIncomingCall, and ignores a new one outright while it holds a call. Handlers run in subscription order.
            _account.Transport.SIPTransportRequestReceived += OnTransportRequest;
            SIPUserAgent agent = new(_account.Transport, null);
            agent.OnIncomingCall += OnIncomingCall;
            agent.OnCallHungup += OnCallHungup;
            agent.OnDtmfTone += OnDtmfTone;
            agent.ClientCallFailed += (_, message, _) => Volatile.Write(ref _lastCallFailure, message ?? "");
            _agent = agent;
        }
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
        string? uri = DialString(destination, _account.Registrar);
        if (uri is null)
        {
            return new CallPlacementResult(CallPlacementStatus.Invalid, "a bare number needs a registrar; give a full sip: URI");
        }
        if (!IsDestinationAllowed(destination, _options.DestinationPrefixes, _account.Registrar))
        {
            return new CallPlacementResult(CallPlacementStatus.NotAllowed, "the destination does not match sip.destinationPrefixes through the registrar");
        }
        lock (_stateLock)
        {
            if (_state != CallState.Idle)
            {
                return new CallPlacementResult(CallPlacementStatus.Busy, "busy");
            }
            if (!_link.IsConnected)
            {
                return new CallPlacementResult(CallPlacementStatus.HostUnavailable, "voice host unavailable");
            }
            _state = CallState.Ringing;
        }
        ActiveCall? call = TryCreateCall(LinkCallDirection.Outbound, _account.Username, destination, "");
        if (call is null)
        {
            _metrics.Failed();
            return new CallPlacementResult(CallPlacementStatus.Failed, "the call's media could not be set up");
        }
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
            return new CallPlacementResult(CallPlacementStatus.NotAnswered, LastCallFailure.Length == 0 ? "not answered" : LastCallFailure);
        }
        call.SipCallId = agent.Dialogue?.CallId ?? "";
        if (!Activate(call))
        {
            EndBeforeAnnouncing(call, agent);
            return new CallPlacementResult(CallPlacementStatus.Failed, "media fault");
        }
        return CallPlacementResult.Ok;
    }

    /// <summary>The SIP URI a destination dials: a bare number goes through <paramref name="registrar"/>, <c>user@host</c>
    /// gains the <c>sip:</c> scheme, a URI is taken as given. Null for a bare number with no registrar.</summary>
    internal static string? DialString(string destination, string registrar)
    {
        if (destination.Contains(':'))
        {
            return destination;
        }
        if (destination.Contains('@'))
        {
            return "sip:" + destination;
        }
        return registrar.Length == 0 ? null : $"sip:{destination}@{registrar}";
    }

    /// <summary>True when <paramref name="destination"/> may be dialled: <paramref name="prefixes"/> is empty, or it dials
    /// through <paramref name="registrar"/> (a bare number, or a URI naming the registrar's host) a number starting with
    /// one of them. The host matters as much as the number: a matching user at another host would take the call, and the
    /// account's digest answer, somewhere else. The agent is steerable by its caller, so an open dial plan on a trunk is
    /// toll fraud waiting to happen.</summary>
    internal static bool IsDestinationAllowed(string destination, IReadOnlyList<string> prefixes, string registrar)
    {
        if (prefixes.Count == 0)
        {
            return true;
        }
        string? dial = DialString(destination, registrar);
        if (dial is null || registrar.Length == 0
            || !SIPURI.TryParse(dial, out SIPURI target) || target is null
            || !SIPURI.TryParse("sip:" + registrar, out SIPURI trunk) || trunk is null)
        {
            return false;
        }
        string number = target.User ?? "";
        return string.Equals(target.HostAddress, trunk.HostAddress, StringComparison.OrdinalIgnoreCase)
            && number.Length > 0 && prefixes.Any(prefix => number.StartsWith(prefix, StringComparison.Ordinal));
    }

    /// <summary>Ends the live call from our side with <paramref name="reason"/> and tells the host.</summary>
    public void HangUp(LinkCallEndReason reason) => HangUp(reason, tellHost: true, only: null);

    /// <summary>Hangs up the live call, or only <paramref name="only"/> when given, so a late request never ends a newer call.</summary>
    private void HangUp(LinkCallEndReason reason, bool tellHost, ActiveCall? only)
    {
        SIPUserAgent? agent;
        lock (_stateLock)
        {
            if (_state is CallState.Idle or CallState.Ending || (only is not null && !ReferenceEquals(_current, only)))
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

    /// <summary>First look at every request, before the user agent's: refuses a new INVITE the gateway will not take,
    /// and answers a copy of an already-refused INVITE the same way. Re-INVITEs (with a To tag), the INVITE being
    /// answered and one to offer pass through.</summary>
    private Task OnTransportRequest(SIPEndPoint localEndPoint, SIPEndPoint remoteEndPoint, SIPRequest request)
    {
        if (!IsNewInvite(request))
        {
            return Task.CompletedTask;
        }
        if (_rejections.TryGet(request, out InviteRejection known))
        {
            // The refusal's transaction absorbs copies while it lives; this one got past it, so answer it here.
            _ = SendStatelessAsync(request, known.Status, known.Reason);
            return Task.CompletedTask;
        }
        InviteDecision decision = Screen(request);
        if (decision is not (InviteDecision.Offer or InviteDecision.OwnCall))
        {
            RejectInvite(request, decision);
        }
        return Task.CompletedTask;
    }

    private void OnIncomingCall(SIPUserAgent agent, SIPRequest request)
    {
        if (_rejections.TryGet(request, out _))
        {
            // Refused by the transport screen, which runs first.
            return;
        }
        InviteDecision decision = Screen(request);
        if (decision == InviteDecision.Offer)
        {
            lock (_stateLock)
            {
                if (_state == CallState.Idle)
                {
                    _state = CallState.Ringing;
                    _ringingSipCallId = request.Header.CallId;
                }
                else
                {
                    // Another INVITE got in since the screen ran.
                    decision = IsOwnCallLocked(request.Header.CallId) ? InviteDecision.OwnCall : InviteDecision.Busy;
                }
            }
        }
        switch (decision)
        {
            case InviteDecision.Offer:
                _ = AnswerAsync(agent, request, request.Header.From?.FromURI?.User, request.URI?.User);
                return;
            case InviteDecision.OwnCall:
                return;
            default:
                RejectInvite(request, decision);
                return;
        }
    }

    private InviteDecision Screen(SIPRequest request)
    {
        lock (_stateLock)
        {
            if (_state != CallState.Idle)
            {
                return IsOwnCallLocked(request.Header.CallId) ? InviteDecision.OwnCall : InviteDecision.Busy;
            }
        }
        if (!_link.IsConnected)
        {
            return InviteDecision.HostDown;
        }
        return IsAllowed(request.Header.From?.FromURI?.User) ? InviteDecision.Offer : InviteDecision.Declined;
    }

    private bool IsOwnCallLocked(string? sipCallId) =>
        sipCallId is not null && (sipCallId == _ringingSipCallId || sipCallId == _current?.SipCallId);

    private static bool IsNewInvite(SIPRequest request) =>
        request.Method == SIPMethodsEnum.INVITE && string.IsNullOrEmpty(request.Header.To?.ToTag);

    /// <summary>Refuses an INVITE on its own server transaction, which then answers the caller's retransmissions and
    /// absorbs the ACK. The ledger makes the refusal count once, whichever path saw the INVITE first.</summary>
    private void RejectInvite(SIPRequest request, InviteDecision decision)
    {
        (SIPResponseStatusCodesEnum status, string reason) = decision switch
        {
            InviteDecision.Busy => (SIPResponseStatusCodesEnum.BusyHere, "Busy Here"),
            InviteDecision.HostDown => (SIPResponseStatusCodesEnum.ServiceUnavailable, "Voice host unavailable"),
            _ => (SIPResponseStatusCodesEnum.Decline, "Decline"),
        };
        if (!_rejections.TryRecord(request, status, reason, out InviteRejection first))
        {
            // A copy that raced the first one past the transaction layer: same answer, not counted again.
            _ = SendStatelessAsync(request, first.Status, first.Reason);
            return;
        }
        string caller = request.Header.From?.FromURI?.User ?? "anonymous";
        switch (decision)
        {
            case InviteDecision.Busy:
                _metrics.RejectedBusy();
                Logs.Info($"[PhoneGateway] INVITE from {caller} while a call is up: 486 Busy Here.");
                break;
            case InviteDecision.HostDown:
                _metrics.RejectedHostDown();
                Logs.Warning($"[PhoneGateway] INVITE from {caller} while the voice host is down: 503.");
                break;
            default:
                _metrics.Declined();
                Logs.Info($"[PhoneGateway] INVITE from {caller} declined by policy {_options.InboundPolicy}: 603.");
                break;
        }
        RejectOnTransaction(request, status, reason);
    }

    /// <summary>Sends a final response on a new server transaction for <paramref name="request"/>.</summary>
    private void RejectOnTransaction(SIPRequest request, SIPResponseStatusCodesEnum status, string reason)
    {
        try
        {
            UASInviteTransaction transaction = new(_account.Transport, request, null);
            SIPServerUserAgent uas = new(_account.Transport, null, transaction, null);
            uas.Reject(status, reason);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Could not send {status} to an INVITE", ex);
        }
    }

    /// <summary>Answers an INVITE outside any transaction, for a copy the transaction layer did not catch.</summary>
    private async Task SendStatelessAsync(SIPRequest request, SIPResponseStatusCodesEnum status, string reason)
    {
        try
        {
            SIPResponse response = SIPResponse.GetResponse(request, status, reason);
            SocketError result = await _account.Transport.SendResponseAsync(response).ConfigureAwait(false);
            if (result != SocketError.Success)
            {
                Logs.Debug($"[PhoneGateway] Resending {status} for an INVITE copy failed: {result}.");
            }
        }
        catch (Exception ex) when (ex is SocketException or ObjectDisposedException or InvalidOperationException)
        {
            Logs.Debug($"[PhoneGateway] Resending {status} for an INVITE copy failed: {ex.Message}");
        }
    }

    private async Task AnswerAsync(SIPUserAgent agent, SIPRequest request, string? caller, string? called)
    {
        ActiveCall? call = TryCreateCall(LinkCallDirection.Inbound, caller, called, request.Header.CallId ?? "");
        if (call is null)
        {
            // Nothing was answered: refuse the INVITE so the caller is not left ringing.
            RejectOnTransaction(request, SIPResponseStatusCodesEnum.InternalServerError, "Media setup failed");
            _metrics.Failed();
            return;
        }
        SIPServerUserAgent? uas = null;
        bool answered;
        try
        {
            uas = agent.AcceptCall(request);
            answered = await agent.Answer(uas, call.Session, _account.PublicAddress.LastResolved).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Answering the call from {caller} failed", ex);
            answered = false;
        }
        if (!answered)
        {
            FinishUnanswered(request, uas);
            AbandonRinging(call);
            _metrics.Failed();
            return;
        }
        if (!Activate(call))
        {
            await WaitForAckAsync(uas).ConfigureAwait(false);
            EndBeforeAnnouncing(call, agent);
        }
    }

    /// <summary>A UAS must not send BYE before the ACK for its 200 (RFC 3261 §15): if the 200 was lost, the caller would
    /// pick up a retransmission later and sit in a call already ended here. Waits for the ACK, bounded.</summary>
    private static async Task WaitForAckAsync(SIPServerUserAgent? uas)
    {
        if (uas is null)
        {
            return;
        }
        long deadline = Environment.TickCount64 + AckWaitMs;
        while (uas.ClientTransaction.TransactionState == SIPTransactionStatesEnum.Completed && Environment.TickCount64 < deadline)
        {
            await Task.Delay(AckPollMs).ConfigureAwait(false);
        }
    }

    /// <summary>After a failed answer, makes sure the caller is not left ringing: a final response when none went out
    /// (sipsorcery sends its own for an SDP mismatch), a BYE when the 200 already did.</summary>
    private void FinishUnanswered(SIPRequest request, SIPServerUserAgent? uas)
    {
        try
        {
            if (uas is null)
            {
                _ = SendStatelessAsync(request, SIPResponseStatusCodesEnum.InternalServerError, "Media setup failed");
                return;
            }
            if (uas.IsUASAnswered)
            {
                uas.SIPDialogue?.Hangup(_account.Transport, null);
                return;
            }
            if (!uas.IsCancelled && uas.ClientTransaction.TransactionState is SIPTransactionStatesEnum.Proceeding or SIPTransactionStatesEnum.Trying)
            {
                uas.Reject(SIPResponseStatusCodesEnum.InternalServerError, "Media setup failed");
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] Could not end an INVITE the gateway failed to answer", ex);
        }
    }

    private bool IsAllowed(string? caller) => _options.InboundPolicy switch
    {
        InboundPolicy.AllowAll => true,
        InboundPolicy.Allowlist => caller is not null && _options.Allowlist.Contains(caller, StringComparer.Ordinal),
        _ => false,
    };

    /// <summary>Builds a ringing call's media. On failure (no free RTP port, for one) it logs, puts the controller back to
    /// idle and returns null, so one failed setup cannot leave the gateway answering 486 until it restarts.</summary>
    private ActiveCall? TryCreateCall(LinkCallDirection direction, string? callerId, string? called, string sipCallId)
    {
        try
        {
            return CreateCall(direction, callerId, called, sipCallId);
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] Could not set up media for the {direction} call; staying idle", ex);
            lock (_stateLock)
            {
                _state = CallState.Idle;
                _ringingSipCallId = null;
            }
            return null;
        }
    }

    private ActiveCall CreateCall(LinkCallDirection direction, string? callerId, string? called, string sipCallId)
    {
        uint callId = Interlocked.Increment(ref _nextCallId);
        CallRecorder? recorder = _options.Recording.Enabled ? CallRecorder.TryOpen(_options.Recording.Directory, sipCallId.Length == 0 ? callId.ToString() : sipCallId) : null;
        ClockedAudioSource source = new(_options.Tick with { Codec = _options.Codec });
        PhoneMediaSession? session = null;
        try
        {
            InboundAudioSink sink = new(_options.Codec);
            session = new PhoneMediaSession(
                new MediaEndPoints { AudioSource = source, AudioSink = sink },
                _options.BindAddress,
                new PortRange(_options.RtpPortStart, _options.RtpPortEnd, shuffle: true),
                _account.PublicAddress.LastResolved);
            RtpJitterBuffer jitter = new();
            InboundAudioPath inbound = new(jitter, _link, callId, recorder);
            OutboundAudioPath outbound = new(source, recorder);
            session.OnRtpPacketReceived += inbound.HandleRtpPacket;
            ActiveCall call = new(callId, direction, callerId, called, sipCallId, source, session, jitter, inbound, outbound, recorder);
            source.TickFaulted += fault => OnMediaFault(call, "RTP tick", fault);
            inbound.PumpFaulted += fault => OnMediaFault(call, "RTP pump", fault);
            MediaCreated?.Invoke(source, inbound);
            return call;
        }
        catch
        {
            // Nothing has started yet; release the sockets and the recording files the half-built call holds.
            session?.Dispose();
            source.Dispose();
            recorder?.Dispose();
            throw;
        }
    }

    /// <summary>Media is up: announce the call, start the pump, play the greeting. False, and nothing announced, when a
    /// media thread has already faulted; the caller then ends the answered call. A fault teardown waits for this to
    /// finish (<see cref="ActiveCall.Settled"/>), so the host never sees <c>CallEnd</c> before <c>CallStart</c>.</summary>
    private bool Activate(ActiveCall call)
    {
        lock (_stateLock)
        {
            if (call.Source.Faulted || call.Inbound.Faulted)
            {
                return false;
            }
            _current = call;
            _state = CallState.Active;
            _pendingEndReason = null;
            _pendingTellHost = true;
            _ringingSipCallId = null;
        }
        if (_link.IsConnected)
        {
            call.Outbound.Configure(_link.OutboundRate);
        }
        _link.SendCallStart(call.CallId, call.ToCallStart(resume: false));
        // After the announcement, so the host never receives caller audio for a call it has not been told about.
        call.Inbound.Start();
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
        call.Settled.TrySetResult();
        return true;
    }

    /// <summary>Ends an answered call that <see cref="Activate"/> refused: BYE, then release its media.</summary>
    private void EndBeforeAnnouncing(ActiveCall call, SIPUserAgent agent)
    {
        try
        {
            agent.Hangup();
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] SIP hangup failed; releasing the call anyway", ex);
        }
        AbandonRinging(call);
    }

    /// <summary>A call that never became active: release its media and go idle.</summary>
    private void AbandonRinging(ActiveCall call)
    {
        lock (_stateLock)
        {
            call.Ended = true;
            _state = CallState.Idle;
            _current = null;
            _ringingSipCallId = null;
        }
        call.Session.OnRtpPacketReceived -= call.Inbound.HandleRtpPacket;
        call.Session.Close("not answered");
        call.Dispose();
        call.Settled.TrySetResult();
    }

    /// <summary>One of a call's media threads (the RTP tick or the inbound pump) died. Runs on that thread, so it only
    /// logs, counts and hands the teardown, which joins this thread, to the pool. At most once per call.</summary>
    private void OnMediaFault(ActiveCall call, string thread, Exception fault)
    {
        lock (_stateLock)
        {
            if (call.Ended || call.FaultReported)
            {
                Logs.Debug($"[PhoneGateway] Call {call.CallId}: {thread} thread faulted after the call began ending: {fault.Message}");
                return;
            }
            call.FaultReported = true;
        }
        Logs.Error($"[PhoneGateway] Call {call.CallId} (SIP {call.SipCallId}): the {thread} thread faulted; ending the call", fault);
        _metrics.MediaFault();
        _ = Task.Run(() => EndFaultedCallAsync(call));
    }

    private async Task EndFaultedCallAsync(ActiveCall call)
    {
        // A call still being answered, placed or announced is left to finish that first: Activate refuses a faulted call
        // (the answer or placement path then sends the BYE), and an announced one is hung up below.
        await call.Settled.Task.ConfigureAwait(false);
        bool active;
        lock (_stateLock)
        {
            active = _state == CallState.Active && ReferenceEquals(_current, call);
        }
        if (active)
        {
            HangUp(LinkCallEndReason.Failed, tellHost: true, only: call);
        }
    }

    private void OnCallHungup(SIPDialogue dialogue)
    {
        LinkCallEndReason reason;
        bool tellHost;
        lock (_stateLock)
        {
            if (_current is null || (dialogue?.CallId is string ended && _current.SipCallId.Length > 0 && ended != _current.SipCallId))
            {
                // A dialogue that is not the live call's (one refused before it was announced, or a stale one).
                return;
            }
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
            call.Ended = true;
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
            $"catchUp={call.Source.CatchUpFrames} resyncs={call.Source.Resyncs} fifo={call.Source.FifoActive} tickFault={call.Source.Faulted} " +
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
        _ = Task.Run(() => HangUp(LinkCallEndReason.Completed, tellHost: false, only: call));
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
                    HangUp(LinkCallEndReason.Completed, tellHost: false, only: call);
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
                    string? dial = DialString(target, _account.Registrar);
                    if (dial is null)
                    {
                        Reply(call.CallId, requestId, LinkToolStatus.Failed, "a bare number needs a registrar; give a full sip: URI");
                        return;
                    }
                    if (!IsDestinationAllowed(target, _options.DestinationPrefixes, _account.Registrar))
                    {
                        Reply(call.CallId, requestId, LinkToolStatus.Failed, "the transfer target does not match sip.destinationPrefixes through the registrar");
                        return;
                    }
                    if (agent is null)
                    {
                        throw new InvalidOperationException("no SIP agent");
                    }
                    SIPURI uri = SIPURI.ParseSIPURI(dial);
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
        _link.SendToolResult(callId, requestId, new ToolResultMessage { Status = status, Message = message });
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

    /// <summary>What the screen decided for a new INVITE.</summary>
    private enum InviteDecision
    {
        /// <summary>Offer it as a call.</summary>
        Offer,
        /// <summary>A copy of the INVITE being answered, or of the live call's: leave it to the user agent.</summary>
        OwnCall,
        /// <summary>486: a call is up or being set up.</summary>
        Busy,
        /// <summary>503: the voice host link is down.</summary>
        HostDown,
        /// <summary>603: refused by the inbound policy.</summary>
        Declined,
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

        /// <summary>Set under the controller's state lock once teardown of this call has begun.</summary>
        public bool Ended { get; set; }

        /// <summary>Set under the controller's state lock when a media fault of this call has been reported.</summary>
        public bool FaultReported { get; set; }

        /// <summary>Completes once the call is announced (end of <see cref="Activate"/>) or abandoned.</summary>
        public TaskCompletionSource Settled { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

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
