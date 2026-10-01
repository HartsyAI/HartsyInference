using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Calls;

namespace HartsyInference.VoiceHost.Link;

/// <summary>One gateway connection: the handshake, the calls it carries, and the two threads the protocol allows, one
/// reader and one writer.</summary>
/// <remarks>The reader thread (<c>voice-link-reader</c>) checks <c>Hello</c> (first frame, version, 16 kHz, token in
/// constant time) and answers <c>HelloAck</c> or <c>Error</c> and closes; after that it reads every frame, checks the
/// per-direction sequence, and hands calls their audio, DTMF and tool results. The sender thread
/// (<c>voice-link-sender</c>) is the only writer once the handshake is done. It wakes on absolute
/// <see cref="MonotonicClock"/> deadlines every 20 ms, first writes the queued control frames (so a <c>Flush(T)</c> is on
/// the wire before anything it read after the barge-in), then moves each call's reply audio, one turn per frame and at most
/// 20 ms per frame, into <c>OutboundAudio(turnId)</c>. A burst of reply audio starts with a prebuffer so the gateway's
/// own 20 ms clock never waits on this one; a late wake-up is made up with catch-up frames, a very late one re-bases the
/// clock. After its warm-up ticks the audio path allocates nothing (<see cref="AudioPathAllocatedBytes"/>). A watchdog
/// closes a connection that has received nothing, or has had one write stuck, for the liveness timeout. Closing fails
/// both threads; the reader then ends this connection's calls (no <c>CallEnd</c>: the link is gone, and the gateway
/// re-announces live calls with <c>CallStart(resume)</c> when it reconnects).</remarks>
internal sealed class LinkConnection
{
    private const int SenderStackBytes = 256 * 1024;
    private const int WatchdogPeriodMs = 1000;
    private const float ToPcm16 = 32767f;

    private readonly PhoneLinkServer _server;
    private readonly PhoneLinkServerOptions _options;
    private readonly Socket _socket;
    private readonly NetworkStream _stream;
    private readonly LinkFrameReader _reader;
    private readonly LinkFrameWriter _writer;
    private readonly ConcurrentQueue<HostControlItem> _control = new();
    private readonly object _callsLock = new();
    private readonly Dictionary<uint, VoiceCall> _calls = [];
    private readonly LatencyHistogram _lateness = new();
    private readonly TaskCompletionSource _closed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly int _frameSamples;
    private readonly int _prebufferSamples;
    private readonly float[] _audio;
    private readonly short[] _pcm;
    private VoiceCall[] _callSnapshot = [];
    private Thread? _senderThread;
    private Timer? _watchdog;
    private volatile bool _closing;
    private volatile bool _senderStop;
    private string _closeReason = "";
    private uint _expectedSequence;
    private long _lastFrameNs;
    private long _writeStartedNs;
    private long _controlWritten;
    private long _audioAllocated;
    private long _catchUpFrames;
    private long _resyncs;
    private long _unknownCallFrames;
    private int _closeStarted;
    private int _unknownTypeLogged;

    public LinkConnection(PhoneLinkServer server, Socket socket, PhoneLinkServerOptions options, int outboundRate)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(socket);
        ArgumentNullException.ThrowIfNull(options);
        if (!LinkProtocol.IsOutboundSampleRate((uint)outboundRate))
        {
            throw new ArgumentOutOfRangeException(nameof(outboundRate), outboundRate, "Not a PhoneLink outbound rate.");
        }
        _server = server;
        _socket = socket;
        _options = options;
        OutboundRate = outboundRate;
        _frameSamples = outboundRate / 50;
        _prebufferSamples = (int)((long)outboundRate * options.PrebufferMs / 1000);
        _audio = new float[_frameSamples];
        _pcm = new short[_frameSamples];
        _stream = new NetworkStream(socket, ownsSocket: false);
        _reader = new LinkFrameReader(_stream);
        _writer = new LinkFrameWriter(_stream);
    }

    public int OutboundRate { get; }

    /// <summary>Completes when the reader thread has torn the connection down and ended its calls.</summary>
    public Task Closed => _closed.Task;

    /// <summary>Why the connection closed; empty while it is open.</summary>
    public string CloseReason => Volatile.Read(ref _closeReason);

    /// <summary>The calls this connection carries right now.</summary>
    public IReadOnlyList<VoiceCall> Calls => Volatile.Read(ref _callSnapshot);

    /// <summary>Managed bytes the sender's audio path allocated after its warm-up ticks; zero by design.</summary>
    public long AudioPathAllocatedBytes => Volatile.Read(ref _audioAllocated);

    /// <summary>Control frames the sender has written.</summary>
    public long ControlFramesWritten => Volatile.Read(ref _controlWritten);

    /// <summary>Sender wake-up lateness per tick.</summary>
    public LatencyHistogram Lateness => _lateness;

    public long CatchUpFrames => Volatile.Read(ref _catchUpFrames);

    public long Resyncs => Volatile.Read(ref _resyncs);

    /// <summary>Starts the reader thread, which runs the handshake.</summary>
    public void Start()
    {
        Volatile.Write(ref _lastFrameNs, MonotonicClock.NowNs());
        new Thread(ReaderMain) { Name = "voice-link-reader", IsBackground = true }.Start();
    }

    /// <summary>Queues a control frame for the sender; dropped once the connection is closing.</summary>
    public void Enqueue(in HostControlItem item)
    {
        if (!_closing)
        {
            _control.Enqueue(item);
        }
    }

    /// <summary>A request id unique on this host, for a <c>ToolRequest</c>.</summary>
    public uint NextRequestId() => _server.NextRequestId();

    /// <summary>Drops <paramref name="call"/> from the set the sender and the reader serve.</summary>
    public void RemoveCall(VoiceCall call)
    {
        lock (_callsLock)
        {
            if (_calls.TryGetValue(call.CallId, out VoiceCall? current) && ReferenceEquals(current, call))
            {
                _calls.Remove(call.CallId);
                PublishCalls();
            }
        }
    }

    /// <summary>Waits until every control frame queued so far is on the wire, or <paramref name="timeoutMs"/> passes.</summary>
    public async Task DrainControlAsync(int timeoutMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (!_control.IsEmpty && !_closing && Environment.TickCount64 < deadline)
        {
            await Task.Delay(5).ConfigureAwait(false);
        }
    }

    /// <summary>Closes the socket, which ends both threads; the reader then cleans up. Idempotent.</summary>
    public void Close(string reason)
    {
        if (Interlocked.Exchange(ref _closeStarted, 1) != 0)
        {
            return;
        }
        Volatile.Write(ref _closeReason, reason);
        _closing = true;
        try
        {
            _socket.Shutdown(SocketShutdown.Both);
        }
        catch (SocketException)
        {
            // Already closed by the gateway; the dispose below is what matters.
        }
        catch (ObjectDisposedException)
        {
            return;
        }
        _socket.Dispose();
    }

    private void ReaderMain()
    {
        string reason = "the gateway closed the socket";
        bool adopted = false;
        try
        {
            if (!Handshake())
            {
                reason = "handshake refused";
                return;
            }
            if (!_server.Adopt(this))
            {
                Refuse("the host is shutting down");
                reason = "the host is shutting down";
                return;
            }
            adopted = true;
            Await(_writer.WriteHelloAckAsync(new LinkHelloAck((uint)OutboundRate, (ushort)(PhoneLinkServerOptions.FramePeriodNs / 1_000_000)), CancellationToken.None));
            Logs.Info($"[VoiceHost] Phone gateway connected; replying at {OutboundRate} Hz in frames of up to 20 ms.");
            _senderThread = new Thread(SenderMain, SenderStackBytes) { Name = "voice-link-sender", IsBackground = true };
            _senderThread.Start();
            int watch = Math.Clamp(_options.LivenessTimeoutMs / 4, 10, WatchdogPeriodMs);
            _watchdog = new Timer(static state => ((LinkConnection)state!).Watch(), this, watch, watch);
            while (!_closing)
            {
                LinkFrame? next = Await(_reader.ReadAsync(CancellationToken.None));
                if (next is null)
                {
                    break;
                }
                Volatile.Write(ref _lastFrameNs, MonotonicClock.NowNs());
                LinkFrame frame = next.Value;
                CheckSequence(frame.Header);
                Dispatch(frame);
            }
        }
        catch (LinkProtocolException ex)
        {
            reason = "protocol error: " + ex.Message;
            Logs.Warning($"[VoiceHost] PhoneLink {reason}; closing the connection.");
        }
        catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or OperationCanceledException)
        {
            reason = _closing ? CloseReason : ex.GetType().Name + ": " + ex.Message;
        }
        catch (Exception ex)
        {
            // Anything else escaping here would end the host from a background thread, every call with it.
            reason = "unexpected " + ex.GetType().Name;
            Logs.Error("[VoiceHost] The PhoneLink reader failed unexpectedly; closing the connection", ex);
        }
        finally
        {
            TearDown(reason, adopted);
        }
    }

    private void TearDown(string reason, bool adopted)
    {
        if (_closing && CloseReason.Length > 0)
        {
            reason = CloseReason;
        }
        Close(reason);
        _watchdog?.Dispose();
        _senderStop = true;
        if (_senderThread is not null && _senderThread != Thread.CurrentThread)
        {
            _senderThread.Join();
        }
        VoiceCall[] calls;
        lock (_callsLock)
        {
            calls = [.. _calls.Values];
            _calls.Clear();
            PublishCalls();
        }
        foreach (VoiceCall call in calls)
        {
            _ = call.EndAsync(null, "the link to the gateway closed");
        }
        _reader.Dispose();
        _writer.Dispose();
        _stream.Dispose();
        if (adopted)
        {
            LatencyHistogram.Summary late = _lateness.Snapshot();
            Logs.Info($"[VoiceHost] Phone gateway link closed ({reason}); sender ticks={late.Count} lateness p50={late.P50Us}us p99={late.P99Us}us "
                + $"max={late.MaxUs}us catchUp={CatchUpFrames} resyncs={Resyncs} audioPathAllocated={AudioPathAllocatedBytes}B; {calls.Length} call(s) ended.");
        }
        _server.Released(this);
        Task.WhenAll(calls.Select(c => c.Ended)).ContinueWith(static (_, state) => ((TaskCompletionSource)state!).TrySetResult(), _closed,
            CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
    }

    /// <summary>Reads and checks <c>Hello</c>; on any refusal answers <c>Error</c> (callId 0) and returns false.</summary>
    private bool Handshake()
    {
        LinkFrame? first;
        using (CancellationTokenSource timeout = new(_options.HandshakeTimeoutMs))
        {
            try
            {
                first = Await(_reader.ReadAsync(timeout.Token));
            }
            catch (OperationCanceledException)
            {
                Refuse($"no Hello within {_options.HandshakeTimeoutMs} ms");
                return false;
            }
        }
        if (first is null)
        {
            return false;
        }
        LinkFrame frame = first.Value;
        if (frame.Header.Sequence != 0)
        {
            Refuse($"the first frame carries sequence {frame.Header.Sequence}, not 0");
            return false;
        }
        if (frame.Type != LinkMessageType.Hello)
        {
            Refuse($"the first frame must be Hello, got {frame.Type}");
            return false;
        }
        LinkHello hello;
        try
        {
            hello = frame.ReadHello();
        }
        catch (LinkProtocolException ex)
        {
            Refuse(ex.Message);
            return false;
        }
        if (hello.Version != LinkProtocol.Version)
        {
            Refuse($"PhoneLink version {hello.Version} is not supported; this host speaks version {LinkProtocol.Version}");
            return false;
        }
        if (!TokenMatches(hello.Token, _options.Token))
        {
            Refuse("the link token does not match");
            return false;
        }
        _expectedSequence = 1;
        return true;
    }

    private void Refuse(string why)
    {
        Logs.Warning($"[VoiceHost] Refused a PhoneLink connection: {why}.");
        _server.CountRefused();
        try
        {
            Await(_writer.WriteErrorAsync(LinkProtocol.ConnectionCallId, new LinkErrorMessage { Text = why }, CancellationToken.None));
        }
        catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException)
        {
            Logs.Debug($"[VoiceHost] The refused connection was already gone: {ex.Message}");
        }
    }

    /// <summary>Constant-time comparison of SHA-256 digests, so neither the content nor the length of the expected token
    /// leaks through timing.</summary>
    internal static bool TokenMatches(string offered, string expected) =>
        CryptographicOperations.FixedTimeEquals(SHA256.HashData(Encoding.UTF8.GetBytes(offered)), SHA256.HashData(Encoding.UTF8.GetBytes(expected)));

    private void CheckSequence(in LinkFrameHeader header)
    {
        if (header.Sequence != _expectedSequence)
        {
            throw new LinkProtocolException($"Frame sequence {header.Sequence} where {_expectedSequence} was due; a frame was lost or duplicated.");
        }
        _expectedSequence++;
    }

    private void Dispatch(in LinkFrame frame)
    {
        uint callId = frame.Header.CallId;
        switch (frame.Type)
        {
            case LinkMessageType.InboundAudio:
                if (Find(callId) is { } audioCall)
                {
                    audioCall.PushInbound(frame);
                }
                else
                {
                    Interlocked.Increment(ref _unknownCallFrames);
                }
                return;
            case LinkMessageType.CallStart:
                OnCallStart(callId, frame.ReadCallStart());
                return;
            case LinkMessageType.CallEnd:
            {
                LinkCallEndReason reason = frame.ReadCallEnd();
                if (Find(callId) is { } ended)
                {
                    _ = ended.EndAsync(null, "the gateway ended it: " + reason);
                }
                return;
            }
            case LinkMessageType.DtmfEvent:
            {
                LinkDtmf dtmf = frame.ReadDtmf();
                Find(callId)?.PushDtmf(dtmf);
                return;
            }
            case LinkMessageType.ToolResult:
            {
                ToolResultMessage result = frame.ReadToolResult(out uint requestId);
                Find(callId)?.CompleteTool(requestId, result);
                return;
            }
            case LinkMessageType.FlushAck:
            {
                LinkFlushAck ack = frame.ReadFlushAck();
                Find(callId)?.OnFlushAck(ack);
                return;
            }
            case LinkMessageType.Ping:
                Enqueue(new HostControlItem(LinkMessageType.Pong, LinkProtocol.ConnectionCallId, 0, frame.ReadTimestampNs(), null, null));
                return;
            case LinkMessageType.Pong:
                frame.ReadTimestampNs();
                return;
            case LinkMessageType.Error:
            {
                LinkErrorMessage error = frame.ReadError();
                Logs.Error($"[VoiceHost] The gateway reported an error (call {callId}): {error.Text}");
                if (callId == LinkProtocol.ConnectionCallId)
                {
                    throw new LinkProtocolException("The gateway reported a connection error: " + error.Text);
                }
                return;
            }
            case LinkMessageType.Hello or LinkMessageType.HelloAck or LinkMessageType.OutboundAudio or LinkMessageType.OutboundEnd
                or LinkMessageType.Flush or LinkMessageType.Event or LinkMessageType.ToolRequest:
                throw new LinkProtocolException($"The gateway sent {frame.Type}, which only the host sends or which comes once per connection.");
            default:
                if (Interlocked.Exchange(ref _unknownTypeLogged, 1) == 0)
                {
                    Logs.Warning($"[VoiceHost] Ignoring unknown PhoneLink frame type 0x{(byte)frame.Type:X2} (a newer gateway?).");
                }
                return;
        }
    }

    private void OnCallStart(uint callId, CallStartMessage message)
    {
        if (callId == LinkProtocol.ConnectionCallId)
        {
            throw new LinkProtocolException("CallStart carries call id 0, which is reserved for the connection.");
        }
        VoiceCall call = new(this, callId, message, _server.Factory, _options);
        VoiceCall? previous;
        lock (_callsLock)
        {
            _calls.TryGetValue(callId, out previous);
            _calls[callId] = call;
            PublishCalls();
        }
        if (previous is not null)
        {
            Logs.Warning($"[VoiceHost] Call {callId} was announced again; replacing its session.");
            _ = previous.EndAsync(null, "announced again");
        }
        _ = call.StartAsync();
    }

    private VoiceCall? Find(uint callId)
    {
        lock (_callsLock)
        {
            return _calls.GetValueOrDefault(callId);
        }
    }

    private void PublishCalls() => Volatile.Write(ref _callSnapshot, [.. _calls.Values]);

    /// <summary>Liveness, checked off the I/O threads so it fires while a read waits on a gateway that went quiet or a
    /// write is blocked on one that stopped reading.</summary>
    private void Watch()
    {
        long now = MonotonicClock.NowNs();
        long limit = _options.LivenessTimeoutMs * 1_000_000L;
        long writeStarted = Volatile.Read(ref _writeStartedNs);
        string? why = now - Volatile.Read(ref _lastFrameNs) > limit ? "sent nothing"
            : writeStarted != 0 && now - writeStarted > limit ? "stopped reading (a write is blocked)"
            : null;
        if (why is null || _closing)
        {
            return;
        }
        Logs.Warning($"[VoiceHost] The phone gateway {why} for {_options.LivenessTimeoutMs} ms; closing the link.");
        Close($"the gateway {why}");
    }

    private void SenderMain()
    {
        long period = _options.SenderPeriodNs;
        int maxCatchUp = _options.MaxCatchUpFrames;
        long t0 = MonotonicClock.NowNs();
        long n = 1;
        long ticks = 0;
        try
        {
            while (!_senderStop)
            {
                long deadline = t0 + n * period;
                MonotonicClock.SleepUntil(deadline);
                if (_senderStop)
                {
                    break;
                }
                long now = MonotonicClock.NowNs();
                long late = now - deadline;
                _lateness.Record(late);
                int frames = 1;
                n++;
                if (late >= period)
                {
                    long behind = late / period;
                    if (behind <= maxCatchUp)
                    {
                        frames += (int)behind;
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
                WriteControlFrames();
                SendAudio(frames, ++ticks);
            }
        }
        catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or LinkProtocolException)
        {
            if (!_closing)
            {
                Logs.Warning($"[VoiceHost] Writing to the phone gateway failed: {ex.Message}");
                Close("a write failed");
            }
        }
        catch (Exception ex)
        {
            // Anything else would end the host from this background thread; drop the connection instead.
            Logs.Error("[VoiceHost] The PhoneLink sender failed unexpectedly; closing the connection", ex);
            Close("the sender failed");
        }
    }

    private void WriteControlFrames()
    {
        while (_control.TryDequeue(out HostControlItem item))
        {
            WriteControl(item);
            Volatile.Write(ref _controlWritten, _controlWritten + 1);
        }
    }

    private void WriteControl(in HostControlItem item)
    {
        VoiceCall? call = item.Call;
        if (call is { EndSent: true })
        {
            return;
        }
        CancellationToken none = CancellationToken.None;
        Volatile.Write(ref _writeStartedNs, MonotonicClock.NowNs());
        switch (item.Type)
        {
            case LinkMessageType.Event:
                Await(_writer.WriteEventAsync(item.CallId, (LinkEventMessage)item.Payload!, none));
                break;
            case LinkMessageType.Flush:
                Await(_writer.WriteFlushAsync(item.CallId, item.Arg, none));
                call?.MarkFlushed((int)item.Arg);
                break;
            case LinkMessageType.ToolRequest:
                Await(_writer.WriteToolRequestAsync(item.CallId, item.Arg, (ToolRequestMessage)item.Payload!, none));
                break;
            case LinkMessageType.CallEnd:
                Await(_writer.WriteCallEndAsync(item.CallId, (LinkCallEndReason)item.Arg, none));
                if (call is not null)
                {
                    call.EndSent = true;
                }
                break;
            case LinkMessageType.Pong:
                Await(_writer.WritePongAsync(item.Timestamp, none));
                break;
            case LinkMessageType.Error:
                Await(_writer.WriteErrorAsync(item.CallId, (LinkErrorMessage)item.Payload!, none));
                break;
            default:
                throw new InvalidOperationException($"{item.Type} is not a control frame the host sends.");
        }
        Volatile.Write(ref _writeStartedNs, 0);
    }

    /// <summary>Moves reply audio for every call; measures the path's allocations once the warm-up ticks are done.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void SendAudio(int frames, long tick)
    {
        bool measured = tick > _options.SenderWarmUpTicks;
        long before = measured ? GC.GetAllocatedBytesForCurrentThread() : 0;
        VoiceCall[] calls = Volatile.Read(ref _callSnapshot);
        for (int i = 0; i < calls.Length; i++)
        {
            VoiceCall call = calls[i];
            if (call.AudioReady && !call.EndSent)
            {
                PumpCall(call, frames);
            }
        }
        if (measured)
        {
            long spent = GC.GetAllocatedBytesForCurrentThread() - before;
            if (spent != 0)
            {
                Volatile.Write(ref _audioAllocated, _audioAllocated + spent);
            }
        }
    }

    /// <summary>Up to <paramref name="frames"/> frames of one call's reply audio, plus the prebuffer when a burst starts.
    /// Audio of a flushed turn is dropped; a turn's <c>OutboundEnd</c> goes out before the next turn's first frame, or once
    /// the session reports the turn finished and nothing of it is left, and the call's <see cref="VoiceCall.DrainedTurn"/>
    /// follows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PumpCall(VoiceCall call, int frames)
    {
        // Taken before the read: a session reports a turn finished only after writing all of its audio, so an empty read
        // after this snapshot means everything up to it has been read.
        int completed = call.CompletedTurn;
        int budget = frames * _frameSamples + (call.Bursting ? 0 : _prebufferSamples);
        bool any = false;
        while (budget > 0)
        {
            int got;
            int turnId;
            try
            {
                got = call.ReadOutbound(_audio.AsSpan(0, Math.Min(budget, _frameSamples)), out turnId);
            }
            catch (Exception ex)
            {
                call.Fault("reading its reply audio", ex);
                return;
            }
            if (got == 0)
            {
                break;
            }
            any = true;
            budget -= got;
            if (call.OpenTurn != 0 && turnId != call.OpenTurn)
            {
                EndTurn(call);
            }
            if (turnId <= call.FlushedTurn)
            {
                call.CountOutbound(0, got);
                continue;
            }
            for (int s = 0; s < got; s++)
            {
                _pcm[s] = (short)Math.Clamp(_audio[s] * ToPcm16, short.MinValue, short.MaxValue);
            }
            Volatile.Write(ref _writeStartedNs, MonotonicClock.NowNs());
            Await(_writer.WriteOutboundAudioAsync(call.CallId, (uint)turnId, _pcm.AsMemory(0, got), CancellationToken.None));
            Volatile.Write(ref _writeStartedNs, 0);
            call.OpenTurn = turnId;
            call.CountOutbound(got, 0);
        }
        call.Bursting = any;
        if (any)
        {
            return;
        }
        if (call.OpenTurn != 0 && completed >= call.OpenTurn)
        {
            EndTurn(call);
        }
        if (call.OpenTurn == 0)
        {
            // Nothing open and nothing queued: every finished turn is out, including one that never had audio.
            call.MarkDrained(completed);
        }
    }

    private void EndTurn(VoiceCall call)
    {
        int turn = call.OpenTurn;
        call.OpenTurn = 0;
        if (turn > call.FlushedTurn)
        {
            Volatile.Write(ref _writeStartedNs, MonotonicClock.NowNs());
            Await(_writer.WriteOutboundEndAsync(call.CallId, (uint)turn, CancellationToken.None));
            Volatile.Write(ref _writeStartedNs, 0);
        }
        call.MarkDrained(turn);
    }

    private static void Await(ValueTask task)
    {
        if (task.IsCompleted)
        {
            task.GetAwaiter().GetResult();
            return;
        }
        task.AsTask().GetAwaiter().GetResult();
    }

    private static T Await<T>(ValueTask<T> task) => task.IsCompleted ? task.GetAwaiter().GetResult() : task.AsTask().GetAwaiter().GetResult();
}
