using System.Net.Sockets;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Transport;

/// <summary>The gateway's side of the PhoneLink socket: dials the voice host, keeps the connection alive and carries
/// audio and control frames both ways on two dedicated threads.</summary>
/// <remarks>The reader thread owns the connection: it dials with exponential backoff and full jitter, sends
/// <c>Hello</c>, waits for <c>HelloAck</c>, then reads frames until the socket fails, and raises the callbacks below
/// on itself. The writer thread drains <see cref="LinkSendQueue"/> (control first, then audio) and sends a <c>Ping</c>
/// every <see cref="EngineLinkOptions.PingIntervalMs"/>; a timer watchdog closes a connection that has received nothing,
/// or has had one write stuck, for <see cref="EngineLinkOptions.LivenessTimeoutMs"/>. Callbacks that throw are logged,
/// never allowed to end a link thread, and a wedged control lane restarts the connection instead of throwing into the
/// caller. A failed connection empties both lanes: their frames belonged
/// to a session the host no longer has, and the controller re-sends <c>CallStart(resume)</c> from
/// <see cref="Connected"/>. The flush epoch lives here because the spec makes it the reader's job: once
/// <c>Flush(T)</c> has been seen, every <c>OutboundAudio</c> or <c>OutboundEnd</c> with a turn at or below T is dropped
/// on arrival. Callbacks run on the reader thread and must return quickly; none of them may block on the link.</remarks>
public sealed class EngineLink : IDisposable
{
    private const int WriterWaitSliceMs = 250;
    private const int WatchdogMaxPeriodMs = 1000;
    private const int MaxOutboundScratchSamples = LinkProtocol.MaxPayloadBytes / 2;

    private readonly EngineLinkOptions _options;
    private readonly LinkSendQueue _queue;
    private readonly ManualResetEventSlim _backoffWake = new(false);
    private readonly object _connectionLock = new();
    private readonly short[] _audioScratch = new short[LinkProtocol.InboundFrameSamples];
    private Thread? _readerThread;
    private Thread? _writerThread;
    private Socket? _socket;
    private CancellationTokenSource? _readCancel;
    private volatile bool _stopping;
    private volatile bool _connected;
    private volatile bool _handshaken;
    private volatile bool _writerStop;
    private short[] _outboundScratch = [];
    private uint _outboundRate;
    private ushort _maxFrameMs;
    private uint _flushedTurn;
    private uint _flushedCallId;
    private long _lastFrameNs;
    private long _writeStartedNs;
    private int _watchdogFired;
    private long _lastRttNs;
    private long _connectionsMade;
    private long _reconnects;
    private long _staleOutboundDropped;
    private long _inboundDroppedWhileDown;
    private long _framesReceived;
    private long _framesSent;
    private int _unknownTypeLogged;

    public EngineLink(EngineLinkOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrWhiteSpace(options.SocketPath))
        {
            throw new ArgumentException("SocketPath is required.", nameof(options));
        }
        ArgumentOutOfRangeException.ThrowIfLessThan(options.PingIntervalMs, 100);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.LivenessTimeoutMs, options.PingIntervalMs);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.ReconnectBaseMs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.ReconnectCapMs, options.ReconnectBaseMs);
        _options = options;
        _queue = new LinkSendQueue(options.AudioLaneDepth, options.ControlLaneDepth, options.ControlEnqueueTimeoutMs);
    }

    /// <summary>A connection completed its handshake; carries the host's outbound rate and frame bound.</summary>
    public Action<LinkHelloAck>? Connected { get; set; }

    /// <summary>A connection that had completed its handshake is gone; the reason is for the log.</summary>
    public Action<string>? Disconnected { get; set; }

    public OutboundAudioHandler? OutboundAudio { get; set; }

    /// <summary>(callId, turnId): the host finished sending a turn.</summary>
    public Action<uint, uint>? OutboundEnd { get; set; }

    /// <summary>(callId, turnId) → milliseconds discarded; invoked only for a turn above the current epoch.</summary>
    public Func<uint, uint, uint>? Flush { get; set; }

    public Action<uint, LinkCallEndReason>? CallEnd { get; set; }

    public Action<uint, LinkEventMessage>? Event { get; set; }

    /// <summary>(callId, requestId, request): the host asks for a telephony tool; answer with <see cref="SendToolResult"/>.</summary>
    public Action<uint, uint, ToolRequestMessage>? ToolRequest { get; set; }

    public Action<uint, LinkErrorMessage>? Error { get; set; }

    public bool IsConnected => _connected;

    /// <summary>Outbound rate from the latest <c>HelloAck</c>; zero before the first connection.</summary>
    public uint OutboundRate => Volatile.Read(ref _outboundRate);

    public ushort MaxFrameMs => _maxFrameMs;

    /// <summary>Round trip of the latest answered ping, in nanoseconds.</summary>
    public long LastRttNs => Volatile.Read(ref _lastRttNs);

    /// <summary>Connections established after the first one.</summary>
    public long Reconnects => Volatile.Read(ref _reconnects);

    /// <summary>Inbound frames dropped from the audio lane because the writer fell behind.</summary>
    public long AudioLaneDropped => _queue.AudioDropped;

    /// <summary>Inbound frames refused because no connection was up.</summary>
    public long InboundDroppedWhileDown => Volatile.Read(ref _inboundDroppedWhileDown);

    /// <summary>Outbound frames dropped by the flush epoch rule.</summary>
    public long StaleOutboundDropped => Volatile.Read(ref _staleOutboundDropped);

    public long FramesReceived => Volatile.Read(ref _framesReceived);

    public long FramesSent => Volatile.Read(ref _framesSent);

    /// <summary>The highest flushed turn of the current call.</summary>
    public uint FlushedTurn => Volatile.Read(ref _flushedTurn);

    public void Start()
    {
        lock (_connectionLock)
        {
            if (_readerThread is not null)
            {
                throw new InvalidOperationException("EngineLink is already started.");
            }
            _stopping = false;
            _readerThread = new Thread(ReaderMain) { Name = "phone-link-reader", IsBackground = true };
            _readerThread.Start();
        }
    }

    public void Stop()
    {
        Thread? reader;
        lock (_connectionLock)
        {
            _stopping = true;
            reader = _readerThread;
            _readerThread = null;
            _backoffWake.Set();
            CloseSocketLocked();
        }
        reader?.Join();
    }

    /// <summary>Queues one 20 ms inbound frame; false (and counted) when no connection is up.</summary>
    public bool TryEnqueueInboundAudio(uint callId, ReadOnlySpan<short> pcm, bool concealed)
    {
        if (!_connected)
        {
            Interlocked.Increment(ref _inboundDroppedWhileDown);
            return false;
        }
        _queue.EnqueueAudio(callId, pcm, concealed);
        return true;
    }

    /// <summary>Announces a call and resets the flush epoch for it.</summary>
    public void SendCallStart(uint callId, CallStartMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        Volatile.Write(ref _flushedCallId, callId);
        Volatile.Write(ref _flushedTurn, 0);
        Control(new LinkControlItem(LinkMessageType.CallStart, callId, 0, 0, 0, message));
    }

    public void SendCallEnd(uint callId, LinkCallEndReason reason) =>
        Control(new LinkControlItem(LinkMessageType.CallEnd, callId, (uint)reason, 0, 0, null));

    public void SendDtmf(uint callId, LinkDtmf dtmf) =>
        Control(new LinkControlItem(LinkMessageType.DtmfEvent, callId, dtmf.Digit, dtmf.DurationMs, 0, null));

    public void SendToolResult(uint callId, uint requestId, ToolResultMessage result)
    {
        ArgumentNullException.ThrowIfNull(result);
        Control(new LinkControlItem(LinkMessageType.ToolResult, callId, requestId, 0, 0, result));
    }

    public void SendError(uint callId, LinkErrorMessage error)
    {
        ArgumentNullException.ThrowIfNull(error);
        Control(new LinkControlItem(LinkMessageType.Error, callId, 0, 0, 0, error));
    }

    /// <summary>Queues a control frame; silently dropped when no connection is up (its session is gone). Never throws:
    /// a lane the writer has not drained for the whole wait means the link is wedged, so the frame is dropped and the
    /// connection forced to restart, rather than failing a caller that may be in the middle of tearing a call down.</summary>
    private void Control(in LinkControlItem item)
    {
        if (!_connected)
        {
            return;
        }
        if (!_queue.TryEnqueueControl(item))
        {
            Logs.Warning($"[PhoneGateway] PhoneLink control lane stayed full; dropping {item.Type} and reconnecting.");
            ForceReconnect();
        }
    }

    /// <summary>Closes the current socket, which fails the reader and the writer and restarts the connection.</summary>
    private void ForceReconnect()
    {
        Socket? socket;
        lock (_connectionLock)
        {
            socket = _socket;
        }
        if (socket is not null)
        {
            CloseSocketQuietly(socket);
        }
    }

    private void ReaderMain()
    {
        int attempt = 0;
        bool loggedFailure = false;
        while (!_stopping)
        {
            string reason;
            try
            {
                RunConnection(out reason);
                attempt = 0;
                loggedFailure = false;
            }
            catch (Exception ex) when (ex is SocketException or IOException or LinkProtocolException or ObjectDisposedException or OperationCanceledException)
            {
                reason = ex is LinkProtocolException ? $"protocol error: {ex.Message}" : ex.GetType().Name + ": " + ex.Message;
                if (!loggedFailure && !_stopping)
                {
                    Logs.Warning($"[PhoneGateway] Voice host link {_options.SocketPath} unavailable: {reason}. Retrying with backoff.");
                    loggedFailure = true;
                }
            }
            catch (Exception ex)
            {
                // Anything else escaping here would end the whole process from this background thread, calls included.
                reason = "unexpected " + ex.GetType().Name + ": " + ex.Message;
                Logs.Error("[PhoneGateway] Voice host link failed unexpectedly; reconnecting", ex);
            }
            bool wasConnected = _handshaken;
            _handshaken = false;
            TearDownConnection();
            if (wasConnected)
            {
                if (_stopping)
                {
                    Logs.Debug("[PhoneGateway] Voice host link closed on stop.");
                }
                else
                {
                    Logs.Warning($"[PhoneGateway] Voice host link dropped: {reason}");
                }
                RaiseCallback("Disconnected", () => Disconnected?.Invoke(reason));
            }
            if (_stopping)
            {
                break;
            }
            Backoff(attempt++);
        }
    }

    /// <summary>Dials, handshakes and reads until the connection fails. Returns normally only on a clean end of stream.</summary>
    private void RunConnection(out string reason)
    {
        Socket socket = new(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        CancellationTokenSource readCancel = new();
        lock (_connectionLock)
        {
            if (_stopping)
            {
                socket.Dispose();
                readCancel.Dispose();
                reason = "stopping";
                return;
            }
            _socket = socket;
            _readCancel = readCancel;
        }
        socket.Connect(new UnixDomainSocketEndPoint(_options.SocketPath));
        NetworkStream stream = new(socket, ownsSocket: false);
        LinkFrameWriter writer = new(stream);
        LinkFrameReader reader = new(stream);
        Timer? watchdog = null;
        try
        {
            Await(writer.WriteHelloAsync(new LinkHello(LinkProtocol.Version, LinkProtocol.InboundSampleRate, _options.Token), readCancel.Token));
            LinkFrame? first = Await(reader.ReadAsync(readCancel.Token));
            if (first is null)
            {
                throw new LinkProtocolException("The host closed the connection before answering Hello.");
            }
            if (first.Value.Type == LinkMessageType.Error)
            {
                throw new LinkProtocolException($"The host refused Hello: {first.Value.ReadError().Text}");
            }
            LinkHelloAck ack = first.Value.ReadHelloAck();
            Volatile.Write(ref _outboundRate, ack.OutboundRate);
            _maxFrameMs = ack.MaxFrameMs;
            int scratch = (int)Math.Min(MaxOutboundScratchSamples, (long)ack.OutboundRate * ack.MaxFrameMs / 1000 + ack.OutboundRate / 50);
            if (_outboundScratch.Length < scratch)
            {
                _outboundScratch = new short[scratch];
            }
            Volatile.Write(ref _lastFrameNs, MonotonicClock.NowNs());
            Volatile.Write(ref _writeStartedNs, 0);
            Volatile.Write(ref _watchdogFired, 0);
            _queue.Clear();
            _writerStop = false;
            Thread writerThread = new(() => WriterMain(writer, socket)) { Name = "phone-link-writer", IsBackground = true };
            lock (_connectionLock)
            {
                _writerThread = writerThread;
            }
            writerThread.Start();
            int watchPeriod = Math.Clamp(_options.LivenessTimeoutMs / 4, 10, WatchdogMaxPeriodMs);
            watchdog = new Timer(_ => Watch(socket), null, watchPeriod, watchPeriod);
            if (Interlocked.Increment(ref _connectionsMade) > 1)
            {
                Interlocked.Increment(ref _reconnects);
            }
            _handshaken = true;
            _connected = true;
            Logs.Info($"[PhoneGateway] Voice host link up: outbound {ack.OutboundRate} Hz, frames up to {ack.MaxFrameMs} ms.");
            RaiseCallback("Connected", () => Connected?.Invoke(ack));
            while (!_stopping)
            {
                LinkFrame? next = Await(reader.ReadAsync(readCancel.Token));
                if (next is null)
                {
                    reason = "host closed the socket";
                    return;
                }
                Volatile.Write(ref _lastFrameNs, MonotonicClock.NowNs());
                Interlocked.Increment(ref _framesReceived);
                try
                {
                    Dispatch(next.Value);
                }
                catch (Exception ex) when (ex is not (LinkProtocolException or SocketException or IOException or ObjectDisposedException or OperationCanceledException))
                {
                    // A consumer callback failed: log it and keep the link; only a malformed frame is fatal here.
                    Logs.Error($"[PhoneGateway] Handling a {next.Value.Type} frame from the voice host failed", ex);
                }
            }
            reason = "stopping";
        }
        finally
        {
            _connected = false;
            watchdog?.Dispose();
            StopWriter();
            reader.Dispose();
            writer.Dispose();
            stream.Dispose();
        }
    }

    /// <summary>Liveness, checked off the I/O threads so it still fires while a write is blocked on a host that stopped
    /// reading or a read waits on one that stopped writing: nothing received, or one write in flight, for the liveness
    /// timeout closes the socket, which fails both threads and restarts the connection.</summary>
    private void Watch(Socket socket)
    {
        long now = MonotonicClock.NowNs();
        long limit = _options.LivenessTimeoutMs * 1_000_000L;
        long writeStarted = Volatile.Read(ref _writeStartedNs);
        string? why = now - Volatile.Read(ref _lastFrameNs) > limit ? "silent"
            : writeStarted != 0 && now - writeStarted > limit ? "not reading (a write is blocked)"
            : null;
        if (why is null || Interlocked.Exchange(ref _watchdogFired, 1) != 0)
        {
            return;
        }
        Logs.Warning($"[PhoneGateway] Voice host link {why} for {_options.LivenessTimeoutMs} ms; closing it.");
        CloseSocketQuietly(socket);
    }

    /// <summary>Runs a consumer callback so that its exception is logged instead of ending the link thread.</summary>
    private static void RaiseCallback(string name, Action callback)
    {
        try
        {
            callback();
        }
        catch (Exception ex)
        {
            Logs.Error($"[PhoneGateway] PhoneLink {name} callback failed", ex);
        }
    }

    private void Dispatch(in LinkFrame frame)
    {
        uint callId = frame.Header.CallId;
        switch (frame.Type)
        {
            case LinkMessageType.OutboundAudio:
            {
                uint turnId = frame.ReadTurnId();
                if (IsStale(callId, turnId))
                {
                    Interlocked.Increment(ref _staleOutboundDropped);
                    return;
                }
                int samples = frame.PcmSampleCount;
                if (samples > _outboundScratch.Length)
                {
                    throw new LinkProtocolException($"OutboundAudio carries {samples} samples, above the {_maxFrameMs} ms the host promised.");
                }
                int count = frame.ReadPcm(_outboundScratch);
                OutboundAudio?.Invoke(callId, turnId, _outboundScratch.AsSpan(0, count));
                return;
            }
            case LinkMessageType.OutboundEnd:
            {
                uint turnId = frame.ReadTurnId();
                if (IsStale(callId, turnId))
                {
                    Interlocked.Increment(ref _staleOutboundDropped);
                    return;
                }
                OutboundEnd?.Invoke(callId, turnId);
                return;
            }
            case LinkMessageType.Flush:
            {
                uint turnId = frame.ReadTurnId();
                uint discardedMs = 0;
                if (!IsStale(callId, turnId))
                {
                    Volatile.Write(ref _flushedCallId, callId);
                    Volatile.Write(ref _flushedTurn, turnId);
                    discardedMs = Flush?.Invoke(callId, turnId) ?? 0;
                }
                Control(new LinkControlItem(LinkMessageType.FlushAck, callId, turnId, discardedMs, 0, null));
                return;
            }
            case LinkMessageType.CallEnd:
                CallEnd?.Invoke(callId, frame.ReadCallEnd());
                return;
            case LinkMessageType.Event:
                Event?.Invoke(callId, frame.ReadEvent());
                return;
            case LinkMessageType.ToolRequest:
            {
                ToolRequestMessage request = frame.ReadToolRequest(out uint requestId);
                ToolRequest?.Invoke(callId, requestId, request);
                return;
            }
            case LinkMessageType.Ping:
                Control(new LinkControlItem(LinkMessageType.Pong, LinkProtocol.ConnectionCallId, 0, 0, frame.ReadTimestampNs(), null));
                return;
            case LinkMessageType.Pong:
                Volatile.Write(ref _lastRttNs, MonotonicClock.NowNs() - (long)frame.ReadTimestampNs());
                return;
            case LinkMessageType.Error:
            {
                LinkErrorMessage error = frame.ReadError();
                Error?.Invoke(callId, error);
                if (callId == LinkProtocol.ConnectionCallId)
                {
                    throw new LinkProtocolException($"The host reported a connection error: {error.Text}");
                }
                return;
            }
            case LinkMessageType.HelloAck:
                throw new LinkProtocolException("Unexpected HelloAck after the handshake.");
            default:
                if (Interlocked.Exchange(ref _unknownTypeLogged, 1) == 0)
                {
                    Logs.Warning($"[PhoneGateway] Ignoring unknown PhoneLink frame type 0x{(byte)frame.Type:X2} (a newer host?).");
                }
                return;
        }
    }

    private bool IsStale(uint callId, uint turnId) =>
        callId == Volatile.Read(ref _flushedCallId) && turnId <= Volatile.Read(ref _flushedTurn);

    /// <summary>Pings and drains the send lanes. Liveness is the watchdog's job (<see cref="Watch"/>): a write blocked on a
    /// host that stopped reading never returns here to check anything, so every write is stamped for it to see.</summary>
    private void WriterMain(LinkFrameWriter writer, Socket socket)
    {
        long pingInterval = _options.PingIntervalMs * 1_000_000L;
        long nextPing = MonotonicClock.NowNs() + pingInterval;
        try
        {
            while (!_writerStop)
            {
                long now = MonotonicClock.NowNs();
                if (now >= nextPing)
                {
                    Volatile.Write(ref _writeStartedNs, now);
                    Await(writer.WritePingAsync((ulong)now, CancellationToken.None));
                    Volatile.Write(ref _writeStartedNs, 0);
                    Interlocked.Increment(ref _framesSent);
                    nextPing = now + pingInterval;
                }
                int wait = (int)Math.Clamp((nextPing - now) / 1_000_000L, 1, WriterWaitSliceMs);
                if (!_queue.TryDequeue(wait, out LinkControlItem control, _audioScratch, out uint audioCallId, out bool concealed, out bool isAudio))
                {
                    continue;
                }
                Volatile.Write(ref _writeStartedNs, MonotonicClock.NowNs());
                if (isAudio)
                {
                    Await(writer.WriteInboundAudioAsync(audioCallId, _audioScratch, concealed, CancellationToken.None));
                }
                else
                {
                    WriteControl(writer, control);
                }
                Volatile.Write(ref _writeStartedNs, 0);
                Interlocked.Increment(ref _framesSent);
            }
        }
        catch (Exception ex) when (ex is SocketException or IOException or ObjectDisposedException or LinkProtocolException)
        {
            // The reader owns teardown: closing the socket makes its read fail and the connection loop restart.
            if (!_writerStop)
            {
                Logs.Warning($"[PhoneGateway] Voice host link write failed: {ex.Message}");
                CloseSocketQuietly(socket);
            }
        }
        catch (Exception ex)
        {
            // Anything else would end the process from this background thread; restart the connection instead.
            Logs.Error("[PhoneGateway] Voice host link writer failed unexpectedly; reconnecting", ex);
            CloseSocketQuietly(socket);
        }
    }

    private static void WriteControl(LinkFrameWriter writer, in LinkControlItem item)
    {
        CancellationToken none = CancellationToken.None;
        switch (item.Type)
        {
            case LinkMessageType.CallStart:
                Await(writer.WriteCallStartAsync(item.CallId, (CallStartMessage)item.Payload!, none));
                break;
            case LinkMessageType.CallEnd:
                Await(writer.WriteCallEndAsync(item.CallId, (LinkCallEndReason)item.Arg, none));
                break;
            case LinkMessageType.FlushAck:
                Await(writer.WriteFlushAckAsync(item.CallId, new LinkFlushAck(item.Arg, item.Arg2), none));
                break;
            case LinkMessageType.DtmfEvent:
                Await(writer.WriteDtmfAsync(item.CallId, new LinkDtmf((char)item.Arg, (ushort)item.Arg2), none));
                break;
            case LinkMessageType.ToolResult:
                Await(writer.WriteToolResultAsync(item.CallId, item.Arg, (ToolResultMessage)item.Payload!, none));
                break;
            case LinkMessageType.Pong:
                Await(writer.WritePongAsync(item.Timestamp, none));
                break;
            case LinkMessageType.Error:
                Await(writer.WriteErrorAsync(item.CallId, (LinkErrorMessage)item.Payload!, none));
                break;
            default:
                throw new InvalidOperationException($"{item.Type} is not a control frame the gateway sends.");
        }
    }

    private void StopWriter()
    {
        Thread? writerThread;
        lock (_connectionLock)
        {
            writerThread = _writerThread;
            _writerThread = null;
        }
        _writerStop = true;
        _queue.Wake();
        if (writerThread is not null && writerThread != Thread.CurrentThread)
        {
            writerThread.Join();
        }
        _queue.Clear();
    }

    private void TearDownConnection()
    {
        _connected = false;
        StopWriter();
        lock (_connectionLock)
        {
            CloseSocketLocked();
        }
    }

    private void CloseSocketLocked()
    {
        CancellationTokenSource? cancel = _readCancel;
        _readCancel = null;
        Socket? socket = _socket;
        _socket = null;
        if (cancel is not null)
        {
            cancel.Cancel();
            cancel.Dispose();
        }
        if (socket is not null)
        {
            CloseSocketQuietly(socket);
        }
    }

    private static void CloseSocketQuietly(Socket socket)
    {
        try
        {
            socket.Shutdown(SocketShutdown.Both);
        }
        catch (SocketException)
        {
            // Already closed by the peer; the dispose below is what matters.
        }
        catch (ObjectDisposedException)
        {
            return;
        }
        socket.Dispose();
    }

    /// <summary>Exponential backoff with full jitter: a uniform delay in [0, min(cap, base·2^attempt)].</summary>
    private void Backoff(int attempt)
    {
        long ceiling = Math.Min(_options.ReconnectCapMs, (long)_options.ReconnectBaseMs << Math.Min(attempt, 20));
        int delay = (int)Random.Shared.NextInt64(ceiling + 1);
        _backoffWake.Reset();
        if (delay > 0 && !_stopping)
        {
            _backoffWake.Wait(delay);
        }
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

    public void Dispose()
    {
        Stop();
        _backoffWake.Dispose();
    }
}
