using System.Net.Sockets;
using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Tests.Support;

/// <summary>A voice host stand-in on a temporary Unix socket: answers <c>Hello</c> and <c>Ping</c>, records every frame
/// it receives (payloads copied, since the reader buffer is reused) and can echo inbound audio back as outbound
/// audio for one turn. Test code sends frames through <see cref="Send"/>, which serializes with the echo path on the
/// single-writer <see cref="LinkFrameWriter"/>.</summary>
internal sealed class FakeLinkHost : IDisposable
{
    private readonly object _framesLock = new();
    private readonly List<RecordedFrame> _frames = [];
    private readonly Socket _listener;
    private Thread? _acceptThread;
    private Connection? _current;
    private volatile bool _stopped;
    private int _connections;
    private int _audioFrames;

    public FakeLinkHost()
    {
        SocketPath = Path.Combine(Path.GetTempPath(), "hpl-" + Guid.NewGuid().ToString("N")[..8] + ".sock");
        File.Delete(SocketPath);
        _listener = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        _listener.Bind(new UnixDomainSocketEndPoint(SocketPath));
        _listener.Listen(4);
    }

    public string SocketPath { get; }

    public uint OutboundRate { get; set; } = 16000;

    public ushort MaxFrameMs { get; set; } = 20;

    public bool Echo { get; set; }

    public uint EchoTurnId { get; set; } = 1;

    public bool AnswerPings { get; set; } = true;

    /// <summary>When set, a Hello with another token is answered with Error and the socket closed.</summary>
    public string? ExpectedToken { get; set; }

    public int Connections => Volatile.Read(ref _connections);

    public int AudioFramesReceived => Volatile.Read(ref _audioFrames);

    /// <summary>Raised on the host reader thread for every frame after it has been recorded.</summary>
    public event Action<LinkFrame>? FrameReceived;

    public void Start()
    {
        _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "fake-host-accept" };
        _acceptThread.Start();
    }

    /// <summary>Closes the current connection; the listener stays up so the gateway can reconnect.</summary>
    public void DropConnection()
    {
        Connection? current = Interlocked.Exchange(ref _current, null);
        current?.Close();
    }

    /// <summary>Closes the listener and the current connection: the host is gone until <see cref="Dispose"/>.</summary>
    public void Shutdown()
    {
        _stopped = true;
        _listener.Dispose();
        DropConnection();
    }

    public bool WaitForConnections(int count, int timeoutMs) => WaitUntil(() => Connections >= count, timeoutMs);

    public bool WaitUntil(Func<bool> condition, int timeoutMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (Environment.TickCount64 < deadline)
        {
            if (condition())
            {
                return true;
            }
            Thread.Sleep(5);
        }
        return condition();
    }

    public List<RecordedFrame> FramesOf(LinkMessageType type)
    {
        lock (_framesLock)
        {
            return _frames.Where(f => f.Header.Type == type).ToList();
        }
    }

    public List<RecordedFrame> AllFrames()
    {
        lock (_framesLock)
        {
            return [.. _frames];
        }
    }

    /// <summary>Writes one frame on the current connection, synchronously, serialized with the echo path.</summary>
    public void Send(Func<LinkFrameWriter, ValueTask> write)
    {
        Connection current = _current ?? throw new InvalidOperationException("No gateway connection is up.");
        current.Send(write);
    }

    public void SendOutboundAudio(uint callId, uint turnId, short[] pcm) =>
        Send(w => w.WriteOutboundAudioAsync(callId, turnId, pcm, CancellationToken.None));

    public void SendOutboundEnd(uint callId, uint turnId) => Send(w => w.WriteOutboundEndAsync(callId, turnId, CancellationToken.None));

    public void SendFlush(uint callId, uint turnId) => Send(w => w.WriteFlushAsync(callId, turnId, CancellationToken.None));

    public void SendToolRequest(uint callId, uint requestId, ToolRequestMessage request) =>
        Send(w => w.WriteToolRequestAsync(callId, requestId, request, CancellationToken.None));

    public void SendCallEnd(uint callId, LinkCallEndReason reason) => Send(w => w.WriteCallEndAsync(callId, reason, CancellationToken.None));

    public void SendEvent(uint callId, LinkEventMessage message) => Send(w => w.WriteEventAsync(callId, message, CancellationToken.None));

    private void AcceptLoop()
    {
        while (!_stopped)
        {
            Socket socket;
            try
            {
                socket = _listener.Accept();
            }
            catch (SocketException)
            {
                return;
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            Connection connection = new(this, socket);
            Connection? previous = Interlocked.Exchange(ref _current, connection);
            previous?.Close();
            Interlocked.Increment(ref _connections);
            connection.Start();
        }
    }

    private void Record(in LinkFrame frame)
    {
        lock (_framesLock)
        {
            _frames.Add(new RecordedFrame(frame.Header, frame.Payload.ToArray()));
        }
        if (frame.Type == LinkMessageType.InboundAudio)
        {
            Interlocked.Increment(ref _audioFrames);
        }
        FrameReceived?.Invoke(frame);
    }

    public void Dispose()
    {
        Shutdown();
        File.Delete(SocketPath);
    }

    /// <summary>One received frame with its payload copied out of the reader buffer.</summary>
    public readonly record struct RecordedFrame(LinkFrameHeader Header, byte[] Payload)
    {
        public LinkFrame AsFrame() => new(Header, Payload);
    }

    private sealed class Connection
    {
        private readonly FakeLinkHost _host;
        private readonly Socket _socket;
        private readonly NetworkStream _stream;
        private readonly LinkFrameReader _reader;
        private readonly LinkFrameWriter _writer;
        private readonly object _sendLock = new();
        private readonly short[] _echo = new short[LinkProtocol.InboundFrameSamples];
        private volatile bool _closed;

        public Connection(FakeLinkHost host, Socket socket)
        {
            _host = host;
            _socket = socket;
            _stream = new NetworkStream(socket, ownsSocket: false);
            _reader = new LinkFrameReader(_stream);
            _writer = new LinkFrameWriter(_stream);
        }

        public void Start() => new Thread(ReadLoop) { IsBackground = true, Name = "fake-host-reader" }.Start();

        public void Send(Func<LinkFrameWriter, ValueTask> write)
        {
            lock (_sendLock)
            {
                if (_closed)
                {
                    throw new InvalidOperationException("The gateway connection is closed.");
                }
                ValueTask task = write(_writer);
                if (!task.IsCompleted)
                {
                    task.AsTask().GetAwaiter().GetResult();
                }
                else
                {
                    task.GetAwaiter().GetResult();
                }
            }
        }

        public void Close()
        {
            if (_closed)
            {
                return;
            }
            _closed = true;
            try
            {
                _socket.Shutdown(SocketShutdown.Both);
            }
            catch (SocketException)
            {
                // Peer already gone.
            }
            _socket.Dispose();
        }

        private void ReadLoop()
        {
            try
            {
                while (!_closed)
                {
                    ValueTask<LinkFrame?> pending = _reader.ReadAsync(CancellationToken.None);
                    LinkFrame? next = pending.IsCompleted ? pending.GetAwaiter().GetResult() : pending.AsTask().GetAwaiter().GetResult();
                    if (next is null)
                    {
                        break;
                    }
                    LinkFrame frame = next.Value;
                    _host.Record(frame);
                    Handle(frame);
                }
            }
            catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or LinkProtocolException)
            {
                // Connection torn down by either side; the test observes counters, not this.
            }
            finally
            {
                Close();
            }
        }

        private void Handle(in LinkFrame frame)
        {
            switch (frame.Type)
            {
                case LinkMessageType.Hello:
                {
                    LinkHello hello = frame.ReadHello();
                    if (_host.ExpectedToken is not null && hello.Token != _host.ExpectedToken)
                    {
                        Send(w => w.WriteErrorAsync(LinkProtocol.ConnectionCallId, new LinkErrorMessage { Text = "bad token" }, CancellationToken.None));
                        Close();
                        return;
                    }
                    Send(w => w.WriteHelloAckAsync(new LinkHelloAck(_host.OutboundRate, _host.MaxFrameMs), CancellationToken.None));
                    return;
                }
                case LinkMessageType.Ping:
                    if (_host.AnswerPings)
                    {
                        ulong ts = frame.ReadTimestampNs();
                        Send(w => w.WritePongAsync(ts, CancellationToken.None));
                    }
                    return;
                case LinkMessageType.InboundAudio:
                    if (_host.Echo && _host.OutboundRate == LinkProtocol.InboundSampleRate)
                    {
                        frame.ReadPcm(_echo);
                        uint callId = frame.Header.CallId;
                        Send(w => w.WriteOutboundAudioAsync(callId, _host.EchoTurnId, _echo, CancellationToken.None));
                    }
                    return;
                default:
                    return;
            }
        }
    }
}
