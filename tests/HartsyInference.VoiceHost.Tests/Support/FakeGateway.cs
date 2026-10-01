using System.Net.Sockets;
using System.Text.Json;
using HartsyInference.Core.Runtime;
using HartsyInference.PhoneLink;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>The gateway's side of a host connection for unit tests: dials the host's socket, writes frames through one
/// locked <see cref="LinkFrameWriter"/>, and records every frame the host sends (payload copied, with the monotonic time
/// it arrived) on a reader thread, in order.</summary>
internal sealed class FakeGateway : IDisposable
{
    public const int WaitMs = 5000;

    private readonly Socket _socket;
    private readonly NetworkStream _stream;
    private readonly LinkFrameReader _reader;
    private readonly LinkFrameWriter _writer;
    private readonly object _sendLock = new();
    private readonly object _framesLock = new();
    private readonly List<RecordedFrame> _frames = [];
    private readonly ManualResetEventSlim _closed = new(false);
    private readonly Thread _readThread;
    private volatile bool _disposed;
    private int _disposeStarted;

    private FakeGateway(Socket socket)
    {
        _socket = socket;
        _stream = new NetworkStream(socket, ownsSocket: false);
        _reader = new LinkFrameReader(_stream);
        _writer = new LinkFrameWriter(_stream);
        _readThread = new Thread(ReadLoop) { IsBackground = true, Name = "fake-gateway-reader" };
        _readThread.Start();
    }

    /// <summary>True once the host closed the connection.</summary>
    public bool IsClosed => _closed.IsSet;

    /// <summary>Raised on the reader thread for every frame after it has been recorded.</summary>
    public event Action<RecordedFrame>? FrameReceived;

    public static FakeGateway Connect(string socketPath)
    {
        Socket socket = new(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        socket.Connect(new UnixDomainSocketEndPoint(socketPath));
        return new FakeGateway(socket);
    }

    /// <summary>Sends <c>Hello</c> and waits for the host's first answer, <c>HelloAck</c> or <c>Error</c>.</summary>
    public RecordedFrame Hello(string token, uint inboundRate = LinkProtocol.InboundSampleRate, ushort version = LinkProtocol.Version)
    {
        Send(w => w.WriteHelloAsync(new LinkHello(version, inboundRate, token), CancellationToken.None));
        if (!WaitUntil(() => AllFrames().Count > 0 || IsClosed, WaitMs) || AllFrames().Count == 0)
        {
            throw new InvalidOperationException("The host answered Hello with nothing.");
        }
        return AllFrames()[0];
    }

    public void Send(Func<LinkFrameWriter, ValueTask> write)
    {
        lock (_sendLock)
        {
            ValueTask task = write(_writer);
            if (task.IsCompleted)
            {
                task.GetAwaiter().GetResult();
            }
            else
            {
                task.AsTask().GetAwaiter().GetResult();
            }
        }
    }

    /// <summary>Writes one frame through a fresh writer, whose sequence restarts at 0: to the host, frames went missing.</summary>
    public void SendWithRestartedSequence(Func<LinkFrameWriter, ValueTask> write)
    {
        lock (_sendLock)
        {
            using LinkFrameWriter restarted = new(_stream);
            ValueTask task = write(restarted);
            if (task.IsCompleted)
            {
                task.GetAwaiter().GetResult();
            }
            else
            {
                task.AsTask().GetAwaiter().GetResult();
            }
        }
    }

    public void SendCallStart(uint callId, bool resume = false, LinkCallDirection direction = LinkCallDirection.Inbound) =>
        Send(w => w.WriteCallStartAsync(callId, new CallStartMessage { Direction = direction, SipCallId = "sip-" + callId, CallerId = "+15550100", Resume = resume },
            CancellationToken.None));

    public void SendInbound(uint callId, short[] pcm, bool concealed = false) =>
        Send(w => w.WriteInboundAudioAsync(callId, pcm, concealed, CancellationToken.None));

    public void SendCallEnd(uint callId, LinkCallEndReason reason) => Send(w => w.WriteCallEndAsync(callId, reason, CancellationToken.None));

    public void SendDtmf(uint callId, char digit) => Send(w => w.WriteDtmfAsync(callId, new LinkDtmf(digit, 120), CancellationToken.None));

    public void SendToolResult(uint callId, uint requestId, LinkToolStatus status, string? message = null) =>
        Send(w => w.WriteToolResultAsync(callId, requestId, new ToolResultMessage { Status = status, Message = message }, CancellationToken.None));

    public void SendPing(ulong timestampNs) => Send(w => w.WritePingAsync(timestampNs, CancellationToken.None));

    public List<RecordedFrame> AllFrames()
    {
        lock (_framesLock)
        {
            return [.. _frames];
        }
    }

    public List<RecordedFrame> FramesOf(LinkMessageType type) => AllFrames().Where(f => f.Header.Type == type).ToList();

    public bool WaitUntil(Func<bool> condition, int timeoutMs = WaitMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (Environment.TickCount64 < deadline)
        {
            if (condition())
            {
                return true;
            }
            Thread.Sleep(2);
        }
        return condition();
    }

    /// <summary>Waits until the host has sent at least <paramref name="count"/> frames of <paramref name="type"/>.</summary>
    public List<RecordedFrame> WaitFor(LinkMessageType type, int count = 1, int timeoutMs = WaitMs)
    {
        if (!WaitUntil(() => FramesOf(type).Count >= count, timeoutMs))
        {
            throw new TimeoutException($"Expected {count} {type} frame(s) from the host, got {FramesOf(type).Count}; saw "
                + string.Join(", ", AllFrames().Select(f => f.Header.Type)));
        }
        return FramesOf(type);
    }

    public bool WaitForClose(int timeoutMs = WaitMs) => _closed.Wait(timeoutMs);

    private void ReadLoop()
    {
        try
        {
            while (!_disposed)
            {
                ValueTask<LinkFrame?> pending = _reader.ReadAsync(CancellationToken.None);
                LinkFrame? next = pending.IsCompleted ? pending.GetAwaiter().GetResult() : pending.AsTask().GetAwaiter().GetResult();
                if (next is null)
                {
                    break;
                }
                RecordedFrame frame = new(next.Value.Header, next.Value.Payload.ToArray(), MonotonicClock.NowNs());
                lock (_framesLock)
                {
                    _frames.Add(frame);
                }
                FrameReceived?.Invoke(frame);
            }
        }
        catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException or LinkProtocolException)
        {
            // The host closed the connection; tests observe IsClosed.
        }
        finally
        {
            _closed.Set();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeStarted, 1) != 0)
        {
            return;
        }
        _disposed = true;
        try
        {
            _socket.Shutdown(SocketShutdown.Both);
        }
        catch (SocketException)
        {
            // Already closed by the host.
        }
        _socket.Dispose();
        _readThread.Join(WaitMs);
        _reader.Dispose();
        _writer.Dispose();
        _stream.Dispose();
        _closed.Dispose();
    }

    /// <summary>One frame from the host with its payload copied out of the reader buffer and the time it arrived.</summary>
    public readonly record struct RecordedFrame(LinkFrameHeader Header, byte[] Payload, long ReceivedNs)
    {
        public LinkFrame AsFrame() => new(Header, Payload);

        public LinkEventMessage Event => AsFrame().ReadEvent();

        public uint TurnId => AsFrame().ReadTurnId();

        public short[] Pcm
        {
            get
            {
                short[] samples = new short[AsFrame().PcmSampleCount];
                AsFrame().ReadPcm(samples);
                return samples;
            }
        }

        public ToolRequestMessage ToolRequest(out uint requestId) => AsFrame().ReadToolRequest(out requestId);

        public string? ToolArgument(string name) =>
            ToolRequest(out _).Arguments is JsonElement args && args.TryGetProperty(name, out JsonElement value) ? value.GetString() : null;
    }
}
