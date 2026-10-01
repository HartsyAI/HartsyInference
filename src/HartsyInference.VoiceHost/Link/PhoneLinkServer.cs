using System.Net.Sockets;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Calls;

namespace HartsyInference.VoiceHost.Link;

/// <summary>The host's side of the PhoneLink socket: listens on a Unix socket, accepts the gateway, and keeps exactly one
/// handshaken connection (a new one that passes <c>Hello</c> replaces the old, whose calls end).</summary>
/// <remarks>At start a socket file left by a host that died is removed, after a connect probe shows nobody listens on it;
/// a live listener there makes the start fail rather than steal the path. The socket file gets
/// <see cref="PhoneLinkServerOptions.SocketMode"/> right after bind, inside a directory the systemd unit creates for the
/// host alone (<c>RuntimeDirectory</c>). The accept thread does nothing but accept; every connection's handshake runs on
/// its own reader thread, so a client that never sends <c>Hello</c> holds up nobody. At most
/// <see cref="MaxPendingHandshakes"/> connections may be in their handshake at once; more are closed at accept, so a
/// misbehaving local process cannot pile up reader threads.</remarks>
internal sealed class PhoneLinkServer : IAsyncDisposable
{
    /// <summary>Connections allowed in their handshake at once.</summary>
    public const int MaxPendingHandshakes = 4;

    private const int AcceptBacklog = 4;
    private const int AcceptRetryMs = 100;
    private const int DrainMs = 1000;

    private readonly PhoneLinkServerOptions _options;
    private readonly object _lock = new();
    private readonly ManualResetEventSlim _stopping = new(false);
    private Socket? _listener;
    private Thread? _acceptThread;
    private LinkConnection? _current;
    private uint _nextRequestId;
    private long _refused;
    private long _turnedAway;
    private int _pendingHandshakes;
    private int _stopped;

    public PhoneLinkServer(PhoneLinkServerOptions options, IVoiceCallSessionFactory factory)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(factory);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.SocketPath);
        if (!LinkProtocol.IsOutboundSampleRate((uint)factory.OutboundSampleRate))
        {
            throw new ArgumentException($"Sessions reply at {factory.OutboundSampleRate} Hz, which PhoneLink cannot carry.", nameof(factory));
        }
        ArgumentOutOfRangeException.ThrowIfNegative(options.PrebufferMs);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.LivenessTimeoutMs, 100);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.HandshakeTimeoutMs, 10);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.SenderPeriodNs, 1_000_000L);
        _options = options;
        Factory = factory;
    }

    public string SocketPath => _options.SocketPath;

    /// <summary>The factory every call's session comes from.</summary>
    public IVoiceCallSessionFactory Factory { get; }

    /// <summary>The handshaken connection, or null.</summary>
    public LinkConnection? Current
    {
        get
        {
            lock (_lock)
            {
                return _current;
            }
        }
    }

    /// <summary>Connections refused at the handshake (bad token, rate, version, no Hello).</summary>
    public long Refused => Interlocked.Read(ref _refused);

    /// <summary>Connections closed at accept because <see cref="MaxPendingHandshakes"/> others were in their handshake.</summary>
    public long TurnedAway => Interlocked.Read(ref _turnedAway);

    /// <summary>Connections in their handshake right now.</summary>
    internal int PendingHandshakes => Volatile.Read(ref _pendingHandshakes);

    /// <summary>Binds the socket and starts accepting.</summary>
    /// <exception cref="InvalidOperationException">Another process listens on the path, or the path is a directory.</exception>
    public void Start()
    {
        lock (_lock)
        {
            if (_listener is not null)
            {
                throw new InvalidOperationException("The PhoneLink server is already started.");
            }
            string? directory = Path.GetDirectoryName(_options.SocketPath);
            if (directory is null || !Directory.Exists(directory))
            {
                throw new InvalidOperationException(
                    $"The directory of link.socketPath {_options.SocketPath} does not exist; the systemd unit's RuntimeDirectory creates it, or create it yourself.");
            }
            RemoveStaleSocket(_options.SocketPath);
            Socket listener = new(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
            try
            {
                listener.Bind(new UnixDomainSocketEndPoint(_options.SocketPath));
                if (!OperatingSystem.IsWindows())
                {
                    File.SetUnixFileMode(_options.SocketPath, _options.SocketMode);
                }
                listener.Listen(AcceptBacklog);
            }
            catch
            {
                listener.Dispose();
                throw;
            }
            _listener = listener;
            _acceptThread = new Thread(AcceptMain) { Name = "voice-link-accept", IsBackground = true };
            _acceptThread.Start();
        }
        Logs.Info($"[VoiceHost] Listening for the phone gateway on {_options.SocketPath} (mode 0{Convert.ToString((int)_options.SocketMode, 8)}).");
        if (_options.Token.Length == 0)
        {
            Logs.Warning("[VoiceHost] link.tokenFile is not set: any local process that can open the socket can drive calls. Set the same token on both sides.");
        }
    }

    /// <summary>Ends every call with <paramref name="reason"/> (sent to the gateway), lets the sender put those frames on
    /// the wire, closes the connection, stops accepting and removes the socket file. Idempotent.</summary>
    public async Task StopAsync(LinkCallEndReason reason)
    {
        if (Interlocked.Exchange(ref _stopped, 1) != 0)
        {
            return;
        }
        _stopping.Set();
        LinkConnection? connection;
        Socket? listener;
        Thread? acceptThread;
        lock (_lock)
        {
            connection = _current;
            listener = _listener;
            acceptThread = _acceptThread;
        }
        listener?.Dispose();
        acceptThread?.Join();
        if (connection is not null)
        {
            IReadOnlyList<VoiceCall> calls = connection.Calls;
            await Task.WhenAll(calls.Select(call => call.EndAsync(reason, "the host is stopping"))).ConfigureAwait(false);
            await connection.DrainControlAsync(DrainMs).ConfigureAwait(false);
            connection.Close("the host is stopping");
            await connection.Closed.ConfigureAwait(false);
        }
        if (listener is not null)
        {
            File.Delete(_options.SocketPath);
        }
        Logs.Info("[VoiceHost] PhoneLink server stopped.");
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync(LinkCallEndReason.LocalHangup).ConfigureAwait(false);
        _stopping.Dispose();
    }

    /// <summary>A connection passed its handshake: it becomes the current one and replaces any other. False while the
    /// server is stopping.</summary>
    internal bool Adopt(LinkConnection connection)
    {
        LinkConnection? previous;
        lock (_lock)
        {
            if (_stopping.IsSet)
            {
                return false;
            }
            previous = _current;
            _current = connection;
        }
        if (previous is not null && !ReferenceEquals(previous, connection))
        {
            Logs.Warning("[VoiceHost] A new phone gateway connection replaces the previous one; its calls end.");
            previous.Close("replaced by a new connection");
        }
        return true;
    }

    /// <summary>A connection closed.</summary>
    internal void Released(LinkConnection connection)
    {
        lock (_lock)
        {
            if (ReferenceEquals(_current, connection))
            {
                _current = null;
            }
        }
    }

    internal uint NextRequestId() => Interlocked.Increment(ref _nextRequestId);

    internal void CountRefused() => Interlocked.Increment(ref _refused);

    /// <summary>A connection's handshake is over, passed or not; its slot is free.</summary>
    internal void HandshakeEnded() => Interlocked.Decrement(ref _pendingHandshakes);

    /// <summary>Removes a socket file nobody listens on (a host that was killed); refuses a path someone does.</summary>
    private static void RemoveStaleSocket(string path)
    {
        if (Directory.Exists(path))
        {
            throw new InvalidOperationException($"link.socketPath {path} is a directory.");
        }
        if (!File.Exists(path))
        {
            return;
        }
        using (Socket probe = new(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified))
        {
            try
            {
                probe.Connect(new UnixDomainSocketEndPoint(path));
            }
            catch (SocketException)
            {
                File.Delete(path);
                Logs.Info($"[VoiceHost] Removed a stale socket file at {path} (no process was listening on it).");
                return;
            }
        }
        throw new InvalidOperationException($"Another process is already listening on {path}; stop it or choose another link.socketPath.");
    }

    private void AcceptMain()
    {
        Socket listener = _listener!;
        while (!_stopping.IsSet)
        {
            Socket client;
            try
            {
                client = listener.Accept();
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            catch (SocketException ex)
            {
                if (_stopping.IsSet)
                {
                    return;
                }
                Logs.Warning($"[VoiceHost] Accepting a PhoneLink connection failed: {ex.SocketErrorCode}; retrying.");
                _stopping.Wait(AcceptRetryMs);
                continue;
            }
            if (Interlocked.Increment(ref _pendingHandshakes) > MaxPendingHandshakes)
            {
                Interlocked.Decrement(ref _pendingHandshakes);
                client.Dispose();
                long turnedAway = Interlocked.Increment(ref _turnedAway);
                if (turnedAway == 1 || turnedAway % 100 == 0)
                {
                    Logs.Warning($"[VoiceHost] Closed a PhoneLink connection at accept: {MaxPendingHandshakes} others are still in their handshake ({turnedAway} so far).");
                }
                continue;
            }
            try
            {
                new LinkConnection(this, client, _options, Factory.OutboundSampleRate).Start();
            }
            catch (Exception ex)
            {
                // Its reader never started, so nothing else frees the slot; an exception here would end the host.
                Interlocked.Decrement(ref _pendingHandshakes);
                client.Dispose();
                Logs.Error("[VoiceHost] Starting a PhoneLink connection failed", ex);
            }
        }
    }
}
