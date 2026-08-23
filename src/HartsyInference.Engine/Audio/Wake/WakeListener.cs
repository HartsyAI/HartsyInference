using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Accepts satellite connections and pumps their audio into <see cref="WakeSession"/>s.
///
/// <para>A plain TCP listener rather than a Kestrel connection handler, so the same code serves both hosts the
/// engine runs under: the standalone API server and the SwarmUI extension, which loads the engine in-process
/// and has no Kestrel pipeline of its own to hang a handler off.</para>
///
/// <para>The server half of self-healing lives here: sessions are keyed by device id so a reconnect resumes the
/// device's configuration, a ping/pong keeps half-open sockets from lingering invisibly (a satellite that lost
/// its AP keeps a socket that looks alive to the sender indefinitely), and every per-connection failure is
/// contained to that connection. The client half — backoff with jitter, hello on reconnect, a hardware watchdog
/// — belongs to the device and is specified in <c>docs/Research/WAKE_SATELLITE_PROTOCOL.md</c>.</para></summary>
public sealed class WakeListener : IDisposable
{
    private readonly ConcurrentDictionary<string, WakeSession> _sessions;
    private readonly Func<string, WakeSession> _sessionFactory;
    private readonly WakeServiceOptions _options;
    private readonly CancellationTokenSource _stopping = new();
    private TcpListener? _listener;
    private Task? _acceptLoop;
    private int _disposed;

    /// <summary>The port actually bound, which differs from the configured one when port 0 was requested.</summary>
    public int Port { get; private set; }

    public WakeListener(ConcurrentDictionary<string, WakeSession> sessions, Func<string, WakeSession> sessionFactory, WakeServiceOptions options)
    {
        _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
        _sessionFactory = sessionFactory ?? throw new ArgumentNullException(nameof(sessionFactory));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public void Start()
    {
        _listener = new TcpListener(IPAddress.Parse(_options.BindAddress), _options.Port);
        _listener.Start();
        Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
        _acceptLoop = Task.Run(() => AcceptLoopAsync(_stopping.Token));
        Logs.Info($"[Audio][Wake] Listening for satellites on {_options.BindAddress}:{Port}.");
    }

    private async Task AcceptLoopAsync(CancellationToken cancel)
    {
        while (!cancel.IsCancellationRequested)
        {
            TcpClient client;
            try
            {
                client = await _listener!.AcceptTcpClientAsync(cancel).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { return; }
            catch (Exception ex)
            {
                Logs.Error("[Audio][Wake] Accept failed; the listener continues.", ex);
                continue;
            }
            _ = Task.Run(() => ServeAsync(client, cancel), CancellationToken.None);
        }
    }

    private async Task ServeAsync(TcpClient client, CancellationToken cancel)
    {
        string remote = client.Client.RemoteEndPoint?.ToString() ?? "?";
        try
        {
            // Small writes must go out immediately; Nagle would coalesce 80 ms audio frames into
            // bursts and add latency for nothing on a LAN.
            client.NoDelay = true;
            ConfigureKeepAlive(client.Client);
            using NetworkStream stream = client.GetStream();
            await ServeStreamAsync(stream, remote, cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio][Wake] Connection from {remote} ended: {ex.Message}");
        }
        finally
        {
            client.Dispose();
        }
    }

    /// <summary>Runs the satellite protocol over any duplex stream, so the same session handling serves both the raw TCP listener and a WebSocket-tunnelled connection. The transport only has to deliver bytes in order.</summary>
    public async Task ServeStreamAsync(Stream stream, string remote, CancellationToken cancel)
    {
        WakeSession? session = null;
        try
        {
            WakeFrameCodec codec = new(stream, _options.MaxPayloadBytes);
            using CancellationTokenSource connectionCancel = CancellationTokenSource.CreateLinkedTokenSource(cancel);

            float[] samples = new float[_options.MaxPayloadBytes / 2];
            Task? pingLoop = null;

            while (!connectionCancel.IsCancellationRequested)
            {
                WakeFrame? maybe = await codec.ReadAsync(connectionCancel.Token).ConfigureAwait(false);
                if (maybe is not WakeFrame frame) break;

                if (session is null && frame.Type != "hello")
                    throw new InvalidOperationException($"First frame from {remote} was '{frame.Type}', expected 'hello'.");

                switch (frame.Type)
                {
                    case "hello":
                    {
                        string deviceId = frame.Data.DeviceId ?? throw new InvalidOperationException($"hello from {remote} has no device_id.");
                        if (!IsTokenValid(frame.Data.Token))
                        {
                            // Deliberately vague to the peer, specific in the log: a client that can distinguish
                            // "wrong token" from "no such device" learns which device ids exist.
                            Logs.Warning($"[Audio][Wake] Rejected '{deviceId}' from {remote}: bad or missing auth token.");
                            await codec.WriteAsync("error", "{\"text\":\"unauthorized\"}", connectionCancel.Token).ConfigureAwait(false);
                            return;
                        }
                        ValidateFormat(frame.Data, remote);
                        session = _sessions.GetOrAdd(deviceId, _sessionFactory);
                        if (session.Codec is not null)
                        {
                            // Two live connections claiming one device id — usually cloned firmware that never
                            // got a unique id. Both would write into one single-producer buffer, so the older
                            // one is dropped rather than letting them interleave into corrupt audio.
                            Logs.Warning($"[Audio][Wake] Device id '{deviceId}' reconnected from {remote} while another connection was live; dropping the older one. Give each satellite a unique device_id.");
                        }
                        session.OnReconnected(codec);
                        Logs.Info($"[Audio][Wake] Device '{deviceId}' connected from {remote} ({string.Join(", ", session.Pipeline.Words)}).");
                        await codec.WriteAsync("hello-ack", $"{{\"words\":[{string.Join(",", session.Pipeline.Words.Select(WakeFrameCodec.Escape))}]}}", connectionCancel.Token).ConfigureAwait(false);
                        pingLoop ??= Task.Run(() => PingLoopAsync(codec, connectionCancel), CancellationToken.None);
                        break;
                    }
                    case "audio-chunk":
                    {
                        int count = frame.ReadPcm(samples);
                        session!.Enqueue(samples.AsSpan(0, count), frame.Data.Sequence);
                        session.LastActivityUtc = DateTimeOffset.UtcNow;
                        break;
                    }
                    case "pong":
                        session!.LastActivityUtc = DateTimeOffset.UtcNow;
                        break;
                    case "ping":
                        await codec.WriteAsync("pong", null, connectionCancel.Token).ConfigureAwait(false);
                        break;
                    case "bye":
                        Logs.Info($"[Audio][Wake] Device '{session!.DeviceId}' disconnected cleanly.");
                        await connectionCancel.CancelAsync().ConfigureAwait(false);
                        break;
                    default:
                        Logs.Verbose($"[Audio][Wake] Ignoring unknown frame '{frame.Type}' from {remote}.");
                        break;
                }
            }
        }
        finally
        {
            // The session object stays registered so the device keeps its words and config across the gap;
            // only the transport is torn down. Detection stops because no audio arrives.
            if (session is not null)
            {
                session.Codec = null;
                session.State = WakeSessionState.Handshake;
            }
        }
    }

    /// <summary>Constant-time comparison of the presented token against the configured one. No token configured means the check is disabled, which is the LAN default.</summary>
    private bool IsTokenValid(string? presented)
    {
        if (string.IsNullOrEmpty(_options.AuthToken)) return true;
        if (string.IsNullOrEmpty(presented)) return false;
        // Fixed-time so a timing side channel cannot be used to recover the token byte by byte.
        return CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(presented), Encoding.UTF8.GetBytes(_options.AuthToken));
    }

    private void ValidateFormat(WakeFrameData data, string remote)
    {
        // Zero means "unspecified"; anything else must match, because resampling on the always-on path
        // would be a silent CPU cost and a silent accuracy change.
        if (data.Rate != 0 && data.Rate != 16_000)
            throw new InvalidOperationException($"{remote} offered {data.Rate} Hz; this endpoint requires 16000.");
        if (data.Width != 0 && data.Width != 2)
            throw new InvalidOperationException($"{remote} offered {data.Width}-byte samples; this endpoint requires 2 (signed 16-bit).");
        if (data.Channels != 0 && data.Channels != 1)
            throw new InvalidOperationException($"{remote} offered {data.Channels} channels; this endpoint requires mono.");
    }

    private async Task PingLoopAsync(WakeFrameCodec codec, CancellationTokenSource connectionCancel)
    {
        try
        {
            while (!connectionCancel.IsCancellationRequested)
            {
                await Task.Delay(_options.PingInterval, connectionCancel.Token).ConfigureAwait(false);
                await codec.WriteAsync("ping", null, connectionCancel.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            // A failed write is how a half-open socket finally reveals itself.
            Logs.Verbose($"[Audio][Wake] Ping loop ended: {ex.Message}");
            await connectionCancel.CancelAsync().ConfigureAwait(false);
        }
    }

    private static void ConfigureKeepAlive(Socket socket)
    {
        try
        {
            socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime, 30);
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval, 5);
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount, 3);
        }
        catch (Exception ex)
        {
            // Not fatal: the application-level ping is the primary liveness check, this is the backstop.
            Logs.Verbose($"[Audio][Wake] TCP keep-alive tuning unavailable: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stopping.Cancel();
        try { _listener?.Stop(); } catch (Exception ex) { Logs.Verbose($"[Audio][Wake] Listener stop: {ex.Message}"); }
        try { _acceptLoop?.Wait(TimeSpan.FromSeconds(5)); } catch (Exception ex) { Logs.Verbose($"[Audio][Wake] Accept loop stop: {ex.Message}"); }
        _stopping.Dispose();
    }
}
