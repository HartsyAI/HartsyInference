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
        // Bytes per sample this connection is sending, from its hello. Two until told otherwise, so a
        // satellite that predates the µ-law option is read exactly as it always was.
        int width = 2;
        // Declared outside the try so the finally below can identify which codec THIS connection installed,
        // and only clear session state that is still this connection's.
        WakeFrameCodec? codec = null;
        try
        {
            codec = new WakeFrameCodec(stream, _options.MaxPayloadBytes);
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
                        // Per connection, not per session: a satellite that reconnects may have been reflashed
                        // with a different build, and the format it declares now is the one it is sending now.
                        width = frame.Data.Width == 0 ? 2 : frame.Data.Width;
                        session = _sessions.GetOrAdd(deviceId, _sessionFactory);
                        if (session.Codec is not null)
                        {
                            // Two live connections claiming one device id — usually cloned firmware that never
                            // got a unique id. Both would write into one single-producer buffer, so the older
                            // one is dropped rather than letting them interleave into corrupt audio.
                            Logs.Warning($"[Audio][Wake] Device id '{deviceId}' reconnected from {remote} while another connection was live; dropping the older one. Give each satellite a unique device_id.");
                        }
                        // A claim belongs to the connection it was made against, not to the device id for all
                        // time: OnReconnected (right below) resets the ring buffer, the sequence counter and
                        // the pipeline because this is a new, discontinuous turn, and WakeDeviceClaim.OnFrame's
                        // contract is continuous audio for one turn. Without this, the stale-disconnect fix
                        // above makes the claim silently outlive the connection it was made on -- the old
                        // connection's teardown now correctly does nothing, but nothing else would ever clear
                        // the claim either, so it would carry over to this new connection's audio with no
                        // OnDisconnected to tell the host its turn's connection is gone. Ending it here, the
                        // same way an explicit disconnect does, lets the host notice and decide whether to
                        // re-claim once detection resumes on this connection.
                        //
                        // Taken, and notified, before OnReconnected publishes the new codec, not after -- for
                        // two separate reasons, not one:
                        //
                        // Correctness: a claim installed by another thread's WakeService.Claim call is only
                        // valid once it has seen session.Codec non-null, so clearing first means any claim this
                        // Exchange can observe necessarily predates this connection and is safe to end
                        // unconditionally. Clearing after (the first version of this fix) left a window, a few
                        // instructions wide, where a claim installed against the brand-new codec right after
                        // OnReconnected published it could be this Exchange's victim instead -- a spurious
                        // disconnect for a connection that was never replaced. A narrower residual remains: a
                        // claim racing in the gap between this block and OnReconnected still attaches to the
                        // dying old connection and is not re-ended here, so it silently carries over to this
                        // new connection exactly once more. Closing that would need the codec swap itself gated
                        // behind this Exchange, which is a bigger change for a window this narrow; accepted.
                        //
                        // Visibility: Codec is volatile and this Exchange is a full fence, but a release fence
                        // only carries writes that happen BEFORE it (in this thread's program order) to a
                        // thread that acquire-reads the released value -- it says nothing about writes AFTER
                        // it. A reader polling Codec until it changes, then immediately checking whether
                        // OnDisconnected ran, needs that invocation to be one of the writes Codec's release
                        // carries, which only holds if it runs before OnReconnected, not after. A regression
                        // test polling exactly that way caught this: moving only the Exchange earlier (and
                        // leaving the invoke after OnReconnected) left it still flaky under full-suite load,
                        // just less often.
                        WakeDeviceClaim? staleClaim = Interlocked.Exchange(ref session.Claim, null);
                        if (staleClaim is not null)
                        {
                            try
                            {
                                staleClaim.OnDisconnected?.Invoke();
                            }
                            catch (Exception ex)
                            {
                                Logs.Error($"[Audio][Wake] WakeDeviceClaim.OnDisconnected threw for '{deviceId}' on reconnect.", ex);
                            }
                        }
                        session.OnReconnected(codec);
                        Logs.Info($"[Audio][Wake] Device '{deviceId}' connected from {remote} ({string.Join(", ", session.Pipeline.Words)}).");
                        await codec.WriteAsync("hello-ack", $"{{\"words\":[{string.Join(",", session.Pipeline.Words.Select(WakeFrameCodec.Escape))}]}}", connectionCancel.Token).ConfigureAwait(false);
                        pingLoop ??= Task.Run(() => PingLoopAsync(codec, connectionCancel), CancellationToken.None);
                        break;
                    }
                    case "audio-chunk":
                    {
                        int count = frame.ReadPcm(samples, width);
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
            //
            // session is shared across connections for one device id: a reconnect can install a new codec
            // (WakeSession.OnReconnected, called from this device's NEW connection, on another thread) while
            // THIS connection is still unwinding here — the "dropping the older one" log above describes
            // exactly this overlap. Clearing unconditionally would then tear down the new connection's codec,
            // state and claim out from under it and fire a spurious disconnect for a device that is, in fact,
            // still connected. The CAS below only clears what THIS connection installed: if session.Codec is
            // no longer this connection's own codec, a newer one has already taken over and this connection's
            // teardown must do nothing.
            if (session is not null && codec is not null && Interlocked.CompareExchange(ref session.Codec, null, codec) == codec)
            {
                // A residual, few-instruction race remains here: a reconnect's OnReconnected can land between
                // the CompareExchange above and this write, setting State to Listening only for this line to
                // put it back to Handshake moments later. WakeWorker.Run only reads State to skip a session
                // that has never completed a handshake, so at worst that reconnect's audio is skipped for one
                // ~10 ms idle poll before its own next frame corrects it — not the race this guards against.
                session.State = WakeSessionState.Handshake;
                WakeDeviceClaim? claim = Interlocked.Exchange(ref session.Claim, null);
                if (claim is not null)
                {
                    try
                    {
                        claim.OnDisconnected?.Invoke();
                    }
                    catch (Exception ex)
                    {
                        // Last statement in the block, so nothing here is skipped by letting it propagate --
                        // but an unguarded throw at this point would still replace whatever exception (if any)
                        // unwound the try above, hiding the reason this connection actually ended. Log and
                        // swallow the host's bug instead.
                        Logs.Error($"[Audio][Wake] WakeDeviceClaim.OnDisconnected threw for '{session.DeviceId}'.", ex);
                    }
                }
            }
            else if (session is not null)
            {
                Logs.Verbose($"[Audio][Wake] Connection from {remote} ended after a newer connection took over '{session.DeviceId}'; leaving its state alone.");
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
        // One byte per sample means µ-law, which halves what a satellite has to push. See WakeFrame.ReadPcm.
        if (data.Width != 0 && data.Width != 2 && data.Width != 1)
            throw new InvalidOperationException($"{remote} offered {data.Width}-byte samples; this endpoint accepts 2 (signed 16-bit) or 1 (G.711 µ-law).");
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
