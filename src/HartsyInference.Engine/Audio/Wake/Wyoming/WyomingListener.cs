using System.Net;
using System.Net.Sockets;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Home Assistant / Wyoming compatibility endpoint: accepts the connections Home Assistant's Wyoming
/// integration dials and serves its wake, speech-to-text and text-to-speech domains from this engine.
///
/// <para>Wire it by constructing one with the engine and options, calling <see cref="Start"/>, and disposing it at
/// shutdown; then add a Wyoming integration in Home Assistant pointing at this host and <see cref="Port"/>
/// (defaults to 10600, deliberately not the satellite listener's 10800 so both can run in one process). Home
/// Assistant sends <c>describe</c> on connect and lists whatever the reply advertises.</para>
///
/// <para>A plain <see cref="TcpListener"/> for the same reason the satellite listener is one: the engine runs both
/// under the standalone API server and inside the SwarmUI extension, and the latter has no Kestrel pipeline of its
/// own to hang a handler off. Every connection is independent — Home Assistant opens one per operation and drops
/// it afterwards — so a failure is contained to that connection and never touches the accept loop.</para>
///
/// <para>Inference itself is not re-entrant per backend, so a host with other traffic must supply
/// <see cref="WyomingOptions.TranscribeGate"/> and <see cref="WyomingOptions.SynthesizeGate"/>; without them a
/// Home Assistant request can run the shared backend underneath an in-flight generation.</para></summary>
public sealed class WyomingListener : IDisposable
{
    private readonly IInferenceEngine? _engine;
    private readonly WyomingOptions _options;
    private readonly CancellationTokenSource _stopping = new();
    private TcpListener? _listener;
    private Task? _acceptLoop;
    private int _disposed;

    /// <summary>The port actually bound, which differs from the configured one when port 0 was requested.</summary>
    public int Port { get; private set; }

    /// <param name="engine">Null serves <c>describe</c> and <c>ping</c> but answers every ASR/TTS request with an
    /// <c>error</c> event — useful for wiring the endpoint up before an engine exists.</param>
    public WyomingListener(IInferenceEngine? engine, WyomingOptions? options = null)
    {
        _engine = engine;
        _options = options ?? new WyomingOptions();
    }

    /// <summary>Binds the port and starts accepting. Safe to call once.</summary>
    public void Start()
    {
        _listener = new TcpListener(IPAddress.Parse(_options.BindAddress), _options.Port);
        _listener.Start();
        Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
        _acceptLoop = Task.Run(() => AcceptLoopAsync(_stopping.Token));
        Logs.Info($"[Audio][Wyoming] Home Assistant endpoint listening on {_options.BindAddress}:{Port}.");
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
                Logs.Error("[Audio][Wyoming] Accept failed; the listener continues.", ex);
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
            // Audio chunks are small and latency-sensitive in both directions; Nagle would coalesce them.
            client.NoDelay = true;
            ConfigureKeepAlive(client.Client);

            using NetworkStream stream = client.GetStream();
            WyomingFrameCodec codec = new(stream, _options.MaxPayloadBytes);
            using WyomingConnection connection = new(_engine, _options, codec, remote);
            await connection.RunAsync(cancel).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            // Home Assistant closes abruptly once it has its answer, so most of these are ordinary hangups.
            Logs.Warning($"[Audio][Wyoming] Connection from {remote} ended: {ex.Message}");
        }
        finally
        {
            client.Dispose();
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
            // Wake streams can idle for minutes between words; without this a NAT in the path can drop them.
            Logs.Verbose($"[Audio][Wyoming] TCP keep-alive tuning unavailable: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stopping.Cancel();
        try { _listener?.Stop(); } catch (Exception ex) { Logs.Verbose($"[Audio][Wyoming] Listener stop: {ex.Message}"); }
        try { _acceptLoop?.Wait(TimeSpan.FromSeconds(5)); } catch (Exception ex) { Logs.Verbose($"[Audio][Wyoming] Accept loop stop: {ex.Message}"); }
        _stopping.Dispose();
    }
}
