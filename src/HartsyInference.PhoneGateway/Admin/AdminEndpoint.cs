using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.PhoneGateway.Metrics;
using HartsyInference.PhoneGateway.Sip;

namespace HartsyInference.PhoneGateway.Admin;

/// <summary>Loopback-only HTTP endpoint: <c>GET /health</c> (JSON), <c>GET /metrics</c> (Prometheus text) and
/// <c>POST /calls</c> (bearer-token gated; disabled when no token is configured). Bound to 127.0.0.1 only; anything
/// wider is a reverse proxy's job.</summary>
public sealed class AdminEndpoint : IDisposable
{
    private const int MaxBodyBytes = 4096;
    private const int StopWaitMs = 1000;

    private readonly int _port;
    private readonly byte[]? _token;
    private readonly GatewayMetrics _metrics;
    private readonly Func<HealthStatus> _health;
    private readonly Func<string, Task<CallPlacementResult>> _placeCall;
    private readonly HttpListener _listener = new();
    private Task? _loop;

    public AdminEndpoint(int port, string? token, GatewayMetrics metrics, Func<HealthStatus> health, Func<string, Task<CallPlacementResult>> placeCall)
    {
        if (port is < 1 or > 65535)
        {
            throw new ArgumentOutOfRangeException(nameof(port), port, "admin.port must be 1..65535 (0 disables the endpoint before construction).");
        }
        ArgumentNullException.ThrowIfNull(metrics);
        ArgumentNullException.ThrowIfNull(health);
        ArgumentNullException.ThrowIfNull(placeCall);
        _port = port;
        _token = string.IsNullOrEmpty(token) ? null : Encoding.UTF8.GetBytes(token);
        _metrics = metrics;
        _health = health;
        _placeCall = placeCall;
    }

    public string Prefix => $"http://127.0.0.1:{_port}/";

    public void Start()
    {
        _listener.Prefixes.Add(Prefix);
        _listener.Start();
        _loop = Task.Run(AcceptLoopAsync);
        Logs.Info($"[PhoneGateway] Admin endpoint on {Prefix} (POST /calls {(_token is null ? "disabled: no admin token" : "enabled")}).");
    }

    public void Stop()
    {
        if (_listener.IsListening)
        {
            _listener.Stop();
        }
        // The accept loop ends as soon as the listener stops; let it finish before Dispose closes the listener.
        _loop?.Wait(StopWaitMs);
    }

    private async Task AcceptLoopAsync()
    {
        while (_listener.IsListening)
        {
            HttpListenerContext context;
            try
            {
                context = await _listener.GetContextAsync().ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException or InvalidOperationException)
            {
                return;
            }
            _ = Task.Run(() => HandleAsync(context));
        }
    }

    private async Task HandleAsync(HttpListenerContext context)
    {
        HttpListenerRequest request = context.Request;
        HttpListenerResponse response = context.Response;
        try
        {
            string path = request.Url?.AbsolutePath ?? "/";
            if (request.HttpMethod == "GET" && path == "/health")
            {
                HealthStatus health = _health();
                await WriteJsonAsync(response, health.Status == "ok" ? 200 : 503, JsonSerializer.SerializeToUtf8Bytes(health, AdminJsonContext.Default.HealthStatus)).ConfigureAwait(false);
            }
            else if (request.HttpMethod == "GET" && path == "/metrics")
            {
                byte[] body = Encoding.UTF8.GetBytes(PrometheusTextWriter.Render(_metrics));
                await WriteAsync(response, 200, "text/plain; version=0.0.4; charset=utf-8", body).ConfigureAwait(false);
            }
            else if (request.HttpMethod == "POST" && path == "/calls")
            {
                await PlaceCallAsync(request, response).ConfigureAwait(false);
            }
            else
            {
                await WriteAsync(response, 404, "text/plain", "not found\n"u8.ToArray()).ConfigureAwait(false);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("[PhoneGateway] Admin request failed", ex);
            try
            {
                await WriteAsync(response, 500, "text/plain", "internal error\n"u8.ToArray()).ConfigureAwait(false);
            }
            catch (Exception inner) when (inner is HttpListenerException or ObjectDisposedException or IOException)
            {
                // The client is gone; nothing to answer.
            }
        }
    }

    private async Task PlaceCallAsync(HttpListenerRequest request, HttpListenerResponse response)
    {
        if (_token is null)
        {
            await WriteAsync(response, 403, "text/plain", "POST /calls is disabled: no admin token configured\n"u8.ToArray()).ConfigureAwait(false);
            return;
        }
        if (!Authorized(request))
        {
            response.AddHeader("WWW-Authenticate", "Bearer");
            await WriteAsync(response, 401, "text/plain", "unauthorized\n"u8.ToArray()).ConfigureAwait(false);
            return;
        }
        if (request.ContentLength64 > MaxBodyBytes)
        {
            await WriteAsync(response, 413, "text/plain", "body too large\n"u8.ToArray()).ConfigureAwait(false);
            return;
        }
        // A chunked body declares no length, so the cap is enforced on what is actually read.
        byte[] buffer = new byte[MaxBodyBytes + 1];
        int length = 0;
        int read;
        while (length < buffer.Length && (read = await request.InputStream.ReadAsync(buffer.AsMemory(length)).ConfigureAwait(false)) > 0)
        {
            length += read;
        }
        if (length > MaxBodyBytes)
        {
            await WriteAsync(response, 413, "text/plain", "body too large\n"u8.ToArray()).ConfigureAwait(false);
            return;
        }
        PlaceCallRequest? body;
        try
        {
            body = JsonSerializer.Deserialize(buffer.AsSpan(0, length), AdminJsonContext.Default.PlaceCallRequest);
        }
        catch (JsonException)
        {
            body = null;
        }
        if (body is null || string.IsNullOrWhiteSpace(body.Destination))
        {
            await WriteAsync(response, 400, "text/plain", "body must be {\"destination\":\"sip:...\"}\n"u8.ToArray()).ConfigureAwait(false);
            return;
        }
        CallPlacementResult result = await _placeCall(body.Destination).ConfigureAwait(false);
        PlaceCallResponse reply = new() { Placed = result.Placed, Message = result.Message };
        await WriteJsonAsync(response, HttpStatusFor(result.Status), JsonSerializer.SerializeToUtf8Bytes(reply, AdminJsonContext.Default.PlaceCallResponse)).ConfigureAwait(false);
    }

    /// <summary>The HTTP status <c>POST /calls</c> answers for each placement outcome.</summary>
    internal static int HttpStatusFor(CallPlacementStatus status) => status switch
    {
        CallPlacementStatus.Placed => 202,
        CallPlacementStatus.Busy => 409,
        CallPlacementStatus.HostUnavailable => 503,
        CallPlacementStatus.NotAllowed => 403,
        CallPlacementStatus.Invalid => 400,
        _ => 502,
    };

    private bool Authorized(HttpListenerRequest request)
    {
        string? header = request.Headers["Authorization"];
        if (_token is null || header is null || !header.StartsWith("Bearer ", StringComparison.Ordinal))
        {
            return false;
        }
        byte[] presented = Encoding.UTF8.GetBytes(header.Substring(7).Trim());
        return CryptographicOperations.FixedTimeEquals(presented, _token);
    }

    private static Task WriteJsonAsync(HttpListenerResponse response, int status, byte[] body) =>
        WriteAsync(response, status, "application/json; charset=utf-8", body);

    private static async Task WriteAsync(HttpListenerResponse response, int status, string contentType, byte[] body)
    {
        response.StatusCode = status;
        response.ContentType = contentType;
        response.ContentLength64 = body.Length;
        await response.OutputStream.WriteAsync(body).ConfigureAwait(false);
        response.Close();
    }

    public void Dispose()
    {
        Stop();
        _listener.Close();
    }
}
