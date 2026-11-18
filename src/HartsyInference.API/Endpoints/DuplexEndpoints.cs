using System.Buffers.Binary;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>The live conversation route for full-duplex speech-to-speech models: <c>GET /v1/native/duplex/ws</c> upgrades to a WebSocket.</summary>
/// <remarks>
/// <para><b>Protocol.</b> The client's first message is text, a <see cref="NativeDuplexOpenRequest"/> as JSON. The server opens the model through the
/// long-running queue, runs the card and the voice reference, and replies <c>{"type":"ready","sampleRate":24000,"frameSamples":3200,"frameMs":133.3,
/// "leadFrames":2,"format":"pcm16"}</c>. From then on the client sends binary messages of microphone audio (mono, at the rate and encoding it asked for, any
/// length per message) and the server sends one binary message per model frame (mono PCM at <c>sampleRate</c>, same encoding) plus text events:
/// <c>{"type":"text","frame":n,"text":"..."}</c> when the model speaks words, <c>{"type":"stats",...}</c> every few seconds and at the end, and
/// <c>{"type":"error","message":"..."}</c>. The client closes the socket (or sends <c>{"type":"close"}</c>) to end the conversation; the server also
/// ends it when the model reaches its frame limit.</para>
/// <para><b>Real time.</b> The server ticks once per frame whatever the client does, so the client should play frames at a steady rate after holding
/// <c>leadFrames</c> of them: a frame that computes late still arrives in time while it is within the lead. There is no echo cancellation: use headphones.
/// The conversation holds the model for its whole life (a second one is refused until it ends) but only the open runs in the long-running
/// queue: once it is open it no longer holds a slot, so other long-running jobs can start on the same device while it talks, and a busy device is what makes
/// frames late. Text messages other than <c>{"type":"close"}</c> are ignored.</para></remarks>
public static class DuplexEndpoints
{
    private const string Route = "/v1/native/duplex/ws";

    /// <summary>How many frames between statistics events (about 5 s of Haru).</summary>
    private const int StatsEveryFrames = 38;

    /// <summary>Maps <c>/v1/native/duplex/ws</c>.</summary>
    public static void MapDuplexEndpoints(this WebApplication app)
    {
        app.MapGet(Route, async (HttpContext ctx, IInferenceEngine engine, [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue) =>
        {
            if (!ctx.WebSockets.IsWebSocketRequest)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(
                    ctx, StatusCodes.Status400BadRequest, "This route is a WebSocket: connect with ws:// or wss:// and send the open message first.", "invalid_request_error");
                return;
            }
            using WebSocket socket = await ctx.WebSockets.AcceptWebSocketAsync();
            await ServeAsync(socket, ctx, engine, queue);
        });
    }

    private static async Task ServeAsync(WebSocket socket, HttpContext ctx, IInferenceEngine engine, InferenceQueue queue)
    {
        CancellationToken aborted = ctx.RequestAborted;
        JsonSerializerOptions json = SseHelpers.ResolveJsonOptions(ctx);
        DuplexVoiceSession? voice = null;
        WebSocketCloseStatus status = WebSocketCloseStatus.NormalClosure;
        try
        {
            NativeDuplexOpenRequest open = await ReadOpenAsync(socket, json, aborted);
            bool pcm16 = ParseFormat(open.Format);
            ModelSpec spec = ModelResolver.Resolve(open.Model, open.ModelPath, Modality.Speech);
            IDuplexSpeechSession session = await queue.EnqueueAsync(() => engine.Speech.OpenDuplexAsync(spec, open.Request, aborted), aborted);
            try
            {
                voice = new DuplexVoiceSession(session, new DuplexVoiceOptions { LeadFrames = open.LeadFrames, InputSampleRate = open.InputSampleRate });
            }
            catch
            {
                // The model allows one conversation at a time: a session that never reached the wrapper would hold it until the process ends.
                session.Dispose();
                throw;
            }
            await Task.Run(voice.Start, aborted);
            await SendJsonAsync(socket, json, new
            {
                type = "ready",
                sampleRate = voice.SampleRate,
                frameSamples = voice.FrameSamples,
                frameMs = voice.FrameMs,
                leadFrames = open.LeadFrames,
                format = pcm16 ? "pcm16" : "f32",
            }, aborted);

            using CancellationTokenSource done = CancellationTokenSource.CreateLinkedTokenSource(aborted);
            Task receive = ReceiveAsync(socket, voice, pcm16, done.Token);
            Task send = SendAsync(socket, voice, json, pcm16, done.Token);
            Task first = await Task.WhenAny(receive, send);
            if (first == receive)
            {
                // The client ended it: the sender is waiting on the frame queue, and cancelling that does not touch the socket.
                await done.CancelAsync();
                await Swallow(send);
                await Swallow(receive);
            }
            else
            {
                // The model ended it (its frame limit), the pacer stopped with an error, or a send failed. The receiver is waiting on the socket, and
                // cancelling that would abort the connection and lose the last messages, so it is left to finish when the client answers the close.
                Exception? failure = send.IsFaulted ? send.Exception!.GetBaseException() : null;
                if (failure is not null && failure is not (WebSocketException or OperationCanceledException or ObjectDisposedException))
                {
                    Logs.Warning($"[Duplex] A conversation ended with an error: {failure.Message}");
                    await SendJsonAsync(socket, json, new { type = "error", message = failure.Message }, CancellationToken.None);
                    status = WebSocketCloseStatus.InternalServerError;
                }
            }
            await SendJsonAsync(socket, json, new { type = "stats", final = true, stats = voice.Stats() }, CancellationToken.None);
            if (first != receive)
            {
                await CloseOutputAsync(socket, status);
                // Give the client a moment to answer with its own close frame; after that the read is cancelled (which ends the connection).
                if (await Task.WhenAny(receive, Task.Delay(TimeSpan.FromSeconds(5))) != receive)
                {
                    await done.CancelAsync();
                }
                await Swallow(receive);
            }
        }
        catch (OperationCanceledException)
        {
            // The client went away or the server is stopping.
        }
        catch (Exception ex) when (ex is not WebSocketException)
        {
            Logs.Warning($"[Duplex] A conversation ended with an error: {ex.Message}");
            await SendJsonAsync(socket, json, new { type = "error", message = ex.Message }, CancellationToken.None);
            status = WebSocketCloseStatus.InternalServerError;
        }
        catch (WebSocketException)
        {
            // The connection dropped.
        }
        finally
        {
            voice?.Dispose();
            await CloseAsync(socket, status);
        }
    }

    private static async Task<NativeDuplexOpenRequest> ReadOpenAsync(WebSocket socket, JsonSerializerOptions json, CancellationToken ct)
    {
        using MemoryStream message = new();
        byte[] buffer = new byte[16 * 1024];
        WebSocketReceiveResult result;
        do
        {
            result = await socket.ReceiveAsync(buffer, ct);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                throw new OperationCanceledException("The client closed the socket before opening a conversation.");
            }
            message.Write(buffer, 0, result.Count);
            if (message.Length > 64L * 1024 * 1024)
            {
                throw new InvalidOperationException("The open message is too large (64 MB at most, including the voice reference).");
            }
        }
        while (!result.EndOfMessage);
        if (result.MessageType != WebSocketMessageType.Text)
        {
            throw new InvalidOperationException("The first message must be text: the JSON open request.");
        }
        NativeDuplexOpenRequest open = JsonSerializer.Deserialize<NativeDuplexOpenRequest>(message.ToArray(), json)
            ?? throw new InvalidOperationException("The open request is empty.");
        if (open.LeadFrames is < 0 or > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(open), "leadFrames is 0 to 8.");
        }
        if (open.InputSampleRate < 0 || open.InputSampleRate % 50 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(open), "inputSampleRate is 0 (the model's own) or a rate that divides by 50, such as 8000, 16000, 44100 or 48000.");
        }
        return open;
    }

    private static bool ParseFormat(string? format) => (format ?? "pcm16").ToLowerInvariant() switch
    {
        "pcm16" => true,
        "f32" => false,
        _ => throw new ArgumentException("format is pcm16 or f32."),
    };

    /// <summary>Client audio to the conversation: bytes of any message length, a half sample carried across messages.</summary>
    private static async Task ReceiveAsync(WebSocket socket, DuplexVoiceSession voice, bool pcm16, CancellationToken ct)
    {
        int width = pcm16 ? 2 : 4;
        byte[] buffer = new byte[width * 8192];
        float[] samples = new float[buffer.Length / width];
        int carry = 0;
        while (!ct.IsCancellationRequested && socket.State == WebSocketState.Open)
        {
            ValueWebSocketReceiveResult result = await socket.ReceiveAsync(buffer.AsMemory(carry), ct);
            if (result.MessageType == WebSocketMessageType.Close)
            {
                return;
            }
            if (result.MessageType == WebSocketMessageType.Text)
            {
                if (IsCloseMessage(buffer.AsSpan(carry, result.Count)))
                {
                    return;
                }
                continue;
            }
            int bytes = carry + result.Count;
            int whole = bytes / width;
            for (int i = 0; i < whole; i++)
            {
                samples[i] = pcm16
                    ? BinaryPrimitives.ReadInt16LittleEndian(buffer.AsSpan(i * 2, 2)) / 32768f
                    : BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(i * 4, 4));
            }
            voice.PushUserAudio(samples.AsSpan(0, whole));
            carry = bytes - whole * width;
            if (carry > 0)
            {
                buffer.AsSpan(whole * width, carry).CopyTo(buffer);
            }
        }
    }

    private static bool IsCloseMessage(ReadOnlySpan<byte> text)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(text.ToArray());
            return doc.RootElement.TryGetProperty("type", out JsonElement type) && type.GetString() == "close";
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>The model's frames to the client, with the words it spoke and the statistics now and then.</summary>
    private static async Task SendAsync(WebSocket socket, DuplexVoiceSession voice, JsonSerializerOptions json, bool pcm16, CancellationToken ct)
    {
        int width = pcm16 ? 2 : 4;
        byte[] bytes = new byte[voice.FrameSamples * width];
        await foreach (DuplexVoiceFrame frame in voice.Frames.ReadAllAsync(ct))
        {
            for (int i = 0; i < frame.Audio.Length; i++)
            {
                if (pcm16)
                {
                    BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(i * 2, 2), (short)Math.Clamp((int)MathF.Round(frame.Audio[i] * 32767f), -32768, 32767));
                }
                else
                {
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), frame.Audio[i]);
                }
            }
            await socket.SendAsync(bytes.AsMemory(0, frame.Audio.Length * width), WebSocketMessageType.Binary, true, ct);
            if (frame.Text.Length > 0)
            {
                await SendJsonAsync(socket, json, new { type = "text", frame = frame.Index, text = frame.Text }, ct);
            }
            if (frame.Index > 0 && frame.Index % StatsEveryFrames == 0)
            {
                await SendJsonAsync(socket, json, new { type = "stats", stats = voice.Stats() }, ct);
            }
        }
    }

    private static async Task SendJsonAsync(WebSocket socket, JsonSerializerOptions json, object message, CancellationToken ct)
    {
        if (socket.State != WebSocketState.Open)
        {
            return;
        }
        try
        {
            byte[] payload = JsonSerializer.SerializeToUtf8Bytes(message, json);
            await socket.SendAsync(payload, WebSocketMessageType.Text, true, ct);
        }
        catch (Exception ex) when (ex is WebSocketException or OperationCanceledException or ObjectDisposedException)
        {
            // The client is gone; there is nobody to tell.
        }
    }

    private static async Task CloseOutputAsync(WebSocket socket, WebSocketCloseStatus status)
    {
        try
        {
            if (socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
            {
                using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(5));
                await socket.CloseOutputAsync(status, "conversation ended", timeout.Token);
            }
        }
        catch (Exception ex) when (ex is WebSocketException or OperationCanceledException or ObjectDisposedException)
        {
            // Already closed from the other side.
        }
    }

    private static async Task CloseAsync(WebSocket socket, WebSocketCloseStatus status)
    {
        try
        {
            if (socket.State is WebSocketState.Open or WebSocketState.CloseReceived or WebSocketState.CloseSent)
            {
                using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(5));
                await socket.CloseAsync(status, "conversation ended", timeout.Token);
            }
        }
        catch (Exception ex) when (ex is WebSocketException or OperationCanceledException or ObjectDisposedException)
        {
            // Already closed from the other side.
        }
    }

    private static async Task Swallow(Task task)
    {
        try
        {
            await task;
        }
        catch (Exception ex) when (ex is OperationCanceledException or WebSocketException or ObjectDisposedException)
        {
        }
    }
}
