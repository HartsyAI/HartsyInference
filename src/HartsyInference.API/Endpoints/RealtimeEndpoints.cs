using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>Native live vision-language session routes (MOSS-VL-Realtime): open a session, push frames and prompts, read the reply as
/// server-sent events, close it. A session lives across requests, so <see cref="RealtimeSessionRegistry"/> finds it again by id.</summary>
/// <remarks>
/// <para>Opening loads the model through the long-running queue. After that the session runs on its own loop: it holds its device's text
/// slot until closed and takes the device gate per forward, so it does not hold a queue slot for its lifetime.</para>
/// <para>Events are buffered until a stream reads them: a client that only pushes input grows the session's memory, and a later stream
/// replays the backlog (which is what a reconnect wants). One stream at a time; a second gets 409. A session whose loop failed
/// (a cap reached, the decoder faulted) reports an <c>error</c> event and is closed, and later input to it is a 404.</para>
/// <para>TODO(websocket): one duplex WebSocket (frames and prompts up, events down) would remove the per-frame HTTP round trip and the
/// separate SSE connection. The session contract (<see cref="IRealtimeVlmSession"/>) is transport-neutral, so a WebSocket route can drive
/// it directly; measure the per-frame POST cost at camera rates before adding one.</para>
/// </remarks>
public static class RealtimeEndpoints
{
    private const string Base = "/v1/native/realtime/sessions";

    /// <summary>How often the event stream sends an SSE comment when the model is silent, so proxies keep the connection open.</summary>
    private static readonly TimeSpan KeepAlive = TimeSpan.FromSeconds(15);

    /// <summary>Maps session open/frame/prompt/stream/stats/close.</summary>
    public static void MapRealtimeEndpoints(this WebApplication app)
    {
        app.MapPost(Base, async (
            NativeRealtimeRequest req, IInferenceEngine engine, RealtimeSessionRegistry registry,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Text);
                IRealtimeVlmSession session = await queue.EnqueueAsync(
                    () => engine.Text.OpenRealtimeAsync(spec, req.Request ?? new RealtimeVlmRequest(), ct), ct);
                return Results.Ok(new { sessionId = registry.Register(session) });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        // TODO(websocket): at camera rates this route is one HTTP request per frame; a duplex socket would carry frames on one connection.
        app.MapPost($"{Base}/{{id}}/frame", (string id, RealtimeFrameRequest req, RealtimeSessionRegistry registry) =>
        {
            IRealtimeVlmSession? session = registry.Get(id);
            if (session is null)
                return SessionNotFound(id);
            try
            {
                Engine.Requests.ImageData frame = CompatEndpoints.DecodeImageUrl(AsDataUri(req.Image));
                bool dropped = session.PushFrame(frame, req.Timestamp);
                return Results.Ok(new { accepted = true, droppedOldest = dropped });
            }
            catch (Exception ex) when (ex is ArgumentException or HartsyInferenceException)
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
            catch (ObjectDisposedException)
            {
                return SessionNotFound(id);
            }
        });

        app.MapPost($"{Base}/{{id}}/prompt", (string id, RealtimePromptRequest req, RealtimeSessionRegistry registry) =>
        {
            IRealtimeVlmSession? session = registry.Get(id);
            if (session is null)
                return SessionNotFound(id);
            if (string.IsNullOrWhiteSpace(req.Prompt))
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "A prompt must not be empty.", "invalid_request_error");
            try
            {
                session.PushPrompt(req.Prompt);
                return Results.Ok(new { accepted = true });
            }
            catch (ObjectDisposedException)
            {
                return SessionNotFound(id);
            }
        });

        app.MapGet($"{Base}/{{id}}", (string id, RealtimeSessionRegistry registry) =>
            registry.Get(id) is { } session ? Results.Ok(session.Stats) : SessionNotFound(id));

        app.MapGet($"{Base}/{{id}}/stream", async (string id, RealtimeSessionRegistry registry, HttpContext ctx, CancellationToken ct) =>
        {
            IRealtimeVlmSession? session = registry.Get(id);
            if (session is null)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(ctx, StatusCodes.Status404NotFound, NotFoundMessage(id), "invalid_request_error");
                return;
            }
            // The session's events are consumed, so two readers would each get an arbitrary share of the tokens.
            if (!ActiveStreams.TryAdd(id, 0))
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(
                    ctx, StatusCodes.Status409Conflict, $"Realtime session '{id}' already has a stream connected; close it before opening another.",
                    "invalid_request_error");
                return;
            }
            JsonSerializerOptions json = SseHelpers.ResolveJsonOptions(ctx);
            ctx.Response.Headers.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            // Cancelled on every exit so a read still pending can finish before the enumerator is disposed.
            using CancellationTokenSource streamCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            IAsyncEnumerator<RealtimeVlmChunk> events = session.StreamAsync(streamCts.Token).GetAsyncEnumerator(streamCts.Token);
            Task<bool>? next = null;
            try
            {
                next = events.MoveNextAsync().AsTask();
                while (true)
                {
                    // A connected stream keeps the session alive; a silent model still gets a keep-alive comment.
                    using CancellationTokenSource tick = CancellationTokenSource.CreateLinkedTokenSource(streamCts.Token);
                    Task done = await Task.WhenAny(next, Task.Delay(KeepAlive, tick.Token));
                    tick.Cancel();   // stop the timer when the event won, so a chatty model leaves none running
                    registry.Touch(id);
                    if (done != next)
                    {
                        await ctx.Response.WriteAsync(": keep-alive\n\n", ct);
                        await ctx.Response.Body.FlushAsync(ct);
                        continue;
                    }
                    if (!await next) break;
                    RealtimeVlmChunk chunk = events.Current;
                    await ctx.Response.WriteAsync(Format(chunk, json), ct);
                    await ctx.Response.Body.FlushAsync(ct);
                    if (chunk.Kind == RealtimeVlmChunkKind.Error)
                    {
                        // The loop has ended and the session can take no more input: free its device slot now instead of holding
                        // it until the client closes or the idle sweep finds it.
                        await registry.CloseAsync(id);
                        break;
                    }
                    next = events.MoveNextAsync().AsTask();
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // The client disconnected; the session stays open for a reconnect until it is closed or idles out.
            }
            catch (Exception ex)
            {
                await ctx.Response.WriteAsync(SseHelpers.ErrorEvent(ex.Message, json), CancellationToken.None);
                await ctx.Response.Body.FlushAsync(CancellationToken.None);
            }
            finally
            {
                streamCts.Cancel();
                if (next is not null)
                {
                    try { await next; }
                    catch (Exception) { /* the read ended with the cancellation or the session's end */ }
                }
                await events.DisposeAsync();
                ActiveStreams.TryRemove(id, out _);
            }
        });

        app.MapDelete($"{Base}/{{id}}", async (string id, RealtimeSessionRegistry registry) =>
            await registry.CloseAsync(id) ? Results.Ok(new { closed = id }) : SessionNotFound(id));
    }

    /// <summary>Sessions with a stream connected, so a second reader is refused.</summary>
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> ActiveStreams = new(StringComparer.Ordinal);

    /// <summary>One session event as an SSE frame: <c>round_start</c>, <c>text</c>, <c>silence</c>, <c>control</c>,
    /// <c>frames_dropped</c> or <c>error</c>.</summary>
    internal static string Format(RealtimeVlmChunk chunk, JsonSerializerOptions json) => chunk.Kind switch
    {
        RealtimeVlmChunkKind.RoundStart => SseHelpers.Event("round_start", new { prompt = chunk.Text }, json),
        RealtimeVlmChunkKind.Text => SseHelpers.Event("text", new { text = chunk.Text }, json),
        RealtimeVlmChunkKind.Silence => SseHelpers.Event("silence", new { token = chunk.Text }, json),
        RealtimeVlmChunkKind.Control => SseHelpers.Event("control", new { token = chunk.Text }, json),
        RealtimeVlmChunkKind.FramesDropped => SseHelpers.Event("frames_dropped", new { count = chunk.Count }, json),
        _ => SseHelpers.ErrorEvent(chunk.Text, json),
    };

    /// <summary>A bare base64 PNG as a data URI, so one decoder (with its size limits) serves both spellings.</summary>
    private static string AsDataUri(string image) =>
        image.StartsWith("data:", StringComparison.OrdinalIgnoreCase) ? image : "data:image/png;base64," + image;

    private static string NotFoundMessage(string id) => $"Realtime session '{id}' is not open (never existed, closed, or idle-timed-out).";

    private static IResult SessionNotFound(string id) =>
        HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, NotFoundMessage(id), "invalid_request_error");
}
