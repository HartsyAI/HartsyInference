using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>Native interactive-world session routes. The genuinely novel piece of Phase 5: <see cref="IWorldSession"/> is stateful and long-lived (open → repeatedly action/stream → close) instead of every other route's one-request-in-one-response-out shape, so an HTTP client needs a way to find the same session again across separate requests — that's what <see cref="WorldSessionRegistry"/> is for. <para><b>Scope boundary, stated plainly:</b> opening a session (which loads a model) is gated through the long-running queue as a bounded operation, same as every other queue-gated call. Once open, an interactive session's ongoing action/stream traffic does NOT hold a queue slot for its whole lifetime — arbitrating GPU contention across multiple simultaneously-open sessions is left to <see cref="IWorldSession"/>'s own implementation, not enforced at this HTTP layer. This is a deliberate v1 scope cut, not an oversight. The exception is a session that <see cref="IWorldSession.DefersDeviceWork"/>: its Open reserves nothing, so its stream does the whole job (load, sample, decode) and holds a long-running slot until it ends.</para></summary>
public static class WorldEndpoints
{
    /// <summary>Maps session open/action/stream/close.</summary>
    public static void MapWorldEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/world/sessions", async (
            NativeWorldRequest req, IInferenceEngine engine, WorldSessionRegistry registry,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.World);
            try
            {
                // Planning refuses a request the model cannot serve before the queue slot is taken and before any weight loads.
                WorldPlan plan = WorldPlanResolver.Resolve(spec, req.Request, engine.BackendSelector);
                IWorldSession session = await queue.EnqueueAsync(() => Task.FromResult(engine.World.Open(spec, plan.Request)), ct);
                return Results.Ok(new { sessionId = registry.Register(session, plan.Fps), frames = plan.Frames, fps = plan.Fps, seed = plan.Seed });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/world/sessions/{id}/action", (string id, WorldActionRequest req, WorldSessionRegistry registry) =>
        {
            IWorldSession? session = registry.Get(id);
            if (session is null)
                return SessionNotFound(id);
            try
            {
                session.SendAction(req.Action);
            }
            catch (ObjectDisposedException)
            {
                // The idle sweep or a concurrent DELETE closed the session between Get and SendAction.
                return SessionNotFound(id);
            }
            catch (InvalidOperationException ex)
            {
                // A batch session has already started its run: a late action conflicts with the session's state.
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status409Conflict, ex.Message, "invalid_request_error");
            }
            return Results.Ok(new { accepted = true });
        });

        app.MapGet("/v1/native/world/sessions/{id}/stream", async (
            string id, WorldSessionRegistry registry, [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue,
            HttpContext ctx, CancellationToken ct) =>
        {
            IWorldSession? session = registry.Get(id);
            int? fps = registry.FrameRate(id);
            if (session is null || fps is null)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(
                    ctx, StatusCodes.Status404NotFound, $"World session '{id}' is not open (never existed, closed, or idle-timed-out).", "invalid_request_error");
                return;
            }

            JsonSerializerOptions jsonOptions = SseHelpers.ResolveJsonOptions(ctx);
            string FrameEvent(VideoFrame frame) => SseHelpers.Event("frame", new
            {
                index = frame.Index,
                timestampMs = WorldPlan.FrameTimestampMs(frame.Index, fps.Value),
                png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
            }, jsonOptions);

            // Every frame is written and flushed before the next is pulled from the session, so a slow client holds the model back
            // (a chunked session blocks when its consumer does) instead of frames piling up in memory.
            async Task<bool> StreamFramesAsync()
            {
                ctx.Response.Headers.ContentType = "text/event-stream";
                try
                {
                    await foreach (VideoFrame frame in session.StreamAsync(ct))
                    {
                        await ctx.Response.WriteAsync(FrameEvent(frame), ct);
                        await ctx.Response.Body.FlushAsync(ct);
                    }
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    // The client disconnected: nothing to send, and the cancelled token must not be used to say so.
                }
                catch (Exception ex)
                {
                    Logs.Error($"[World] Session {id} failed while streaming: {ex}");
                    await ctx.Response.WriteAsync(SseHelpers.ErrorEvent(ex.Message, jsonOptions), CancellationToken.None);
                    await ctx.Response.Body.FlushAsync(CancellationToken.None);
                }
                return true;
            }

            if (!session.DefersDeviceWork)
            {
                // An interactive session reserved its resources under the long-running gate when it opened, so streaming does not take
                // a new slot (see this class's doc).
                await StreamFramesAsync();
                return;
            }

            // Open reserved nothing for this session: this stream loads the model and runs it, so it takes a long-running slot like any
            // other generation. It waits its turn before the response starts, and a full queue is a 429.
            try
            {
                await queue.EnqueueAsync(StreamFramesAsync, ct);
            }
            catch (QueueFullException ex)
            {
                ctx.Response.Headers["Retry-After"] = "1";
                await HartsyInferenceServiceExtensions.WriteErrorAsync(ctx, StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error");
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                // The client left while waiting for a slot.
            }
        });

        app.MapDelete("/v1/native/world/sessions/{id}", (string id, WorldSessionRegistry registry) =>
            registry.Close(id) ? Results.Ok(new { closed = id }) : SessionNotFound(id));
    }

    private static IResult SessionNotFound(string id) => HartsyInferenceServiceExtensions.Problem(
        StatusCodes.Status404NotFound, $"World session '{id}' is not open (never existed, closed, or idle-timed-out).", "invalid_request_error");
}
