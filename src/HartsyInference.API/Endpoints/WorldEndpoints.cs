using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>Native interactive-world session routes. The genuinely novel piece of Phase 5: <see cref="IWorldSession"/>
/// is stateful and long-lived (open → repeatedly action/stream → close) instead of every other route's one-request-
/// in-one-response-out shape, so an HTTP client needs a way to find the same session again across separate
/// requests — that's what <see cref="WorldSessionRegistry"/> is for.
///
/// <para><b>Scope boundary, stated plainly:</b> opening a session (which loads a model) is gated through the
/// long-running queue as a bounded operation, same as every other queue-gated call. Once open, the session's
/// ongoing action/stream traffic does NOT hold a queue slot for its whole lifetime — arbitrating GPU contention
/// across multiple simultaneously-open sessions is left to <see cref="IWorldSession"/>'s own implementation, not
/// enforced at this HTTP layer. This is a deliberate v1 scope cut, not an oversight.</para></summary>
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
                IWorldSession session = await queue.EnqueueAsync(() => Task.FromResult(engine.World.Open(spec, req.Request)), ct);
                return Results.Ok(new { sessionId = registry.Register(session) });
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
            session.SendAction(req.Action);
            return Results.Ok(new { accepted = true });
        });

        app.MapGet("/v1/native/world/sessions/{id}/stream", async (string id, WorldSessionRegistry registry, HttpContext ctx, CancellationToken ct) =>
        {
            IWorldSession? session = registry.Get(id);
            if (session is null)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(
                    ctx, StatusCodes.Status404NotFound, $"World session '{id}' is not open (never existed, closed, or idle-timed-out).", "invalid_request_error");
                return;
            }

            // NOT routed through a queue: Open() already reserved this session's resources under the
            // long-running gate. Streaming its already-resident frames doesn't compete for a NEW slot the way
            // constructing a session or a fresh one-shot generation call does — see this file's class doc.
            JsonSerializerOptions jsonOptions = SseHelpers.ResolveJsonOptions(ctx);
            ctx.Response.Headers.ContentType = "text/event-stream";
            try
            {
                await foreach (VideoFrame frame in session.StreamAsync(ct))
                {
                    string evt = SseHelpers.Event("frame", new
                    {
                        index = frame.Index,
                        png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                    }, jsonOptions);
                    await ctx.Response.WriteAsync(evt, ct);
                    await ctx.Response.Body.FlushAsync(ct);
                }
            }
            catch (Exception ex)
            {
                await ctx.Response.WriteAsync(SseHelpers.ErrorEvent(ex.Message, jsonOptions), ct);
                await ctx.Response.Body.FlushAsync(ct);
            }
        });

        app.MapDelete("/v1/native/world/sessions/{id}", (string id, WorldSessionRegistry registry) =>
            registry.Close(id) ? Results.Ok(new { closed = id }) : SessionNotFound(id));
    }

    private static IResult SessionNotFound(string id) => HartsyInferenceServiceExtensions.Problem(
        StatusCodes.Status404NotFound, $"World session '{id}' is not open (never existed, closed, or idle-timed-out).", "invalid_request_error");
}
