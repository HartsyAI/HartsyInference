using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API.Endpoints;

/// <summary>Native restore routes (SeedVR2). The streaming variant mirrors <see cref="VideoEndpoints"/>
/// (SSE <c>progress</c>/<c>frame</c>/<c>complete</c>/<c>error</c>, frames as base64 PNG); the non-streaming
/// variant buffers the restored clip and returns one JSON array — fine for the short clips restoration
/// typically handles, and the payload-size tradeoff is the caller's. Both run through the long-running
/// queue: a restore is one DiT step but the whole-clip VAE work still takes minutes at 720p-area.</summary>
public static class RestoreEndpoints
{
    /// <summary>Maps <c>/v1/native/restore</c> and <c>/v1/native/restore/stream</c>.</summary>
    public static void MapRestoreEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/restore/stream", async (
            NativeRestoreRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Restore);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs inside the queue's held slot — call the service directly, not through a second EnqueueAsync.
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", new { step = p.Step, total = p.TotalSteps }, jsonOptions)));
                int frameCount = 0;
                await foreach (VideoFrame frame in engine.Restore.RestoreAsync(spec, req.Request, progress, ct))
                {
                    writer.TryWrite(SseHelpers.Event("frame", new
                    {
                        index = frame.Index,
                        png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                    }, jsonOptions));
                    frameCount++;
                }
                writer.TryWrite(SseHelpers.Event("complete", new { frames = frameCount }, jsonOptions));
            }, ct);
        });

        app.MapPost("/v1/native/restore", async (
            NativeRestoreRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Restore);
                List<object> frames = await queue.EnqueueAsync(async () =>
                {
                    List<object> collected = new List<object>();
                    await foreach (VideoFrame frame in engine.Restore.RestoreAsync(spec, req.Request, null, ct))
                    {
                        collected.Add(new
                        {
                            index = frame.Index,
                            width = frame.Width,
                            height = frame.Height,
                            png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                        });
                    }
                    return collected;
                }, ct);
                return Results.Ok(new { frames });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }
}
