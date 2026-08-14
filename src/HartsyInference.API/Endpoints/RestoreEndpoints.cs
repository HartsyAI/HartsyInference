using HartsyInference.Core.Logging;
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
                List<VideoFrame> collected = new List<VideoFrame>();
                await foreach (VideoFrame frame in engine.Restore.RestoreAsync(spec, req.Request, progress, ct))
                {
                    writer.TryWrite(SseHelpers.Event("frame", new
                    {
                        index = frame.Index,
                        png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                    }, jsonOptions));
                    // Only retained when the caller wants the sequence on disk — restore clips are large.
                    if (req.Save != false) { collected.Add(frame); }
                    frameCount++;
                }
                writer.TryWrite(SseHelpers.Event("complete", new
                {
                    frames = frameCount,
                    savedPath = PersistFrames(req, collected),
                }, jsonOptions));
            }, ct);
        });

        app.MapPost("/v1/native/restore", async (
            NativeRestoreRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Restore);
                List<VideoFrame> raw = new List<VideoFrame>();
                List<object> frames = await queue.EnqueueAsync(async () =>
                {
                    List<object> collected = new List<object>();
                    await foreach (VideoFrame frame in engine.Restore.RestoreAsync(spec, req.Request, null, ct))
                    {
                        raw.Add(frame);
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
                return Results.Ok(new { frames, savedPath = PersistFrames(req, raw) });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }

    /// <summary>Writes a restored frame sequence into a numbered directory under the output root, matching the
    /// CLI's layout. Null when the request opted out, nothing was produced, or the write failed.</summary>
    private static string? PersistFrames(NativeRestoreRequest req, IReadOnlyList<VideoFrame> frames)
    {
        if (req.Save == false || frames.Count == 0)
            return null;
        try
        {
            byte[][] rgb = new byte[frames.Count][];
            for (int i = 0; i < frames.Count; i++)
                rgb[i] = frames[i].Rgb;
            return VideoOutputWriter.Write(rgb, frames[0].Width, frames[0].Height,
                OutputWriter.ResolveDir(req.OutputDir), "restored", audio: null,
                req.Request.Fps is { } f ? (int)Math.Round(f) : 24).Directory;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[api] restored frames could not be saved: {ex.Message}");
            return null;
        }
    }
}
