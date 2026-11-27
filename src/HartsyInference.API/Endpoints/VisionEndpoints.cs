using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native vision routes. <c>/v1/native/vision</c> runs one image in any single-image mode
/// (<see cref="VisionRequest.Mode"/> selects which). <c>/v1/native/vision/sequence</c> runs a
/// <see cref="VisionBatchRequest"/>: views of one scene come back as one JSON result, video frames stream as SSE. Both
/// validate with <see cref="VisionRequestValidator"/> first, so a malformed request is a 400 before any model loads, and
/// a mode no model implements is a 501. Response shape: <see cref="VisionWire"/>.</summary>
public static class VisionEndpoints
{
    /// <summary>Maps <c>/v1/native/vision</c> and <c>/v1/native/vision/sequence</c>.</summary>
    public static void MapVisionEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/vision", async (NativeVisionRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            try
            {
                VisionRequestValidator.Validate(req.Request);
                ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Vision);
                VisionResult result = await queue.EnqueueAsync(() => engine.Vision.RunAsync(spec, req.Request, ct), ct);
                return Results.Ok(VisionWire.Result(result));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/vision/sequence", async Task<IResult> (
            NativeVisionSequenceRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec;
            try
            {
                VisionRequestValidator.Validate(req.Request);
                spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Vision);
                if (req.Request.Kind == VisionSequenceKind.Views)
                {
                    VisionResult views = await queue.EnqueueAsync(() => engine.Vision.RunViewsAsync(spec, req.Request, ct), ct);
                    return Results.Ok(VisionWire.Result(views));
                }
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }

            // Frames stream. Asking for the stream before SSE opens surfaces a refusal (no model, wrong kind) as a
            // proper status; errors once frames are flowing arrive as an `error` event.
            IAsyncEnumerable<VisionFrameResult> stream;
            try
            {
                stream = engine.Vision.RunFramesAsync(spec, req.Request, ct);
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                int count = 0;
                await foreach (VisionFrameResult frame in stream.WithCancellation(ct))
                {
                    await writer.WriteAsync(SseHelpers.Event("frame", new NativeVisionFrameEvent
                    {
                        Index = frame.Index,
                        TimestampSeconds = frame.TimestampSeconds,
                        SourceIndex = frame.SourceIndex,
                        Result = VisionWire.Result(frame.Result),
                    }, jsonOptions), ct);
                    count++;
                }
                await writer.WriteAsync(SseHelpers.Event("complete", new NativeVisionSequenceCompleteEvent { Frames = count }, jsonOptions), ct);
            }, ct);
            return Results.Empty;
        })
            .Produces(StatusCodes.Status200OK, contentType: "text/event-stream");
    }
}
