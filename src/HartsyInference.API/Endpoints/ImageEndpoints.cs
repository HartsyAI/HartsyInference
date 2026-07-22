using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native image-generation routes: a byte-for-byte pass-through of <see cref="ImageRequest"/> (LoRA/
/// ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional composition included) to <see cref="ImageResult"/>.</summary>
public static class ImageEndpoints
{
    /// <summary>Maps <c>/v1/native/images</c> and its SSE step-preview variant.</summary>
    public static void MapImageEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/images", async (NativeImageRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Image);
            try
            {
                ImageResult result = await GenerateAsync(engine, queue, spec, req.Request, progress: null, ct);
                return Results.Ok(ToResponse(result));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/images/stream", async (NativeImageRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Image);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs INSIDE the queue's held slot (SseHelpers wraps this whole delegate in EnqueueAsync) — call
                // the service directly here, never the GenerateAsync helper below, or the second EnqueueAsync
                // would deadlock waiting on the semaphore this frame already holds.
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", new { step = p.Step, total = p.TotalSteps }, jsonOptions)));
                ImageResult result = await engine.Images.GenerateAsync(spec, req.Request, progress, ct);
                writer.TryWrite(SseHelpers.Event("complete", ToResponse(result), jsonOptions));
            }, ct);
        });
    }

    /// <summary>Queue-gated generation, shared with the OpenAI-compat <c>/v1/images/generations</c> wrapper.</summary>
    internal static Task<ImageResult> GenerateAsync(
        IInferenceEngine engine, InferenceQueue queue, ModelSpec spec, ImageRequest request,
        IProgress<StepPreview>? progress, CancellationToken ct) =>
        queue.EnqueueAsync(() => engine.Images.GenerateAsync(spec, request, progress, ct), ct);

    /// <summary>PNG-encodes the raw RGB result for HTTP transport (base64 JSON — the native contract carries no
    /// codec of its own).</summary>
    internal static object ToResponse(ImageResult result) => new
    {
        png = Convert.ToBase64String(PngEncoder.Encode(result.Rgb, result.Width, result.Height)),
        result.Width,
        result.Height,
        result.Seed,
        result.Meta,
    };
}
