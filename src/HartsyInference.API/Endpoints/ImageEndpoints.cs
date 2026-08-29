using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native image-generation routes: a byte-for-byte pass-through of <see cref="ImageRequest"/> (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional composition included) to <see cref="ImageResult"/>.</summary>
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
                ImageResult result = await GenerateAsync(engine, queue, spec, ApplyImageComposition(req), progress: null, ct);
                return Results.Ok(ToResponse(result, Persist(req, result, req.Request.Prompt)));
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
                    writer.TryWrite(SseHelpers.Event("progress", StepPreviewPayload.Create(p), jsonOptions)));
                ImageResult result = await engine.Images.GenerateAsync(spec, ApplyImageComposition(req), progress, ct);
                writer.TryWrite(SseHelpers.Event("complete", ToResponse(result, Persist(req, result, req.Request.Prompt)), jsonOptions));
            }, ct);
        });
    }

    /// <summary>Folds the envelope's base64 init image / mask conveniences into the native request. Returns the request untouched when none are set, so a client that populates <c>request.img2img</c> with raw RGB24 directly keeps working.</summary>
    public static ImageRequest ApplyImageComposition(NativeImageRequest req)
    {
        ImageRequest request = req.Request;
        if (req.InitImageBase64 is not null)
        {
            Img2Img img2img = new Img2Img { InitImage = ImageDataCodec.DecodeBase64(req.InitImageBase64) };
            request = request with { Img2Img = req.Creativity is null ? img2img : img2img with { Creativity = req.Creativity.Value } };
        }
        else if (req.Creativity is not null && request.Img2Img is not null)
        {
            request = request with { Img2Img = request.Img2Img with { Creativity = req.Creativity.Value } };
        }

        if (req.MaskBase64 is not null)
        {
            Inpaint inpaint = new Inpaint { Mask = ImageDataCodec.DecodeBase64(req.MaskBase64) };
            if (req.MaskGrow is not null) { inpaint = inpaint with { Grow = req.MaskGrow.Value }; }
            if (req.MaskBlur is not null) { inpaint = inpaint with { Blur = req.MaskBlur.Value }; }
            if (req.MaskShrinkGrow is not null) { inpaint = inpaint with { ShrinkGrow = req.MaskShrinkGrow.Value }; }
            request = request with { Inpaint = inpaint };
        }
        return request;
    }

    /// <summary>Queue-gated generation, shared with the OpenAI-compat <c>/v1/images/generations</c> wrapper.</summary>
    internal static Task<ImageResult> GenerateAsync(
        IInferenceEngine engine, InferenceQueue queue, ModelSpec spec, ImageRequest request,
        IProgress<StepPreview>? progress, CancellationToken ct) =>
        queue.EnqueueAsync(() => engine.Images.GenerateAsync(spec, request, progress, ct), ct);

    /// <summary>PNG-encodes the raw RGB result for HTTP transport (base64 JSON — the native contract carries no codec of its own), and reports where it was saved.</summary>
    internal static object ToResponse(ImageResult result, string? savedPath = null) => new
    {
        png = Convert.ToBase64String(PngEncoder.Encode(result.Rgb, result.Width, result.Height)),
        result.Width,
        result.Height,
        result.Seed,
        result.Meta,
        savedPath,
    };

    /// <summary>Persists the PNG unless the request opted out; null when nothing was written.</summary>
    internal static string? Persist(NativeArtifactRequest req, ImageResult result, string slugSource) =>
        ArtifactPersistence.Save(req, PngEncoder.Encode(result.Rgb, result.Width, result.Height), slugSource, "png");
}
